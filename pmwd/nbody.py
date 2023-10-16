from functools import partial

import jax
from jax import value_and_grad, jit, vjp, custom_vjp
import jax.numpy as jnp
from jax.tree_util import tree_map
from jax.lax import cond, fori_loop, scan

from pmwd.boltzmann import growth
from pmwd.cosmology import E2, H_deriv
from pmwd.gravity import gravity
from pmwd.obs_util import interptcl, itp_prev_adj, itp_next_adj
from pmwd.particles import Particles


def _G_D(a, cosmo, conf):
    """Growth factor of ZA canonical velocity in [H_0]."""
    return a**2 * jnp.sqrt(E2(a, cosmo)) * growth(a, cosmo, conf, deriv=1)


def _G_K(a, cosmo, conf):
    """Growth factor of ZA accelerations in [H_0^2]."""
    return a**3 * E2(a, cosmo) * (
        growth(a, cosmo, conf, deriv=2)
        + (2 + H_deriv(a, cosmo)) * growth(a, cosmo, conf, deriv=1)
    )


def drift_factor(a_vel, a_prev, a_next, cosmo, conf):
    """Drift time step factor of conf.float_dtype in [1/H_0]."""
    factor = growth(a_next, cosmo, conf) - growth(a_prev, cosmo, conf)
    factor /= _G_D(a_vel, cosmo, conf)
    return factor


def kick_factor(a_acc, a_prev, a_next, cosmo, conf):
    """Kick time step factor of conf.float_dtype in [1/H_0]."""
    factor = _G_D(a_next, cosmo, conf) - _G_D(a_prev, cosmo, conf)
    factor /= _G_K(a_acc, cosmo, conf)
    return factor


def drift(a_vel, a_prev, a_next, ptcl, cosmo, conf):
    """Drift."""
    factor = drift_factor(a_vel, a_prev, a_next, cosmo, conf)
    factor = factor.astype(conf.float_dtype)

    disp = ptcl.disp + ptcl.vel * factor

    return ptcl.replace(disp=disp)


def drift_adj(a_vel, a_prev, a_next, ptcl, ptcl_cot, cosmo, cosmo_cot, conf):
    """Drift, and particle and cosmology adjoints."""
    factor_valgrad = value_and_grad(drift_factor, argnums=3)
    factor, cosmo_cot_drift = factor_valgrad(a_vel, a_prev, a_next, cosmo, conf)
    factor = factor.astype(conf.float_dtype)

    # drift
    disp = ptcl.disp + ptcl.vel * factor
    ptcl = ptcl.replace(disp=disp)

    # particle adjoint
    vel_cot = ptcl_cot.vel - ptcl_cot.disp * factor
    ptcl_cot = ptcl_cot.replace(vel=vel_cot)

    # cosmology adjoint
    cosmo_cot_drift *= (ptcl_cot.disp * ptcl.vel).sum()
    cosmo_cot -= cosmo_cot_drift

    return ptcl, ptcl_cot, cosmo_cot


def kick(a_acc, a_prev, a_next, ptcl, cosmo, conf):
    """Kick."""
    factor = kick_factor(a_acc, a_prev, a_next, cosmo, conf)
    factor = factor.astype(conf.float_dtype)

    vel = ptcl.vel + ptcl.acc * factor

    return ptcl.replace(vel=vel)


def kick_adj(a_acc, a_prev, a_next, ptcl, ptcl_cot, cosmo, cosmo_cot, cosmo_cot_force, conf):
    """Kick, and particle and cosmology adjoints."""
    factor_valgrad = value_and_grad(kick_factor, argnums=3)
    factor, cosmo_cot_kick = factor_valgrad(a_acc, a_prev, a_next, cosmo, conf)
    factor = factor.astype(conf.float_dtype)

    # kick
    vel = ptcl.vel + ptcl.acc * factor
    ptcl = ptcl.replace(vel=vel)

    # particle adjoint
    disp_cot = ptcl_cot.disp - ptcl_cot.acc * factor
    ptcl_cot = ptcl_cot.replace(disp=disp_cot)

    # cosmology adjoint
    cosmo_cot_kick *= (ptcl_cot.vel * ptcl.acc).sum()
    cosmo_cot_force *= factor
    cosmo_cot -= cosmo_cot_kick + cosmo_cot_force

    return ptcl, ptcl_cot, cosmo_cot


def force(a, ptcl, cosmo, conf):
    """Force."""
    acc = gravity(a, ptcl, cosmo, conf)
    return ptcl.replace(acc=acc)


def force_adj(a, ptcl, ptcl_cot, cosmo, conf):
    """Force, and particle and cosmology vjp."""
    # force
    acc, gravity_vjp = vjp(gravity, a, ptcl, cosmo, conf)
    ptcl = ptcl.replace(acc=acc)

    # particle and cosmology vjp
    _, ptcl_cot_force, cosmo_cot_force, _ = gravity_vjp(ptcl_cot.vel)
    ptcl_cot = ptcl_cot.replace(acc=ptcl_cot_force.disp)

    return ptcl, ptcl_cot, cosmo_cot_force


def integrate(a_prev, a_next, ptcl, cosmo, conf):
    """Symplectic integration for one step."""
    D = K = 0
    a_disp = a_vel = a_acc = a_prev
    for d, k in conf.symp_splits:
        if d != 0:
            D += d
            a_disp_next = a_prev * (1 - D) + a_next * D
            ptcl = drift(a_vel, a_disp, a_disp_next, ptcl, cosmo, conf)
            a_disp = a_disp_next
            ptcl = force(a_disp, ptcl, cosmo, conf)
            a_acc = a_disp

        if k != 0:
            K += k
            a_vel_next = a_prev * (1 - K) + a_next * K
            ptcl = kick(a_acc, a_vel, a_vel_next, ptcl, cosmo, conf)
            a_vel = a_vel_next

    return ptcl


def integrate_adj(a_prev, a_next, ptcl, ptcl_cot, obsvbl_cot, cosmo, cosmo_cot, cosmo_cot_force, conf):
    """Symplectic integration adjoint for one step."""
    K = D = 0
    a_disp = a_vel = a_acc = a_prev
    for d, k in reversed(conf.symp_splits):
        if k != 0:
            K += k
            a_vel_next = a_prev * (1 - K) + a_next * K
            ptcl, ptcl_cot, cosmo_cot = kick_adj(a_acc, a_vel, a_vel_next, ptcl, ptcl_cot, cosmo, cosmo_cot, cosmo_cot_force, conf)
            a_vel = a_vel_next

        if d != 0:
            D += d
            a_disp_next = a_prev * (1 - D) + a_next * D
            ptcl, ptcl_cot, cosmo_cot = drift_adj(a_vel, a_disp, a_disp_next, ptcl, ptcl_cot, cosmo, cosmo_cot, conf)
            a_disp = a_disp_next
            ptcl, ptcl_cot, cosmo_cot_force = force_adj(a_disp, ptcl, ptcl_cot, cosmo, conf)
            a_acc = a_disp

    return ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force


def form(a_prev, a_next, ptcl, cosmo, conf):
    pass


def form_init(a, ptcl, cosmo, conf):
    pass  # TODO necessary?


def coevolve(a_prev, a_next, ptcl, cosmo, conf):
    attr = form(a_prev, a_next, ptcl, cosmo, conf)
    return ptcl.replace(attr=attr)


def coevolve_init(a, ptcl, cosmo, conf):
    if ptcl.attr is None:
        attr = form_init(a, ptcl, cosmo, conf)
        ptcl = ptcl.replace(attr=attr)
    return ptcl


def observe(a_prev, a_next, ptcl, obsvbl, cosmo, conf):
    def itp(a_snap, snap):
        snap_itp = interptcl(obsvbl['ptcl_prev'], ptcl, a_prev, a_next, a_snap, cosmo)
        return snap_itp

    def f_cond(carry, x):
        a_snap, snap = x
        y = cond(jnp.logical_and(a_prev < a_snap, a_snap <= a_next),
                 itp, lambda *args: snap, a_snap, snap)
        return None, y

    carry, obsvbl['snaps'] = scan(f_cond, None, (obsvbl['a_snaps'], obsvbl['snaps']))

    obsvbl['ptcl_prev'] = ptcl

    return obsvbl


def observe_init(a, ptcl, obsvbl, cosmo, conf):
    # a dict to carry all observables and related useful information
    obsvbl = {}

    # to carry the prev ptcl, starting with lpt ptcl
    obsvbl['ptcl_prev'] = ptcl

    if conf.a_snapshots is not None:
        obsvbl['a_snaps'] = jnp.array(conf.a_snapshots)
        # all output snapshots, at times given by conf.a_snapshots
        obsvbl['snaps'] = [Particles(ptcl.conf, ptcl.pmid, jnp.zeros_like(ptcl.disp),
                           vel=jnp.zeros_like(ptcl.vel))] * len(conf.a_snapshots)
        # transposed pytree with leading axis for scan
        obsvbl['snaps'] = tree_map(lambda *xs: jnp.stack(xs), *obsvbl['snaps'])

        # the nbody a step of output snapshots, (,]
        idx = jnp.searchsorted(conf.a_nbody, jnp.asarray(conf.a_snapshots), side='left')
        obsvbl['snap_a_step'] = jnp.array((conf.a_nbody[idx-1], conf.a_nbody[idx])).T

    return obsvbl


def observe_adj(a_prev, a_next, ptcl, ptcl_cot, obsvbl, obsvbl_cot, cosmo, cosmo_cot, conf):

    if conf.a_snapshots is not None:
        for a_snap, a_step in zip(conf.a_snapshots, obsvbl['snap_a_step']):
            ptcl_cot, cosmo_cot = cond(a_step[1] == a_next, itp_next_adj,
                                       lambda *args: (ptcl_cot, cosmo_cot),
                                       ptcl_cot, cosmo_cot, obsvbl_cot['snapshots'][a_snap],
                                       ptcl, a_step[0], a_step[1], a_snap, cosmo)
            ptcl_cot, cosmo_cot = cond(a_step[1] == a_prev, itp_prev_adj,
                                       lambda *args: (ptcl_cot, cosmo_cot),
                                       ptcl_cot, cosmo_cot, obsvbl_cot['snapshots'][a_snap],
                                       ptcl, a_step[0], a_step[1], a_snap, cosmo)

    return ptcl_cot, cosmo_cot


def observe_adj_init(a, ptcl, ptcl_cot, obsvbl, obsvbl_cot, cosmo, cosmo_cot, conf):

    if conf.a_snapshots is not None:
        # check if the last ptcl is used in interpolation
        for a_snap, a_step in zip(conf.a_snapshots, obsvbl['snap_a_step']):
            ptcl_cot, cosmo_cot = cond(a_step[1] == a, itp_next_adj,
                                       lambda *args: (ptcl_cot, cosmo_cot),
                                       ptcl_cot, cosmo_cot, obsvbl_cot['snapshots'][a_snap],
                                       ptcl, a_step[0], a_step[1], a_snap, cosmo)

    return ptcl_cot, cosmo_cot


@jit
def nbody_init(a, ptcl, obsvbl, cosmo, conf):
    ptcl = force(a, ptcl, cosmo, conf)

    ptcl = coevolve_init(a, ptcl, cosmo, conf)

    obsvbl = observe_init(a, ptcl, obsvbl, cosmo, conf)

    return ptcl, obsvbl


@jit
def nbody_step(a_prev, a_next, ptcl, obsvbl, cosmo, conf):
    ptcl = integrate(a_prev, a_next, ptcl, cosmo, conf)

    ptcl = coevolve(a_prev, a_next, ptcl, cosmo, conf)

    obsvbl = observe(a_prev, a_next, ptcl, obsvbl, cosmo, conf)

    return ptcl, obsvbl


@partial(custom_vjp, nondiff_argnums=(4,))
def nbody(ptcl, obsvbl, cosmo, conf, reverse=False):
    """N-body time integration."""
    a_nbody = conf.a_nbody[::-1] if reverse else conf.a_nbody

    ptcl, obsvbl = nbody_init(a_nbody[0], ptcl, obsvbl, cosmo, conf)
    for a_prev, a_next in zip(a_nbody[:-1], a_nbody[1:]):
        ptcl, obsvbl = nbody_step(a_prev, a_next, ptcl, obsvbl, cosmo, conf)
    return ptcl, obsvbl


@jit
def nbody_adj_init(a, ptcl, ptcl_cot, obsvbl, obsvbl_cot, cosmo, conf):

    #ptcl, ptcl_cot = coevolve_adj(a_prev, a_next, ptcl, ptcl_cot, cosmo)

    ptcl, ptcl_cot, cosmo_cot_force = force_adj(a, ptcl, ptcl_cot, cosmo, conf)

    cosmo_cot = tree_map(jnp.zeros_like, cosmo)

    ptcl_cot, cosmo_cot = observe_adj_init(a, ptcl, ptcl_cot, obsvbl, obsvbl_cot,
                                           cosmo, cosmo_cot, conf)

    return ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force


@jit
def nbody_adj_step(a_prev, a_next, ptcl, ptcl_cot, obsvbl, obsvbl_cot,
                   cosmo, cosmo_cot, cosmo_cot_force, conf):

    #ptcl, ptcl_cot = coevolve_adj(a_prev, a_next, ptcl, ptcl_cot, cosmo, conf)

    ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force = integrate_adj(
        a_prev, a_next, ptcl, ptcl_cot, obsvbl_cot, cosmo, cosmo_cot, cosmo_cot_force, conf)

    ptcl_cot, cosmo_cot = observe_adj(a_prev, a_next, ptcl, ptcl_cot, obsvbl, obsvbl_cot,
                                      cosmo, cosmo_cot, conf)

    return ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force


def nbody_adj(ptcl, ptcl_cot, obsvbl, obsvbl_cot, cosmo, conf, reverse=False):
    """N-body time integration with adjoint equation."""
    a_nbody = conf.a_nbody[::-1] if reverse else conf.a_nbody

    ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force = nbody_adj_init(
        a_nbody[-1], ptcl, ptcl_cot, obsvbl, obsvbl_cot, cosmo, conf)

    for a_prev, a_next in zip(a_nbody[:0:-1], a_nbody[-2::-1]):
        ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force = nbody_adj_step(
            a_prev, a_next, ptcl, ptcl_cot, obsvbl, obsvbl_cot,
            cosmo, cosmo_cot, cosmo_cot_force, conf)

    return ptcl, ptcl_cot, cosmo_cot


def nbody_fwd(ptcl, obsvbl, cosmo, conf, reverse):
    ptcl, obsvbl = nbody(ptcl, obsvbl, cosmo, conf, reverse)
    return (ptcl, obsvbl), (ptcl, obsvbl, cosmo, conf)

def nbody_bwd(reverse, res, cotangents):
    ptcl, obsvbl, cosmo, conf = res
    ptcl_cot, obsvbl_cot = cotangents

    ptcl, ptcl_cot, cosmo_cot = nbody_adj(
        ptcl, ptcl_cot, obsvbl, obsvbl_cot, cosmo, conf, reverse=reverse)

    return ptcl_cot, None, cosmo_cot, None

nbody.defvjp(nbody_fwd, nbody_bwd)
