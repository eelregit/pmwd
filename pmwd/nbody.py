from functools import partial

import numpy as np
from jax import value_and_grad, jit, vjp, custom_vjp
import jax.numpy as jnp
from jax.tree_util import tree_map

from pmwd.boltzmann import growth
from pmwd.cosmology import E2, H_deriv
from pmwd.gravity import gravity


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
    return factor.astype(conf.float_dtype)


def kick_factor(a_acc, a_prev, a_next, cosmo, conf):
    """Kick time step factor of conf.float_dtype in [1/H_0]."""
    factor = _G_D(a_next, cosmo, conf) - _G_D(a_prev, cosmo, conf)
    factor /= _G_K(a_acc, cosmo, conf)
    return factor.astype(conf.float_dtype)


def drift(a_vel, a_prev, a_next, ptcl, cosmo, conf):
    """Drift."""
    disp = ptcl.disp + ptcl.vel * drift_factor(a_vel, a_prev, a_next, cosmo, conf)
    return ptcl.replace(disp=disp)


# TODO deriv wrt a
def drift_adj(a_vel, a_prev, a_next, ptcl, ptcl_cot, cosmo, cosmo_cot, conf):
    """Drift, and particle and cosmology adjoints."""
    factor_valgrad = value_and_grad(drift_factor, argnums=3)
    factor, cosmo_cot_drift = factor_valgrad(a_vel, a_prev, a_next, cosmo, conf)

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
    vel = ptcl.vel + ptcl.acc * kick_factor(a_acc, a_prev, a_next, cosmo, conf)
    return ptcl.replace(vel=vel)


# TODO deriv wrt a
def kick_adj(a_acc, a_prev, a_next, ptcl, ptcl_cot, cosmo, cosmo_cot, cosmo_cot_force, conf):
    """Kick, and particle and cosmology adjoints."""
    factor_valgrad = value_and_grad(kick_factor, argnums=3)
    factor, cosmo_cot_kick = factor_valgrad(a_acc, a_prev, a_next, cosmo, conf)

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
    a_cot, ptcl_cot_force, cosmo_cot_force, conf_cot = gravity_vjp(ptcl_cot.vel)
    ptcl_cot = ptcl_cot.replace(acc=ptcl_cot_force.disp)

    return ptcl, ptcl_cot, cosmo_cot_force


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
    pass


def observe_init(a, ptcl, obsvbl, cosmo, conf):
    pass


@jit
def nbody_init(a, ptcl, obsvbl, cosmo, conf):
    # ptcl = force(a, ptcl, cosmo, conf)

    # ptcl = coevolve_init(a, ptcl, cosmo, conf)

    # obsvbl = observe_init(a, ptcl, obsvbl, cosmo, conf)

    return ptcl, obsvbl


@jit
def nbody_step(i, C, D, a_nbody, ptcl, obsvbl, cosmo, conf):
    # symplectic integrator
    # TODO speed up by skipping nothing-happens operations
    for j in range(conf.symp_order):
        ax0, ax1, ap0, ap1 = (a_nbody[i] * (1 - x) + a_nbody[i+1] * x for x in
                              (C[j], C[j+1], D[j], D[j+1]))
        ptcl = drift(ap0, ax0, ax1, ptcl, cosmo, conf)
        ptcl = force(ax1, ptcl, cosmo, conf)
        ptcl = kick(ax1, ap0, ap1, ptcl, cosmo, conf)

    # ptcl = coevolve(i, a_nbody, ptcl, cosmo, conf)

    # obsvbl = observe(i, a_nbody, ptcl, obsvbl, cosmo, conf)

    return ptcl, obsvbl


@partial(custom_vjp, nondiff_argnums=(4,))
def nbody(ptcl, obsvbl, cosmo, conf, reverse=False):
    """N-body time integration."""
    a_nbody = conf.a_nbody[::-1] if reverse else conf.a_nbody
    C, D = conf.symp_cd

    ptcl, obsvbl = nbody_init(a_nbody[0], ptcl, obsvbl, cosmo, conf)
    for i, a in enumerate(a_nbody[:-1]):  # i denotes ptcl before the step
        ptcl, obsvbl = nbody_step(i, C, D, a_nbody, ptcl, obsvbl, cosmo, conf)
    return ptcl, obsvbl


@jit
def nbody_adj_init(a, ptcl, ptcl_cot, obsvbl_cot, cosmo, conf):
    #ptcl_cot = observe_adj(a_prev, a_next, ptcl, ptcl_cot, obsvbl_cot, cosmo)

    #ptcl, ptcl_cot = coevolve_adj(a_prev, a_next, ptcl, ptcl_cot, cosmo)

    ptcl, ptcl_cot, cosmo_cot_force = force_adj(a, ptcl, ptcl_cot, cosmo, conf)

    cosmo_cot = tree_map(lambda x: jnp.zeros_like(x), cosmo)

    # TODO conf_cot

    return ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force


@jit
def nbody_adj_step(i, C, D, a_nbody, ptcl, ptcl_cot, obsvbl_cot, cosmo, cosmo_cot, cosmo_cot_force, conf):
    #ptcl_cot = observe_adj(a_prev, a_next, ptcl, ptcl_cot, obsvbl_cot, cosmo, conf)

    #ptcl, ptcl_cot = coevolve_adj(a_prev, a_next, ptcl, ptcl_cot, cosmo, conf)

    # symplectic integrator and its adjoint
    for j in range(conf.symp_order-1, -1, -1):
        ax0, ax1, ap0, ap1 = (a_nbody[i-1] * (1 - x) + a_nbody[i] * x for x in
                              (C[j], C[j+1], D[j], D[j+1]))
        ptcl, ptcl_cot, cosmo_cot = kick_adj(ax1, ap1, ap0, ptcl, ptcl_cot, cosmo, cosmo_cot, cosmo_cot_force, conf)
        ptcl, ptcl_cot, cosmo_cot = drift_adj(ap0, ax1, ax0, ptcl, ptcl_cot, cosmo, cosmo_cot, conf)
        ptcl, ptcl_cot, cosmo_cot_force = force_adj(ax0, ptcl, ptcl_cot, cosmo, conf)

    return ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force


def nbody_adj(ptcl, ptcl_cot, obsvbl_cot, cosmo, conf, reverse=False):
    """N-body time integration with adjoint equation."""
    a_nbody = conf.a_nbody[::-1] if reverse else conf.a_nbody
    C, D = conf.symp_cd

    ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force = nbody_adj_init(
        a_nbody[-1], ptcl, ptcl_cot, obsvbl_cot, cosmo, conf)
    for i in range(len(a_nbody)-1, 0, -1):  # i denotes ptcl before the step
        ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force = nbody_adj_step(
            i, C, D, a_nbody, ptcl, ptcl_cot, obsvbl_cot, cosmo, cosmo_cot, cosmo_cot_force, conf)
    return ptcl, ptcl_cot, cosmo_cot


def nbody_fwd(ptcl, obsvbl, cosmo, conf, reverse):
    ptcl, obsvbl = nbody(ptcl, obsvbl, cosmo, conf, reverse)
    return (ptcl, obsvbl), (ptcl, cosmo, conf)

def nbody_bwd(reverse, res, cotangents):
    ptcl, cosmo, conf = res
    ptcl_cot, obsvbl_cot = cotangents

    ptcl, ptcl_cot, cosmo_cot = nbody_adj(ptcl, ptcl_cot, obsvbl_cot, cosmo, conf,
                                          reverse=reverse)

    return ptcl_cot, obsvbl_cot, cosmo_cot, None  # FIXME HACK on conf_cot

nbody.defvjp(nbody_fwd, nbody_bwd)
