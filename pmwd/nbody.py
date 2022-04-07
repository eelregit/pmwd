import numpy as np
from jax import value_and_grad, jit, vjp, custom_vjp, float0
import jax.numpy as jnp
from jax.tree_util import tree_map

from pmwd.boltzmann import growth
from pmwd.cosmology import E2, H_deriv
from pmwd.gravity import gravity


def _G_D(a, cosmo):
    """Growth factor of ZA canonical momenta in [H_0]."""
    return a**2 * jnp.sqrt(E2(a, cosmo)) * growth(a, cosmo, deriv=1)


def _G_K(a, cosmo):
    """Growth factor of ZA accelerations in [H_0^2]."""
    return a**3 * E2(a, cosmo) * (
        growth(a, cosmo, deriv=2) + (2. + H_deriv(a, cosmo)) * growth(a, cosmo, deriv=1)
    )


def drift_factor(a_vel, a_prev, a_next, cosmo):
    """Drift time step factor of cosmo.conf.float_dtype in [1/H_0]."""
    factor = (growth(a_next, cosmo) - growth(a_prev, cosmo)) / _G_D(a_vel, cosmo)
    return factor.astype(cosmo.conf.float_dtype)


def kick_factor(a_acc, a_prev, a_next, cosmo):
    """Kick time step factor of cosmo.conf.float_dtype in [1/H_0]."""
    factor = (_G_D(a_next, cosmo) - _G_D(a_prev, cosmo)) / _G_K(a_acc, cosmo)
    return factor.astype(cosmo.conf.float_dtype)


def drift(a_vel, a_prev, a_next, ptcl, cosmo):
    """Drift."""
    disp = ptcl.disp + ptcl.vel * drift_factor(a_vel, a_prev, a_next, cosmo)
    return ptcl.replace(disp=disp)


# TODO deriv wrt a
def drift_adj(a_vel, a_prev, a_next, ptcl, ptcl_cot, cosmo, cosmo_cot):
    """Drift, and particle and cosmology adjoints."""
    factor_valgrad = value_and_grad(drift_factor, argnums=3)
    factor, cosmo_cot_drift = factor_valgrad(a_vel, a_prev, a_next, cosmo)

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


def kick(a_acc, a_prev, a_next, ptcl, cosmo):
    """Kick."""
    vel = ptcl.vel + ptcl.acc * kick_factor(a_acc, a_prev, a_next, cosmo)
    return ptcl.replace(vel=vel)


# TODO deriv wrt a
def kick_adj(a_acc, a_prev, a_next, ptcl, ptcl_cot, cosmo, cosmo_cot, cosmo_cot_force):
    """Kick, and particle and cosmology adjoints."""
    factor_valgrad = value_and_grad(kick_factor, argnums=3)
    factor, cosmo_cot_kick = factor_valgrad(a_acc, a_prev, a_next, cosmo)

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


def force(a, ptcl, cosmo):
    """Force."""
    acc = gravity(ptcl, cosmo)
    return ptcl.replace(acc=acc)


def force_adj(a, ptcl, ptcl_cot, cosmo):
    """Force, and particle and cosmology vjp."""
    # force
    acc, gravity_vjp = vjp(gravity, ptcl, cosmo)
    ptcl = ptcl.replace(acc=acc)

    # particle and cosmology vjp
    ptcl_cot_force, cosmo_cot_force = gravity_vjp(ptcl_cot.vel)
    ptcl_cot = ptcl_cot.replace(acc=ptcl_cot_force.disp)

    return ptcl, ptcl_cot, cosmo_cot_force


def form(a_prev, a_next, ptcl, cosmo):
    pass


def form_init(a, ptcl, cosmo):
    pass  # TODO necessary?


def coevolve(a_prev, a_next, ptcl, cosmo):
    attr = form(a_prev, a_next, ptcl, cosmo)
    return ptcl.replace(attr=attr)


def coevolve_init(a, ptcl, cosmo):
    if ptcl.attr is None:
        attr = form_init(a, ptcl, cosmo)
        ptcl = ptcl.replace(attr=attr)
    return ptcl


def observe(a_prev, a_next, ptcl, obsvbl, cosmo):
    pass


def observe_init(a, ptcl, obsvbl, cosmo):
    pass


@jit
def nbody_init(a, ptcl, obsvbl, cosmo):
    ptcl = force(a, ptcl, cosmo)

    ptcl = coevolve_init(a, ptcl, cosmo)

    obsvbl = observe_init(a, ptcl, obsvbl, cosmo)

    return ptcl, obsvbl


@jit
def nbody_step(a_prev, a_next, ptcl, obsvbl, cosmo):
    # leapfrog
    a_half = 0.5 * (a_prev + a_next)
    ptcl = kick(a_prev, a_prev, a_half, ptcl, cosmo)
    ptcl = drift(a_half, a_prev, a_next, ptcl, cosmo)
    ptcl = force(a_next, ptcl, cosmo)
    ptcl = kick(a_next, a_half, a_next, ptcl, cosmo)

    ptcl = coevolve(a_prev, a_next, ptcl, cosmo)

    obsvbl = observe(a_prev, a_next, ptcl, obsvbl, cosmo)

    return ptcl, obsvbl


@custom_vjp
def nbody(ptcl, obsvbl, cosmo):
    """N-body time integration."""
    conf = cosmo.conf
    ptcl, obsvbl = nbody_init(conf.a_nbody[0], ptcl, obsvbl, cosmo)
    for a_prev, a_next in zip(conf.a_nbody[:-1], conf.a_nbody[1:]):
        ptcl, obsvbl = nbody_step(a_prev, a_next, ptcl, obsvbl, cosmo)
    return ptcl, obsvbl


@jit
def nbody_adj_init(a, ptcl, ptcl_cot, obsvbl_cot, cosmo):
    def zeros_float0_like(x):
        if issubclass(x.dtype.type, (jnp.bool_, jnp.integer)):
            # FIXME after jax issues #4433 is addressed
            return np.empty(x.shape, dtype=float0)
        else:
            return jnp.zeros_like(x)

    cosmo_cot = tree_map(zeros_float0_like, cosmo)

    #ptcl_cot = observe_adj(a_prev, a_next, ptcl, ptcl_cot, obsvbl_cot, cosmo)

    #ptcl, ptcl_cot = coevolve_adj(a_prev, a_next, ptcl, ptcl_cot, cosmo)

    ptcl, ptcl_cot, cosmo_cot_force = force_adj(a, ptcl, ptcl_cot, cosmo)

    return ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force


@jit
def nbody_adj_step(a_prev, a_next, ptcl, ptcl_cot, obsvbl_cot, cosmo, cosmo_cot, cosmo_cot_force):
    #ptcl_cot = observe_adj(a_prev, a_next, ptcl, ptcl_cot, obsvbl_cot, cosmo)

    #ptcl, ptcl_cot = coevolve_adj(a_prev, a_next, ptcl, ptcl_cot, cosmo)

    # leapfrog and its adjoint
    a_half = 0.5 * (a_prev + a_next)
    ptcl, ptcl_cot, cosmo_cot = kick_adj(a_prev, a_prev, a_half, ptcl, ptcl_cot, cosmo, cosmo_cot, cosmo_cot_force)
    ptcl, ptcl_cot, cosmo_cot = drift_adj(a_half, a_prev, a_next, ptcl, ptcl_cot, cosmo, cosmo_cot)
    ptcl, ptcl_cot, cosmo_cot_force = force_adj(a_next, ptcl, ptcl_cot, cosmo)
    ptcl, ptcl_cot, cosmo_cot = kick_adj(a_next, a_half, a_next, ptcl, ptcl_cot, cosmo, cosmo_cot, cosmo_cot_force)

    return ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force


def nbody_adj(ptcl, ptcl_cot, obsvbl_cot, cosmo):
    """N-body time integration with adjoint equation."""
    conf = cosmo.conf
    ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force = nbody_adj_init(
        conf.a_nbody[-1], ptcl, ptcl_cot, obsvbl_cot, cosmo)
    for a_prev, a_next in zip(conf.a_nbody[:0:-1], conf.a_nbody[-2::-1]):
        ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force = nbody_adj_step(
            a_prev, a_next, ptcl, ptcl_cot, obsvbl_cot, cosmo, cosmo_cot, cosmo_cot_force)
    return ptcl, ptcl_cot, cosmo_cot


def nbody_fwd(ptcl, obsvbl, cosmo):
    ptcl, obsvbl = nbody(ptcl, obsvbl, cosmo)
    return (ptcl, obsvbl), (ptcl, cosmo)

def nbody_bwd(res, cotangents):
    ptcl, cosmo = res
    ptcl_cot, obsvbl_cot = cotangents

    ptcl, ptcl_cot, cosmo_cot = nbody_adj(ptcl, ptcl_cot, obsvbl_cot, cosmo)

    return ptcl_cot, obsvbl_cot, cosmo_cot

nbody.defvjp(nbody_fwd, nbody_bwd)
