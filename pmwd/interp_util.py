from jax import jit, vjp, custom_vjp
import jax.numpy as jnp
from functools import partial

from pmwd.particles import Particles
from pmwd.cosmology import E2


def coefs_prev(a0, a1, a, cosmo):
    Da = a1 - a0
    t = (a - a0) / Da
    a3E0 = a0**3 * jnp.sqrt(E2(a0, cosmo))
    a3E = a**3 * jnp.sqrt(E2(a, cosmo))

    # Hermite basis functions and derivatives
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h00p = 6 * t**2 - 6 * t
    h10p = 3 * t**2 - 4 * t + 1

    dtype = cosmo.conf.float_dtype
    dd = h00.astype(dtype)
    dv = (h10 * Da / a3E0).astype(dtype)
    vd = (a3E * h00p / Da).astype(dtype)
    vv = (a3E * h10p / a3E0).astype(dtype)
    return dd, dv, vd, vv


def coefs_next(a0, a1, a, cosmo):
    Da = a1 - a0
    t = (a - a0) / Da
    a3E1 = a1**3 * jnp.sqrt(E2(a1, cosmo))
    a3E = a**3 * jnp.sqrt(E2(a, cosmo))

    # Hermite basis functions and derivatives
    h01 = - 2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    h01p = - 6 * t**2 + 6 * t
    h11p = 3 * t**2 - 2 * t

    dtype = cosmo.conf.float_dtype
    dd = h01.astype(dtype)
    dv = (h11 * Da / a3E1).astype(dtype)
    vd = (a3E * h01p / Da).astype(dtype)
    vv = (a3E * h11p / a3E1).astype(dtype)
    return dd, dv, vd, vv


@partial(custom_vjp, nondiff_argnums=(0,))
def itp_snap(order, disp, vel, a0, a1, a, cosmo):
    if order == 'prev':
        dd, dv, vd, vv = coefs_prev(a0, a1, a, cosmo)
    if order == 'next':
        dd, dv, vd, vv = coefs_next(a0, a1, a, cosmo)

    disp_itp = dd * disp + dv * vel
    vel_itp = vd * disp + vv * vel

    return disp_itp, vel_itp

def itp_snap_fwd(order, disp, vel, a0, a1, a, cosmo):
    return itp_snap(order, disp, vel, a0, a1, a, cosmo), (disp, vel, a0, a1, a, cosmo)

def itp_snap_bwd(order, res, cots):
    disp, vel, a0, a1, a, cosmo = res
    disp_itp_cot, vel_itp_cot = cots

    if order == 'prev':
        (dd, dv, vd, vv), coefs_vjp = vjp(coefs_prev, a0, a1, a, cosmo)
    if order == 'next':
        (dd, dv, vd, vv), coefs_vjp = vjp(coefs_next, a0, a1, a, cosmo)

    disp_cot = disp_itp_cot * dd + vel_itp_cot * vd
    vel_cot = disp_itp_cot * dv + vel_itp_cot * vv

    dd_cot = (disp_itp_cot * disp).sum()
    dv_cot = (disp_itp_cot * vel).sum()
    vd_cot = (vel_itp_cot * disp).sum()
    vv_cot = (vel_itp_cot * vel).sum()
    a0_cot, a1_cot, a_cot, cosmo_cot = coefs_vjp((dd_cot, dv_cot, vd_cot, vv_cot))

    return (disp_cot, vel_cot, a0_cot, a1_cot, a_cot, cosmo_cot)

itp_snap.defvjp(itp_snap_fwd, itp_snap_bwd)



def interptcl(ptcl0, ptcl1, a0, a1, a, cosmo):
    """Given two ptcl snapshots, get the interpolated one at a given time using
       cubic Hermite interpolation."""
    dtype = cosmo.conf.float_dtype
    Da = a1 - a0
    t = (a - a0) / Da
    a3E0 = a0**3 * jnp.sqrt(E2(a0, cosmo))
    a3E1 = a1**3 * jnp.sqrt(E2(a1, cosmo))
    a3E = a**3 * jnp.sqrt(E2(a, cosmo))
    # Hermite basis functions and derivatives
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = - 2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    h00p = 6 * t**2 - 6 * t
    h10p = 3 * t**2 - 4 * t + 1
    h01p = - 6 * t**2 + 6 * t
    h11p = 3 * t**2 - 2 * t

    disp = (h00.astype(dtype) * ptcl0.disp +
            (h10 * Da / a3E0).astype(dtype) * ptcl0.vel +
            h01.astype(dtype) * ptcl1.disp +
            (h11 * Da / a3E1).astype(dtype) * ptcl1.vel)
    vel = ((a3E * h00p / Da).astype(dtype) * ptcl0.disp +
           (a3E * h10p / a3E0).astype(dtype) * ptcl0.vel +
           (a3E * h01p / Da).astype(dtype) * ptcl1.disp +
           (a3E * h11p / a3E1).astype(dtype) * ptcl1.vel)

    iptcl = Particles(ptcl0.conf, ptcl0.pmid, disp, vel=vel)
    return iptcl


def interptcl_adj(iptcl_cot, ptcl0, ptcl1, a0, a1, a, cosmo):
    iptcl, interptcl_vjp = vjp(interptcl, ptcl0, ptcl1, a0, a1, a, cosmo)
    ptcl0_cot, ptcl1_cot, a0_cot, a1_cot, a_cot, cosmo_cot_itp = interptcl_vjp(iptcl_cot)
    return ptcl0_cot, ptcl1_cot, cosmo_cot_itp
