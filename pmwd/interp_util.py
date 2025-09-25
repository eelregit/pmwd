from jax import jit, vjp
import jax.numpy as jnp

from pmwd.particles import Particles
from pmwd.cosmology import E2


def itp_prev(ptcl0, a0, a1, a, cosmo):
    """Cubic Hermite interpolation is a linear combination of two ptcls, this
       function returns the disp and vel from the first ptcl at a0."""
    dtype = cosmo.conf.float_dtype
    Da = a1 - a0
    t = (a - a0) / Da
    a3E0 = a0**3 * jnp.sqrt(E2(a0, cosmo))
    a3E = a**3 * jnp.sqrt(E2(a, cosmo))
    # Hermite basis functions and derivatives
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h00p = 6 * t**2 - 6 * t
    h10p = 3 * t**2 - 4 * t + 1

    disp = (h00.astype(dtype) * ptcl0.disp +
            (h10 * Da / a3E0).astype(dtype) * ptcl0.vel)
    vel = ((a3E * h00p / Da).astype(dtype) * ptcl0.disp +
           (a3E * h10p / a3E0).astype(dtype) * ptcl0.vel)

    return disp, vel


def itp_prev_adj(ptcl_cot, cosmo_cot, iptcl_cot, ptcl0, a0, a1, a, cosmo):
    """Update ptcl_cot and cosmo_cot given the iptcl_cot and the vjp with itp_prev."""
    # iptcl_cot is the cotangent of the interpolated ptcl
    (disp, vel), itp_prev_vjp = vjp(itp_prev, ptcl0, a0, a1, a, cosmo)
    ptcl0_cot, a0_cot, a1_cot, a_cot, cosmo_cot_itp = itp_prev_vjp(
                                            (iptcl_cot.disp, iptcl_cot.vel))

    disp_cot = ptcl_cot.disp + ptcl0_cot.disp
    vel_cot = ptcl_cot.vel + ptcl0_cot.vel
    ptcl_cot = ptcl_cot.replace(disp=disp_cot, vel=vel_cot)
    cosmo_cot += cosmo_cot_itp  # cosmo used in interp, TODO keep or comment out
    return ptcl_cot, cosmo_cot


def itp_next(ptcl1, a0, a1, a, cosmo):
    """Cubic Hermite interpolation is a linear combination of two ptcls, this
       function returns the disp and vel from the second ptcl at a1."""
    dtype = cosmo.conf.float_dtype
    Da = a1 - a0
    t = (a - a0) / Da
    a3E1 = a1**3 * jnp.sqrt(E2(a1, cosmo))
    a3E = a**3 * jnp.sqrt(E2(a, cosmo))
    # Hermite basis functions and derivatives
    h01 = - 2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    h01p = - 6 * t**2 + 6 * t
    h11p = 3 * t**2 - 2 * t

    disp = (h01.astype(dtype) * ptcl1.disp +
            (h11 * Da / a3E1).astype(dtype) * ptcl1.vel)
    vel = ((a3E * h01p / Da).astype(dtype) * ptcl1.disp +
           (a3E * h11p / a3E1).astype(dtype) * ptcl1.vel)

    return disp, vel


def itp_next_adj(ptcl_cot, cosmo_cot, iptcl_cot, ptcl1, a0, a1, a, cosmo):
    """Update ptcl_cot and cosmo_cot given the iptcl_cot and the vjp with itp_next."""
    # iptcl_cot is the cotangent of the interpolated ptcl
    (disp, vel), itp_next_vjp = vjp(itp_next, ptcl1, a0, a1, a, cosmo)
    ptcl1_cot, a0_cot, a1_cot, a_cot, cosmo_cot_itp = itp_next_vjp(
                                            (iptcl_cot.disp, iptcl_cot.vel))

    disp_cot = ptcl_cot.disp + ptcl1_cot.disp
    vel_cot = ptcl_cot.vel + ptcl1_cot.vel
    ptcl_cot = ptcl_cot.replace(disp=disp_cot, vel=vel_cot)
    cosmo_cot += cosmo_cot_itp  # cosmo used in interp, TODO keep or comment out
    return ptcl_cot, cosmo_cot


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
