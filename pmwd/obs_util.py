import jax.numpy as jnp

from pmwd.particles import Particles
from pmwd.cosmology import E2


def interp_ptcl(ptcl0, ptcl1, a0, a1, a, cosmo, conf):
    """Given two ptcl snapshots, get the interpolated one
       at a given time using cubic Hermite interpolation.
    """
    Da = a1 - a0
    t = (a - a0) / Da
    a3E0 = a0**3 * jnp.sqrt(E2(a0, cosmo))
    a3E1 = a1**3 * jnp.sqrt(E2(a1, cosmo))
    # displacement
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = - 2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    disp = (h00 * ptcl0.disp + h10 * Da / a3E0 * ptcl0.vel +
            h01 * ptcl1.disp + h11 * Da / a3E1 * ptcl1.vel)
    # velocity
    # derivatives of the Hermite basis functions
    h00 = 6 * t**2 - 6 * t
    h10 = 3 * t**2 - 4 * t + 1
    h01 = - 6 * t**2 + 6 * t
    h11 = 3 * t**2 - 2 * t
    vel = (h00 / Da * ptcl0.disp + h10 / a3E0 * ptcl0.vel +
           h01 / Da * ptcl1.disp + h11 / a3E1 * ptcl1.vel)
    vel *= a**3 * jnp.sqrt(E2(a, cosmo))

    ptcl = Particles(conf, ptcl0.pmid, disp, vel=vel)
    return ptcl
