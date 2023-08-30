import jax.numpy as jnp

from pmwd.configuration import Configuration
from pmwd.cosmology import Cosmology, H_deriv, Omega_m_a, SimpleLCDM
from pmwd.boltzmann import boltzmann, growth, linear_power


def _sotheta(cosmo, conf, a):
    """Physical quantities to be used in SO input features along with k."""
    theta = jnp.asarray([
        a,
        cosmo.A_s_1e9,
        cosmo.n_s,
        cosmo.Omega_m,
        cosmo.Omega_b / cosmo.Omega_m,
        cosmo.Omega_k,
        cosmo.h,
        conf.box_size[0],
        conf.ptcl_spacing,
        conf.cell_size,
        conf.softening_length,
    ])
    return theta


def _sotheta_names():
    # str names of the variables returned by the sotheta function above
    # currently hardcoded, should be updated along with sotheta function
    pass


def _soft_len(k_fac=1):
    # get the length of SO input features with dummy conf and cosmo
    conf = Configuration(1., (128,)*3)
    cosmo = SimpleLCDM(conf)
    cosmo = boltzmann(cosmo, conf)
    theta = _sotheta(cosmo, conf, conf.a_start)
    return k_fac + len(theta)


def soft(k, theta):
    """SO features for neural nets input, with k being a scalar."""
    return jnp.concatenate((jnp.atleast_1d(k), theta))


def soft_k(k, theta, logk=True):
    """Get SO input features (k, theta)."""
    theta = jnp.broadcast_to(theta, k.shape+theta.shape)
    k = k.reshape(k.shape + (1,))
    if logk:
        k = jnp.log(jnp.where(k > 0., k, 1.))
    ft = jnp.concatenate((k, theta), axis=-1)
    return ft


def soft_kvec(kv, theta, logk=True):
    """Get SO input features (k1, k2, k3, theta)."""
    theta = jnp.broadcast_to(theta, kv.shape[:-1]+theta.shape)
    if logk:
        kv = jnp.log(jnp.where(kv > 0., kv, 1.))
    ft = jnp.concatenate((kv, theta), axis=-1)
    return ft
