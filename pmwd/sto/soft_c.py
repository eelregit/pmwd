"""SO input features consists of simple Sobol parameters, a and k etc."""
import jax.numpy as jnp

from pmwd.configuration import Configuration
from pmwd.cosmology import Cosmology, H_deriv, Omega_m_a, SimpleLCDM
from pmwd.boltzmann import boltzmann, growth, linear_power


def sotheta(cosmo, conf, a):
    """Physical quantities to be used in SO input features along with k."""
    # quantities of dim L
    theta_l = jnp.asarray([
        conf.ptcl_spacing,
        conf.cell_size,
        conf.softening_length,
    ])

    # dimensionless quantities
    theta_o = jnp.asarray([
        a,
        cosmo.A_s_1e9,
        cosmo.n_s,
        cosmo.Omega_m,
        cosmo.Omega_b / cosmo.Omega_m,
        cosmo.Omega_k / (1 - cosmo.Omega_k),
        cosmo.h,
    ])
    return (theta_l, theta_o)


def soft_len(k_fac=1):
    # get the length of SO input features with dummy conf and cosmo
    conf = Configuration(1., (128,)*3)
    cosmo = SimpleLCDM(conf)
    cosmo = boltzmann(cosmo, conf)
    theta_l, theta_o = sotheta(cosmo, conf, conf.a_start)
    return len(theta_l) * k_fac + len(theta_o)


def soft(k, theta, log_k_theta=True):
    """SO features for neural nets input, with k being a scalar."""
    theta_l, theta_o = theta
    k_theta_l = k * theta_l
    if log_k_theta:
        k_theta_l = jnp.log(jnp.where(k_theta_l > 0., k_theta_l, 1.))
    return jnp.concatenate((k_theta_l, theta_o))


def soft_k(k, theta, log_k_theta=True):
    """Get SO input features (k * l, o)."""
    theta_l, theta_o = theta  # e.g. (8,), (6,)
    k_shape = k.shape  # e.g. (128, 1, 1)
    k = k.reshape(k_shape + (1,))  # (128, 1, 1, 1)
    theta_l = theta_l.reshape((1,) * len(k_shape) + theta_l.shape)  # (1, 1, 1, 8)
    ft = k * theta_l  # (128, 1, 1, 8)
    if log_k_theta:
        ft = jnp.log(jnp.where(ft > 0., ft, 1.))
    theta_o = jnp.broadcast_to(theta_o, k_shape+theta_o.shape)  # (128, 1, 1, 6)
    ft = jnp.concatenate((ft, theta_o), axis=-1)  # (128, 1, 1, 8+6)
    return ft


def soft_kvec(kv, theta, log_k_theta=True):
    """Get SO input features (k1 * l, k2 * l, k3 * l, o)."""
    kv_shape = kv.shape  # e.g. (128, 128, 65, 3)
    kv = kv.reshape(kv_shape + (1,))  # (128, 128, 65, 3, 1)

    theta_l, theta_o = theta  # e.g. (8,), (6,)
    theta_l = theta_l.reshape((1,) * len(kv_shape) + theta_l.shape)  # (1, 1, 1, 1, 8)
    ft = kv * theta_l  # (128, 128, 65, 3, 8)
    ft = ft.reshape(kv_shape[:-1] + (-1,))  # (128, 128, 65, 3*8)
    if log_k_theta:
        ft = jnp.log(jnp.where(ft > 0., ft, 1.))
    theta_o = jnp.broadcast_to(theta_o, kv_shape[:-1]+theta_o.shape)  # (128, 128, 65, 6)
    ft = jnp.concatenate((ft, theta_o), axis=-1)  # (128, 128, 65, 3*8+6)
    return ft


def soft_names(net):
    # str names of input features of the SO neural nets
    # currently hardcoded, should be updated along with functions above
    pass
