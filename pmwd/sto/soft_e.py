import jax.numpy as jnp

from pmwd.configuration import Configuration
from pmwd.cosmology import Omega_m_a, SimpleLCDM
from pmwd.boltzmann import growth, boltzmann


def nonlinear_scales(cosmo, conf, a):
    D = growth(a, cosmo, conf)
    # TopHat variance, var is decreasing with R
    # but for jnp.interp, xp must be increasing, thus the reverse [::-1]
    R_TH = jnp.interp(1 / D**2, cosmo.varlin[::-1], conf.varlin_R[::-1])
    return (R_TH,)


def sotheta(cosmo, conf, a):
    """Physical quantities to be used in SO input features along with k."""
    # quantities of dim L
    theta_l = jnp.asarray([
        conf.cell_size,
        conf.softening_length,
        *nonlinear_scales(cosmo, conf, a),
    ])

    # dimensionless quantities
    theta_o = jnp.asarray([
        growth(a, cosmo, conf),
        Omega_m_a(a, cosmo),
    ])
    return (theta_l, theta_o)


def soft_len(k_fac=1):
    # get the length of SO input features with dummy conf and cosmo
    conf = Configuration(1., (128,)*3)
    cosmo = SimpleLCDM(conf)
    cosmo = boltzmann(cosmo, conf)
    theta_l, theta_o = sotheta(cosmo, conf, conf.a_start)
    return len(theta_l) * k_fac + len(theta_o)


def soft(k, theta):
    """SO features for neural nets input, with k being a scalar."""
    theta_l, theta_o = theta
    k_theta_l = k * theta_l
    return jnp.concatenate((k_theta_l, theta_o))


def soft_k(k, theta):
    """Get SO input features (k * l, o)."""
    theta_l, theta_o = theta  # e.g. (8,), (6,)
    k_shape = k.shape  # e.g. (128, 1, 1)
    k = k.reshape(k_shape + (1,))  # (128, 1, 1, 1)
    theta_l = theta_l.reshape((1,) * len(k_shape) + theta_l.shape)  # (1, 1, 1, 8)
    ft = k * theta_l  # (128, 1, 1, 8)
    theta_o = jnp.broadcast_to(theta_o, k_shape+theta_o.shape)  # (128, 1, 1, 6)
    ft = jnp.concatenate((ft, theta_o), axis=-1)  # (128, 1, 1, 8+6)
    return ft


def soft_kvec(kv, theta):
    """Get SO input features (k1 * l, k2 * l, k3 * l, o)."""
    kv_shape = kv.shape  # e.g. (128, 128, 65, 3)
    kv = kv.reshape(kv_shape + (1,))  # (128, 128, 65, 3, 1)

    theta_l, theta_o = theta  # e.g. (8,), (6,)
    theta_l = theta_l.reshape((1,) * len(kv_shape) + theta_l.shape)  # (1, 1, 1, 1, 8)
    ft = kv * theta_l  # (128, 128, 65, 3, 8)
    ft = ft.reshape(kv_shape[:-1] + (-1,))  # (128, 128, 65, 3*8)
    theta_o = jnp.broadcast_to(theta_o, kv_shape[:-1]+theta_o.shape)  # (128, 128, 65, 6)
    ft = jnp.concatenate((ft, theta_o), axis=-1)  # (128, 128, 65, 3*8+6)
    return ft


def soft_names(net):
    # str names of input features of the SO neural nets
    # currently hardcoded, should be updated along with functions above
    theta_l = ['cell size', 'softening length', 'R_TH']
    theta_l_k = []
    if net == 'f':
        for v in theta_l:
            theta_l_k.append(f'k * {v}')
    if net == 'g':
        for n in range(3):
            for v in theta_l:
                theta_l_k.append(f'k_{n} * {v}')

    theta_o = ['D(a)', 'Omega_m(a)']

    return theta_l_k + theta_o


def soft_names_tex(net):
    # soft_names in latex math expressions
    theta_l = ['l_c', 'l_s', 'R_{\\rm TH}']
    theta_l_k = []
    if net == 'f':
        for v in theta_l:
            theta_l_k.append(f'k {v}')
    if net == 'g':
        for n in range(3):
            for v in theta_l:
                theta_l_k.append(f'k_{n} {v}')

    theta_o = ['D(a)', '\\Omega_m(a)']

    return theta_l_k + theta_o
