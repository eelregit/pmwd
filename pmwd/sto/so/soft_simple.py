"""SO input features consists of simple Sobol parameters, a and k etc."""
import jax.numpy as jnp

from pmwd.boltzmann import growth


def nonlinear_scales(cosmo, conf, a):
    D = growth(a, cosmo, conf)
    # TopHat variance, var is decreasing with R
    # but for jnp.interp, xp must be increasing, thus the reverse [::-1]
    R_TH = jnp.interp(1 / D**2, cosmo.varlin[::-1], conf.varlin_R[::-1])
    return R_TH


def sotheta(cosmo, conf, a):
    """Physical quantities to be used in SO input features along with k."""
    R_TH = nonlinear_scales(cosmo, conf, a)

    # quantities to be multiplied with k
    theta_l = jnp.array([
        conf.ptcl_spacing,
        R_TH,
        cosmo.softening_length,
    ], dtype=conf.float_dtype)

    # other quantities
    theta_o = jnp.asarray([
        a,
        cosmo.A_s_1e9,
        cosmo.n_s,
        cosmo.Omega_m,
        cosmo.Omega_b,
        cosmo.Omega_k,
        cosmo.h,
    ], dtype=conf.float_dtype)

    return (theta_l, theta_o)


def soft_k(k, theta):
    """Get SO input features (k * l, o) with k of shape (...,)."""
    theta_l, theta_o = theta  # e.g. (8,), (6,)

    ft = k[..., None] * theta_l  # (..., 8)

    theta_o = jnp.broadcast_to(theta_o, k.shape + theta_o.shape)  # (..., 6)
    ft = jnp.concatenate((ft, theta_o), axis=-1)  # (..., 8+6)

    return ft


def soft_kv(kv, theta):
    """Get SO input features (k1 * l, k2 * l, k3 * l, o) with kv of shape (..., 3)."""
    theta_l, theta_o = theta  # e.g. (8,), (6,)

    ft = kv[..., None] * theta_l  # (..., 3, 8)
    ft = ft.reshape(kv.shape[:-1] + (-1,))  # (..., 3 * 8)

    theta_o = jnp.broadcast_to(theta_o, kv.shape[:-1] + theta_o.shape)  # (..., 6)
    ft = jnp.concatenate((ft, theta_o), axis=-1)  # (..., 3 * 8 + 6)

    return ft


def soft_names(net):
    # str names of input features of the SO neural nets
    # currently hardcoded, should be updated along with functions above
    theta_l = ['l_p', 'R_TH', 'l_s']
    theta_l_k = []
    if net == 'f':
        for v in theta_l:
            theta_l_k.append(f'k{v}')
    if net == 'g':
        for n in range(3):
            for v in theta_l:
                theta_l_k.append(f'k_{n}{v}')

    theta_o = ['a', 'A_s_1e9', 'n_s', 'Omega_m', 'Omega_b', 'Omega_k', 'h']

    return theta_l_k + theta_o


def soft_names_tex(net):
    # soft_names in latex math expressions
    theta_l = ['l_p', 'R_{\\rm TH}', 'l_s']
    theta_l_k = []
    if net == 'f':
        for v in theta_l:
            theta_l_k.append(f'k {v}')
    if net == 'g':
        for n in range(3):
            for v in theta_l:
                theta_l_k.append(f'k_{n} {v}')

    theta_o = ['a', 'A_s', 'n_s', '\\Omega_m', '\\Omega_b', '\\Omega_k', 'h']

    return theta_l_k + theta_o


def soft_len(net):
    # get the length of SO input features
    return len(soft_names(net))
