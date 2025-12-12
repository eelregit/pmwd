"""SO input features consists of all kinds of nonlinear scales etc."""
import jax
import jax.numpy as jnp

from pmwd.cosmology import H_deriv, Omega_m_a
from pmwd.boltzmann import growth, linear_power


def nonlinear_scales(cosmo, conf, a):
    """Some nonlinear scales and their time derivatives."""
    k = conf.transfer_k[1:]
    D = growth(a, cosmo, conf)
    dD = growth(a, cosmo, conf, deriv=1)  # dD / dlna
    dD2i = -2 * D**(-3) * dD  # d(1/D^2) / dlna
    interp_valgrad = jax.value_and_grad(jnp.interp, argnums=0)

    # dimensionless linear power
    Plin = linear_power(k, None, cosmo, conf)  # no a dependence
    k_P, dk_P = interp_valgrad(1 / D**2, k**3 * Plin / (2 * jnp.pi**2), k)
    dk_P *= dD2i
    R_P = 1 / k_P
    dR_P = - dk_P / k_P**2

    # TopHat variance, var is decreasing with R
    # but for jnp.interp, xp must be increasing, thus the reverse [::-1]
    R_TH, dR_TH = interp_valgrad(1 / D**2, cosmo.varlin[::-1], conf.varlin_R[::-1])
    dR_TH *= dD2i

    # Gaussian variance
    R_G, dR_G = interp_valgrad(1 / D**2, cosmo.varlin_g[::-1], conf.varlin_R_g[::-1])
    dR_G *= dD2i

    # rms linear theory displacement
    R_d = (jnp.trapezoid(k * Plin, x=jnp.log(k)) + k[0] * Plin[0] / 2) / (6 * jnp.pi**2)
    R_d = jnp.sqrt(R_d)
    dR_d = R_d * dD
    R_d *= D

    return (R_P, R_TH, R_G, R_d, dR_P, dR_TH, dR_G, dR_d)


def sotheta(cosmo, conf, a):
    """Physical quantities to be used in SO input features along with k."""
    # quantities of dim L
    theta_l = jnp.array([
        *nonlinear_scales(cosmo, conf, a),
        conf.ptcl_spacing,
        conf.cell_size,
        conf.softening_length,
    ], dtype=conf.float_dtype)

    # dimensionless quantities
    D1 = growth(a, cosmo, conf, order=1)
    dlnD1 = growth(a, cosmo, conf, order=1, deriv=1) / D1
    D2 = growth(a, cosmo, conf, order=2)
    dlnD2 = growth(a, cosmo, conf, order=2, deriv=1) / D2
    theta_o = jnp.array([
        D1 / a,     # G1
        D2 / a**2,  # G2
        dlnD1 - 1,  # dlnG1 / dlna
        dlnD2 - 2,  # dlnG2 / dlna
        Omega_m_a(a, cosmo),
        H_deriv(a, cosmo),
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
    theta_l = ['R_P', 'R_TH', 'R_G', 'R_d',
               'dR_P', 'dR_TH', 'dR_G', 'dR_d']
    theta_l += ['l_p', 'l_c', 'l_s']
    theta_l_k = []
    if net == 'f':
        for v in theta_l:
            theta_l_k.append(f'k{v}')
    if net == 'g':
        for n in range(3):
            for v in theta_l:
                theta_l_k.append(f'k_{n}{v}')

    theta_o = ['G1', 'G2', 'dlnG1', 'dlnG2', 'Omega_m_a', 'dlnH']

    return theta_l_k + theta_o


def soft_names_tex(net):
    # soft_names in latex math expressions
    theta_l = ['R_P', 'R_{\\rm TH}', 'R_{\\rm G}', 'R_d',
               'R_P\'', 'R_{\\rm TH}\'', 'R_{\\rm G}\'', 'R_d\'']
    theta_l += ['l_p', 'l_c', 'l_s']
    theta_l_k = []
    if net == 'f':
        for v in theta_l:
            theta_l_k.append(f'k {v}')
    if net == 'g':
        for n in range(3):
            for v in theta_l:
                theta_l_k.append(f'k_{n} {v}')

    theta_o = ['G_1', 'G_2', 'G_1\'', 'G_2\'', '\\Omega_m(a)',
               '\\ln H\'']

    return theta_l_k + theta_o


def soft_len(net):
    # get the length of SO input features
    return len(soft_names(net))
