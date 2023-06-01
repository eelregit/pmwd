import jax
import jax.numpy as jnp
from jax import vmap
from jax.tree_util import tree_map
import math

from pmwd.configuration import Configuration
from pmwd.cosmology import Cosmology, H_deriv, Omega_m_a, SimpleLCDM
from pmwd.boltzmann import boltzmann, growth, linear_power
from pmwd.sto.mlp import MLP


def nonlinear_scales(cosmo, conf, a):
    k = conf.transfer_k[1:]
    D = growth(a, cosmo, conf)
    dD = growth(a, cosmo, conf, deriv=1)
    dD2i = -2 * D**(-3) * dD  # d(1/D^2) / dlna
    interp_valgrad = jax.value_and_grad(jnp.interp, argnums=0)

    # dimensionless linear power
    Plin = linear_power(k, None, cosmo, conf)  # no a dependence
    k_P, dk_P = interp_valgrad(1 / D**2, k**3 * Plin / (2 * jnp.pi**2), k)
    dk_P *= dD2i

    # TopHat variance, var is decreasing with R
    # but for jnp.interp, xp must be increasing, thus the reverse [::-1]
    R_TH, dR_TH = interp_valgrad(1 / D**2, cosmo.varlin[::-1], conf.varlin_R[::-1])
    dR_TH *= dD2i

    # Gaussian variance
    R_G, dR_G = interp_valgrad(1 / D**2, cosmo.varlin_g[::-1], conf.varlin_R_g[::-1])
    dR_G *= dD2i

    # rms linear theory displacement
    Rd = (jnp.trapz(k * Plin, x=jnp.log(k)) + k[0] * Plin[0] / 2) / (6 * jnp.pi**2)
    Rd = jnp.sqrt(Rd)
    dRd = Rd * dD
    Rd *= D

    return (1/k_P, R_TH, R_G, Rd, -dk_P/k_P**2, dR_TH, dR_G, dRd)


def sotheta(cosmo, conf, a):
    # quantities of dim L
    theta_l = jnp.asarray([
        *nonlinear_scales(cosmo, conf, a),
        conf.ptcl_spacing,
        conf.cell_size,
        conf.softening_length,
    ])

    # dimensionless quantities
    D1 = growth(a, cosmo, conf, order=1)
    dlnD1 = growth(a, cosmo, conf, order=1, deriv=1) / D1
    D2 = growth(a, cosmo, conf, order=2)
    dlnD2 = growth(a, cosmo, conf, order=2, deriv=1) / D2
    theta_o = jnp.asarray([
        D1 / a,
        D2 / a**2,
        dlnD1 - 1,
        dlnD2 - 2,
        Omega_m_a(a, cosmo),
        H_deriv(a, cosmo),
        # TODO time step size?
    ])

    return (theta_l, theta_o)


def soft_len(l_fac=1):
    # get the length of SO input features with dummy conf and cosmo
    conf = Configuration(1., (128,)*3)
    cosmo = SimpleLCDM(conf)
    cosmo = boltzmann(cosmo, conf)
    theta_l, theta_o = sotheta(cosmo, conf, conf.a_start)
    return len(theta_l) * l_fac + len(theta_o)


def soft(k, theta):
    """SO features for neural nets input, with k being a scalar."""
    theta_l, theta_o = theta
    k_theta_l = k * theta_l
    # k_theta_l = jnp.log(jnp.where(k_theta_l != 0., jnp.abs(k_theta_l), 1.))
    return jnp.concatenate((k_theta_l, theta_o))


def soft_bc(k, theta):
    """SO features for neural nets input, broadcast with k being an array."""
    # multiply each element of k with theta_l, and append theta_o
    theta_l, theta_o = theta
    k_shape = k.shape
    k = k.reshape(k_shape + (1,))
    theta_l = theta_l.reshape((1,) * len(k_shape) + theta_l.shape)
    ft = k * theta_l
    # ft = jnp.log(jnp.where(ft != 0., jnp.abs(ft), 1.))
    ft = jnp.concatenate((ft, jnp.broadcast_to(theta_o, k_shape+theta_o.shape)),
                         axis=-1)
    return ft


def sonn_bc(k, theta, cosmo, conf, nid):
    """Evaluate the neural net, broadcast with k being processed in one pass."""
    ft = soft_bc(k, theta)
    net = MLP(features=conf.so_nodes[nid])
    return net.apply(cosmo.so_params[nid], ft)[..., 0]  # rm the trailing axis of dim one


def sonn_vmap(k, theta, cosmo, conf, nid):
    """Evaluate the neural net, using vmap over k."""
    net = MLP(features=conf.so_nodes[nid])
    def _sonn(_k):
        _ft = soft(_k, theta)
        return net.apply(cosmo.so_params[nid], _ft)[0]
    return vmap(_sonn)(k.ravel()).reshape(k.shape)


def sonn_g(kv, theta, cosmo, conf, nid):
    # kv shape: e.g. (128, 128, 65, 3)
    kv_shape = kv.shape
    kv = kv.reshape(kv_shape + (1,))

    theta_l, theta_o = theta
    theta_l = theta_l.reshape((1,) * len(kv_shape) + theta_l.shape)
    ft = (kv * theta_l).reshape(kv_shape[:-1] + (-1,))
    ft = jnp.concatenate((ft, jnp.broadcast_to(theta_o, kv_shape[:-1]+theta_o.shape)),
                         axis=-1)
    net = MLP(features=conf.so_nodes[nid])
    return net.apply(cosmo.so_params[nid], ft)[..., 0]  # rm the trailing axis of dim one


def pot_sharp(kvec, theta, pot, cosmo, conf, a):
    """Spatial optimization of the laplace potential."""
    if conf.so_type == 3:
        if conf.so_nodes[0] is not None:  # apply f net
            f = [sonn_bc(k_, theta, cosmo, conf, 0) for k_ in kvec]
            pot *= math.prod(f)

        if conf.so_nodes[1] is not None:  # apply g net
            k = jnp.sqrt(sum(k_**2 for k_ in kvec))
            g = sonn_bc(k, theta, cosmo, conf, 1)
            # ks = jnp.array_split(k, 16)
            # g = []
            # for k in ks:
            #     g.append(sonn_vmap(k, theta, cosmo, conf, 1))
            # g = jnp.concatenate(g, axis=0)
            pot *= g

    if conf.so_type == 2:
        kv = jnp.stack([jnp.broadcast_to(k_, pot.shape) for k_ in kvec], axis=-1)
        kv = jnp.sort(kv, axis=-1)
        g = sonn_g(kv, theta, cosmo, conf, 0)
        pot *= g

    return pot


def grad_sharp(k, theta, grad, cosmo, conf, a):
    """Spatial optimization of the gradient."""
    if conf.so_type == 3:
        if conf.so_nodes[2] is not None:  # apply h net
            grad *= sonn_bc(k, theta, cosmo, conf, 2)

    if conf.so_type == 2:
        grad *= sonn_bc(k, theta, cosmo, conf, 1)

    return grad
