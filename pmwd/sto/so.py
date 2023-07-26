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
        conf.a_nbody_step / a,  # time step size dlna ~ da/a
    ])

    return (theta_l, theta_o)


def sotheta_names():
    # str names of the variables returned by the sotheta function above
    # currently hardcoded, should be updated along with sotheta function
    pass


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
    return jnp.concatenate((k_theta_l, theta_o))


def apply_net(nid, conf, cosmo, x):
    net = MLP(features=conf.so_nodes[nid])
    if conf.dropout_rate is not None:
        dropout = True
        rngs = {'dropout': jnp.asarray(conf.dropout_key, dtype=jnp.uint32)}
    else:
        dropout = False
        rngs = None
    return net.apply(cosmo.so_params[nid], x, dropout=dropout,
                     dropout_rate=conf.dropout_rate, rngs=rngs)


def sonn_vmap(k, theta, cosmo, conf, nid):
    """Evaluate the neural net, using vmap over k."""
    def _sonn(_k):
        _ft = soft(_k, theta)
        return apply_net(nid, conf, cosmo, _ft)[0]
    return vmap(_sonn)(k.ravel()).reshape(k.shape)


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


def sonn_k(k, theta, cosmo, conf, nid):
    """SO net with input: (k * l, o)."""
    ft = soft_k(k, theta)
    return apply_net(nid, conf, cosmo, ft)[..., 0]  # rm the trailing axis of dim one


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


def sonn_kvec(kv, theta, cosmo, conf, nid):
    """SO net with input: (k1 * l, k2 * l, k3 * l, o)."""
    ft = soft_kvec(kv, theta)
    return apply_net(nid, conf, cosmo, ft)[..., 0]  # rm the trailing axis of dim one


def pot_sharp(pot, kvec, theta, cosmo, conf, a):
    """SO of the laplace potential, function of 3D k vector."""
    kvec = map(jnp.abs, kvec)

    if conf.so_type == 3:
        if conf.so_nodes[0] is not None:
            pot *= math.prod([sonn_k(k_, theta, cosmo, conf, 0) for k_ in kvec])

        if conf.so_nodes[1] is not None:
            k = jnp.sqrt(sum(k_**2 for k_ in kvec))
            g = sonn_k(k, theta, cosmo, conf, 1)
            # ks = jnp.array_split(k, 16)
            # g = []
            # for k in ks:
            #     g.append(sonn_vmap(k, theta, cosmo, conf, 1))
            # g = jnp.concatenate(g, axis=0)
            pot *= g

    if conf.so_type == 2:
        if conf.so_nodes[0] is not None:
            kv = jnp.stack([jnp.broadcast_to(k_, pot.shape) for k_ in kvec], axis=-1)
            kv = jnp.sort(kv, axis=-1)  # sort for permutation symmetry
            g = sonn_kvec(kv, theta, cosmo, conf, 0)
            pot *= g

    return pot


def grad_sharp(grad, k, theta, cosmo, conf, a):
    """SO of the gradient, function of 1D k component."""
    k = jnp.abs(k)

    if conf.so_type == 3:
        if conf.so_nodes[2] is not None:
            grad *= sonn_k(k, theta, cosmo, conf, 2)

    if conf.so_type == 2:
        if conf.so_nodes[1] is not None:
            grad *= sonn_k(k, theta, cosmo, conf, 1)

    return grad
