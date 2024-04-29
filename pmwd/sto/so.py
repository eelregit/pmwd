import jax
import jax.numpy as jnp
from jax import jit, vmap, checkpoint
from functools import partial

from pmwd.sto.mlp import MLP
from pmwd.sto import (
    soft_v1,
    soft_v2, soft_v2_1, soft_v2_2,
    soft_v3,
    soft_v4)


def mod_soft_i(soft_i):
    match soft_i:
        case 'soft_v1': soft_i = soft_v1
        case 'soft_v2': soft_i = soft_v2
        case 'soft_v2_1': soft_i = soft_v2_1
        case 'soft_v2_2': soft_i = soft_v2_2
        case 'soft_v3': soft_i = soft_v3
        case 'soft_v4': soft_i = soft_v4
    return soft_i


def sotheta(cosmo, conf, a, soft_i=None):
    if soft_i is None:
        soft_i = conf.soft_i
    soft_i = mod_soft_i(soft_i)
    return soft_i.sotheta(cosmo, conf, a)


def soft_names(soft_i, net):
    soft_i = mod_soft_i(soft_i)
    return soft_i.soft_names(net)


def soft_names_tex(soft_i, net):
    soft_i = mod_soft_i(soft_i)
    return soft_i.soft_names_tex(net)


def soft_len(soft_i, net):
    soft_i = mod_soft_i(soft_i)
    return soft_i.soft_len(net)


def soft(soft_i, k, theta):
    soft_i = mod_soft_i(soft_i)
    return soft_i.soft(k, theta)


def soft_k(soft_i, k, theta):
    soft_i = mod_soft_i(soft_i)
    return soft_i.soft_k(k, theta)


def soft_kvec(soft_i, kvec, theta):
    soft_i = mod_soft_i(soft_i)
    return soft_i.soft_kvec(kvec, theta)


def apply_net(nid, conf, cosmo, x):
    net = MLP(features=conf.so_nodes[nid])
    return net.apply(cosmo.so_params[nid], x)


def sonn_vmap(k, theta, cosmo, conf, nid):
    """Evaluate the neural net, using vmap over k."""
    def _sonn(_k):
        _ft = soft(conf.soft_i, _k, theta)
        return apply_net(nid, conf, cosmo, _ft)[0]
    return vmap(_sonn)(k.ravel()).reshape(k.shape)


def sonn_k(k, theta, cosmo, conf, nid):
    """SO net of 1D k input."""
    ft = soft_k(conf.soft_i, k, theta)
    return apply_net(nid, conf, cosmo, ft)[..., 0]  # rm the trailing axis of dim one


def sonn_kvec(kv, theta, cosmo, conf, nid):
    """SO net of 3D k input, with permutation symmetry."""
    kv = jnp.sort(kv, axis=-1)  # sort for permutation symmetry of kv components
    ft = soft_kvec(conf.soft_i, kv, theta)
    return apply_net(nid, conf, cosmo, ft)[..., 0]  # rm the trailing axis of dim one


def pot_sharp(pot, kvec, theta, cosmo, conf, a):
    """SO of the laplace potential, function of 3D k vector (g function)."""
    kvec = map(jnp.abs, kvec)  # even function

    if conf.so_type == 'NN' and conf.so_nodes[0] is not None:
        kv = jnp.stack([jnp.broadcast_to(k_, pot.shape) for k_ in kvec], axis=-1)
        @checkpoint  # checkpoint for saving memory in backward AD
        def sonn_kvec_slice(k_):
            return sonn_kvec(k_, theta, cosmo, conf, 0)
        # map for reduced memeory usage in the forward run
        g = jax.lax.map(sonn_kvec_slice, kv)
        pot *= g

    return pot


def grad_sharp(grad, k, theta, cosmo, conf, a):
    """SO of the gradient, function of 1D k component (f function)."""
    k = jnp.abs(k)  # even function

    if conf.so_type == 'NN' and conf.so_nodes[1] is not None:
        grad *= sonn_k(k, theta, cosmo, conf, 1)

    return grad
