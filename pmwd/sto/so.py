import jax.numpy as jnp
from jax import vmap
import math

from pmwd.sto.mlp import MLP
from pmwd.sto.soft_a import (
    _sotheta, _soft_names, _soft_len, soft, soft_k, soft_kvec)


def sotheta(cosmo, conf, a):
    return _sotheta(cosmo, conf, a)


def soft_names():
    return _soft_names()


def soft_len(k_fac=1):
    return _soft_len(k_fac=k_fac)


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


def sonn_k(k, theta, cosmo, conf, nid):
    """SO net of 1D k input."""
    ft = soft_k(k, theta)
    return apply_net(nid, conf, cosmo, ft)[..., 0]  # rm the trailing axis of dim one


def sonn_kvec(kv, theta, cosmo, conf, nid):
    """SO net of 3D k input, with permutation symmetry."""
    kv = jnp.sort(kv, axis=-1)  # sort for permutation symmetry of kv components
    ft = soft_kvec(kv, theta)
    return apply_net(nid, conf, cosmo, ft)[..., 0]  # rm the trailing axis of dim one


def pot_sharp(pot, kvec, theta, cosmo, conf, a):
    """SO of the laplace potential, function of 3D k vector."""
    kvec = map(jnp.abs, kvec)

    if conf.so_type == 'NN':
        if conf.so_nodes[0] is not None:
            kv = jnp.stack([jnp.broadcast_to(k_, pot.shape) for k_ in kvec], axis=-1)
            g = sonn_kvec(kv, theta, cosmo, conf, 0)
            pot *= g

    return pot


def grad_sharp(grad, k, theta, cosmo, conf, a):
    """SO of the gradient, function of 1D k component."""
    k = jnp.abs(k)

    if conf.so_type == 'NN':
        if conf.so_nodes[1] is not None:
            grad *= sonn_k(k, theta, cosmo, conf, 1)

    return grad
