import jax.numpy as jnp
from jax import vmap
import math

from pmwd.sto.mlp import MLP
from pmwd.sto.soft import (
    _sotheta, _sotheta_names, _soft_len, soft, soft_k, soft_kvec)


def sotheta(cosmo, conf, a):
    return _sotheta(cosmo, conf, a)


def sotheta_names():
    return _sotheta_names()


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
    """SO net with 1D k input."""
    ft = soft_k(k, theta)
    return apply_net(nid, conf, cosmo, ft)[..., 0]  # rm the trailing axis of dim one


def sonn_kvec(kv, theta, cosmo, conf, nid):
    """SO net with 3D k input."""
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
