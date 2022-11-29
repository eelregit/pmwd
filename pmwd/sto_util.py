import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.core.frozen_dict import unfreeze, freeze
from typing import Sequence
import math
from pmwd.pm_util import rfftnfreq


class MLP(nn.Module):
    features: Sequence[int]

    def setup(self):
        self.layers = [nn.Dense(f) for f in self.features]

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers)-1:
                x = nn.relu(x)
        return x


def init_mlp_params(n_input, nodes, zero_last_weights=False):
    """Initialized MLP parameters."""
    nets = [MLP(features=n) for n in nodes]
    xs = [jnp.ones(n) for n in n_input]  # dummy inputs
    keys = random.split(random.PRNGKey(0), len(n_input))
    params = [nn.init(key, x) for nn, key, x in zip(nets, keys, xs)]

    if zero_last_weights:  # set the weights of the last layer to zero
        for i, p in enumerate(params):
            p = unfreeze(p)
            p['params'][f'layers_{len(nodes)-1}']['kernel'] = (
                jnp.zeros((nodes[i][-2], 1)))
            params[i] = freeze(p)

    return params


def sofea_k(k, P):
    """Input features for g(k) net, with k being 3D wavenumber."""
    cosmo, conf, a = P
    fts = jnp.array([
        k * conf.cell_size,
        k * conf.box_size[0],  # TODO more features
    ])
    return fts


def sofea_kvec(k, P):
    """Input features for f(k) net, with k being 1D wavenumber."""
    cosmo, conf, a = P
    fts = jnp.array([
        k * conf.cell_size,
        k * conf.box_size[0],  # TODO more features
    ])
    return fts


def sofeatures(kvec, cosmo, conf, a):
    """Input spatial optim features for the neural nets.

    Returns
    -------
        ft_k: array
            features for k
        ft_kvec: list
            list of feature arrays for kvec
    """
    k = jnp.sqrt(sum(k**2 for k in kvec))
    P = (cosmo, conf, a)
    ft_k = jax.vmap(sofea_k, (0, None), 0)(
        k.ravel(), P).reshape(k.shape+(-1,))
    ft_kvec = [jax.vmap(sofea_kvec, (0, None), 0)(
        kv.ravel(), P).reshape(kv.shape+(-1,))
        for kv in kvec]
    return ft_k, ft_kvec


def sharpening(pot, cosmo, conf, a):
    """Apply the spatial optimization to the laplace potential.
    """
    kvec = rfftnfreq(conf.mesh_shape, conf.cell_size, dtype=conf.float_dtype)
    # input features of the neural nets
    # 0 for k and 1 for kvec
    fts = sofeatures(kvec, cosmo, conf, a)
    # neural nets
    nets = [MLP(features=n) for n in conf.so_nodes]
    # modification factors to the laplace potential
    g_k = 1 + nets[0].apply(cosmo.so_params[0], fts[0])
    g_k = g_k.reshape(g_k.shape[:-1])  # remove the trailing axis of dim one
    f_kvec = [1 + nets[1].apply(cosmo.so_params[1], ft) for ft in fts[1]]
    f_kvec = [f.reshape(f.shape[:-1]) for f in f_kvec]
    pot *= g_k * math.prod(f_kvec)
    return pot
