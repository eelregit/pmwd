import jax
import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map
import flax.linen as nn
from flax.core.frozen_dict import unfreeze, freeze
from typing import Sequence, Callable
import math

from pmwd.pm_util import rfftnfreq


class MLP(nn.Module):
    features: Sequence[int]
    activator: Callable[[jnp.ndarray], jnp.ndarray] = nn.softplus
    outivator: Callable[[jnp.ndarray], jnp.ndarray] = nn.softplus

    def setup(self):
        self.layers = [nn.Dense(f) for f in self.features]

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers)-1:
                x = self.activator(x)
            else:
                if self.outivator is not None:
                    x = self.outivator(x)

        return x


def init_mlp_params(n_input, nodes, zero_params=None):
    """Initialized MLP parameters."""
    nets = [MLP(features=n) for n in nodes]
    xs = [jnp.ones(n) for n in n_input]  # dummy inputs
    keys = random.split(random.PRNGKey(0), len(n_input))

    # by default in flax.linen.Dense, kernel: lecun_norm, bias: 0
    params = [nn.init(key, x) for nn, key, x in zip(nets, keys, xs)]

    # set all params to zero
    if zero_params == 'all':
        params = tree_map(lambda x: jnp.zeros_like(x), params)

    # set the params of the last layer to zero, bias = zero by default
    if zero_params == 'last':
        for i, p in enumerate(params):
            p = unfreeze(p)
            p['params'][f'layers_{len(nodes)-1}']['kernel'] = (
                jnp.zeros((nodes[i][-2], 1)))
            params[i] = freeze(p)

    return params


def soft_3d(k, P):
    """Input SO features for g(k) net, with k being 3D wavenumber.

    Paramters
    ---------
        k: float
            3D wavenumber
        P: (cosmo, conf, a)
    """
    cosmo, conf, a = P
    fts = jnp.array([
        k * conf.cell_size,
        k * conf.box_size[0],  # TODO more features
    ])
    return fts


def soft_1d(k, P):
    """Input SO features for f(k) net, with k being 1D wavenumber.

    Paramters
    ---------
        k: float
            1D wavenumber
        P: (cosmo, conf, a)
    """
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
    k = jnp.sqrt(sum(kv**2 for kv in kvec))
    P = (cosmo, conf, a)
    ft_k = jax.vmap(soft_3d, (0, None), 0)(k.ravel(), P).reshape(k.shape+(-1,))
    ft_kvec = [jax.vmap(soft_1d, (0, None), 0)(kv.ravel(), P).reshape(
        kv.shape+(-1,)) for kv in kvec]
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
    g_k = nets[0].apply(cosmo.so_params[0], fts[0])
    g_k = g_k.reshape(g_k.shape[:-1])  # remove the trailing axis of dim one
    f_kvec = [nets[1].apply(cosmo.so_params[1], ft) for ft in fts[1]]
    f_kvec = [f.reshape(f.shape[:-1]) for f in f_kvec]
    pot *= g_k * math.prod(f_kvec)
    return pot


def gnet(k, params, cosmo, conf, a):
    """Function g(k) neural net, where k is 3D wavenumber."""
    if k == 'rfftn':
        kvec = rfftnfreq(conf.mesh_shape, conf.cell_size, dtype=conf.float_dtype)
        k = jnp.sort(jnp.sqrt(sum(kv**2 for kv in kvec)).ravel())
    fts = jax.vmap(soft_3d, (0, None), 0)(k, (cosmo, conf, a))
    net = MLP(features=conf.so_nodes[0])
    return net.apply(params, fts), k


def fnet(k, params, cosmo, conf, a):
    """Function f(k) neural net, where k is 1D wavenumber."""
    if k == 'rfftn':
        kvec = rfftnfreq(conf.mesh_shape, conf.cell_size, dtype=conf.float_dtype)
        k = jnp.sort(kvec[0].ravel())
    fts = jax.vmap(soft_1d, (0, None), 0)(k, (cosmo, conf, a))
    net = MLP(features=conf.so_nodes[1])
    return net.apply(params, fts), k
