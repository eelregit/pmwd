import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence


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


def sofea_k(k, P):
    """Input features for g(k) net, with k being 3D wavenumber."""
    cosmo, conf = P
    fts = jnp.array([
        k * conf.cell_size,
        k * conf.box_size[0],  #TODO more features
    ])
    return fts


def sofea_kvec(k, P):
    """Input features for f(k) net, with k being 1D wavenumber."""
    cosmo, conf = P
    fts = jnp.array([
        k * conf.cell_size,
        k * conf.box_size[0],  #TODO more features
    ])
    return fts


def sofeatures(kvec, cosmo, conf):
    """Input spatial optim features for the neural nets.

    Returns
    -------
        ft_k: array
            features for k
        ft_kvec: list
            list of feature arrays for kvec
    """
    k = jnp.sqrt(sum(k**2 for k in kvec))
    P = cosmo, conf
    ft_k = jax.vmap(sofea_k, (0, None), 0)(
        k.ravel(), P).reshape(k.shape+(-1,))
    ft_kvec = [jax.vmap(sofea_kvec, (0, None), 0)(
        kv.ravel(), P).reshape(kv.shape+(-1,))
        for kv in kvec]
    return ft_k, ft_kvec
