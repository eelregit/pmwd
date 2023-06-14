import jax.numpy as jnp
from jax import random
import flax.linen as nn
from typing import Sequence, Callable
from flax.core.frozen_dict import unfreeze, freeze


class MLP(nn.Module):
    features: Sequence[int]
    activator: Callable[[jnp.ndarray], jnp.ndarray] = nn.softplus
    outivator: Callable[[jnp.ndarray], jnp.ndarray] = None

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


def init_mlp_params(n_input, nodes, scheme=None):
    """Initialize MLP parameters."""
    nets = [MLP(features=n) for n in nodes]
    xs = [jnp.ones(n) for n in n_input]  # dummy inputs
    keys = random.split(random.PRNGKey(0), len(n_input))

    # by default in flax.linen.Dense, kernel: lecun_norm, bias: 0
    params = [nn.init(key, x) for nn, key, x in zip(nets, keys, xs)]

    # for the last layer: set bias to one & weights to zero
    if scheme == 'last_w0_b1':
        for i, p in enumerate(params):
            p = unfreeze(p)
            p['params'][f'layers_{len(nodes[i])-1}']['kernel'] = (
                jnp.zeros((nodes[i][-2], nodes[i][-1])))
            p['params'][f'layers_{len(nodes[i])-1}']['bias'] = (
                jnp.ones(nodes[i][-1]))
            params[i] = freeze(p)

    # for the last layer: set bias to one & weights to small random values
    if scheme == 'last_ws_b1':
        keys = random.split(random.PRNGKey(1), len(params))
        for i, p in enumerate(params):
            p = unfreeze(p)
            p['params'][f'layers_{len(nodes[i])-1}']['kernel'] = (
                random.normal(keys[i], (nodes[i][-2], nodes[i][-1]))) * 1e-4
            p['params'][f'layers_{len(nodes[i])-1}']['bias'] = (
                jnp.ones(nodes[i][-1]))
            params[i] = freeze(p)

    return params


def mlp_size(mlp_params):
    """Infer the sizes of input and hidden layers given a list of MLP params."""
    n_input, n_nodes = [], []
    for params in mlp_params:
        dic = params['params']
        n_input.append(dic['layers_0']['kernel'].shape[0])
        n_nodes.append([dic[f'layers_{i}']['kernel'].shape[1] for i in range(len(dic))])

    return n_input, n_nodes
