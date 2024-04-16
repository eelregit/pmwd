import jax.numpy as jnp
from jax import random, jit
from typing import Sequence, Callable
import flax.linen as nn
from flax.core.frozen_dict import unfreeze, freeze
from flax.linen.initializers import he_normal, zeros_init


@jit
def squareplus(x, b=4):
  """Squareplus activation function in https://arxiv.org/abs/2112.11687."""
  y = x + jnp.sqrt(jnp.square(x) + b)
  return y / 2


class MLP(nn.Module):
    features: Sequence[int]
    activator: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    regulator: Callable[[jnp.ndarray], jnp.ndarray] = jnp.exp
    kernel_init: Callable = he_normal()
    bias_init: Callable = zeros_init()

    @nn.compact
    def __call__(self, x):
        # hidden layers
        for i, fts in enumerate(self.features[:-1]):
            x = nn.Dense(fts, param_dtype=jnp.float64, kernel_init=self.kernel_init,
                         bias_init=self.bias_init)(x)
            x = self.activator(x)

        # output layer
        x = nn.Dense(self.features[-1], param_dtype=jnp.float64,
                     kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        if self.regulator is not None:
            x = self.regulator(x)

        return x


def init_mlp_params(n_input, nodes, kernel_init=he_normal(), bias_init=zeros_init(),
                    scheme=None, last_ws=1e-8, last_b=0, seed=42):
    """Initialize MLP parameters."""
    nets = [MLP(features=n, kernel_init=kernel_init, bias_init=bias_init) for n in nodes]
    xs = [jnp.ones(n) for n in n_input]  # dummy inputs
    keys = random.split(random.PRNGKey(seed), len(n_input))

    params = [nn.init(key, x) for nn, key, x in zip(nets, keys, xs)]

    # for the last layer: set bias to the given value & weights to zero
    if scheme == 'last_w0':
        for i, p in enumerate(params):
            p = unfreeze(p)
            p['params'][f'Dense_{len(nodes[i])-1}']['kernel'] = (
                jnp.zeros((nodes[i][-2], nodes[i][-1])))
            p['params'][f'Dense_{len(nodes[i])-1}']['bias'] = (
                jnp.full(nodes[i][-1], last_b, dtype=jnp.float64))
            params[i] = freeze(p)

    # for the last layer: set bias to the given value & weights to small random values
    if scheme == 'last_ws':
        keys = random.split(random.PRNGKey(seed+1), len(params))
        for i, p in enumerate(params):
            p = unfreeze(p)
            p['params'][f'Dense_{len(nodes[i])-1}']['kernel'] = (
                random.normal(keys[i], (nodes[i][-2], nodes[i][-1]))) * last_ws
            p['params'][f'Dense_{len(nodes[i])-1}']['bias'] = (
                jnp.full(nodes[i][-1], last_b, dtype=jnp.float64))
            params[i] = freeze(p)

    return params


def mlp_size(mlp_params):
    """Infer the sizes of input and hidden layers given a list of MLP params."""
    n_input, n_nodes = [], []
    for params in mlp_params:
        dic = params['params']
        n_input.append(dic['Dense_0']['kernel'].shape[0])
        n_nodes.append([dic[f'Dense_{i}']['kernel'].shape[1] for i in range(len(dic))])

    return n_input, n_nodes
