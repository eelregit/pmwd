import jax.numpy as jnp
from jax import random, jit
from typing import Sequence, Callable
from jax.typing import DTypeLike
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
    dtype: DTypeLike = jnp.float32  # dtype for computation
    param_dtype: DTypeLike = jnp.float64  # dtype for parameters
    kernel_init: Callable = he_normal()
    bias_init: Callable = zeros_init()
    activator: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    regulator: Callable[[jnp.ndarray], jnp.ndarray] = None

    @nn.compact
    def __call__(self, x):
        # hidden layers
        for i, fts in enumerate(self.features[:-1]):
            x = nn.Dense(fts, dtype=self.dtype, param_dtype=self.param_dtype,
                         kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
            x = self.activator(x)

        # output layer
        x = nn.Dense(self.features[-1], dtype=self.dtype, param_dtype=self.param_dtype,
                     kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        if self.regulator is not None:
            x = self.regulator(x)

        return x


def init_mlp_params(n_input_list, nodes_list, seed=42,
                    dtype=jnp.float32, param_dtype=jnp.float64,
                    kernel_init=he_normal(), bias_init=zeros_init(),
                    scheme=None, last_ws=1e-8, last_b=0.):
    """Initialize parameters for a list of MLPs."""
    nets = [MLP(features=nodes, kernel_init=kernel_init, bias_init=bias_init,
                dtype=dtype, param_dtype=param_dtype) for nodes in nodes_list]

    # initialize parameters with dummy inputs
    xs = [jnp.ones(n, dtype=dtype) for n in n_input_list]
    keys = random.split(random.PRNGKey(seed), len(n_input_list))
    params = [nn.init(key, x) for nn, key, x in zip(nets, keys, xs)]

    # for the last layer: set bias to the given value & weights to zero
    if scheme == 'last_w0':
        for i, (p, nodes) in enumerate(zip(params, nodes_list)):
            p = unfreeze(p)
            p['params'][f'Dense_{len(nodes)-1}']['kernel'] = (
                jnp.zeros((nodes[-2], nodes[-1]), dtype=param_dtype))
            p['params'][f'Dense_{len(nodes)-1}']['bias'] = (
                jnp.full(nodes[-1], last_b, dtype=param_dtype))
            params[i] = freeze(p)

    # for the last layer: set bias to the given value & weights to small random values
    if scheme == 'last_ws':
        keys = random.split(random.PRNGKey(seed+1), len(params))
        for i, (p, nodes) in enumerate(zip(params, nodes_list)):
            p = unfreeze(p)
            p['params'][f'Dense_{len(nodes)-1}']['kernel'] = (
                random.normal(keys[i], (nodes[-2], nodes[-1]), dtype=param_dtype)
                ) * last_ws
            p['params'][f'Dense_{len(nodes)-1}']['bias'] = (
                jnp.full(nodes[-1], last_b, dtype=param_dtype))
            params[i] = freeze(p)

    return params


def mlp_size(params_list):
    """Infer the sizes of input and hidden layers given a list of MLP params."""
    n_input, n_nodes = [], []
    for params in params_list:
        dic = params['params']
        n_input.append(dic['Dense_0']['kernel'].shape[0])
        n_nodes.append([dic[f'Dense_{i}']['kernel'].shape[1] for i in range(len(dic))])

    return tuple(n_input), tuple(n_nodes)
