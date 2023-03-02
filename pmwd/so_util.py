import jax
import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map
import flax.linen as nn
from flax.core.frozen_dict import unfreeze, freeze
from typing import Sequence, Callable
import math

from pmwd import H_deriv, Omega_m_a, growth


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
    """Initialize MLP parameters."""
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


# TODO add more relevant factors
def sotheta(cosmo, conf, a):
    theta_l = jnp.asarray([  # quantities of dim L
        conf.ptcl_spacing,
        conf.cell_size,
    ])
    theta_o = jnp.asarray([  # dimensionless quantities
        a,
        Omega_m_a(a, cosmo),
        growth(a, cosmo, conf),
        growth(a, cosmo, conf, deriv=1),
        H_deriv(a, cosmo),
    ])
    return (theta_l, theta_o)


def soft_len(cosmo, conf):
    tl, to = sotheta(cosmo, conf, conf.a_start)
    return len(tl) + len(to)


def soft(k, theta):
    """SO features for neural nets input."""
    # multiply each element of k with theta_l, and append theta_o
    # return is of shape k.shape + (len(theta_l) + len(theta_o),)
    theta_l, theta_o = theta
    k_shape = k.shape
    f = k.reshape(k_shape+(1,)) * theta_l.reshape((1,)*len(k_shape)+theta_l.shape)
    f = jnp.concatenate((f, jnp.broadcast_to(theta_o, k_shape+theta_o.shape)), axis=-1)
    return f


def sonn(k, theta, cosmo, conf, nid):
    """Evaluate the neural net."""
    ft = soft(k, theta)
    net = MLP(features=conf.so_nodes[nid])
    return net.apply(cosmo.so_params[nid], ft)[..., 0]  # rm the trailing axis of dim one


def pot_sharp(kvec, theta, pot, cosmo, conf, a):
    """Spatial optimization of the laplace potential."""
    f = [sonn(k_, theta, cosmo, conf, 0) for k_ in kvec]

    k = jnp.sqrt(sum(k_**2 for k_ in kvec))
    g = sonn(k, theta, cosmo, conf, 1)

    pot *= g * math.prod(f)
    return pot


def grad_sharp(k, theta, grad, cosmo, conf, a):
    """Spatial optimization of the gradient."""
    grad *= sonn(k, theta, cosmo, conf, 2)
    return grad
