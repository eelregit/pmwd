import numpy as np
from jax import float0
import jax.numpy as jnp


def is_float0_array(x):
    return isinstance(x, np.ndarray) and x.dtype == float0


def float0_like(x):
    return np.empty(x.shape, dtype=float0)  # see jax issue #4433


def bincount(x, weights, *, length, dtype=None):
    """Quick fix before is released."""
    if dtype is None:
        dtype = jnp.dtype(weights)
    return jnp.zeros(length, dtype=dtype).at[jnp.clip(x, 0)].add(weights)
