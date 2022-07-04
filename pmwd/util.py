import numpy as np
from jax import float0


def is_float0_array(x):
    return isinstance(x, np.ndarray) and x.dtype == float0


def float0_like(x):
    return np.empty(x.shape, dtype=float0)  # see jax issue #4433
