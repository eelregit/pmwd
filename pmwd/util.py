import numpy as np
from jax.dtypes import float0


def is_float0_array(obj):
    return isinstance(obj, np.ndarray) and obj.dtype == float0


def float0_like(array):
    return np.empty(array.shape, dtype=float0)  # see JAX issues #4433, #19386, #20620


def add(a, b):
    """Like ``operator.add`` with simplest ``float0`` handling (no broadcasting)."""
    if is_float0_array(a) and is_float0_array(b):
        if a.shape != b.shape:
            raise ValueError(f'float0 array shapes mismatch: {a.shape} != {b.shape}')
        return a
    return a + b


def sub(a, b):
    """Like ``operator.sub`` with simplest ``float0`` handling (no broadcasting)."""
    if is_float0_array(a) and is_float0_array(b):
        if a.shape != b.shape:
            raise ValueError(f'float0 array shapes mismatch: {a.shape} != {b.shape}')
        return a
    return a - b


def neg(obj):
    """Like ``operator.neg`` with ``float0`` handling."""
    return obj if is_float0_array(obj) else -obj


def scalar_mul(scalar, array):
    """Scalar multiplication with ``float0`` handling."""
    if is_float0_array(array):
        return array
    return scalar * array


def scalar_div(scalar, array):
    """Division by scalar with ``float0`` handling."""
    if is_float0_array(array):
        return array
    return array / scalar
