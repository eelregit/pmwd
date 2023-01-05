from functools import partial

import numpy as np
from jax import core, dtypes, lax
from jax import numpy as jnp
from jax.abstract_arrays import ShapedArray
from jax.interpreters import ad, batching, mlir, xla
from jax.lib import xla_client
from jaxlib.mhlo_helpers import custom_call

from . import _jaxpmwd



def scatter(x,y):
    x_, y_ = jnp.broadcast_arrays(x, y)

    # Then we need to wrap into the range [0, 2*pi)

    return _scatter_prim.bind(x_, y_)

def _scatter_abstract_eval(x, y):
    shape = x.shape
    dtype = dtypes.canonicalize_dtype(x.dtype)
    assert dtypes.canonicalize_dtype(y.dtype) == dtype
    assert y.shape == shape
    return (ShapedArray(shape, dtype), ShapedArray(shape, dtype))

def _scatter_lowering(ctx, x, y, *, platform="gpu"):

    # Checking that input types and shape agree
    assert x.type == y.type

    # Extract the numpy type of the inputs
    x_aval, _ = ctx.avals_in
    np_dtype = np.dtype(x_aval.dtype)

    # The inputs and outputs all have the same shape and memory layout
    # so let's predefine this specification
    dtype = mlir.ir.RankedTensorType(x.type)
    dims = dtype.shape
    layout = tuple(range(len(dims) - 1, -1, -1))

    # The total size of the input is the product across dimensions
    size = np.prod(dims).astype(np.int64)

    # We dispatch a different call depending on the dtype
    if np_dtype == np.float32:
        op_name = platform + "_scatter_f32"
    elif np_dtype == np.float64:
        op_name = platform + "_scatter_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {np_dtype}")

    # And then the following is what changes between the GPU and CPU
    if platform == "cpu":
        # On the CPU, we pass the size of the data as a the first input
        # argument
        return custom_call(
            op_name,
            # Output types
            out_types=[dtype, dtype],
            # The inputs:
            operands=[mlir.ir_constant(size), x, y],
            # Layout specification:
            operand_layouts=[(), layout, layout],
            result_layouts=[layout, layout]
        )

    elif platform == "gpu":
        if gpu_ops is None:
            raise ValueError(
                "The 'kepler_jax' module was not compiled with CUDA support"
            )
        # On the GPU, we do things a little differently and encapsulate the
        # dimension using the 'opaque' parameter
        opaque = gpu_ops.build_kepler_descriptor(size)

        return custom_call(
            op_name,
            # Output types
            out_types=[dtype, dtype],
            # The inputs:
            operands=[x, y],
            # Layout specification:
            operand_layouts=[layout, layout],
            result_layouts=[layout, layout],
            # GPU specific additional data
            backend_config=opaque
        )

    raise ValueError(
        "Unsupported platform; this must be either 'cpu' or 'gpu'"
    )

_scatter_prim = Primitive("scatter")
_scatter_prim.def_impl(partial(xla.apply_primitive, _scatter_prim))
_scatter_prim.def_abstract_eval(_scatter_abstract_eval)
mlir.register_lowering(scatter_prim, _scatter_lowering, platform="gpu")
