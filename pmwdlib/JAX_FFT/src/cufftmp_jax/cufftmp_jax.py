# -*- coding: utf-8 -*-

__all__ = ["cufftmp"]

from functools import partial
import math

import jax
from jax.lib import xla_client
from jax import core, dtypes
from jax.interpreters import xla, mlir
from jax.abstract_arrays import ShapedArray
# from jax._src.sharding_impls import NamedSharding
from jax.experimental.custom_partitioning import custom_partitioning
from jaxlib.hlo_helpers import custom_call

from fft_common import Dir, Dist

from . import gpu_ops
for _name, _value in gpu_ops.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")

xops = xla_client.ops

# ************
# * BINDINGS *
# ************


def _cufftmp_bind(input, num_parts, rank, size, dist, dir):

    # param=val means it's a static parameter
    (output,) = _cufftmp_prim.bind(input,
                                   num_parts=num_parts,
                                   rank=rank,
                                   size=size,
                                   dist=dist,
                                   dir=dir)

    return output



def cufftmp(x, rank, size, dist, dir):

    """Compute the DFT using a JAX+cuFFTMp implementation.

    Arguments:
    x    -- the input tensor
    dist -- the data decomposition of x.
            Should be an instance of fft_common.Dist
    dir  -- the direction of the transform.
            Should be an instance of fft_common.Dir

    Returns the transformed tensor.
    The output tensoris distributed according to dist.opposite

    This function should be used with pjit like

        pjit(
             cufftmp,
             in_axis_resources=dist.part_spec,
             out_axis_resources=dist.opposite.part_spec,
             static_argnums=[1, 2]
            )(x, dist, dir)

    """

    # cuFFTMp only supports 1 device per proces
    # assert jax.local_device_count() == 1

    @custom_partitioning
    def _cufftmp_(x):
        return _cufftmp_bind(x, num_parts=1, rank=rank, size=size, dist=dist, dir=dir)

    return _cufftmp_(x)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

# Abstract implementation, i.e., return the shape of the output array
# based on the input array and a number of partitions (ie devices)
def _cufftmp_abstract(input, num_parts, rank, size, dist, dir):
    input_dtype = dtypes.canonicalize_dtype(input.dtype)
    input_shape = input.shape

    if dir == Dir.INV:
        if input_dtype == jax.numpy.complex64:
            output_dtype = jax.numpy.float32
        else:
            output_dtype = jax.numpy.float64

        output_shape = (input_shape[0],
                        input_shape[1],
                        2*(input_shape[2]-1))
    else:
        if input_dtype == jax.numpy.float32:
            output_dtype = jax.numpy.complex64
        else:
            output_dtype = jax.numpy.complex128

        output_shape = (input_shape[0],
                        input_shape[1],
                        (input_shape[2] // 2 + 1))

    return (ShapedArray(output_shape, output_dtype),)


# Implementation calling into the C++ bindings
def _cufftmp_translation(ctx, input, num_parts, rank, size, dist, dir):
    input_type = mlir.ir.RankedTensorType(input.type)
    dims_in = input_type.shape
    f32Type = mlir.ir.F32Type.get()
    c32Type = mlir.ir.ComplexType.get(mlir.ir.F32Type.get())

    if dir == Dir.INV:
        if input_type.element_type == c32Type:
            outType = mlir.ir.F32Type.get()
            op_name = "gpu_cufftmp_f32"
        else:
            outType = mlir.ir.F64Type.get()
            op_name = "gpu_cufftmp_f64"

        dims_out = (dims_in[0], dims_in[1], 2*(dims_in[2]-1))
    else:
        if input_type.element_type == f32Type:
            outType = mlir.ir.ComplexType.get(mlir.ir.F32Type.get())
            op_name = "gpu_cufftmp_f32"
        else:
            outType = mlir.ir.ComplexType.get(mlir.ir.F64Type.get())
            op_name = "gpu_cufftmp_f64"

        dims_out = (dims_in[0], dims_in[1], (dims_in[2] // 2 + 1))

    output_type = mlir.ir.RankedTensorType.get(
        dims_out,
        outType
    )

    layout = tuple(range(len(dims_in) - 1, -1, -1))

    opaque = gpu_ops.build_cufftmp_descriptor(
        dims_in[0],
        dims_in[1],
        dims_in[2],
        rank,
        size,
        dist._C_enum,
        dir._C_enum
    )

    return [custom_call(
        op_name,
        # Output types
        out_types=[output_type],
        # The inputs:
        operands=[input,],
        # Layout specification:
        operand_layouts=[layout,],
        result_layouts=[layout,],
        # GPU specific additional data
        backend_config=opaque
    )]


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_cufftmp_prim = core.Primitive("cufftmp")
_cufftmp_prim.multiple_results = True
_cufftmp_prim.def_impl(partial(xla.apply_primitive, _cufftmp_prim))
_cufftmp_prim.def_abstract_eval(_cufftmp_abstract)

# Register the op with MLIR
mlir.register_lowering(_cufftmp_prim, _cufftmp_translation, platform="gpu")
