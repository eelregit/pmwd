# -*- coding: utf-8 -*-

__all__ = ["nvtransfer"]

from functools import partial
import math

import jax
from jax.lib import xla_client
from jax import core, dtypes
from jax.interpreters import xla, mlir
from jax.abstract_arrays import ShapedArray
from jax.experimental.custom_partitioning import custom_partitioning
from jaxlib.hlo_helpers import custom_call

from . import gpu_ops
for _name, _value in gpu_ops.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")


# ************
# * BINDINGS *
# ************
def _nvtransfer_bind(send_buf, send_buf_size, recv_buf_size, src_rank, dst_rank):

    # param=val means it's a static parameter
    (recv_buf,) = _nvtransfer_prim.bind(send_buf,
                                        send_buf_size=send_buf_size,        # workaround for output_shape is 0, call lowering function fail
                                        recv_buf_size=recv_buf_size+1,     
                                        src_rank=src_rank,
                                        dst_rank=dst_rank
                                        )

    return recv_buf[:-1]                        # recv_buf workaround for output_shape is 0, call lowering function fail


def nvtransfer(send_buf, send_buf_size, recv_buf_size, src_rank, dst_rank):
    """Compute the DFT using a JAX+cuFFTMp implementation.

    Arguments:
    send_buf        -- the input tensor
    send_buf_size   -- the data size need to send
    recv_buf_size   -- the data size need to recv
    src_rank        -- the rank recv from src
    dst_rank        -- the rank send to dst

    Returns the recv data.
    The output tensoris distributed according to dist.opposite

    This function should be used with pjit like

    """

    @custom_partitioning
    def _nvtransfer_(send_buf):
        return _nvtransfer_bind(send_buf, send_buf_size=send_buf_size, recv_buf_size=recv_buf_size, src_rank=src_rank, dst_rank=dst_rank)

    return _nvtransfer_(send_buf)
    # return _nvtransfer_bind(send_buf, send_buf_size=send_buf_size, recv_buf_size=recv_buf_size, src_rank=src_rank, dst_rank=dst_rank)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

# Abstract implementation, i.e., return the shape of the output array
# based on the send array and a number of partitions (ie devices)
def _nvtransfer_abstract(send_buf, send_buf_size, recv_buf_size, src_rank, dst_rank):
    output_dtype = dtypes.canonicalize_dtype(send_buf.dtype)
    # output_shape = send_buf.shape
    # output_shape = (recv_buf_size,)
    output_shape = (recv_buf_size,)

    return (ShapedArray(output_shape, output_dtype),)


# Implementation calling into the C++ bindings
def _nvtransfer_translation(ctx, send_buf, send_buf_size, recv_buf_size, src_rank, dst_rank):
    input_type = mlir.ir.RankedTensorType(send_buf.type)
    dims_in = input_type.shape
    i16Type = mlir.ir.IntegerType.get_signless(16)
    i32Type = mlir.ir.IntegerType.get_signless(32)
    f32Type = mlir.ir.F32Type.get()
    f64Type = mlir.ir.F64Type.get()

    if input_type.element_type == i16Type:
        outType = i16Type
        op_name = "gpu_nvtransfer_i16"
    elif input_type.element_type == i32Type:
        outType = i32Type
        op_name = "gpu_nvtransfer_i32"           
    elif input_type.element_type == f32Type:
        outType = f32Type
        op_name = "gpu_nvtransfer_f32"        
    elif input_type.element_type == f64Type:
        outType = f64Type
        op_name = "gpu_nvtransfer_f64"
    else :
        print('fatal error : Unsupported type')

    dims_out = dims_in
    dims_out[0] = recv_buf_size

    output_type = mlir.ir.RankedTensorType.get(
        dims_out,
        outType)

    layout = tuple(range(len(dims_in) - 1, -1, -1))

    opaque = gpu_ops.build_nvtransfer_descriptor(
        1, send_buf_size, recv_buf_size, src_rank, dst_rank)

    return [custom_call(
        op_name,
        # Output types
        out_types=[output_type],
        # The inputs:
        operands=[send_buf,],
        # Layout specification:
        operand_layouts=[layout,],
        result_layouts=[layout,],
        # GPU specific additional data
        backend_config=opaque
    )]


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_nvtransfer_prim = core.Primitive("nvtransfer")
_nvtransfer_prim.multiple_results = True
_nvtransfer_prim.def_impl(partial(xla.apply_primitive, _nvtransfer_prim))
_nvtransfer_prim.def_abstract_eval(_nvtransfer_abstract)

# Register the op with MLIR
mlir.register_lowering(
    _nvtransfer_prim, _nvtransfer_translation, platform="gpu")
