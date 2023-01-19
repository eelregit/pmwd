from functools import partial

import numpy as np
import jaxlib.mlir.ir as ir
from jax import core, dtypes, lax
from jax.core import Primitive
from jax import numpy as jnp
from jax.abstract_arrays import ShapedArray
from jax.interpreters import ad, batching, mlir, xla
from jax import jit
from jax.lib import xla_client
from jaxlib.mhlo_helpers import custom_call

from . import _jaxpmwd

### define scatter op
@partial(jit, static_argnums=(0,1,2,3,4,5))
def scatter(pmid, disp, val, mesh, offset, ptcl_spacing, cell_size):

    return _scatter_prim.bind(pmid, disp, val, mesh, offset, cell_size)

def _scatter_abstract_eval(pmid, disp, val, mesh, offset, ptcl_spacing, cell_size):
    shape = mesh.shape
    dtype = dtypes.canonicalize_dtype(val.dtype)
    assert dtypes.canonicalize_dtype(disp.dtype) == dtype
    return ShapedArray(shape, dtype)

def _scatter_lowering(ctx, pmid, disp, val, mesh, offset, ptcl_spacing, cell_size, *, platform="gpu"):

    # Extract the numpy type of the inputs
    pmid_aval, disp_aval, _ = ctx.avals_in
    np_dtype = np.dtype(disp_aval.dtype)
    in_type1 = ir.RankedTensorType(pmid.type)
    in_type2 = ir.RankedTensorType(val.type)
    out_type = ir.RankedTensorType(mesh.type)
    out_layout = tuple(range(len(out_type.shape) - 1, -1, -1))
    in_layout1 = tuple(range(len(in_type1.shape) - 1, -1, -1))
    in_layout2 = tuple(range(len(in_type2.shape) - 1, -1, -1))

    # We dispatch a different call depending on the dtype
    if np_dtype == np.float32:
        op_name = platform + "_scatter_f32"
        # dimension using the 'opaque' parameter
        opaque = _jaxpmwd.build_pmwd_descriptor_f32(np.prod(in_type2.shape).astype(np.int64), ptcl_spacing, cell_size, *offset, *out_type.shape)
    elif np_dtype == np.float64:
        op_name = platform + "_scatter_f64"
        # dimension using the 'opaque' parameter
        opaque = _jaxpmwd.build_pmwd_descriptor_f64(np.prod(in_type2.shape).astype(np.int64), ptcl_spacing, cell_size, *offset, *out_type.shape)
    else:
        raise NotImplementedError(f"Unsupported dtype {np_dtype}")

    # And then the following is what changes between the GPU and CPU
    if platform == "cpu":
        raise NotImplementedError(f"Unsupported cpu platform")
    elif platform == "gpu":
        if _jaxpmwd is None:
            raise ValueError(
                "The 'pmwd_jax' module was not compiled with CUDA support"
            )

        # TODO: if we use shared mem with bin sort, bin sort work mem allocate by XLA here and pass to cuda
        return custom_call(
            op_name,
            # Output types
            out_types=[out_type],
            # The inputs:
            operands=[pmid,disp,val,mesh],
            # Layout specification:
            operand_layouts=[in_layout1, in_layout1, in_layout2, out_layout],
            result_layouts=[out_layout],
            operand_output_aliases={3:0},
            # GPU specific additional data
            backend_config=opaque
        )

    raise ValueError(
        "Unsupported platform; this must be either 'cpu' or 'gpu'"
    )

_scatter_prim = Primitive("scatter")
_scatter_prim.def_impl(partial(xla.apply_primitive, _scatter_prim))
_scatter_prim.def_abstract_eval(_scatter_abstract_eval)
mlir.register_lowering(_scatter_prim, _scatter_lowering, platform="gpu")

### define gather op
@partial(jit, static_argnums=(0,1,2,3,4,5))
def gather(pmid, disp, val, mesh, offset, ptcl_spacing, cell_size):

    return _gather_prim.bind(pmid, disp, val, mesh, offset, cell_size)

def _gather_abstract_eval(pmid, disp, val, mesh, offset, ptcl_spacing, cell_size):
    shape = val.shape
    dtype = dtypes.canonicalize_dtype(disp.dtype)
    assert dtypes.canonicalize_dtype(val.dtype) == dtype
    return ShapedArray(shape, dtype)

def _gather_lowering(ctx, pmid, disp, val, mesh, offset, ptcl_spacing, cell_size, *, platform="gpu"):

    # Extract the numpy type of the inputs
    pmid_aval, disp_aval, _ = ctx.avals_in
    np_dtype = np.dtype(disp_aval.dtype)
    in_type1 = ir.RankedTensorType(pmid.type)
    in_type2 = ir.RankedTensorType(val.type)
    out_type = ir.RankedTensorType(mesh.type)
    out_layout = tuple(range(len(out_type.shape) - 1, -1, -1))
    in_layout1 = tuple(range(len(in_type1.shape) - 1, -1, -1))
    in_layout2 = tuple(range(len(in_type2.shape) - 1, -1, -1))

    # We dispatch a different call depending on the dtype
    if np_dtype == np.float32:
        op_name = platform + "_gather_f32"
        # dimension using the 'opaque' parameter
        opaque = _jaxpmwd.build_pmwd_descriptor_f32(np.prod(in_type2.shape).astype(np.int64), ptcl_spacing, cell_size, *offset, *out_type.shape)
    elif np_dtype == np.float64:
        op_name = platform + "_gather_f64"
        # dimension using the 'opaque' parameter
        opaque = _jaxpmwd.build_pmwd_descriptor_f32(np.prod(in_type2.shape).astype(np.int64), ptcl_spacing, cell_size, *offset, *out_type.shape)
    else:
        raise NotImplementedError(f"Unsupported dtype {np_dtype}")

    # And then the following is what changes between the GPU and CPU
    if platform == "cpu":
        raise NotImplementedError(f"Unsupported cpu platform")
    elif platform == "gpu":
        if _jaxpmwd is None:
            raise ValueError(
                "The 'pmwd_jax' module was not compiled with CUDA support"
            )

        # TODO: if we use shared mem with bin sort, bin sort work mem allocate by XLA here and pass to cuda
        return custom_call(
            op_name,
            # Output types
            out_types=[out_type],
            # The inputs:
            operands=[pmid,disp,val,mesh],
            # Layout specification:
            operand_layouts=[in_layout1, in_layout1, in_layout2, out_layout],
            result_layouts=[in_layout2],
            operand_output_aliases={2:0},
            # GPU specific additional data
            backend_config=opaque
        )

    raise ValueError(
        "Unsupported platform; this must be either 'cpu' or 'gpu'"
    )

_gather_prim = Primitive("gather")
_gather_prim.def_impl(partial(xla.apply_primitive, _gather_prim))
_gather_prim.def_abstract_eval(_gather_abstract_eval)
mlir.register_lowering(_gather_prim, _gather_lowering, platform="gpu")
