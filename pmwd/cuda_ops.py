__all__ = ["scatter_cuda", "gather_cuda"]

from functools import partial
import numpy as np
import jaxlib.mlir.ir as ir
from jax import core, dtypes, lax
from jax.core import Primitive
from jax import numpy as jnp
from jax.core import ShapedArray
from jax._src.interpreters import ad, batching, mlir, xla
from jax import jit
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call
from jax._src.lib.mlir.dialects import hlo
from jax._src import dispatch
from . import _jaxpmwd

# Registering ops for XLA
for name, fn in _jaxpmwd.registrations().items():
  xla_client.register_custom_call_target(name, fn, platform="gpu")

### define scatter op
@partial(jit, static_argnums=(4,5,6,7))
def scatter_cuda(pmid, disp, val, mesh, offset, ptcl_grid, ptcl_spacing, cell_size):
    return _scatter_prim.bind(pmid, disp, val, mesh, offset=offset, ptcl_grid=ptcl_grid, ptcl_spacing=ptcl_spacing, cell_size=cell_size)

def _scatter_abstract_eval(pmid, disp, val, mesh, offset, ptcl_grid, ptcl_spacing, cell_size):
    shape = mesh.shape
    dtype = dtypes.canonicalize_dtype(val.dtype)
    assert dtypes.canonicalize_dtype(disp.dtype) == dtype
    return mesh.update()

def _scatter_lowering(ctx, pmid, disp, val, mesh, *, offset, ptcl_grid, ptcl_spacing, cell_size, platform="gpu"):
    # Extract the numpy type of the inputs
    pmid_aval, disp_aval, *_ = ctx.avals_in
    out_aval, *_ = ctx.avals_out
    np_dtype = np.dtype(disp_aval.dtype)
    np_pmidtype = np.dtype(pmid_aval.dtype)
    in_type1 = ir.RankedTensorType(pmid.type)
    in_type2 = ir.RankedTensorType(val.type)
    in_layout1 = tuple(range(len(in_type1.shape) - 1, -1, -1))
    in_layout2 = tuple(range(len(in_type2.shape) - 1, -1, -1))
    out_type = ir.RankedTensorType(mesh.type)
    out_layout = tuple(range(len(out_type.shape) - 1, -1, -1))

    # todo pmid int type should be uint16, uint8 or uint32?
    assert np_pmidtype == np.uint32

    # We dispatch a different call depending on the dtype
    if np_dtype == np.float32:
        op_name = platform + "_scatter_f32"
        # dimension using the 'opaque' parameter
        workspace_size, opaque = _jaxpmwd.build_pmwd_descriptor_f32(np.prod(in_type2.shape).astype(np.int64), ptcl_spacing, cell_size, *offset, *ptcl_grid, *out_type.shape)
    elif np_dtype == np.float64:
        op_name = platform + "_scatter_f64"
        # dimension using the 'opaque' parameter
        workspace_size, opaque = _jaxpmwd.build_pmwd_descriptor_f64(np.prod(in_type2.shape).astype(np.int64), ptcl_spacing, cell_size, *offset, *ptcl_grid, *out_type.shape)
    else:
        raise NotImplementedError(f"Unsupported dtype {np_dtype}")

    workspace = mlir.full_like_aval(ctx, 0, core.ShapedArray(shape=[workspace_size], dtype=np.byte))

    # And then the following is what changes between the GPU and CPU
    if platform == "cpu":
        raise NotImplementedError(f"Unsupported cpu platform")
    elif platform == "gpu":
        if _jaxpmwd is None:
            raise ValueError(
                "The '_jaxpmwd' module was not compiled with CUDA support"
            )

        # TODO: if we use shared mem with bin sort, bin sort work mem allocate by XLA here and pass to cuda
        result = custom_call(
            op_name,
            # Output types
            result_types=[out_type],
            # The inputs:
            operands=[pmid,disp,val,mesh,workspace],
            # Layout specification:
            operand_layouts=[in_layout1, in_layout1, in_layout2, out_layout, (0,)],
            result_layouts=[out_layout],
            operand_output_aliases={3:0},
            # GPU specific additional data
            backend_config=opaque,
        )
        return hlo.ReshapeOp(mlir.aval_to_ir_type(out_aval), result).results

    raise ValueError(
        "Unsupported platform; this must be 'gpu'"
    )

_scatter_prim = Primitive("scatter_cuda")
_scatter_prim.def_impl(partial(dispatch.apply_primitive, _scatter_prim))
_scatter_prim.def_abstract_eval(_scatter_abstract_eval)
mlir.register_lowering(_scatter_prim, _scatter_lowering, platform="gpu")

### define gather op
@partial(jit, static_argnums=(4,5,6))
def gather_cuda(pmid, disp, val, mesh, offset, ptcl_spacing, cell_size):
    return _gather_prim.bind(pmid, disp, val, mesh, offset=offset, ptcl_spacing=ptcl_spacing, cell_size=cell_size)

def _gather_abstract_eval(pmid, disp, val, mesh, offset, ptcl_spacing, cell_size):
    shape = val.shape
    dtype = dtypes.canonicalize_dtype(disp.dtype)
    assert dtypes.canonicalize_dtype(val.dtype) == dtype
    return val.update()

def _gather_lowering(ctx, pmid, disp, val, mesh, *, offset, ptcl_spacing, cell_size, platform="gpu"):

    # Extract the numpy type of the inputs
    pmid_aval, disp_aval, *_ = ctx.avals_in
    out_aval, *_ = ctx.avals_out
    np_dtype = np.dtype(disp_aval.dtype)
    np_pmidtype = np.dtype(pmid_aval.dtype)
    in_type1 = ir.RankedTensorType(pmid.type)
    in_type2 = ir.RankedTensorType(val.type)
    in_layout1 = tuple(range(len(in_type1.shape) - 1, -1, -1))
    in_layout2 = tuple(range(len(in_type2.shape) - 1, -1, -1))
    out_type = ir.RankedTensorType(mesh.type)
    out_layout = tuple(range(len(out_type.shape) - 1, -1, -1))

    # todo pmid int type should be uint16, uint8 or uint32?
    assert np_pmidtype == np.uint32

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
                "The '_jaxpmwd' module was not compiled with CUDA support"
            )

        # TODO: if we use shared mem with bin sort, bin sort work mem allocate by XLA here and pass to cuda
        result = custom_call(
            op_name,
            # Output types
            result_types=[out_type],
            # The inputs:
            operands=[pmid,disp,val,mesh],
            # Layout specification:
            operand_layouts=[in_layout1, in_layout1, in_layout2, out_layout],
            result_layouts=[in_layout2],
            operand_output_aliases={2:0},
            # GPU specific additional data
            backend_config=opaque,
        )

        return hlo.ReshapeOp(mlir.aval_to_ir_type(out_aval), result).results

    raise ValueError(
        "Unsupported platform; this must be 'gpu'"
    )

_gather_prim = Primitive("gather_cuda")
_gather_prim.def_impl(partial(dispatch.apply_primitive, _gather_prim))
_gather_prim.def_abstract_eval(_gather_abstract_eval)
mlir.register_lowering(_gather_prim, _gather_lowering, platform="gpu")

### define sort keys op
@jit
def sort_keys_cuda(keys):
    return _sort_keys_prim.bind(keys)

def _sort_keys_abstract_eval(keys):
    return keys.update(shape=keys.shape, dtype=keys.dtype)

def _sort_keys_lowering(ctx, keys, *, platform="gpu"):
    # Extract the numpy type of the inputs
    keys_aval, *_ = ctx.avals_in
    out_aval, *_ = ctx.avals_out
    np_dtype = np.dtype(keys_aval.dtype)
    in_type = ir.RankedTensorType(keys.type)
    in_layout = tuple(range(len(in_type.shape) - 1, -1, -1))
    out_type = ir.RankedTensorType(keys.type)
    out_layout = tuple(range(len(out_type.shape) - 1, -1, -1))

    # We dispatch a different call depending on the dtype
    if np_dtype == np.float32:
        op_name = platform + "_sort_keys_f32"
        # dimension using the 'opaque' parameter
        workspace_size, opaque = _jaxpmwd.build_sort_keys_descriptor_f32(np.prod(in_type.shape).astype(np.int64))
    elif np_dtype == np.float64:
        op_name = platform + "_sort_keys_f64"
        # dimension using the 'opaque' parameter
        workspace_size, opaque = _jaxpmwd.build_sort_keys_descriptor_f64(np.prod(in_type.shape).astype(np.int64))
    elif np_dtype == np.int32:
        op_name = platform + "_sort_keys_f64"
        # dimension using the 'opaque' parameter
        workspace_size, opaque = _jaxpmwd.build_sort_keys_descriptor_i32(np.prod(in_type.shape).astype(np.int64))
    elif np_dtype == np.int64:
        op_name = platform + "_sort_keys_f64"
        # dimension using the 'opaque' parameter
        workspace_size, opaque = _jaxpmwd.build_sort_keys_descriptor_i64(np.prod(in_type.shape).astype(np.int64))
    else:
        raise NotImplementedError(f"Unsupported dtype {np_dtype}")

    workspace = mlir.full_like_aval(ctx, 0, core.ShapedArray(shape=[workspace_size], dtype=np.byte))

    # And then the following is what changes between the GPU and CPU
    if platform == "cpu":
        raise NotImplementedError(f"Unsupported cpu platform")
    elif platform == "gpu":
        if _jaxpmwd is None:
            raise ValueError(
                "The '_jaxpmwd' module was not compiled with CUDA support"
            )

        # TODO: if we use shared mem with bin sort, bin sort work mem allocate by XLA here and pass to cuda
        result = custom_call(
            op_name,
            # Output types
            result_types=[out_type],
            # The inputs:
            operands=[keys, workspace],
            # Layout specification:
            operand_layouts=[in_layout, (0,)],
            result_layouts=[out_layout],
            operand_output_aliases={0:0},
            # GPU specific additional data
            backend_config=opaque,
        )
        return hlo.ReshapeOp(mlir.aval_to_ir_type(out_aval), result).results

    raise ValueError(
        "Unsupported platform; this must be 'gpu'"
    )

_sort_keys_prim = Primitive("sort_keys_cuda")
_sort_keys_prim.def_impl(partial(dispatch.apply_primitive, _sort_keys_prim))
_sort_keys_prim.def_abstract_eval(_sort_keys_abstract_eval)
mlir.register_lowering(_sort_keys_prim, _sort_keys_lowering, platform="gpu")

### define argsort op
@jit
def argsort_cuda(keys):
    return _argsort_prim.bind(keys)

def _argsort_abstract_eval(keys):
    return (ShapedArray(keys.shape, dtypes.canonicalize_dtype(np.uint32)), ShapedArray(keys.shape, keys.dtype))

def _argsort_lowering(ctx, keys, *, platform="gpu"):
    # Extract the numpy type of the inputs
    keys_aval, *_ = ctx.avals_in
    out_aval, *_ = ctx.avals_out
    #print(out_aval.dtype)
    np_dtype = np.dtype(keys_aval.dtype)
    in_type = ir.RankedTensorType(keys.type)
    in_layout = tuple(range(len(in_type.shape) - 1, -1, -1))
    out_type = ir.RankedTensorType.get(out_aval.shape, ir.IntegerType.get_unsigned(32))
    out_layout = tuple(range(len(out_type.shape) - 1, -1, -1))

    # We dispatch a different call depending on the dtype
    if np_dtype == np.float32:
        op_name = platform + "_argsort_f32"
        # dimension using the 'opaque' parameter
        workspace_size, opaque = _jaxpmwd.build_argsort_descriptor_f32(np.prod(in_type.shape).astype(np.int64))
    elif np_dtype == np.float64:
        op_name = platform + "_argsort_f64"
        # dimension using the 'opaque' parameter
        workspace_size, opaque = _jaxpmwd.build_argsort_descriptor_f64(np.prod(in_type.shape).astype(np.int64))
    elif np_dtype == np.int32:
        op_name = platform + "_argsort_f64"
        # dimension using the 'opaque' parameter
        workspace_size, opaque = _jaxpmwd.build_argsort_descriptor_i32(np.prod(in_type.shape).astype(np.int64))
    elif np_dtype == np.int64:
        op_name = platform + "_argsort_f64"
        # dimension using the 'opaque' parameter
        workspace_size, opaque = _jaxpmwd.build_argsort_descriptor_i64(np.prod(in_type.shape).astype(np.int64))
    else:
        raise NotImplementedError(f"Unsupported dtype {np_dtype}")

    workspace = mlir.full_like_aval(ctx, 0, core.ShapedArray(shape=[workspace_size], dtype=np.byte))

    # And then the following is what changes between the GPU and CPU
    if platform == "cpu":
        raise NotImplementedError(f"Unsupported cpu platform")
    elif platform == "gpu":
        if _jaxpmwd is None:
            raise ValueError(
                "The '_jaxpmwd' module was not compiled with CUDA support"
            )

        # TODO: if we use shared mem with bin sort, bin sort work mem allocate by XLA here and pass to cuda
        return custom_call(
            op_name,
            # Output types
            result_types=[out_type, in_type],
            # The inputs:
            operands=[keys, workspace],
            # Layout specification:
            operand_layouts=[in_layout, (0,)],
            result_layouts=[out_layout, in_layout],
            operand_output_aliases={0:1},
            # GPU specific additional data
            backend_config=opaque,
        )
        #return hlo.ReshapeOp(mlir.aval_to_ir_type(out_aval), result).results

    raise ValueError(
        "Unsupported platform; this must be 'gpu'"
    )

_argsort_prim = Primitive("argsort_cuda")
_argsort_prim.multiple_results = True
_argsort_prim.def_impl(partial(dispatch.apply_primitive, _argsort_prim))
_argsort_prim.def_abstract_eval(_argsort_abstract_eval)
mlir.register_lowering(_argsort_prim, _argsort_lowering, platform="gpu")

### define enmesh op
@partial(jit, static_argnums=(3,4,5,6))
def enmesh_cuda(pmid, disp, mesh, offset, ptcl_grid, ptcl_spacing, cell_size):
    return _enmesh_prim.bind(pmid, disp, mesh, offset=offset, ptcl_grid=ptcl_grid, ptcl_spacing=ptcl_spacing, cell_size=cell_size)

# returns: 1. index of argsort by cellid, 2. ptcl count of each cell, 3. dense edge of each cell
def _enmesh_abstract_eval(pmid, disp, mesh, offset, ptcl_grid, ptcl_spacing, cell_size):
    shape_mesh = [np.prod(mesh.shape)]
    shape_ptcl = [np.prod(pmid.shape)]
    return (ShapedArray(shape_ptcl, dtypes.canonicalize_dtype(np.uint32)), ShapedArray(shape_mesh, dtypes.canonicalize_dtype(np.uint32)), ShapedArray(shape_mesh, dtypes.canonicalize_dtype(np.uint32)))

def _enmesh_lowering(ctx, pmid, disp, mesh, *, offset, ptcl_grid, ptcl_spacing, cell_size, platform="gpu"):
    # Extract the numpy type of the inputs
    pmid_aval, disp_aval, *_ = ctx.avals_in
    ptcl_out_aval, mesh_out_aval, *_ = ctx.avals_out
    np_dtype = np.dtype(disp_aval.dtype)
    np_pmidtype = np.dtype(pmid_aval.dtype)
    in_type1 = ir.RankedTensorType(pmid.type)
    in_type2 = ir.RankedTensorType(mesh.type)
    in_layout1 = tuple(range(len(in_type1.shape) - 1, -1, -1))
    in_layout2 = tuple(range(len(in_type2.shape) - 1, -1, -1))

    ptcl_out_type = ir.RankedTensorType.get(ptcl_out_aval.shape, ir.IntegerType.get_unsigned(32))
    ptcl_out_layout = tuple(range(len(ptcl_out_type.shape) - 1, -1, -1))
    mesh_out_type = ir.RankedTensorType.get(mesh_out_aval.shape, ir.IntegerType.get_unsigned(32))
    mesh_out_layout = tuple(range(len(mesh_out_type.shape) - 1, -1, -1))

    # todo pmid int type should be uint16, uint8 or uint32?
    assert np_pmidtype == np.uint32

    # We dispatch a different call depending on the dtype
    if np_dtype == np.float32:
        op_name = platform + "_enmesh_f32"
        # dimension using the 'opaque' parameter
        workspace_size, opaque = _jaxpmwd.build_enmesh_descriptor_f32(np.prod(in_type2.shape).astype(np.int64), ptcl_spacing, cell_size, *offset, *ptcl_grid, *out_type.shape)
    elif np_dtype == np.float64:
        op_name = platform + "_enmesh_f64"
        # dimension using the 'opaque' parameter
        workspace_size, opaque = _jaxpmwd.build_enmesh_descriptor_f64(np.prod(in_type2.shape).astype(np.int64), ptcl_spacing, cell_size, *offset, *ptcl_grid, *out_type.shape)
    else:
        raise NotImplementedError(f"Unsupported dtype {np_dtype}")

    workspace = mlir.full_like_aval(ctx, 0, core.ShapedArray(shape=[workspace_size], dtype=np.byte))

    # And then the following is what changes between the GPU and CPU
    if platform == "cpu":
        raise NotImplementedError(f"Unsupported cpu platform")
    elif platform == "gpu":
        if _jaxpmwd is None:
            raise ValueError(
                "The '_jaxpmwd' module was not compiled with CUDA support"
            )

        # TODO: if we use shared mem with bin sort, bin sort work mem allocate by XLA here and pass to cuda
        return custom_call(
            op_name,
            # Output types
            result_types=[ptcl_out_type, mesh_out_type, mesh_out_type],
            # The inputs:
            operands=[pmid,disp,mesh,workspace],
            # Layout specification:
            operand_layouts=[in_layout1, in_layout1, in_layout2, (0,)],
            result_layouts=[ptcl_out_layout, mesh_out_layout, mesh_out_layout],
            # GPU specific additional data
            backend_config=opaque,
        )
        #return hlo.ReshapeOp(mlir.aval_to_ir_type(out_aval), result).results

    raise ValueError(
        "Unsupported platform; this must be 'gpu'"
    )

_enmesh_prim = Primitive("enmesh_cuda")
_argsort_prim.multiple_results = True
_enmesh_prim.def_impl(partial(dispatch.apply_primitive, _enmesh_prim))
_enmesh_prim.def_abstract_eval(_enmesh_abstract_eval)
mlir.register_lowering(_enmesh_prim, _enmesh_lowering, platform="gpu")
