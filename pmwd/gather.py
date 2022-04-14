from functools import partial

import jax.numpy as jnp
from jax import custom_vjp
from jax.lax import scan

from pmwd.scatter import _chunk_split, _chunk_cat


def gather(ptcl, mesh, val, cell_size, chunk_size=None):
    """Gather particle values from mesh in n-D with trilinear scheme."""
    return _gather(ptcl.pmid, ptcl.disp, mesh, val, cell_size, chunk_size)


@partial(custom_vjp, nondiff_argnums=(5,))
def _gather(pmid, disp, mesh, val, cell_size, chunk_size):
    ptcl_num = pmid.shape[0]

    val = jnp.asarray(val, dtype=disp.dtype)

    if val.ndim == 0:
        val = jnp.full(ptcl_num, val)

    remainder, chunks = _chunk_split(ptcl_num, chunk_size, pmid, disp, val)

    carry = mesh, cell_size
    val_0 = None
    if remainder is not None:
        val_0 = _gather_chunk(carry, remainder)[1]
    val = scan(_gather_chunk, carry, chunks)[1]

    val = _chunk_cat(val_0, val)

    return val


def _gather_chunk(carry, chunk):
    mesh, cell_size = carry
    pmid, disp, val = chunk

    ptcl_num, spatial_ndim = pmid.shape

    spatial_shape = mesh.shape[:spatial_ndim]
    chan_ndim = mesh.ndim - spatial_ndim
    chan_axis = tuple(range(-chan_ndim, 0))

    disp /= cell_size

    # insert neighbor axis
    pmid = pmid[:, jnp.newaxis]
    disp = disp[:, jnp.newaxis]

    # trilinear
    neighbors = (jnp.arange(2 ** spatial_ndim, dtype=pmid.dtype)[:, jnp.newaxis]
                 >> jnp.arange(spatial_ndim, dtype=pmid.dtype)
                ) & 1
    tgt = jnp.floor(disp).astype(pmid.dtype)
    tgt += neighbors
    frac = 1 - jnp.abs(disp - tgt)
    frac = frac.prod(axis=-1)
    frac = jnp.expand_dims(frac, chan_axis)
    tgt += pmid

    # periodic boundaries
    # TODO no wrapping for parallelization
    tgt %= jnp.array(spatial_shape, dtype=pmid.dtype)

    # gather
    tgt = tuple(tgt[..., i] for i in range(spatial_ndim))
    val += (mesh[tgt] * frac).sum(axis=1)

    return carry, val


def _gather_chunk_adj(carry, chunk):
    """Adjoint of `_gather_chunk`, or equivalently `_gather_adj_chunk`, i.e.
    gather adjoint in chunks

    Gather disp_cot from val_cot and mesh;
    Scatter val_cot to mesh_cot.

    """
    mesh, mesh_cot, cell_size = carry
    pmid, disp, val_cot = chunk

    ptcl_num, spatial_ndim = pmid.shape

    spatial_shape = mesh.shape[:spatial_ndim]
    chan_ndim = mesh.ndim - spatial_ndim
    chan_axis = tuple(range(-chan_ndim, 0))

    disp /= cell_size

    # insert neighbor axis
    pmid = pmid[:, jnp.newaxis]
    disp = disp[:, jnp.newaxis]
    val_cot = val_cot[:, jnp.newaxis]

    # trilinear
    neighbors = (jnp.arange(2 ** spatial_ndim, dtype=pmid.dtype)[:, jnp.newaxis]
                 >> jnp.arange(spatial_ndim, dtype=pmid.dtype)
                ) & 1
    tgt = jnp.floor(disp).astype(pmid.dtype)
    tgt += neighbors
    frac = 1 - jnp.abs(disp - tgt)
    sign = jnp.sign(tgt - disp)
    frac_grad = []
    for i in range(spatial_ndim):
        not_i = tuple(range(0, i)) + tuple(range(i + 1, spatial_ndim))
        frac_grad.append(sign[..., i] * frac[..., not_i].prod(axis=-1))
    frac_grad = jnp.stack(frac_grad, axis=-1)
    frac = frac.prod(axis=-1)
    frac = jnp.expand_dims(frac, chan_axis)
    tgt += pmid

    # periodic boundaries
    # TODO no wrapping for parallelization
    tgt %= jnp.array(spatial_shape, dtype=pmid.dtype)

    # gather disp_cot from val_cot and mesh, and scatter val_cot to mesh_cot
    tgt = tuple(tgt[..., i] for i in range(spatial_ndim))
    val = mesh[tgt]

    disp_cot = (val_cot * val).sum(axis=chan_axis)
    disp_cot = (disp_cot[..., jnp.newaxis] * frac_grad).sum(axis=1)
    disp_cot /= cell_size

    mesh_cot = mesh_cot.at[tgt].add(val_cot * frac)

    carry = mesh, mesh_cot, cell_size
    return carry, disp_cot


def _gather_fwd(pmid, disp, mesh, val, cell_size, chunk_size):
    val = _gather(pmid, disp, mesh, val, cell_size, chunk_size)
    return val, (pmid, disp, mesh, cell_size)

def _gather_bwd(chunk_size, res, val_cot):
    pmid, disp, mesh, cell_size = res

    ptcl_num = pmid.shape[0]

    remainder, chunks = _chunk_split(ptcl_num, chunk_size, pmid, disp, val_cot)

    mesh_cot = jnp.zeros_like(mesh)
    carry = mesh, mesh_cot, cell_size
    disp_cot_0 = None
    if remainder is not None:
        carry, disp_cot_0 = _gather_chunk_adj(carry, remainder)
    carry, disp_cot = scan(_gather_chunk_adj, carry, chunks)
    mesh_cot = carry[1]

    disp_cot = _chunk_cat(disp_cot_0, disp_cot)

    return None, disp_cot, mesh_cot, val_cot, None

_gather.defvjp(_gather_fwd, _gather_bwd)
