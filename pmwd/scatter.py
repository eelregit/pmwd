from functools import partial

import jax.numpy as jnp
from jax import custom_vjp
from jax.lax import scan


def _chunk_split(ptcl_num, chunk_size, *arrays):
    """Split and reshape particle arrays into chunks and a remainder."""
    chunk_size = ptcl_num if chunk_size is None else min(chunk_size, ptcl_num)
    remainder_size = ptcl_num % chunk_size
    chunk_num = ptcl_num // chunk_size

    remainder = None
    chunks = arrays
    if remainder_size:
        remainder = [x[:remainder_size] for x in arrays]
        chunks = [x[remainder_size:] for x in arrays]

    chunks = [x.reshape(chunk_num, chunk_size, *x.shape[1:]) for x in chunks]

    return remainder, chunks

def _chunk_cat(remainder_array, chunks_array):
    """Reshape and concatenate a remainder and a chunked particle arrays."""
    array = chunks_array.reshape(-1, *chunks_array.shape[2:])

    if remainder_array is not None:
        array = jnp.concatenate((remainder_array, array), axis=0)

    return array


def scatter(ptcl, mesh, val, cell_size, chunk_size=None):
    """Scatter particle values to mesh in n-D with trilinear scheme."""
    return _scatter(ptcl.pmid, ptcl.disp, mesh, val, cell_size, chunk_size)


@partial(custom_vjp, nondiff_argnums=(5,))
def _scatter(pmid, disp, mesh, val, cell_size, chunk_size):
    ptcl_num = pmid.shape[0]

    val = jnp.asarray(val, dtype=disp.dtype)

    if val.ndim == 0:
        val = jnp.full(ptcl_num, val)

    remainder, chunks = _chunk_split(ptcl_num, chunk_size, pmid, disp, val)

    carry = mesh, cell_size
    if remainder is not None:
        carry = _scatter_chunk(carry, remainder)[0]
    carry = scan(_scatter_chunk, carry, chunks)[0]
    mesh = carry[0]

    return mesh


def _scatter_chunk(carry, chunk):
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
    val = val[:, jnp.newaxis]

    # trilinear
    neighbors = (jnp.arange(2**spatial_ndim, dtype=pmid.dtype)[:, jnp.newaxis]
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

    # scatter
    tgt = tuple(tgt[..., i] for i in range(spatial_ndim))
    mesh = mesh.at[tgt].add(val * frac)

    carry = mesh, cell_size
    return carry, None


def _scatter_chunk_adj(carry, chunk):
    """Adjoint of `_scatter_chunk`, or equivalently `_scatter_adj_chunk`, i.e. scatter
    adjoint in chunks.

    Gather disp_cot from mesh_cot and val;
    Gather val_cot from mesh_cot.

    """
    mesh_cot, cell_size = carry
    pmid, disp, val = chunk

    ptcl_num, spatial_ndim = pmid.shape

    spatial_shape = mesh_cot.shape[:spatial_ndim]
    chan_ndim = mesh_cot.ndim - spatial_ndim
    chan_axis = tuple(range(-chan_ndim, 0))

    disp /= cell_size

    # insert neighbor axis
    pmid = pmid[:, jnp.newaxis]
    disp = disp[:, jnp.newaxis]
    val = val[:, jnp.newaxis]

    # trilinear
    neighbors = (jnp.arange(2**spatial_ndim, dtype=pmid.dtype)[:, jnp.newaxis]
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

    # gather disp_cot from mesh_cot and val, and gather val_cot from mesh_cot
    tgt = tuple(tgt[..., i] for i in range(spatial_ndim))
    val_cot = mesh_cot[tgt]

    disp_cot = (val_cot * val).sum(axis=chan_axis)
    disp_cot = (disp_cot[..., jnp.newaxis] * frac_grad).sum(axis=1)
    disp_cot /= cell_size

    val_cot = (val_cot * frac).sum(axis=1)

    return carry, (disp_cot, val_cot)


def _scatter_fwd(pmid, disp, mesh, val, cell_size, chunk_size):
    mesh = _scatter(pmid, disp, mesh, val, cell_size, chunk_size)
    return mesh, (pmid, disp, val, cell_size)

def _scatter_bwd(chunk_size, res, mesh_cot):
    pmid, disp, val, cell_size = res

    ptcl_num = pmid.shape[0]

    val = jnp.asarray(val, dtype=disp.dtype)

    if val.ndim == 0:
        val = jnp.full(ptcl_num, val)

    remainder, chunks = _chunk_split(ptcl_num, chunk_size, pmid, disp, val)

    carry = mesh_cot, cell_size
    disp_cot_0, val_cot_0 = None, None
    if remainder is not None:
        disp_cot_0, val_cot_0 = _scatter_chunk_adj(carry, remainder)[1]
    disp_cot, val_cot = scan(_scatter_chunk_adj, carry, chunks)[1]

    disp_cot = _chunk_cat(disp_cot_0, disp_cot)
    val_cot = _chunk_cat(val_cot_0, val_cot)

    return None, disp_cot, mesh_cot, val_cot, None

_scatter.defvjp(_scatter_fwd, _scatter_bwd)
