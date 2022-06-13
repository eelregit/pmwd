import jax.numpy as jnp
from jax import custom_vjp
from jax.lax import scan

from pmwd.pm_util import _chunk_split, _chunk_cat, enmesh


def gather(ptcl, conf, mesh, val=0, offset=0, cell_size=None):
    """Gather particle values from mesh multilinearly in n-D.

    Parameters
    ----------
    ptcl : Particles
    conf : Configuration
    mesh : array_like
        Input mesh.
    val : array_like, optional
        Input values, can be 0D.
    offset : array_like, optional
        Offset of mesh to particle grid. If 0D, the value is used in each dimension.
    cell_size : float, optional
        Mesh cell size in [L]. Default is ``conf.cell_size``.

    Returns
    -------
    val : jax.numpy.ndarray
        Output values.

    """
    return _gather(ptcl.pmid, ptcl.disp, conf, mesh, val, offset, cell_size)


@custom_vjp
def _gather(pmid, disp, conf, mesh, val, offset, cell_size):
    ptcl_num, spatial_ndim = pmid.shape

    mesh = jnp.asarray(mesh, dtype=conf.float_dtype)

    val = jnp.asarray(val, dtype=conf.float_dtype)

    if mesh.shape[spatial_ndim:] != val.shape[1:]:
        raise ValueError('channel shape mismatch: '
                         f'{mesh.shape[spatial_ndim:]} != {val.shape[1:]}')

    remainder, chunks = _chunk_split(ptcl_num, conf.chunk_size, pmid, disp, val)

    carry = conf, mesh, offset, cell_size
    val_0 = None
    if remainder is not None:
        val_0 = _gather_chunk(carry, remainder)[1]
    val = scan(_gather_chunk, carry, chunks)[1]

    val = _chunk_cat(val_0, val)

    return val


def _gather_chunk(carry, chunk):
    conf, mesh, offset, cell_size = carry
    pmid, disp, val = chunk

    spatial_ndim = pmid.shape[1]

    spatial_shape = mesh.shape[:spatial_ndim]
    chan_ndim = mesh.ndim - spatial_ndim
    chan_axis = tuple(range(-chan_ndim, 0))

    # multilinear mesh indices and fractions
    ind, frac = enmesh(pmid, disp, conf.cell_size, conf.mesh_shape,
                       offset, cell_size, spatial_shape, False)

    # gather
    ind = tuple(ind[..., i] for i in range(spatial_ndim))
    frac = jnp.expand_dims(frac, chan_axis)
    val += (mesh.at[ind].get(mode='drop', fill_value=0) * frac).sum(axis=1)

    return carry, val


def _gather_chunk_adj(carry, chunk):
    """Adjoint of `_gather_chunk`, or equivalently `_gather_adj_chunk`, i.e.
    gather adjoint in chunks

    Gather disp_cot from val_cot and mesh;
    Scatter val_cot to mesh_cot.

    """
    conf, mesh, mesh_cot, offset, cell_size = carry
    pmid, disp, val_cot = chunk

    spatial_ndim = pmid.shape[1]

    spatial_shape = mesh.shape[:spatial_ndim]
    chan_ndim = mesh.ndim - spatial_ndim
    chan_axis = tuple(range(-chan_ndim, 0))

    # multilinear mesh indices and fractions
    ind, frac, frac_grad = enmesh(pmid, disp, conf.cell_size, conf.mesh_shape,
                                  offset, cell_size, spatial_shape, True)

    if val_cot.ndim != 0:
        val_cot = val_cot[:, jnp.newaxis]  # insert neighbor axis

    # gather disp_cot from val_cot and mesh, and scatter val_cot to mesh_cot
    ind = tuple(ind[..., i] for i in range(spatial_ndim))
    val = mesh.at[ind].get(mode='drop', fill_value=0)

    disp_cot = (val_cot * val).sum(axis=chan_axis)
    disp_cot = (disp_cot[..., jnp.newaxis] * frac_grad).sum(axis=1)
    disp_cot /= cell_size if cell_size is not None else conf.cell_size

    frac = jnp.expand_dims(frac, chan_axis)
    mesh_cot = mesh_cot.at[ind].add(val_cot * frac)

    carry = conf, mesh, mesh_cot, offset, cell_size
    return carry, disp_cot


def _gather_fwd(pmid, disp, conf, mesh, val, offset, cell_size):
    val = _gather(pmid, disp, conf, mesh, val, offset, cell_size)
    return val, (pmid, disp, conf, mesh, offset, cell_size)

def _gather_bwd(res, val_cot):
    pmid, disp, conf, mesh, offset, cell_size = res

    ptcl_num = len(pmid)

    mesh = jnp.asarray(mesh, dtype=conf.float_dtype)
    mesh_cot = jnp.zeros_like(mesh)

    remainder, chunks = _chunk_split(ptcl_num, conf.chunk_size, pmid, disp, val_cot)

    carry = conf, mesh, mesh_cot, offset, cell_size
    disp_cot_0 = None
    if remainder is not None:
        carry, disp_cot_0 = _gather_chunk_adj(carry, remainder)
    carry, disp_cot = scan(_gather_chunk_adj, carry, chunks)
    mesh_cot = carry[2]

    disp_cot = _chunk_cat(disp_cot_0, disp_cot)

    return None, disp_cot, None, mesh_cot, val_cot, None, None

_gather.defvjp(_gather_fwd, _gather_bwd)
