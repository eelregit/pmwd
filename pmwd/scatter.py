import jax.numpy as jnp
from jax import custom_vjp
from jax.lax import scan

from pmwd.pm_util import _chunk_split, _chunk_cat, enmesh


def scatter(ptcl, conf, mesh=None, val=None, offset=0, cell_size=None):
    """Scatter particle values to mesh multilinearly in n-D.

    Parameters
    ----------
    ptcl : Particles
    conf : Configuration
    mesh : array_like, optional
        Input mesh. Default is a ``zeros`` array of ``conf.mesh_shape + val.shape[1:]``.
    val : array_like, optional
        Input values, can be 0D. Default is ``conf.mesh_size / conf.ptcl_num``.
    offset : array_like, optional
        Offset of mesh to particle grid. If 0D, the value is used in each dimension.
    cell_size : float, optional
        Mesh cell size in [L]. Default is ``conf.cell_size``.

    Returns
    -------
    mesh : jax.numpy.ndarray
        Output mesh.

    """
    return _scatter(ptcl.pmid, ptcl.disp, conf, mesh, val, offset, cell_size)


@custom_vjp
def _scatter(pmid, disp, conf, mesh, val, offset, cell_size):
    ptcl_num, spatial_ndim = pmid.shape

    if val is None:
        val = conf.mesh_size / conf.ptcl_num
    val = jnp.asarray(val, dtype=conf.float_dtype)

    if mesh is None:
        mesh = jnp.zeros(conf.mesh_shape + val.shape[1:], dtype=conf.float_dtype)
    mesh = jnp.asarray(mesh, dtype=conf.float_dtype)

    if mesh.shape[spatial_ndim:] != val.shape[1:]:
        raise ValueError('channel shape mismatch: '
                         f'{mesh.shape[spatial_ndim:]} != {val.shape[1:]}')

    remainder, chunks = _chunk_split(ptcl_num, conf.chunk_size, pmid, disp, val)

    carry = conf, mesh, offset, cell_size
    if remainder is not None:
        carry = _scatter_chunk(carry, remainder)[0]
    carry = scan(_scatter_chunk, carry, chunks)[0]
    mesh = carry[1]

    return mesh


def _scatter_chunk(carry, chunk):
    conf, mesh, offset, cell_size = carry
    pmid, disp, val = chunk

    spatial_ndim = pmid.shape[1]

    spatial_shape = mesh.shape[:spatial_ndim]
    chan_ndim = mesh.ndim - spatial_ndim
    chan_axis = tuple(range(-chan_ndim, 0))

    # multilinear mesh indices and fractions
    ind, frac = enmesh(pmid, disp, conf.cell_size, conf.mesh_shape,
                       offset, cell_size, spatial_shape, False)

    if val.ndim != 0:
        val = val[:, jnp.newaxis]  # insert neighbor axis

    # scatter
    ind = tuple(ind[..., i] for i in range(spatial_ndim))
    frac = jnp.expand_dims(frac, chan_axis)
    mesh = mesh.at[ind].add(val * frac)

    carry = conf, mesh, offset, cell_size
    return carry, None


def _scatter_chunk_adj(carry, chunk):
    """Adjoint of `_scatter_chunk`, or equivalently `_scatter_adj_chunk`, i.e. scatter
    adjoint in chunks.

    Gather disp_cot from mesh_cot and val;
    Gather val_cot from mesh_cot.

    """
    conf, mesh_cot, offset, cell_size = carry
    pmid, disp, val = chunk

    spatial_ndim = pmid.shape[1]

    spatial_shape = mesh_cot.shape[:spatial_ndim]
    chan_ndim = mesh_cot.ndim - spatial_ndim
    chan_axis = tuple(range(-chan_ndim, 0))

    # multilinear mesh indices and fractions
    ind, frac, frac_grad = enmesh(pmid, disp, conf.cell_size, conf.mesh_shape,
                                  offset, cell_size, spatial_shape, True)

    if val.ndim != 0:
        val = val[:, jnp.newaxis]  # insert neighbor axis

    # gather disp_cot from mesh_cot and val, and gather val_cot from mesh_cot
    ind = tuple(ind[..., i] for i in range(spatial_ndim))
    val_cot = mesh_cot.at[ind].get(mode='drop', fill_value=0)

    disp_cot = (val_cot * val).sum(axis=chan_axis)
    disp_cot = (disp_cot[..., jnp.newaxis] * frac_grad).sum(axis=1)
    disp_cot /= cell_size if cell_size is not None else conf.cell_size

    frac = jnp.expand_dims(frac, chan_axis)
    val_cot = (val_cot * frac).sum(axis=1)

    return carry, (disp_cot, val_cot)


def _scatter_fwd(pmid, disp, conf, mesh, val, offset, cell_size):
    mesh = _scatter(pmid, disp, conf, mesh, val, offset, cell_size)
    return mesh, (pmid, disp, conf, val, offset, cell_size)

def _scatter_bwd(res, mesh_cot):
    pmid, disp, conf, val, offset, cell_size = res

    ptcl_num = len(pmid)

    if val is None:
        val = conf.mesh_size / conf.ptcl_num
    val = jnp.asarray(val, dtype=conf.float_dtype)

    remainder, chunks = _chunk_split(ptcl_num, conf.chunk_size, pmid, disp, val)

    carry = conf, mesh_cot, offset, cell_size
    disp_cot_0, val_cot_0 = None, None
    if remainder is not None:
        disp_cot_0, val_cot_0 = _scatter_chunk_adj(carry, remainder)[1]
    disp_cot, val_cot = scan(_scatter_chunk_adj, carry, chunks)[1]

    disp_cot = _chunk_cat(disp_cot_0, disp_cot)
    val_cot = _chunk_cat(val_cot_0, val_cot)

    return None, disp_cot, None, mesh_cot, val_cot, None, None

_scatter.defvjp(_scatter_fwd, _scatter_bwd)
