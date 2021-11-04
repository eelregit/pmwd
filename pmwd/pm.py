from functools import partial
from dataclasses import dataclass, fields
from typing import Callable, Optional, Any

import jax.numpy as jnp
from jax import jit, vjp, custom_vjp
from jax.lax import scan
from jax.tree_util import tree_map

from .dataclasses import pytree_dataclass


@pytree_dataclass
class Particles:
    """Particle state or adjoint particle state

    Attributes:
        pmid: particles' IDs by mesh indices, of signed int dtype
        disp: displacements or adjoint
        vel: velocities (canonical momenta) or adjoint
        acc: accelerations or force vjp
        val: custom feature values or adjoint, as a pytree
    """
    pmid: jnp.ndarray
    disp: jnp.ndarray
    vel: Optional[jnp.ndarray] = None
    acc: Optional[jnp.ndarray] = None
    val: Any = None

    @property
    def num(self):
        return self.pmid.shape[0]

    @property
    def ndim(self):
        return self.pmid.shape[1]

    @property
    def int_dtype(self):
        return self.pmid.dtype

    @property
    def real_dtype(self):
        return self.disp.dtype

    def assert_valid(self):
        for field in fields(self):
            data = getattr(self, field.name)
            if data is not None:
                assert isinstance(data, jnp.ndarray), (
                    f'{field.name} must be jax.numpy.ndarray')

        assert jnp.issubdtype(self.pmid.dtype, jnp.signedinteger), (
            'pmid must be signed integers')

        assert self.disp.shape == self.pmid.shape, 'disp shape mismatch'
        assert jnp.issubdtype(self.disp.dtype, jnp.floating), (
            'disp must be floating point numbers')

        if self.vel is not None:
            assert self.vel.shape == self.pmid.shape, 'vel shape mismatch'
            assert self.vel.dtype == self.disp.dtype, 'vel dtype mismatch'

        if self.acc is not None:
            assert self.acc.shape == self.pmid.shape, 'acc shape mismatch'
            assert self.acc.dtype == self.disp.dtype, 'acc dtype mismatch'

        def assert_valid_val(v):
            assert v.shape[0] == self.num, 'val num mismatch'
            assert v.dtype == self.disp.dtype, 'val dtype mismatch'
        tree_map(assert_valid_val, self.val)


@pytree_dataclass
class State:
    """State of particle species, of integration or observation

    Attributes:
        dm: dark matter particles
    """
    dm: Particles

    # TODO: some parameter- and time-dependent factors that, like particle
    # states, are to be propagated forward and backward


@pytree_dataclass
class Param:
    """Cosmological parameters
    """
    Omega_m: float
    Omega_L: float
    A_s: float
    n_s: float
    h: float

    @property
    def Omega_k(self):
        return 1. - Omega_m - Omega_L


@pytree_dataclass
class DynamicConfig:
    """Configurations that do not need derivatives or affect jit

    Attributes:
        cell_size: mesh cell size
        time_steps: time integration steps
    """
    cell_size: float = 1.
    #max_disp_to_box_size_ratio: float  # shortest axis

    time_steps: Optional[jnp.ndarray] = None


@dataclass(frozen=True)
class StaticConfig:
    """Configurations that affect jit compilations

    Attributes:
        mesh_shape: n-D shape of mesh
        chunk_size: chunk size for scatter and gather, must be a divisor of
            particle numbers
    """
    mesh_shape: tuple

    chunk_size: int = 1<<24

    int_dtype: jnp.dtype = jnp.dtype(jnp.int32)
    real_dtype: jnp.dtype = jnp.dtype(jnp.float32)


def _chunk_split(ptcl_num, chunk_size, *arrays):
    """Split and reshape particle arrays into chunks and a remainder
    """
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
    """Reshape and concatenate a remainder and a chunked particle arrays
    """
    array = chunks_array.reshape(-1, *chunks_array.shape[2:])

    if remainder_array is not None:
        array = jnp.concatenate([remainder_array, array], axis=0)

    return array


def scatter(ptcl, mesh, val=1., cell_size=1., chunk_size=None):
    """Scatter particle values to mesh in n-D with CIC window
    """
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

    # normalize by cell size
    inv_cell_size = 1. / cell_size
    disp = disp * inv_cell_size

    # insert neighbor axis
    pmid = pmid[:, jnp.newaxis]
    disp = disp[:, jnp.newaxis]
    val = val[:, jnp.newaxis]

    # CIC
    neighbors = (jnp.arange(2 ** spatial_ndim)[:, jnp.newaxis]
                 >> jnp.arange(spatial_ndim)
                ) & 1
    tgt = jnp.floor(disp)
    tgt = tgt + neighbors.astype(tgt.dtype)
    frac = 1. - jnp.abs(disp - tgt)
    frac = frac.prod(axis=-1)
    frac = jnp.expand_dims(frac, chan_axis)
    tgt = pmid + tgt.astype(pmid.dtype)

    # periodic boundaries
    tgt = jnp.remainder(tgt, jnp.array(spatial_shape))

    # scatter
    tgt = tuple(tgt[..., i] for i in range(spatial_ndim))
    mesh = mesh.at[tgt].add(val * frac)

    carry = mesh, cell_size
    return carry, None


def _scatter_chunk_adj(carry, chunk):
    """Adjoint of `_scatter_chunk`, or equivalently `_scatter_adj_chunk`, i.e.
    scatter adjoint in chunks

    Gather disp_cot from mesh_cot and val;
    Gather val_cot from mesh_cot.
    """
    mesh_cot, cell_size = carry
    pmid, disp, val = chunk

    ptcl_num, spatial_ndim = pmid.shape

    spatial_shape = mesh_cot.shape[:spatial_ndim]
    chan_ndim = mesh_cot.ndim - spatial_ndim
    chan_axis = tuple(range(-chan_ndim, 0))

    # normalize by cell size
    inv_cell_size = 1. / cell_size
    disp = disp * inv_cell_size

    # insert neighbor axis
    pmid = pmid[:, jnp.newaxis]
    disp = disp[:, jnp.newaxis]
    val = val[:, jnp.newaxis]

    # CIC
    neighbors = (jnp.arange(2 ** spatial_ndim)[:, jnp.newaxis]
                 >> jnp.arange(spatial_ndim)
                ) & 1
    tgt = jnp.floor(disp)
    tgt = tgt + neighbors.astype(tgt.dtype)
    frac = 1. - jnp.abs(disp - tgt)
    sign = jnp.sign(tgt - disp)
    frac_grad = []
    for i in range(spatial_ndim):
        not_i = tuple(range(0, i)) + tuple(range(i + 1, spatial_ndim))
        frac_grad.append(sign[..., i] * frac[..., not_i].prod(axis=-1))
    frac_grad = jnp.stack(frac_grad, axis=-1)
    frac = frac.prod(axis=-1)
    frac = jnp.expand_dims(frac, chan_axis)
    tgt = pmid + tgt.astype(pmid.dtype)

    # periodic boundaries
    tgt = jnp.remainder(tgt, jnp.array(spatial_shape))

    # gather disp_cot from mesh_cot and val, and gather val_cot from mesh_cot
    tgt = tuple(tgt[..., i] for i in range(spatial_ndim))
    val_cot = mesh_cot[tgt]

    disp_cot = (val_cot * val).sum(axis=chan_axis)
    disp_cot = (disp_cot[..., jnp.newaxis] * frac_grad).sum(axis=1)
    disp_cot = disp_cot * inv_cell_size

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


def gather(ptcl, mesh, val=0., cell_size=1., chunk_size=None):
    """Gather particle values from mesh in n-D with CIC window
    """
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

    # normalize by cell size
    inv_cell_size = 1. / cell_size
    disp = disp * inv_cell_size

    # insert neighbor axis
    pmid = pmid[:, jnp.newaxis]
    disp = disp[:, jnp.newaxis]

    # CIC
    neighbors = (jnp.arange(2 ** spatial_ndim)[:, jnp.newaxis]
                 >> jnp.arange(spatial_ndim)
                ) & 1
    tgt = jnp.floor(disp)
    tgt = tgt + neighbors.astype(tgt.dtype)
    frac = 1. - jnp.abs(disp - tgt)
    frac = frac.prod(axis=-1)
    frac = jnp.expand_dims(frac, chan_axis)
    tgt = pmid + tgt.astype(pmid.dtype)

    # periodic boundaries
    tgt = jnp.remainder(tgt, jnp.array(spatial_shape))

    # gather
    tgt = tuple(tgt[..., i] for i in range(spatial_ndim))
    val = val + (mesh[tgt] * frac).sum(axis=1)

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

    # normalize by cell size
    inv_cell_size = 1. / cell_size
    disp = disp * inv_cell_size

    # insert neighbor axis
    pmid = pmid[:, jnp.newaxis]
    disp = disp[:, jnp.newaxis]
    val_cot = val_cot[:, jnp.newaxis]

    # CIC
    neighbors = (jnp.arange(2 ** spatial_ndim)[:, jnp.newaxis]
                 >> jnp.arange(spatial_ndim)
                ) & 1
    tgt = jnp.floor(disp)
    tgt = tgt + neighbors.astype(tgt.dtype)
    frac = 1. - jnp.abs(disp - tgt)
    sign = jnp.sign(tgt - disp)
    frac_grad = []
    for i in range(spatial_ndim):
        not_i = tuple(range(0, i)) + tuple(range(i + 1, spatial_ndim))
        frac_grad.append(sign[..., i] * frac[..., not_i].prod(axis=-1))
    frac_grad = jnp.stack(frac_grad, axis=-1)
    frac = frac.prod(axis=-1)
    frac = jnp.expand_dims(frac, chan_axis)
    tgt = pmid + tgt.astype(pmid.dtype)

    # periodic boundaries
    tgt = jnp.remainder(tgt, jnp.array(spatial_shape))

    # gather disp_cot from val_cot and mesh, and scatter val_cot to mesh_cot
    tgt = tuple(tgt[..., i] for i in range(spatial_ndim))
    val = mesh[tgt]

    disp_cot = (val_cot * val).sum(axis=chan_axis)
    disp_cot = (disp_cot[..., jnp.newaxis] * frac_grad).sum(axis=1)
    disp_cot = disp_cot * inv_cell_size

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


def rfftnfreq(shape, cell_size=1., dtype=float):
    """wavevectors for `numpy.fft.rfftn`

    Returns:
        A list of broadcastable wavevectors as a "`sparse`" `numpy.meshgrid`
    """
    freq_period = 2 * jnp.pi / cell_size

    kvec = []
    for axis, n in enumerate(shape[:-1]):
        k = jnp.fft.fftfreq(n).astype(dtype) * freq_period
        kvec.append(k)

    k = jnp.fft.rfftfreq(shape[-1]).astype(dtype) * freq_period
    kvec.append(k)

    kvec = jnp.meshgrid(*kvec, indexing='ij', sparse=True)

    return kvec


@custom_vjp
def laplace(kvec, dens, param=None):
    """Laplace kernel in Fourier space
    """
    spatial_ndim = len(kvec)
    chan_ndim = dens.ndim - spatial_ndim
    chan_axis = tuple(range(-chan_ndim, 0))

    kk = sum(k**2 for k in kvec)
    kk = jnp.expand_dims(kk, chan_axis)

    pot = jnp.where(kk != 0., - dens / kk, 0.)

    return pot


def laplace_fwd(kvec, dens, param):
    pot = laplace(kvec, dens, param)
    return pot, (kvec, param)

def laplace_bwd(res, pot_cot):
    """Custom vjp to avoid NaN when using where, as well as to save memory

    .. _JAX FAQ:
        https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
    """
    kvec, param = res
    dens_cot = laplace(kvec, pot_cot, param)
    return None, dens_cot, None

laplace.defvjp(laplace_fwd, laplace_bwd)


def negative_gradient(k, pot, cell_size=1.):
    nyquist = jnp.pi / cell_size
    eps = nyquist * jnp.finfo(k.dtype).eps

    spatial_ndim = k.ndim
    chan_ndim = pot.ndim - spatial_ndim
    chan_axis = tuple(range(-chan_ndim, 0))

    k = jnp.expand_dims(k, chan_axis)
    neg_ik = jnp.where(jnp.abs(jnp.abs(k) - nyquist) <= eps, 0j, -1j * k)

    neg_grad = neg_ik * pot

    return neg_grad


def gravity(ptcl, param, dconf, sconf):
    """Compute particles' gravitational forces on a mesh with FFT
    """
    kvec = rfftnfreq(sconf.mesh_shape,
                     cell_size=dconf.cell_size, dtype=ptcl.real_dtype)

    dens = jnp.zeros(sconf.mesh_shape, dtype=ptcl.real_dtype)

    dens = scatter(ptcl, dens, chunk_size=sconf.chunk_size)

    dens = jnp.fft.rfftn(dens)

    pot = laplace(kvec, dens, param)

    acc = []
    for k in kvec:
        neg_grad = negative_gradient(k, pot, cell_size=dconf.cell_size)

        neg_grad = jnp.fft.irfftn(neg_grad, s=sconf.mesh_shape)

        neg_grad = gather(ptcl, neg_grad, chunk_size=sconf.chunk_size)

        acc.append(neg_grad)
    acc = jnp.stack(acc, axis=-1)

    return acc


def force(state, param, dconf, sconf):
    ptcl = state.dm
    ptcl.acc = gravity(ptcl, param, dconf, sconf)
    return state


def force_adj(state, state_cot, param, dconf, sconf):
    ptcl = state.dm
    ptcl.acc, gravity_vjp = vjp(partial(gravity, dconf=dconf, sconf=sconf),
                                ptcl, param)

    ptcl_cot = state_cot.dm
    ptcl_cot.acc = gravity_vjp(ptcl_cot.vel)[0].disp

    return state, state_cot


def init_force(state, param, dconf, sconf):
    ptcl = state.dm
    if ptcl.acc is None:
        state = force(state, param, dconf, sconf)
    return state


def kick(state, param, step, dconf, sconf):
    ptcl = state.dm
    ptcl.vel = ptcl.vel + ptcl.acc * step
    return state


def kick_adj(state, state_cot, param, step, dconf, sconf):
    state = kick(state, param, step, dconf, sconf)

    ptcl_cot = state_cot.dm
    ptcl_cot.disp = ptcl_cot.disp - ptcl_cot.acc * step

    return state, state_cot


def drift(state, param, step, dconf, sconf):
    ptcl = state.dm
    ptcl.disp = ptcl.disp + ptcl.vel * step
    return state


def drift_adj(state, state_cot, param, step, dconf, sconf):
    state = drift(state, param, step, dconf, sconf)

    ptcl_cot = state_cot.dm
    ptcl_cot.vel = ptcl_cot.vel - ptcl_cot.disp * step

    return state, state_cot


def form(ptcl, param, step, dconf, sconf):
    pass


def coevolve(state, param, step, dconf, sconf):
    ptcl = state.dm
    ptcl.val = form(ptcl, param, step, dconf, sconf)
    return state


def init_coevolve(state, param, dconf, sconf):
    ptcl = state.dm
    if ptcl.val is None:
        state = coevolve(state, param, 0., dconf, sconf)  # FIXME HACK step
    return state


def observe(state, obsvbl, param, dconf, sconf):
    pass


def init_observe(state, obsvbl, param, dconf, sconf):
    pass


@partial(jit, static_argnames='sconf')
def integrate_init(state, obsvbl, param, dconf, sconf):
    state = init_force(state, param, dconf, sconf)

    state = init_coevolve(state, param, dconf, sconf)

    obsvbl = init_observe(state, obsvbl, param, dconf, sconf)

    return state, obsvbl


@partial(jit, static_argnames='sconf')
def integrate_step(state, obsvbl, param, step, dconf, sconf):
    # leapfrog
    state = kick(state, param, 0.5*step, dconf, sconf)
    state = drift(state, param, step, dconf, sconf)
    state = force(state, param, dconf, sconf)
    state = kick(state, param, 0.5*step, dconf, sconf)

    state = coevolve(state, param, step, dconf, sconf)

    obsvbl = observe(state, obsvbl, param, dconf, sconf)

    return state, obsvbl


@partial(custom_vjp, nondiff_argnums=(4,))
def integrate(state, obsvbl, param, dconf, sconf):
    """Time integration
    """
    state, obsvbl = integrate_init(state, obsvbl, param, dconf, sconf)
    for step in dconf.time_steps:
        state, obsvbl = integrate_step(state, obsvbl, param, step, dconf, sconf)
    return state, obsvbl


@partial(jit, static_argnames='sconf')
def integrate_adj_init(state, state_cot, obsvbl_cot, param, dconf, sconf):
    """
    Note:
        No need for `init_force_adj` here like the `init_force` in
        `integrate_init` to skip redundant computations, because one probably
        want to recompute force vjp after loading checkpoints
    """
    #state_cot = observe_adj(state, state_cot, obsvbl_cot, param, dconf, sconf)

    #state, state_cot = coevolve_adj(state, state_cot, param, step, dconf, sconf)

    state, state_cot = force_adj(state, state_cot, param, dconf, sconf)

    return state, state_cot


@partial(jit, static_argnames='sconf')
def integrate_adj_step(state, state_cot, obsvbl_cot, param, step, dconf, sconf):
    #state_cot = observe_adj(state, state_cot, obsvbl_cot, param, dconf, sconf)

    #state, state_cot = coevolve_adj(state, state_cot, param, step, dconf, sconf)

    # leapfrog and its adjoint
    # FIXME HACK step
    state, state_cot = kick_adj(state, state_cot, param, 0.5*step, dconf, sconf)
    state, state_cot = drift_adj(state, state_cot, param, step, dconf, sconf)
    state, state_cot = force_adj(state, state_cot, param, dconf, sconf)
    state, state_cot = kick_adj(state, state_cot, param, 0.5*step, dconf, sconf)

    return state, state_cot


def integrate_adj(state, state_cot, obsvbl_cot, param, dconf, sconf):
    """Time integration with adjoint equation
    """
    state, state_cot = integrate_adj_init(state, state_cot, obsvbl_cot,
                                          param, dconf, sconf)
    rev_steps = - dconf.time_steps  # FIXME HACK
    for step in rev_steps:
        state, state_cot = integrate_adj_step(state, state_cot, obsvbl_cot,
                                              param, step, dconf, sconf)
    return state, state_cot


def integrate_fwd(state, obsvbl, param, dconf, sconf):
    state, obsvbl = integrate(state, obsvbl, param, dconf, sconf)
    return (state, obsvbl), (state, param, dconf)

def integrate_bwd(sconf, res, cotangents):
    state, param, dconf = res
    state_cot, obsvbl_cot = cotangents

    state_cot = integrate_adj(state, state_cot, obsvbl_cot,
                              param, dconf, sconf)[1]

    return state_cot, obsvbl_cot, None, None  # FIXME HACK no param grad

integrate.defvjp(integrate_fwd, integrate_bwd)
