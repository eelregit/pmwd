from functools import partial
from dataclasses import dataclass, fields
from typing import Callable, Optional

import jax.numpy as jnp
from jax import vjp, custom_vjp
from jax.lax import scan

from .dataclasses import pytree_dataclass


@pytree_dataclass
class Particles:
    """Particle state or adjoint particle state

    Attributes:
        pmid: particles' IDs by mesh indices, of signed int dtype
        disp: displacements or adjoint
        vel: velocities (canonical momenta) or adjoint
        acc: accelerations or force vjp
        val: custom feature values or adjoint
    """
    pmid: jnp.ndarray
    disp: jnp.ndarray
    vel: Optional[jnp.ndarray] = None
    acc: Optional[jnp.ndarray] = None
    val: Optional[jnp.ndarray] = None

    @property
    def num(self):
        return self.pmid.shape[0]

    @property
    def ndim(self):
        return self.pmid.shape[1]

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

        if self.val is not None:
            assert self.val.shape[0] == self.pmid.shape[0], 'val num mismatch'
            assert self.val.dtype == self.disp.dtype, 'val dtype mismatch'


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


@dataclass(frozen=True)
class Config:
    """Configurations that affect jit compilations or do not need derivatives

    Attributes:
        mesh_shape: n-D shape of mesh
        cell_size: mesh cell size
        chunk_size: chunk size for scatter and gather, must be a divisor of
            particle numbers
    """
    mesh_shape: tuple
    cell_size: float = 1.
    chunk_size: int = 1024**2
    #max_disp_to_box_size_ratio: float  # shortest axis

    @property
    def inv_cell_size(self):
        return 1. / self.cell_size


def _chunk_split(ptcl_num, chunk_size, *arrays):
    """Split and reshape particle arrays into chunks and a remainder
    """
    chunk_size = min(chunk_size, ptcl_num)
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


def scatter(ptcl, mesh, val=1., chunk_size=1024**2):
    """Scatter particle values to mesh in n-D with CIC window
    """
    return _scatter(ptcl.pmid, ptcl.disp, mesh, val, chunk_size)


@partial(custom_vjp, nondiff_argnums=(4,))
def _scatter(pmid, disp, mesh, val, chunk_size):
    ptcl_num = pmid.shape[0]

    val = jnp.asarray(val)

    if val.ndim == 0:
        val = jnp.full(ptcl_num, val)

    remainder, chunks = _chunk_split(ptcl_num, chunk_size, pmid, disp, val)

    if remainder is not None:
        mesh = _scatter_chunk(mesh, remainder)[0]

    mesh = scan(_scatter_chunk, mesh, chunks)[0]

    return mesh


def _scatter_chunk(mesh, chunk):
    pmid, disp, val = chunk

    ptcl_num, spatial_ndim = pmid.shape

    spatial_shape = mesh.shape[:spatial_ndim]
    chan_ndim = mesh.ndim - spatial_ndim
    chan_axis = tuple(range(-chan_ndim, 0))

    pmid = pmid[:, jnp.newaxis]  # insert neighbor axis
    disp = disp[:, jnp.newaxis]  # insert neighbor axis
    val = val[:, jnp.newaxis]  # insert neighbor axis

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

    return mesh, None


def _scatter_chunk_adj(mesh_cot, chunk):
    """Adjoint of _scatter_chunk

    Gather disp_cot from mesh_cot and val;
    Gather val_cot from mesh_cot.
    """
    pmid, disp, val = chunk

    ptcl_num, spatial_ndim = pmid.shape

    spatial_shape = mesh_cot.shape[:spatial_ndim]
    chan_ndim = mesh_cot.ndim - spatial_ndim
    chan_axis = tuple(range(-chan_ndim, 0))

    pmid = pmid[:, jnp.newaxis]  # insert neighbor axis
    disp = disp[:, jnp.newaxis]  # insert neighbor axis
    val = val[:, jnp.newaxis]  # insert neighbor axis

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

    val_cot = (val_cot * frac).sum(axis=1)

    return mesh_cot, (disp_cot, val_cot)


def _scatter_fwd(pmid, disp, mesh, val, chunk_size):
    mesh = _scatter(pmid, disp, mesh, val, chunk_size)
    return mesh, (pmid, disp, val)

def _scatter_bwd(chunk_size, res, mesh_cot):
    pmid, disp, val = res

    ptcl_num = pmid.shape[0]

    val = jnp.asarray(val)

    if val.ndim == 0:
        val = jnp.full(ptcl_num, val)

    remainder, chunks = _chunk_split(ptcl_num, chunk_size, pmid, disp, val)

    disp_cot_0, val_cot_0 = None, None
    if remainder is not None:
        disp_cot_0, val_cot_0 = _scatter_chunk_adj(mesh_cot, remainder)[1]

    disp_cot, val_cot = scan(_scatter_chunk_adj, mesh_cot, chunks)[1]

    disp_cot = _chunk_cat(disp_cot_0, disp_cot)
    val_cot = _chunk_cat(val_cot_0, val_cot)

    return None, disp_cot, mesh_cot, val_cot

_scatter.defvjp(_scatter_fwd, _scatter_bwd)


def gather(ptcl, mesh, val=0., chunk_size=1024**2):
    """Gather particle values from mesh in n-D with CIC window
    """
    return _gather(ptcl.pmid, ptcl.disp, mesh, val, chunk_size)


@partial(custom_vjp, nondiff_argnums=(4,))
def _gather(pmid, disp, mesh, val, chunk_size):
    ptcl_num = pmid.shape[0]

    val = jnp.asarray(val)

    if val.ndim == 0:
        val = jnp.full(ptcl_num, val)

    remainder, chunks = _chunk_split(ptcl_num, chunk_size, pmid, disp, val)

    val_0 = None
    if remainder is not None:
        val_0 = _gather_chunk(mesh, remainder)[1]

    val = scan(_gather_chunk, mesh, chunks)[1]

    val = _chunk_cat(val_0, val)

    return val


def _gather_chunk(mesh, chunk):
    pmid, disp, val = chunk

    ptcl_num, spatial_ndim = pmid.shape

    spatial_shape = mesh.shape[:spatial_ndim]
    chan_ndim = mesh.ndim - spatial_ndim
    chan_axis = tuple(range(-chan_ndim, 0))

    pmid = pmid[:, jnp.newaxis]  # insert neighbor axis
    disp = disp[:, jnp.newaxis]  # insert neighbor axis

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

    return mesh, val


def _gather_chunk_adj(meshes, chunk):
    """Adjoint of _gather_chunk

    Gather disp_cot from val_cot and mesh;
    Scatter val_cot to mesh_cot.
    """
    mesh, mesh_cot = meshes
    pmid, disp, val_cot = chunk

    ptcl_num, spatial_ndim = pmid.shape

    spatial_shape = mesh.shape[:spatial_ndim]
    chan_ndim = mesh.ndim - spatial_ndim
    chan_axis = tuple(range(-chan_ndim, 0))

    pmid = pmid[:, jnp.newaxis]  # insert neighbor axis
    disp = disp[:, jnp.newaxis]  # insert neighbor axis
    val_cot = val_cot[:, jnp.newaxis]  # insert neighbor axis

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

    mesh_cot = mesh_cot.at[tgt].add(val_cot * frac)

    return (mesh, mesh_cot), disp_cot


def _gather_fwd(pmid, disp, mesh, val, chunk_size):
    val = _gather(pmid, disp, mesh, val, chunk_size)
    return val, (pmid, disp, mesh)

def _gather_bwd(chunk_size, res, val_cot):
    pmid, disp, mesh = res

    ptcl_num = pmid.shape[0]

    remainder, chunks = _chunk_split(ptcl_num, chunk_size, pmid, disp, val_cot)

    mesh_cot = jnp.zeros_like(mesh)
    meshes = (mesh, mesh_cot)

    disp_cot_0 = None
    if remainder is not None:
        meshes, disp_cot_0 = _gather_chunk_adj(meshes, remainder)

    meshes, disp_cot = scan(_gather_chunk_adj, meshes, chunks)

    mesh_cot = meshes[1]

    disp_cot = _chunk_cat(disp_cot_0, disp_cot)

    return None, disp_cot, mesh_cot, val_cot

_gather.defvjp(_gather_fwd, _gather_bwd)


def rfftnfreq(shape, dtype=float):
    """wavevectors (angular frequencies) for `numpy.fft.rfftn`

    Returns:
        A list of broadcastable wavevectors as a "`sparse`" `numpy.meshgrid`
    """
    rad_per_cycle = 2 * jnp.pi

    kvec = []
    for axis, n in enumerate(shape[:-1]):
        k = jnp.fft.fftfreq(n).astype(dtype) * rad_per_cycle
        kvec.append(k)

    k = jnp.fft.rfftfreq(shape[-1]).astype(dtype) * rad_per_cycle
    kvec.append(k)

    kvec = jnp.meshgrid(*kvec, indexing='ij', sparse=True)

    return kvec


@custom_vjp
def laplace(kvec, dens, param):
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


def negative_gradient(k, pot):
    nyquist = jnp.pi
    eps = nyquist * jnp.finfo(k.dtype).eps

    spatial_ndim = k.ndim
    chan_ndim = pot.ndim - spatial_ndim
    chan_axis = tuple(range(-chan_ndim, 0))

    k = jnp.expand_dims(k, chan_axis)
    neg_ik = jnp.where(jnp.abs(jnp.abs(k) - nyquist) <= eps, 0j, -1j * k)

    neg_grad = neg_ik * pot

    return neg_grad


def gravity(ptcl, param, config):
    """Compute particles' gravitational forces on a mesh with FFT
    """
    real_dtype = ptcl.disp.dtype

    kvec = rfftnfreq(config.mesh_shape, dtype=real_dtype)

    dens = jnp.zeros(config.mesh_shape, dtype=real_dtype)

    dens = scatter(ptcl, dens, chunk_size=config.chunk_size)

    dens = jnp.fft.rfftn(dens)

    pot = laplace(kvec, dens, param)

    acc = []
    for k in kvec:
        neg_grad = negative_gradient(k, pot)

        neg_grad = jnp.fft.irfftn(neg_grad, s=config.mesh_shape)

        neg_grad = gather(ptcl, neg_grad, chunk_size=config.chunk_size)

        acc.append(neg_grad)
    acc = jnp.stack(acc, axis=-1)

    return acc


def force(state, param, config):
    ptcl = state.dm
    ptcl.acc = gravity(ptcl, param, config)
    return state


def force_adj(state, state_cot, param, config):
    ptcl = state.dm
    ptcl.acc, gravity_vjp = vjp(partial(gravity, config=config), ptcl, param)

    ptcl_cot = state_cot.dm
    ptcl_cot.acc = gravity_vjp(ptcl_cot.vel)[0].disp

    return state, state_cot


def init_force(state, param, config):
    ptcl = state.dm
    if ptcl.acc is None:
        state = force(state, param, config)
    return state


def init_force_adj(state, state_cot, param, config):
    ptcl = state.dm
    ptcl_cot = state_cot.dm
    if ptcl.acc is None or ptcl_cot.acc is None:
        state, state_cot = force_adj(state, state_cot, param, config)
    return state, state_cot


def kick(state, step, param, config):
    ptcl = state.dm
    ptcl.vel = ptcl.vel + ptcl.acc * step
    return state


def kick_adj(state, state_cot, step, param, config):
    state = kick(state, step, param, config)

    ptcl_cot = state_cot.dm
    ptcl_cot.disp = ptcl_cot.disp - ptcl_cot.acc * step

    return state, state_cot


def drift(state, step, param, config):
    ptcl = state.dm
    ptcl.disp = ptcl.disp + ptcl.vel * step
    return state


def drift_adj(state, state_cot, step, param, config):
    state = drift(state, step, param, config)

    ptcl_cot = state_cot.dm
    ptcl_cot.vel = ptcl_cot.vel - ptcl_cot.disp * step

    return state, state_cot


def leapfrog(state, step, param, config):
    """Leapfrog time stepping
    """
    state = kick(state, 0.5 * step, param, config)

    state = drift(state, step, param, config)

    state = force(state, param, config)

    state = kick(state, 0.5 * step, param, config)

    return state


def leapfrog_adj(state, state_cot, step, param, config):
    """Leapfrog with adjoint equation
    """
    # FIXME HACK step
    state, state_cot = kick_adj(state, state_cot, 0.5*step, param, config)

    state, state_cot = drift_adj(state, state_cot, step, param, config)

    state, state_cot = force_adj(state, state_cot, param, config)

    state, state_cot = kick_adj(state, state_cot, 0.5*step, param, config)

    return state, state_cot


def form(ptcl, param, config):
    pass


def coevolve(state, param, config):
    ptcl = state.dm
    ptcl.val = form(ptcl, param, config)
    return state


def init_coevolve(state, param, config):
    ptcl = state.dm
    if ptcl.val is None:
        state = coevolve(state, param, config)
    return state


def observe(state, obsvbl, param, config):
    pass


def init_observe(state, obsvbl, param, config):
    pass


@partial(custom_vjp, nondiff_argnums=(4,))
def integrate(state, obsvbl, steps, param, config):
    """Time integration
    """
    state = init_force(state, param, config)

    state = init_coevolve(state, param, config)

    obsvbl = init_observe(state, obsvbl, param, config)

    def _integrate(carry, step):
        state, obsvbl, param = carry

        state = leapfrog(state, step, param, config)

        state = coevolve(state, param, config)

        obsvbl = observe(state, obsvbl, param, config)

        carry = state, obsvbl, param

        return carry, None

    carry = state, obsvbl, param

    state, obsvbl = scan(_integrate, carry, steps)[0][:2]

    return state, obsvbl


def integrate_adj(state, state_cot, obsvbl_cot, steps, param, config):
    """Time integration with adjoint equation
    """
    state, state_cot = init_force_adj(state, state_cot, param, config)

    #state = init_coevolve_adj(state, param, config)

    #obsvbl = init_observe_adj(state, obsvbl, param, config)

    def _integrate_adj(carry, step):
        state, state_cot, obsvbl_cot, param = carry

        state, state_cot = leapfrog_adj(state, state_cot, step, param, config)

        #state = coevolve_adj(state, param, config)

        #obsvbl = observe_adj(state, obsvbl, param, config)

        carry = state, state_cot, obsvbl_cot, param

        return carry, None

    carry = state, state_cot, obsvbl_cot, param

    state, state_cot, obsvbl_cot = scan(_integrate_adj, carry, steps)[0][:3]

    return state, state_cot, obsvbl_cot


def integrate_fwd(state, obsvbl, steps, param, config):
    state, obsvbl = integrate(state, obsvbl, steps, param, config)
    return (state, obsvbl), (state, steps, param)

def integrate_bwd(config, res, cotangents):
    state, steps, param = res
    state_cot, obsvbl_cot = cotangents

    rev_steps = - steps  # FIXME HACK

    # need state below? need *_adj functions?
    state, state_cot, obsvbl_cot = integrate_adj(
        state, state_cot, obsvbl_cot, rev_steps, param, config)

    return state_cot, obsvbl_cot, None, None  # FIXME HACK no param grad

integrate.defvjp(integrate_fwd, integrate_bwd)
