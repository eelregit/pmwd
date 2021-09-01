from dataclasses import dataclass, fields
from typing import Callable, Optional
from functools import partial

import jax.numpy as jnp
from jax import grad, jit, vjp, custom_vjp
from jax import lax
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
@dataclass
class Particles:
    """Particle state

    Attributes:
        pmid: particles' IDs by mesh indices
        disp: displacements
        vel: velocities (canonical momenta)
        acc: accelerations
        val: custom feature values
    """
    pmid: jnp.ndarray
    disp: jnp.ndarray
    vel: Optional[jnp.ndarray] = None
    acc: Optional[jnp.ndarray] = None
    val: Optional[jnp.ndarray] = None

    def tree_flatten(self):
        #children = astuple(self)  # dataclasses.astuple was slow
        children = tuple(getattr(self, field.name) for field in fields(self))
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @property
    def num(self):
        return self.pmid.shape[0]

    @property
    def ndim(self):
        return self.pmid.shape[1]

    def assert_valid(self):
        for field in fields(self):
            assert isinstance(getattr(self, field.name), jnp.ndarray), \
                   f'{field.name} must be jax.numpy.ndarray'

        assert jnp.issubdtype(self.pmid.dtype, jnp.signedinteger), \
               'pmid must be signed integers'

        assert self.disp.shape == self.pmid.shape, 'disp shape mismatch'
        assert jnp.issubdtype(self.disp.dtype, jnp.floating), \
               'disp must be floating point numbers'

        if self.vel is not None:
            assert self.vel.shape == self.pmid.shape, 'vel shape mismatch'
            assert self.vel.dtype == self.disp.dtype, 'vel dtype mismatch'

        if self.acc is not None:
            assert self.acc.shape == self.pmid.shape, 'acc shape mismatch'
            assert self.acc.dtype == self.disp.dtype, 'acc dtype mismatch'

        if self.val is not None:
            assert self.val.shape[0] == self.pmid.shape[0], 'val num mismatch'
            assert self.val.dtype == self.disp.dtype, 'val dtype mismatch'


@register_pytree_node_class
@dataclass
class State:
    """State of particle species, of integration or observation

    Attributes:
        dm: dark matter particles
    """
    dm: Particles

    # TODO: some parameter- and time-dependent factors that, like particle
    # states, are to be propagated forward and backward

    def tree_flatten(self):
        children = tuple(getattr(self, field.name) for field in fields(self))
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
@dataclass
class Param:
    """Cosmological parameters
    """
    Omega_m: float
    Omega_L: float
    A_s: float
    n_s: float
    h: float

    def tree_flatten(self):
        children = tuple(getattr(self, field.name) for field in fields(self))
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

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


@partial(custom_vjp, nondiff_argnums=(4,))
#@partial(jit, static_argnames='chunk_size')
def scatter(pmid, disp, mesh, val=1., chunk_size=1024**2):
    """Scatter particle values to mesh in n-D with CIC window
    """
    ptcl_num, spatial_ndim = pmid.shape

    chunk_size = min(chunk_size, ptcl_num)
    chunk_num = ptcl_num // chunk_size
    chan_shape = mesh.shape[spatial_ndim:]

    val = jnp.asarray(val)

    if val.ndim == 0:
        val = jnp.full(ptcl_num, val)

    chunks = (
        pmid.reshape(chunk_num, chunk_size, spatial_ndim),
        disp.reshape(chunk_num, chunk_size, spatial_ndim),
        val.reshape(chunk_num, chunk_size, *chan_shape),
    )

    mesh = lax.scan(_scatter, mesh, chunks)[0]

    return mesh


def _scatter(mesh, chunk):
    pmid, disp, val = chunk

    ptcl_num, spatial_ndim = pmid.shape

    spatial_shape = mesh.shape[:spatial_ndim]
    chan_ndim = mesh.ndim - spatial_ndim

    pmid = pmid[:, jnp.newaxis]  # insert neighbor axis
    disp = disp[:, jnp.newaxis]  # insert neighbor axis
    val = val[:, jnp.newaxis]  # insert neighbor axis

    # CIC
    tgt = jnp.floor(disp)
    neighbors = (jnp.arange(2 ** spatial_ndim)[:, jnp.newaxis]
                 >> jnp.arange(spatial_ndim)
                ) & 1
    tgt = tgt + neighbors  # jax type promotion prefers float
    frac = (1.0 - jnp.abs(disp - tgt)).prod(axis=-1)
    frac = frac.reshape(frac.shape + (1,) * chan_ndim)
    tgt = pmid + tgt.astype(pmid.dtype)

    # periodic boundaries
    tgt = jnp.remainder(tgt, jnp.array(spatial_shape))

    # scatter
    tgt = tuple(tgt[..., i] for i in range(spatial_ndim))
    mesh = mesh.at[tgt].add(val * frac)

    return mesh, None


def _scatter_adjoint(mesh_cot, chunk):
    """Adjoint of _scatter

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
    tgt = jnp.floor(disp)
    neighbors = (jnp.arange(2 ** spatial_ndim)[:, jnp.newaxis]
                 >> jnp.arange(spatial_ndim)
                ) & 1
    tgt = tgt + neighbors  # jax type promotion prefers float
    frac = 1.0 - jnp.abs(disp - tgt)
    sign = jnp.sign(tgt - disp)
    frac_grad = []
    for i in range(spatial_ndim):
        not_i = tuple(range(0, i)) + tuple(range(i, spatial_ndim))
        frac_grad.append(sign[..., i] * frac[..., not_i].prod(axis=-1))
    frac_grad = jnp.stack(frac_grad, axis=-1)
    frac = frac.prod(axis=-1)
    frac = frac.reshape(frac.shape + (1,) * chan_ndim)
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


def scatter_fwd(pmid, disp, mesh, val=1., chunk_size=1024**2):
    mesh = scatter(pmid, disp, mesh, val=val, chunk_size=chunk_size)
    return mesh, (pmid, disp, val)


def scatter_bwd(chunk_size, res, mesh_cot):
    pmid, disp, val = res

    ptcl_num, spatial_ndim = pmid.shape

    chunk_size = min(chunk_size, ptcl_num)
    chunk_num = ptcl_num // chunk_size
    chan_shape = mesh_cot.shape[spatial_ndim:]

    val = jnp.asarray(val)

    if val.ndim == 0:
        val = jnp.full(ptcl_num, val)

    chunks = (
        pmid.reshape(chunk_num, chunk_size, spatial_ndim),
        disp.reshape(chunk_num, chunk_size, spatial_ndim),
        val.reshape(chunk_num, chunk_size, *chan_shape),
    )

    disp_cot, val_cot = lax.scan(_scatter_adjoint, mesh_cot, chunks)[1]

    disp_cot = disp_cot.reshape(ptcl_num, spatial_ndim)
    val_cot = val_cot.reshape(ptcl_num, *chan_shape)

    return None, disp_cot, mesh_cot, val_cot


scatter.defvjp(scatter_fwd, scatter_bwd)


@partial(custom_vjp, nondiff_argnums=(4,))
#@partial(jit, static_argnames='chunk_size')
def gather(pmid, disp, mesh, val=0., chunk_size=1024**2):
    """Gather particle values from mesh in n-D with CIC window
    """
    ptcl_num, spatial_ndim = pmid.shape

    chunk_size = min(chunk_size, ptcl_num)
    chunk_num = ptcl_num // chunk_size
    chan_shape = mesh.shape[spatial_ndim:]

    val = jnp.asarray(val)

    if val.ndim == 0:
        val = jnp.full(ptcl_num, val)

    chunks = (
        pmid.reshape(chunk_num, chunk_size, spatial_ndim),
        disp.reshape(chunk_num, chunk_size, spatial_ndim),
        val.reshape(chunk_num, chunk_size, *chan_shape),
    )

    val = lax.scan(_gather, mesh, chunks)[1]

    val = val.reshape(ptcl_num, *chan_shape)

    return val


def _gather(mesh, chunk):
    pmid, disp, val_in = chunk

    ptcl_num, spatial_ndim = pmid.shape

    spatial_shape = mesh.shape[:spatial_ndim]
    chan_ndim = mesh.ndim - spatial_ndim

    pmid = pmid[:, jnp.newaxis]  # insert neighbor axis
    disp = disp[:, jnp.newaxis]  # insert neighbor axis

    # CIC
    tgt = jnp.floor(disp)
    neighbors = (jnp.arange(2 ** spatial_ndim)[:, jnp.newaxis]
                 >> jnp.arange(spatial_ndim)
                ) & 1
    tgt = tgt + neighbors  # jax type promotion prefers float
    frac = (1.0 - jnp.abs(disp - tgt)).prod(axis=-1)
    frac = frac.reshape(frac.shape + (1,) * chan_ndim)
    tgt = pmid + tgt.astype(pmid.dtype)

    # periodic boundaries
    tgt = jnp.remainder(tgt, jnp.array(spatial_shape))

    # gather
    tgt = tuple(tgt[..., i] for i in range(spatial_ndim))
    val = mesh[tgt]

    val = val_in + (val * frac).sum(axis=1)

    return mesh, val


def _gather_adjoint(mesh, chunk):
    """Adjoint of _gather

    Gather disp_cot from val_cot and mesh;
    Scatter val_cot to mesh_cot.
    """
    pmid, disp, val_cot = chunk

    ptcl_num, spatial_ndim = pmid.shape

    spatial_shape = mesh.shape[:spatial_ndim]
    chan_ndim = mesh.ndim - spatial_ndim
    chan_axis = tuple(range(-chan_ndim, 0))

    pmid = pmid[:, jnp.newaxis]  # insert neighbor axis
    disp = disp[:, jnp.newaxis]  # insert neighbor axis
    val_cot = val_cot[:, jnp.newaxis]  # insert neighbor axis

    # CIC
    tgt = jnp.floor(disp)
    neighbors = (jnp.arange(2 ** spatial_ndim)[:, jnp.newaxis]
                 >> jnp.arange(spatial_ndim)
                ) & 1
    tgt = tgt + neighbors  # jax type promotion prefers float
    frac = 1.0 - jnp.abs(disp - tgt)
    sign = jnp.sign(tgt - disp)
    frac_grad = []
    for i in range(spatial_ndim):
        not_i = tuple(range(0, i)) + tuple(range(i, spatial_ndim))
        frac_grad.append(sign[..., i] * frac[..., not_i].prod(axis=-1))
    frac_grad = jnp.stack(frac_grad, axis=-1)
    frac = frac.prod(axis=-1)
    frac = frac.reshape(frac.shape + (1,) * chan_ndim)
    tgt = pmid + tgt.astype(pmid.dtype)

    # periodic boundaries
    tgt = jnp.remainder(tgt, jnp.array(spatial_shape))

    # gather disp_cot from val_cot and mesh, and scatter val_cot to mesh_cot
    tgt = tuple(tgt[..., i] for i in range(spatial_ndim))
    val = mesh[tgt]

    disp_cot = (val_cot * val).sum(axis=chan_axis)
    disp_cot = (disp_cot[..., jnp.newaxis] * frac_grad).sum(axis=1)

    mesh_cot = jnp.zeros_like(mesh)
    mesh_cot = mesh_cot.at[tgt].add(val_cot * frac)

    return mesh_cot, disp_cot


def gather_fwd(pmid, disp, mesh, val=0., chunk_size=1024**2):
    val = gather(pmid, disp, mesh, val=val, chunk_size=chunk_size)
    return val, (pmid, disp, mesh)


def gather_bwd(chunk_size, res, val_cot):
    pmid, disp, mesh = res

    ptcl_num, spatial_ndim = pmid.shape

    chunk_size = min(chunk_size, ptcl_num)
    chunk_num = ptcl_num // chunk_size
    chan_shape = mesh.shape[spatial_ndim:]

    val_cot = jnp.asarray(val_cot)

    if val_cot.ndim == 0:
        val_cot = jnp.full(ptcl_num, val_cot)

    chunks = (
        pmid.reshape(chunk_num, chunk_size, spatial_ndim),
        disp.reshape(chunk_num, chunk_size, spatial_ndim),
        val_cot.reshape(chunk_num, chunk_size, *chan_shape),
    )

    mesh_cot, disp_cot = lax.scan(_gather_adjoint, mesh, chunks)

    disp_cot = disp_cot.reshape(ptcl_num, spatial_ndim)

    return None, disp_cot, mesh_cot, val_cot


gather.defvjp(gather_fwd, gather_bwd)


def rfftnfreq(shape, dtype=float):
    """wavevectors (angular frequencies) for `numpy.fft.rfftn`

    Return:
        A list of broadcastable wavevectors (similar to `numpy.ogrid`)
    """
    ndim = len(shape)
    rad_per_cycle = 2 * jnp.pi

    kvec = []
    for axis, n in enumerate(shape[:-1]):
        k = jnp.fft.fftfreq(n).astype(dtype) * rad_per_cycle

        k = k.reshape((-1,) + (1,) * (ndim - axis - 1))

        kvec.append(k)

    k = jnp.fft.rfftfreq(shape[-1]).astype(dtype) * rad_per_cycle
    kvec.append(k)

    return kvec


@custom_vjp
def laplace(kvec, dens, param):
    """Laplace kernel in Fourier space
    """
    ndim = len(kvec)

    assert dens.ndim == ndim, 'spatial dimension mismatch'
    assert all(kvec[axis].shape[axis-ndim] == dens.shape[axis]
               for axis in range(ndim)), 'spatial shape mismatch'
    assert jnp.iscomplexobj(dens), 'source not in Fourier space'

    kk = sum(k**2 for k in kvec)

    pot = jnp.where(kk != 0., - dens / kk, 0.)

    return pot


def laplace_fwd(kvec, dens, param):
    pot = laplace(kvec, dens, param)
    return pot, kvec


def laplace_bwd(kvec, pot_cot):
    """Custom vjp to avoid NaN when using where

    See also https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
    """
    kk = sum(k**2 for k in kvec)
    dens_cot = - pot_cot * kk
    return None, dens_cot, None


laplace.defvjp(laplace_fwd, laplace_bwd)


def negative_gradient(k, pot):
    assert jnp.iscomplexobj(pot), 'potential not in Fourier space'

    nyquist = jnp.pi
    eps = jnp.finfo(k.dtype).eps

    neg_ik = jnp.where(jnp.abs(jnp.abs(k) - nyquist) > eps, -1j * k, 0j)

    neg_grad = neg_ik * pot

    return neg_grad


#@partial(jit, static_argnames='config')
def gravity(pmid, disp, param, config):
    """Compute particles' gravitational forces on a mesh with FFT
    """
    real_dtype = disp.dtype

    kvec = rfftnfreq(config.mesh_shape, dtype=real_dtype)

    dens = jnp.zeros(config.mesh_shape, dtype=real_dtype)

    dens = scatter(pmid, disp, dens, chunk_size=config.chunk_size)

    dens = jnp.fft.rfftn(dens)

    pot = laplace(kvec, dens, param)

    acc = []

    for k in kvec:
        neg_grad = negative_gradient(k, pot)

        neg_grad = jnp.fft.irfftn(neg_grad, s=config.mesh_shape)

        neg_grad = gather(pmid, disp, neg_grad, chunk_size=config.chunk_size)

        acc.append(neg_grad)

    acc = jnp.stack(acc, axis=-1)

    return acc


def force(state, param, config):
    ptcl = state.dm
    ptcl.acc = gravity(ptcl.pmid, ptcl.disp, param, config)
    return state


def force_adjoint(state, state_cot, param, config):
    ptcl = state.dm
    ptcl_cot = state_cot.dm
    _gravity = partial(gravity, config=config)
    ptcl.acc, acc_vjp = vjp(_gravity, ptcl.pmid, ptcl.disp, param)
    ptcl_cot.acc = acc_vjp(ptcl_cot.vel)[1]
    return state, state_cot


def kick(state, step, param, config):
    ptcl = state.dm
    ptcl.vel = ptcl.vel + ptcl.acc * step
    return state


def kick_adjoint(state, state_cot, step, param, config):
    state = kick(state, step, param, config)

    ptcl_cot = state_cot.dm
    ptcl_cot.disp = ptcl_cot.disp - ptcl_cot.acc * step

    return state, state_cot


def drift(state, step, param, config):
    ptcl = state.dm
    ptcl.disp = ptcl.disp + ptcl.vel * step
    return state


def drift_adjoint(state, state_cot, step, param, config):
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


def leapfrog_adjoint(state, state_cot, step, param, config):
    """Leapfrog with adjoint equation
    """
    # FIXME HACK step
    state, state_cot = kick_adjoint(state, state_cot, 0.5*step, param, config)

    state, state_cot = drift_adjoint(state, state_cot, step, param, config)

    state, state_cot = force_adjoint(state, state_cot, param, config)

    state, state_cot = kick_adjoint(state, state_cot, 0.5*step, param, config)

    return state, state_cot


def form(ptcl, param, config):
    pass


def coevolve(state, param, config):
    ptcl = state.dm
    ptcl.val = form(ptcl, param, config)
    return state


def observe(state, observable, param, config):
    pass


@partial(custom_vjp, nondiff_argnums=(4,))
@partial(jit, static_argnames='config')
def integrate(state, observable, steps, param, config):
    """Time integration
    """
    state = force(state, param, config)  # how to skip this if acc is given?

    state = coevolve(state, param, config)  # same as above

    observable = observe(state, observable, param, config) # same as above

    def _integrate(carry, step):
        state, observable, param = carry

        state = leapfrog(state, step, param, config)

        state = coevolve(state, param, config)

        observable = observe(state, observable, param, config)

        carry = state, observable, param

        return carry, None

    carry = state, observable, param

    state, observable = lax.scan(_integrate, carry, steps)[0][:2]

    return state, observable


@partial(jit, static_argnames='config')
def integrate_adjoint(state, state_cot, observable_cot,
                      steps, param, config):
    """Time integration with adjoint equation
    """
    state, state_cot = force_adjoint(state, state_cot, param, config)

    #state = coevolve(state, param, config)

    #observable = observe(state, observable, param, config)

    def _integrate_adjoint(carry, step):
        state, state_cot, observable_cot, param = carry

        state, state_cot = leapfrog_adjoint(state, state_cot,
                                            step, param, config)

        #state = coevolve(state, param, config)

        #observable = observe(state, observable, param, config)

        carry = state, state_cot, observable_cot, param

        return carry, None

    carry = state, state_cot, observable_cot, param

    state, state_cot, observable_cot = lax.scan(_integrate_adjoint,
                                                carry, steps)[0][:3]

    return state, state_cot, observable_cot


def integrate_fwd(state, observable, steps, param, config):
    state, observable = integrate(state, observable, steps, param, config)
    return (state, observable), (state, steps, param)


def integrate_bwd(config, res, cotangents):
    state, steps, param = res
    state_cot, observable_cot = cotangents

    rev_steps = - steps  # FIXME HACK

    # need state below? need *_adjoint functions?
    state, state_cot, observable_cot = integrate_adjoint(
        state, state_cot, observable_cot, rev_steps, param, config)

    return state_cot, observable_cot, None, None  # FIXME HACK no param grad


integrate.defvjp(integrate_fwd, integrate_bwd)
