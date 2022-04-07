from dataclasses import fields
from functools import partial
from operator import itemgetter
from typing import Optional, Any

import numpy as np
import jax.numpy as jnp
from jax import float0
from jax.tree_util import tree_map

from pmwd.tree_util import pytree_dataclass
from pmwd.cosmology import E2


@partial(pytree_dataclass, frozen=True)
class Particles:
    """Particle state or adjoint particle state.

    Parameters
    ----------
    pmid : jnp.ndarray
        Particles' IDs by mesh indices, of signed int dtype. They are the nearest mesh
        grid points from particles' Lagrangian positions.
    disp : jnp.ndarray
        Particles' (comoving) displacements from pmid in [L], or adjoint. For
        displacements from particles' Lagrangian positions, use ``ptcl.disp -
        gen_ptcl(conf).disp``.
    vel : jnp.ndarray, optional
        Particles' canonical momenta in [H_0 L], or adjoint.
    acc : jnp.ndarray, optional
        Particles' accelerations in [H_0^2 L], or force vjp.
    attr : pytree
        Particles' attributes or adjoint, can be custom features.

    Raises
    ------
    AssertionError
        If any particle field has the wrong types, shapes, or dtypes.

    """

    pmid: jnp.ndarray
    disp: jnp.ndarray
    vel: Optional[jnp.ndarray] = None
    acc: Optional[jnp.ndarray] = None
    attr: Any = None

    @classmethod
    def from_pos(cls, pos, conf, wrap=True):
        """Construct particle state of pmid and disp from positions.

        Parameters
        ----------
        pos : array_like
            Particle positions in [L].
        conf : Configuration
        wrap : bool, optional
            Whether to wrap around the periodic boundaries.

        Notes
        -----
        There may be collisions in particle ``pmid``.

        """
        pmid = jnp.rint(pos / conf.cell_size)
        disp = pos - pmid * conf.cell_size

        pmid = pmid.astype(conf.int_dtype)
        disp = disp.astype(conf.float_dtype)

        if wrap:
            pmid %= jnp.array(conf.mesh_shape, dtype=conf.int_dtype)

        return cls(pmid, disp)

    def __getitem__(self, key):
        return tree_map(itemgetter(key), self)

    @property
    def num(self):
        return self.pmid.shape[0]

    @property
    def dim(self):
        return self.pmid.shape[1]

    @property
    def int_dtype(self):
        return self.pmid.dtype

    @property
    def float_dtype(self):
        return self.disp.dtype

    def assert_valid(self):
        for field in fields(self):
            data = getattr(self, field.name)
            if data is not None:
                # FIXME after jax issues #4433 is addressed
                assert isinstance(data, (jnp.ndarray, np.ndarray)), (
                    f'{field.name} must be jax.numpy.ndarray')

        # FIXME after jax issues #4433 is addressed
        assert (jnp.issubdtype(self.pmid.dtype, jnp.signedinteger)
                or self.pmid.dtype == float0), 'pmid must be signed integers'

        assert self.disp.shape == self.pmid.shape, 'disp shape mismatch'
        assert jnp.issubdtype(self.disp.dtype, jnp.floating), (
            'disp must be floating point numbers')

        if self.vel is not None:
            assert self.vel.shape == self.pmid.shape, 'vel shape mismatch'
            assert self.vel.dtype == self.disp.dtype, 'vel dtype mismatch'

        if self.acc is not None:
            assert self.acc.shape == self.pmid.shape, 'acc shape mismatch'
            assert self.acc.dtype == self.disp.dtype, 'acc dtype mismatch'

        def assert_valid_attr(v):
            assert v.shape[0] == self.num, 'attr num mismatch'
            assert v.dtype == self.disp.dtype, 'attr dtype mismatch'
        tree_map(assert_valid_attr, self.attr)


def gen_ptcl(conf):
    """Generate particles on a uniform grid with zero velocities.

    Parameters
    ----------
    conf : Configuration

    Returns
    -------
    ptcl : Particles
        Particles on a uniform grid with zero velocities.

    """
    pmid, disp = [], []
    for i, (sp, sm) in enumerate(zip(conf.ptcl_grid_shape, conf.mesh_shape)):
        pmid_1d = jnp.linspace(0, sm, num=sp, endpoint=False)
        pmid_1d = jnp.rint(pmid_1d)
        pmid_1d = pmid_1d.astype(conf.int_dtype)
        pmid.append(pmid_1d)

        disp_1d = jnp.arange(sp) * sm - pmid_1d.astype(int) * sp  # exact int arithmetic
        disp_1d *= conf.cell_size / sp
        disp_1d = disp_1d.astype(conf.float_dtype)
        disp.append(disp_1d)

    pmid = jnp.meshgrid(*pmid, indexing='ij')
    pmid = jnp.stack(pmid, axis=-1).reshape(-1, conf.dim)

    disp = jnp.meshgrid(*disp, indexing='ij')
    disp = jnp.stack(disp, axis=-1).reshape(-1, conf.dim)

    vel = jnp.zeros_like(disp)

    ptcl = Particles(pmid, disp, vel=vel)

    return ptcl


def ptcl_pos(ptcl, conf, dtype=None, wrap=True):
    """Particle positions in [L].

    Parameters
    ----------
    ptcl : Particles
    conf : Configuration
    dtype : jax.numpy.dtype, optional
        Output float dtype. Default is conf.float_dtype.
    wrap : bool, optional
        Whether to wrap around the periodic boundaries.

    Returns
    -------
    pos : jax.numpy.ndarray
        Particle positions.

    """
    if dtype is None:
        dtype = conf.float_dtype

    pos = ptcl.pmid.astype(dtype)
    pos *= conf.cell_size
    pos += ptcl.disp.astype(dtype)

    if wrap:
        pos %= jnp.array(conf.box_size, dtype=dtype)

    return pos


def ptcl_rpos(ptcl1, ptcl0, conf, wrap=True):
    """Positions of Particles ptcl1 relative to Particles ptcl0 in [L].

    Parameters
    ----------
    ptcl1 : Particles
    ptcl0 : Particles
    conf : Configuration
    wrap : bool, optional
        Whether to wrap around the periodic boundaries.

    Returns
    -------
    rpos : jax.numpy.ndarray
        Particle relative positions.

    """
    rpos = ptcl1.pmid - ptcl0.pmid
    rpos = rpos.astype(conf.float_dtype)
    rpos *= conf.cell_size
    rpos += ptcl1.disp - ptcl0.disp

    if wrap:
        box_size = jnp.array(conf.box_size, dtype=conf.float_dtype)
        rpos -= jnp.rint(rpos / box_size) * box_size

    return rpos


def ptcl_rsd(ptcl, los, a, cosmo):
    """Particle redshift-space distortion displacements in [L].

    Parameters
    ----------
    ptcl : Particles
    los : array_like
        Line-of-sight **unit vectors**, global or per particle. Vector norms are *not*
        checked.
    a : array_like
        Scale factors, global or per particle.
    cosmo : Cosmology

    Returns
    -------
    rsd : jax.numpy.ndarray
        Particle redshift-space distortion displacements.

    """
    conf = cosmo.conf

    los = jnp.asarray(los, dtype=conf.float_dtype)
    a = jnp.asarray(a, dtype=conf.float_dtype)

    E = jnp.sqrt(E2(a, cosmo))
    E = E.astype(conf.float_dtype)

    rsd = (ptcl.vel * los).sum(axis=1, keepdims=True)
    rsd *= los / (a**2 * E)

    return rsd


def ptcl_los(ptcl, obs, conf):
    """Particle line-of-sight unit vectors.

    Parameters
    ----------
    ptcl : Particles
    obs : array_like or Particles
        Observer position in [L].
    conf : Configuration

    Returns
    -------
    los : jax.numpy.ndarray
        Particles line-of-sight unit vectors.

    """
    if not isinstance(obs, Particles):
        obs = jnp.asarray(obs, conf.float_dtype)
        obs = Particles.from_pos(obs, conf, wrap=False)

    los = ptcl_rpos(ptcl, obs, conf, wrap=False)
    los /= jnp.linalg.norm(los, axis=1, keepdims=True)

    return los
