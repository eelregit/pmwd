from dataclasses import field, fields
from functools import partial
from operator import itemgetter
from typing import Optional, Any

import numpy as np
from numpy.typing import ArrayLike
import jax.numpy as jnp
from jax import float0
from jax.tree_util import tree_map

from pmwd.tree_util import pytree_dataclass
from pmwd.conf import Configuration
from pmwd.cosmology import E2


@partial(pytree_dataclass, aux_fields="conf", frozen=True)
class Particles:
    """Particle state or adjoint particle state.

    Particles are indexable.

    Array_like's are converted to jax.numpy.ndarray of conf.pmid_dtype or
    conf.float_dtype at instantiation.

    Parameters
    ----------
    conf : Configuration
        Configuration parameters.
    pmid : array_like
        Particles' IDs by mesh indices, of signed int dtype. They are the nearest mesh
        grid points from particles' Lagrangian positions.
    disp : array_like
        Particles' (comoving) displacements from pmid in [L], or adjoint. For
        displacements from particles' Lagrangian positions, use ``ptcl.disp -
        gen_ptcl(conf).disp``.
    vel : array_like, optional
        Particles' canonical momenta in [H_0 L], or adjoint.
    acc : array_like, optional
        Particles' accelerations in [H_0^2 L], or force vjp.
    attr : pytree, optional
        Particles' attributes or adjoint, can be custom features.

    """

    conf: Configuration = field(repr=False)

    pmid: ArrayLike
    disp: ArrayLike
    vel: Optional[ArrayLike] = None
    acc: Optional[ArrayLike] = None
    attr: Any = None

    def __post_init__(self):
        if self._is_transforming():
            return

        conf = self.conf
        for field in fields(self):
            value = getattr(self, field.name)
            dtype = conf.pmid_dtype if field.name == 'pmid' else conf.float_dtype
            value = tree_map(
                lambda x: x if isinstance(x, np.ndarray) and x.dtype == float0
                else jnp.asarray(x, dtype=dtype),
                value,
            )
            object.__setattr__(self, field.name, value)

    def __getitem__(self, key):
        return tree_map(itemgetter(key), self)

    @classmethod
    def from_pos(cls, conf, pos, wrap=True):
        """Construct particle state of pmid and disp from positions.

        There may be collisions in particle ``pmid``.

        Parameters
        ----------
        conf : Configuration
        pos : array_like
            Particle positions in [L].
        wrap : bool, optional
            Whether to wrap around the periodic boundaries.

        """
        pos = jnp.asarray(pos)

        pmid = jnp.rint(pos / conf.cell_size)
        disp = pos - pmid * conf.cell_size

        pmid = pmid.astype(conf.pmid_dtype)
        disp = disp.astype(conf.float_dtype)

        if wrap:
            pmid %= jnp.array(conf.mesh_shape, dtype=conf.pmid_dtype)

        return cls(conf, pmid, disp)


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
        pmid_1d = pmid_1d.astype(conf.pmid_dtype)
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

    ptcl = Particles(conf, pmid, disp, vel=vel)

    return ptcl


def ptcl_pos(ptcl, conf, dtype=None, wrap=True):
    """Particle positions in [L].

    Parameters
    ----------
    ptcl : Particles
    conf : Configuration
    dtype : dtype_like, optional
        Output float dtype. If None (default), conf.float_dtype.
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
    rpos : jax.numpy.ndarray of cosmo.conf.float_dtype
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
    rsd : jax.numpy.ndarray of cosmo.conf.float_dtype
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
    los : jax.numpy.ndarray of cosmo.conf.float_dtype
        Particles line-of-sight unit vectors.

    """
    if not isinstance(obs, Particles):
        obs = Particles.from_pos(conf, obs, wrap=False)

    los = ptcl_rpos(ptcl, obs, conf, wrap=False)
    los /= jnp.linalg.norm(los, axis=1, keepdims=True)

    return los
