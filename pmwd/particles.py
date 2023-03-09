from dataclasses import field
from functools import partial
from itertools import accumulate
from operator import itemgetter, mul
from typing import Optional, Any, Union

from numpy.typing import ArrayLike
import jax.numpy as jnp
from jax.tree_util import tree_map

from pmwd.tree_util import pytree_dataclass
from pmwd.configuration import Configuration
from pmwd.cosmology import E2
from pmwd.util import is_float0_array
from pmwd.pm_util import enmesh


ArrayLike = Union[ArrayLike, jnp.ndarray]


@partial(pytree_dataclass, aux_fields="conf", frozen=True)
class Particles:
    """Particle state.

    Particles are indexable.

    Array-likes are converted to ``jax.numpy.ndarray`` of ``conf.pmid_dtype`` or
    ``conf.float_dtype`` at instantiation.

    Parameters
    ----------
    conf : Configuration
        Configuration parameters.
    pmid : array_like
        Particle IDs by mesh indices, of signed int dtype. They are the nearest mesh
        grid points from particles' Lagrangian positions. It can save memory compared to
        the raveled particle IDs, e.g., 6 bytes for 3 times int16 versus 8 bytes for
        uint64. Call ``raveled_id`` for the raveled IDs.
    disp : array_like
        # FIXME after adding the CUDA scatter and gather ops
        Particle comoving displacements from pmid in [L]. For displacements from
        particles' grid Lagrangian positions, use ``ptcl_rpos(ptcl,
        Particles.gen_grid(ptcl.conf), ptcl.conf)``. It can save the particle locations
        with much more uniform precision than positions, whereever they are. Call
        ``pos`` for the positions.
    vel : array_like, optional
        Particle canonical velocities in [H_0 L].
    acc : array_like, optional
        Particle accelerations in [H_0^2 L].
    attr : pytree, optional
        Particle attributes (custom features).

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
        for name, value in self.named_children():
            dtype = conf.pmid_dtype if name == 'pmid' else conf.float_dtype
            if name == 'attr':
                value = tree_map(lambda x: jnp.asarray(x, dtype=dtype), value)
            else:
                value = (value if value is None or is_float0_array(value)
                         else jnp.asarray(value, dtype=dtype))
            object.__setattr__(self, name, value)

    def __len__(self):
        return len(self.pmid)

    def __getitem__(self, key):
        return tree_map(itemgetter(key), self)

    @classmethod
    def from_pos(cls, conf, pos, wrap=True):
        """Construct particle state of ``pmid`` and ``disp`` from positions.

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

    @classmethod
    def gen_grid(cls, conf, vel=False, acc=False):
        """Generate particles on a uniform grid with zero velocities.

        Parameters
        ----------
        conf : Configuration
        vel : bool, optional
            Whether to initialize velocities to zeros.
        acc : bool, optional
            Whether to initialize accelerations to zeros.

        """
        pmid, disp = [], []
        for i, (sp, sm) in enumerate(zip(conf.ptcl_grid_shape, conf.mesh_shape)):
            pmid_1d = jnp.linspace(0, sm, num=sp, endpoint=False)
            pmid_1d = jnp.rint(pmid_1d)
            pmid_1d = pmid_1d.astype(conf.pmid_dtype)
            pmid.append(pmid_1d)

            # exact int arithmetic
            disp_1d = jnp.arange(sp) * sm - pmid_1d.astype(int) * sp
            disp_1d *= conf.cell_size / sp
            disp_1d = disp_1d.astype(conf.float_dtype)
            disp.append(disp_1d)

        pmid = jnp.meshgrid(*pmid, indexing='ij')
        pmid = jnp.stack(pmid, axis=-1).reshape(-1, conf.dim)

        disp = jnp.meshgrid(*disp, indexing='ij')
        disp = jnp.stack(disp, axis=-1).reshape(-1, conf.dim)

        vel = jnp.zeros_like(disp) if vel else None
        acc = jnp.zeros_like(disp) if acc else None

        return cls(conf, pmid, disp, vel=vel, acc=acc)

        #pid = [jnp.arange(s, dtype=conf.id_dtype) for s in conf.ptcl_grid_shape]
        #pid = jnp.meshgrid(*pid, indexing='ij')
        #pid = jnp.stack(pid, axis=-1).reshape(conf.ptcl_num, conf.dim)

        #dis = jnp.zeros_like(pid, dtype=conf.float_dtype)
        #vel = jnp.zeros_like(dis) if vel else None
        #acc = jnp.zeros_like(dis) if acc else None

        #return cls(conf, pid, dis, vel=vel, acc=acc)

    def raveled_id(self, dtype=jnp.uint64, wrap=False):
        """Particle raveled IDs, flattened from ``pmid``.

        Parameters
        ----------
        dtype : dtype_like, optional
            Output int dtype.
        wrap : bool, optional
            Whether to wrap around the periodic boundaries.

        Returns
        -------
        raveled_id : jax.numpy.ndarray
            Particle raveled IDs.

        """
        conf = self.conf

        pmid = self.pmid
        if wrap:
            pmid = pmid % jnp.array(conf.mesh_shape, dtype=conf.pmid_dtype)

        strides = tuple(accumulate((1,) + conf.mesh_shape[:0:-1], mul))[::-1]

        raveled_id = sum(i.astype(dtype) * s for i, s in zip(pmid.T, strides))

        return raveled_id

    def pos(self, dtype=jnp.float64, wrap=True):
        """Particle positions.

        Parameters
        ----------
        dtype : dtype_like, optional
            Output float dtype.
        wrap : bool, optional
            Whether to wrap around the periodic boundaries.

        Returns
        -------
        pos : jax.numpy.ndarray
            Particle positions in [L].

        """
        conf = self.conf

        pos = self.pmid.astype(dtype)
        pos *= conf.cell_size
        pos += self.disp.astype(dtype)

        if wrap:
            pos %= jnp.array(conf.box_size, dtype=dtype)

        return pos


def ptcl_enmesh(ptcl, conf, offset=0, cell_size=None, mesh_shape=None,
                wrap=True, drop=True, grad=False):
    """Compute multilinear mesh indices and fractions given particles.

    See ``pm_util.enmesh``.

    Parameters
    ----------
    ptcl : Particles
    conf : Configuration
    offset : array_like, optional
        Offset of mesh to particle grid. If 0D, the value is used in each dimension.
    cell_size : float, optional
        Mesh cell size in [L]. Default is ``conf.cell_size``.
    mesh_shape : tuple of int, optional
        Mesh shape. Default is ``conf.mesh_shape``.
    wrap : bool, optional
        Whether to wrap around the periodic boundaries.
    drop : bool, optional
        Whether to set negative out-of-bounds indices of ``ind`` to ``mesh_shape``,
        avoiding some of them being treated as in bounds, thus allowing them to be
        dropped by ``add()`` and ``get()`` of ``jax.numpy.ndarray.at``.
    grad : bool, optional
        Whether to return ``frac_grad``, gradients of ``frac``.

    Returns
    -------
    ind : (ptcl_num, 2**dim, dim) jax.numpy.ndarray
        Mesh indices.
    frac : (ptcl_num, 2**dim) jax.numpy.ndarray
        Multilinear fractions on the mesh.
    frac_grad : (ptcl_num, 2**dim, dim) jax.numpy.ndarray
        Multilinear fraction gradients on the mesh.

    """
    wrap_shape = conf.mesh_shape if wrap else None

    if mesh_shape is None:
        mesh_shape = conf.mesh_shape
    drop_shape = mesh_shape if drop else None

    return enmesh(ptcl.pmid, ptcl.disp, conf.cell_size, wrap_shape,
                  offset, cell_size, drop_shape, grad)


def ptcl_pos(ptcl, conf, dtype=float, wrap=True):
    raise RuntimeError('Deprecated and replaced by ptcl.pos')


def ptcl_rpos(ptcl, ref, conf, wrap=True):
    """Particle positions relative to references.

    Parameters
    ----------
    ptcl : Particles
    ref : array_like or Particles
        Reference points or particles.
    conf : Configuration
    wrap : bool, optional
        Whether to wrap around the periodic boundaries.

    Returns
    -------
    rpos : jax.numpy.ndarray of conf.float_dtype
        Particle relative positions in [L].

    """
    if not isinstance(ref, Particles):
        ref = Particles.from_pos(conf, ref, wrap=False)

    rpos = ptcl.pmid - ref.pmid
    rpos = rpos.astype(conf.float_dtype)
    rpos *= conf.cell_size
    rpos += ptcl.disp - ref.disp

    if wrap:
        box_size = jnp.array(conf.box_size, dtype=conf.float_dtype)
        rpos -= jnp.rint(rpos / box_size) * box_size

    return rpos


def ptcl_rsd(ptcl, los, a, cosmo):
    """Particle redshift-space distortion displacements.

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
        Particle redshift-space distortion displacements in [L].

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
        Observer position.
    conf : Configuration

    Returns
    -------
    los : jax.numpy.ndarray of conf.float_dtype
        Particles line-of-sight unit vectors.

    """
    los = ptcl_rpos(ptcl, obs, conf, wrap=False)
    los /= jnp.linalg.norm(los, axis=1, keepdims=True)

    return los
