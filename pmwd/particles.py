from functools import partial
from dataclasses import fields
from typing import Optional, Any

import jax.numpy as jnp
from jax.tree_util import tree_map

from pmwd.dataclasses import pytree_dataclass


@partial(pytree_dataclass, frozen=True)
class Particles:
    """Particle state or adjoint particle state.

    Parameters
    ----------
    pmid : jnp.ndarray
        Particles' IDs by mesh indices, of signed int dtype.
    disp : jnp.ndarray
        Particles' (comoving) displacements in [L], or adjoint.
    vel : jnp.ndarray, optional
        Particles' canonical momenta in [H_0 L], or adjoint.
    acc : jnp.ndarray, optional
        Particles' accelerations in [H_0^2 L], or force vjp.
    val : pytree
        Particles' custom feature values or adjoint.

    Raises
    ------
    AssertionError
        If any particle field has the wrong types, shapes, or dtypes.

    """
    pmid: jnp.ndarray
    disp: jnp.ndarray
    vel: Optional[jnp.ndarray] = None
    acc: Optional[jnp.ndarray] = None
    val: Any = None

    def __post_init__(self):
        self._assert_valid()

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

    def _assert_valid(self):
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


def ptcl_gen(conf):
    """Generate uniformly distributed particles with zero velocities.

    Parameters
    ----------
    conf : Configuration

    Returns
    -------
    ptcl : Particles
        Uniformly distributed particles with zero velocities.

    """
    pmid = (jnp.arange(s, step=conf.ptcl_sample, dtype=conf.int_dtype)
            for s in conf.mesh_shape)
    pmid = jnp.meshgrid(*pmid, indexing='ij')
    pmid = jnp.stack(pmid, axis=-1).reshape(-1, conf.dim)

    disp = jnp.zeros_like(pmid, dtype=conf.float_dtype)
    vel = jnp.zeros_like(disp)

    ptcl = Particles(pmid, disp, vel=vel)

    return ptcl
