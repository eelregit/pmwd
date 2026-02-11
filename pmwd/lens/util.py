from functools import partial

import jax.numpy as jnp

from pmwd.background import distance
from pmwd.tree_util import dyn_field, fxd_field, asarray_of


lens_dyn_field = partial(dyn_field, validate=(asarray_of(field='dtype'),
                                              jnp.atleast_1d))
lens_dyn_field.__doc__ = '`tree_util.dyn_field` for strong gravitational lensing.'
lens_dyn_2d_field = partial(dyn_field, validate=(asarray_of(field='dtype'),
                                                 jnp.atleast_2d))
lens_dyn_2d_field.__doc__ = '`tree_util.dyn_field` for 2D angular positions in strong gravitational lensing.'

lens_fxd_field = partial(fxd_field, validate=(asarray_of(field='dtype'),
                                              jnp.atleast_1d))
lens_fxd_field.__doc__ = '`tree_util.fxd_field` for strong gravitational lensing.'


#TODO worth turning ray arrays into a Rays type in the future?

def _canonicalize_ang_pos(x):
    x = jnp.asarray(x)
    if x.shape[-1] != 2:
        raise ValueError(f'angular position shape {x.shape} not ending with 2')
    return x


def _canonicalize_array(f, shape=None, dtype=None):
    f = jnp.asarray(f)
    if shape is not None and f.shape != shape:
        raise ValueError(f'array shape {f.shape} != {shape}')
    if dtype is not None and f.dtype != dtype:
        raise ValueError(f'array dtype {f.dtype} != {dtype}')
    return f


def Sigma_crit(cosmo, a_lens, a_src):
    """Critical surface density in :math:`M / L^2`."""
    return (cosmo.c**2 / (4 * jnp.pi * cosmo.G)
            * distance(a_src, cosmo, type='transverse')
            / (distance(a_lens, cosmo, type='angdiam')
               * distance(a_src, cosmo, type='transverse', a_ref=a_lens)))
    #FIXME check eq
