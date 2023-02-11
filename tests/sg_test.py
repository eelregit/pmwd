from functools import partial

import pytest
import numpy as np
import jax.numpy as jnp
import jax.test_util as jtu
from jax import vjp
from jax.tree_util import tree_map
from jax.config import config
config.update("jax_enable_x64", True)

import pmwd.pm as pm
from pmwd.pm import *
from pmwd.test_util import (gen_ptcl, gen_pmid, gen_disp, gen_mesh,
                            check_custom_vjp)


@pytest.mark.parametrize(
    'ptcl_grid_shape, disp_std, vel_ratio, acc_ratio, '
    'chan_shape, val_mean, val_std, int_dtype, real_dtype',
    [
        ((64,),     3., 0.,   None, None,   1., 0., 'i4', 'f2'),
        ((4, 16),   2., None, None, (),     1., 1., 'i2', 'f4'),
        ((2, 4, 8), 1., None, 0.,   (1, 1), 0., 1., 'i1', 'f8'),
    ],
    ids=['1d', '2d', '3d'],
)
def test_ptcl(ptcl_grid_shape, disp_std, vel_ratio, acc_ratio,
              chan_shape, val_mean, val_std, int_dtype, real_dtype):
    ptcl = gen_ptcl(
        ptcl_grid_shape, disp_std,
        vel_ratio=vel_ratio, acc_ratio=acc_ratio,
        chan_shape=chan_shape, val_mean=val_mean, val_std=val_std,
        int_dtype=int_dtype, real_dtype=real_dtype,
    )
    ptcl.assert_valid()
    assert ptcl.num == np.prod(ptcl_grid_shape)
    assert ptcl.ndim == len(ptcl_grid_shape)
    assert ptcl.int_dtype == int_dtype
    assert ptcl.real_dtype == real_dtype


@pytest.mark.parametrize(
    'ptcl_num, pos, chan_shape',
    [
        (3, (-1.,),        (2, 1)),
        (5, (1., -3.),     (1, 2, 3)),
        (7, (-3., 5., 7.), None),
        (7, (3., -5., 7.), ()),
        (7, (3., 5., -7.), (1,)),
    ],
    ids=['1d', '2d', '3d1', '3d2', '3d3'],
)
def test_scatter_centered_ptcl(ptcl_num, pos, chan_shape):
    spatial_ndim = len(pos)
    mesh_shape = (2,) * spatial_ndim
    val = 1.
    pmid = jnp.zeros((ptcl_num, spatial_ndim), dtype='i1')
    disp = jnp.array(pos)
    disp = jnp.tile(disp, (ptcl_num, 1))
    ptcl = Particles(pmid, disp)
    if chan_shape is None:
        chan_shape = ()
    else:
        val = jnp.full((ptcl_num,) + chan_shape, val)
    mesh = jnp.zeros(mesh_shape + chan_shape)

    mesh = scatter(ptcl, mesh, val=val, cell_size=2., chunk_size=3)
    mesh_expected = jnp.full(mesh_shape + chan_shape,
                             ptcl_num * 2**-spatial_ndim)
    jtu.check_eq(mesh, mesh_expected)
