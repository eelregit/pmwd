from functools import partial

import pytest
import numpy as np
import jax.numpy as jnp
import jax.test_util as jtu
from jax import vjp
from jax import random
from jax.tree_util import tree_map
from jax.config import config
config.update("jax_enable_x64", True)

import pmwd.pm as pm
from pmwd.pm import *
from pmwd.test_util import check_custom_vjp


def gen_pmid(ptcl_grid_shape):
    ndim = len(ptcl_grid_shape)
    pmid = jnp.meshgrid(*[jnp.arange(s) for s in ptcl_grid_shape], indexing='ij')
    pmid = jnp.stack(pmid, axis=-1).reshape(-1, ndim)
    return pmid

def gen_disp(ptcl_grid_shape, disp_std):
    key = random.PRNGKey(0)
    ndim = len(ptcl_grid_shape)
    disp = disp_std * random.normal(key, ptcl_grid_shape + (ndim,))
    disp = disp.reshape(-1, ndim)
    return disp

def gen_val(ptcl_grid_shape, chan_shape, val_mean, val_std):
    key = random.PRNGKey(0)
    val = val_mean + val_std * random.normal(key, ptcl_grid_shape + chan_shape)
    val = val.reshape(-1, *chan_shape)
    return val

def gen_ptcl(ptcl_grid_shape, disp_std, vel_ratio=None, acc_ratio=None,
             chan_shape=None, val_mean=1., val_std=0.):
    pmid = gen_pmid(ptcl_grid_shape)
    disp = gen_disp(ptcl_grid_shape, disp_std)

    vel = None
    if vel_ratio is not None:
        vel = vel_ratio * disp

    acc = None
    if acc_ratio is not None:
        acc = acc_ratio * disp

    val = None
    if chan_shape is not None:
        val = gen_val(ptcl_grid_shape, chan_shape, val_mean, val_std)

    ptcl = Particles(pmid, disp, vel=vel, acc=acc, val=val)

    return ptcl


@pytest.mark.parametrize(
    'ptcl_grid_shape, disp_std, vel_ratio, acc_ratio, '
    'chan_shape, val_mean, val_std',
    [
        ((64,),     3., 0.,   None, None,   1., 0.),
        ((4, 16),   2., None, None, (),     1., 1.),
        ((2, 4, 8), 1., None, 0.,   (1, 1), 0., 1.),
    ],
    ids=['1d', '2d', '3d'],
)
def test_ptcl(ptcl_grid_shape, disp_std, vel_ratio, acc_ratio,
              chan_shape, val_mean, val_std):
    ptcl = gen_ptcl(
        ptcl_grid_shape, disp_std,
        vel_ratio=vel_ratio, acc_ratio=acc_ratio,
        chan_shape=chan_shape, val_mean=val_mean, val_std=val_std,
    )
    ptcl.assert_valid()
    assert ptcl.num == np.prod(ptcl_grid_shape)
    assert ptcl.ndim == len(ptcl_grid_shape)


@pytest.mark.parametrize(
    'ptcl_num, pos, chan_shape',
    [
        (3, (-0.5,),          (2, 1)),
        (5, (0.5, -1.5),      (1, 2, 3)),
        (7, (-1.5, 2.5, 3.5), None),
        (7, (1.5, -2.5, 3.5), ()),
        (7, (1.5, 2.5, -3.5), (1,)),
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

    mesh = scatter(ptcl, mesh, val=val, chunk_size=3)
    mesh_expected = jnp.full(mesh_shape + chan_shape,
                             ptcl_num * 2**-spatial_ndim)
    jtu.check_eq(mesh, mesh_expected)


@pytest.mark.parametrize(
    'ptcl_num, spatial_ndim, chan_shape',
    [
        (3, 1, (2, 1)),
        (5, 2, (1, 2, 3)),
        (7, 3, None),
        (7, 3, ()),
        (7, 3, (1,)),
    ],
    ids=['1d', '2d', '3d1', '3d2', '3d3'],
)
class TestScatterGather:
    def test_scatter_sum(self, ptcl_num, spatial_ndim, chan_shape):
        mesh_shape = (3,) * spatial_ndim
        val = 1.
        ptcl_grid_shape = (ptcl_num,) + (1,) * (spatial_ndim - 1)
        disp_std = 7.
        ptcl = gen_ptcl(ptcl_grid_shape, disp_std,
                        chan_shape=chan_shape, val_mean=val, val_std=0.)
        if chan_shape is None:
            chan_shape = ()
        else:
            val = ptcl.val
        mesh = jnp.zeros(mesh_shape + chan_shape)

        mesh = scatter(ptcl, mesh, val=val, chunk_size=3)
        sum_expected = ptcl_num * np.prod(chan_shape)
        jtu.check_close(mesh.sum(), sum_expected)

    def test_gather_uniform(self, ptcl_num, spatial_ndim, chan_shape):
        mesh_shape = (5,) * spatial_ndim
        val = 0.
        ptcl_grid_shape = (ptcl_num,) + (1,) * (spatial_ndim - 1)
        disp_std = 7.
        ptcl = gen_ptcl(ptcl_grid_shape, disp_std,
                        chan_shape=chan_shape, val_mean=val, val_std=0.)
        if chan_shape is None:
            chan_shape = ()
        else:
            val = ptcl.val
        mesh = jnp.ones(mesh_shape + chan_shape)

        val = gather(ptcl, mesh, val=val, chunk_size=3)
        val_expected = jnp.ones((ptcl_num,) + chan_shape)
        jtu.check_eq(val, val_expected)


@pytest.mark.parametrize('chunk_size', [16, None], ids=['16', 'ptcl_num'])
class TestScatterGatherCustomVJP:
    def test_scatter_custom_vjp(self, chunk_size):
        mesh_shape = (4, 9)
        chan_shape = (2, 1)
        ptcl_grid_shape = mesh_shape
        disp_std = 7.
        ptcl = gen_ptcl(ptcl_grid_shape, disp_std,
                        chan_shape=chan_shape, val_mean=1., val_std=1.)
        mesh = jnp.zeros(mesh_shape + chan_shape)

        primals = ptcl.disp, mesh, ptcl.val
        args = (ptcl.pmid,)
        kwargs = {'chunk_size': chunk_size}
        check_custom_vjp(pm._scatter, primals, args=args, kwargs=kwargs)

    def test_gather_custom_vjp(self, chunk_size):
        mesh_shape = (4, 9)
        chan_shape = (2, 1)
        ptcl_grid_shape = mesh_shape
        disp_std = 7.
        ptcl = gen_ptcl(ptcl_grid_shape, disp_std,
                        chan_shape=chan_shape, val_mean=1., val_std=1.)
        mesh = jnp.ones(mesh_shape + chan_shape)

        primals = ptcl.disp, mesh, ptcl.val
        args = (ptcl.pmid,)
        kwargs = {'chunk_size': chunk_size}
        check_custom_vjp(pm._gather, primals, args=args, kwargs=kwargs)


@pytest.mark.parametrize('mesh_shape', [(4, 9), (7, 8)],
                         ids=['evenodd', 'oddeven'])
def test_gravity_vjp(mesh_shape):
    ptcl_grid_shape = mesh_shape
    disp_std = 7.
    pmid = gen_pmid(ptcl_grid_shape)
    disp = gen_disp(ptcl_grid_shape, disp_std)
    param = 0.
    dconf = DynamicConfig()
    sconf = StaticConfig(mesh_shape, chunk_size=49)

    def _gravity(disp, param):
        ptcl = Particles(pmid, disp)
        acc = gravity(ptcl, param, dconf, sconf)
        return acc
    _gravity_vjp = partial(vjp, _gravity)
    eps = jnp.sqrt(jnp.finfo(disp.dtype).eps)
    jtu.check_vjp(_gravity, _gravity_vjp, (disp, param), eps=eps)


class TestIntegrate:
    @pytest.mark.parametrize('mesh_shape', [(4, 9), (7, 8)],
                             ids=['evenodd', 'oddeven'])
    def test_integrate_custom_vjp(self, mesh_shape):
        ptcl_grid_shape = mesh_shape
        disp_std = 7.
        ptcl = gen_ptcl(ptcl_grid_shape, disp_std, vel_ratio=0.1)
        state = State(ptcl)
        obsvbl = None
        param = 0.
        dconf = DynamicConfig(time_steps=jnp.full(9, 0.1))
        sconf = StaticConfig(mesh_shape, chunk_size=None)

        primals = state, obsvbl, param
        cot_out_std = tree_map(lambda x: 1., (state, obsvbl))
        # otherwise acc cot also backprops in the automatic vjp
        cot_out_std[0].dm.acc = 0.
        # fix dconf here otherwise it gets cot in the automatic vjp
        kwargs = {'dconf': dconf, 'sconf': sconf}
        check_custom_vjp(integrate, primals,
                         cot_out_std=cot_out_std, kwargs=kwargs)






# benchmark with block_until_ready


# test reversibility
