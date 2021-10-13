from functools import partial

from absl.testing import absltest
from absl.testing import parameterized
import jax.test_util as jtu
import chex

import jax.numpy as jnp
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


class TestParticles(parameterized.TestCase):
    @parameterized.named_parameters(
            ('1d', (64,),     3., 0.,   None, None,   1., 0.),
            ('2d', (4, 16),   2., 0.,   None, None,   1., 0.),
            ('3d', (2, 4, 8), 1., None, 0.,   (2, 1), 0., 1.))
    def test_ptcl(self, ptcl_grid_shape, disp_std, vel_ratio, acc_ratio,
                  chan_shape, val_mean, val_std):
        ptcl = gen_ptcl(
            ptcl_grid_shape,
            disp_std, vel_ratio=vel_ratio, acc_ratio=acc_ratio,
            chan_shape=chan_shape, val_mean=val_mean, val_std=val_std,
        )
        ptcl.assert_valid()


class TestScatterGather(parameterized.TestCase):
    data_centered_ptcl = [
        ('1d', 3, (-0.5,), (2, 1)),
        ('2d', 5, (0.5, -1.5), (1, 2, 3)),
        ('3d1', 7, (-1.5, 2.5, 3.5), None),
        ('3d2', 7, (1.5, -2.5, 3.5), ()),
        ('3d3', 7, (1.5, 2.5, -3.5), (1,)),
    ]

    def gen_centered_ptcl_mesh(self, ptcl_num, pos, chan_shape,
                      val, mesh_size, mesh_val):
        spatial_ndim = len(pos)

        pmid = jnp.zeros((ptcl_num, spatial_ndim), dtype='i1')
        disp = jnp.array(pos)
        disp = jnp.tile(disp, (ptcl_num, 1))
        ptcl = Particles(pmid, disp)

        if chan_shape is None:
            chan_shape = ()
            val = val
        else:
            val = jnp.full((ptcl_num,) + chan_shape, val)
        val_shape = (ptcl_num,) + chan_shape
        mesh_shape = (mesh_size,) * spatial_ndim + chan_shape

        mesh = jnp.full(mesh_shape, mesh_val)

        return ptcl, chan_shape, val, val_shape, mesh, mesh_shape

    @parameterized.named_parameters(*data_centered_ptcl)
    def test_scatter_centered_ptcl(self, ptcl_num, pos, chan_shape):
        ptcl, chan_shape, val, val_shape, mesh, mesh_shape = self.gen_centered_ptcl_mesh(
            ptcl_num, pos, chan_shape, 1., 2, 0.)

        mesh = scatter(ptcl, mesh, val=val, chunk_size=3)
        spatial_ndim = len(pos)
        mesh_expected = jnp.full(mesh_shape, ptcl_num * 2**-spatial_ndim)
        jtu.check_eq(mesh, mesh_expected)

    @parameterized.named_parameters(*data_centered_ptcl)
    def test_scatter_offcentered_ptcl(self, ptcl_num, pos, chan_shape):
        ptcl, chan_shape, val, val_shape, mesh, mesh_shape = self.gen_centered_ptcl_mesh(
            ptcl_num, pos, chan_shape, 1., 3, 0.)
        ptcl.disp = ptcl.disp + 0.7

        mesh = scatter(ptcl, mesh, val=val, chunk_size=3)
        sum_expected = jnp.full(chan_shape, ptcl_num).sum()
        jtu.check_close(mesh.sum(), sum_expected)

    @parameterized.named_parameters(*data_centered_ptcl)
    def test_gather_offcentered_ptcl(self, ptcl_num, pos, chan_shape):
        ptcl, chan_shape, val, val_shape, mesh, mesh_shape = self.gen_centered_ptcl_mesh(
            ptcl_num, pos, chan_shape, 0., 5, 1.)
        ptcl.disp = ptcl.disp - 0.8

        val = gather(ptcl, mesh, val=val, chunk_size=3)
        val_expected = jnp.ones(val_shape)
        jtu.check_eq(val, val_expected)

    @parameterized.named_parameters(('16', 16), ('ptcl_num', None))
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

    @parameterized.named_parameters(('16', 16), ('ptcl_num', None))
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


class TestGravity(parameterized.TestCase):
    data_bad_shape = [
        ('y_shape_mismatch', (4, 4)),
        ('chan_axis_first', (1, 4, 3)),
    ]

    @parameterized.named_parameters(*data_bad_shape)
    def test_laplace_raise_shapes(self, dens_shape):
        kvec = rfftnfreq((4, 4))
        dens = jnp.ones(dens_shape)
        param = 0.

        with self.assertRaisesRegex(
            TypeError,
            "div got incompatible shapes for broadcasting:",
        ):
            laplace(kvec, dens, param)

    @parameterized.named_parameters(*data_bad_shape)
    def test_negative_gradient_raise_shapes(self, pot_shape):
        kvec = rfftnfreq((4, 4))
        k = kvec[1]
        pot = jnp.ones(pot_shape)

        with self.assertRaisesRegex(
            TypeError,
            "mul got incompatible shapes for broadcasting:",
        ):
            negative_gradient(k, pot)

    @parameterized.named_parameters(('evenodd', (4, 9)), ('oddeven', (7, 8)))
    def test_gravity_vjp(self, mesh_shape):
        ptcl_grid_shape = mesh_shape
        disp_std = 7.
        param = 0.
        pmid = gen_pmid(ptcl_grid_shape)
        disp = gen_disp(ptcl_grid_shape, disp_std)
        config = Config(mesh_shape, chunk_size=16)

        def _gravity(disp, param):
            ptcl = Particles(pmid, disp)
            acc = gravity(ptcl, param=param, config=config)
            return acc
        _gravity_vjp = partial(vjp, _gravity)
        eps = jnp.sqrt(jnp.finfo(disp.dtype).eps)
        jtu.check_vjp(_gravity, _gravity_vjp, (disp, param), eps=eps)


class TestIntegrate(parameterized.TestCase):
    @parameterized.named_parameters(('evenodd', (4, 9)), ('oddeven', (7, 8)))
    def test_integrate_custom_vjp(self, mesh_shape):
        ptcl_grid_shape = mesh_shape
        disp_std = 7.
        ptcl = gen_ptcl(ptcl_grid_shape, disp_std, vel_ratio=0.1)
        state = State(ptcl)
        obsvbl = None
        steps = jnp.full(9, 0.1)
        param = 0.

        primals = state, obsvbl, steps, param
        cot_out_std = tree_map(lambda x: 1., (state, obsvbl))
        # otherwise acc cot also backprops in the automatic vjp
        cot_out_std[0].dm.acc = 0.
        cot_skip = tree_map(lambda x: False, primals)
        cot_skip = list(cot_skip)
        cot_skip[2] = True  # otherwise steps gets cot in the automatic vjp
        cot_skip = tuple(cot_skip)
        kwargs = {'config': Config(mesh_shape, chunk_size=16)}
        check_custom_vjp(integrate, primals,
                         cot_out_std=cot_out_std, cot_skip=cot_skip,
                         kwargs=kwargs)






# benchmark with block_until_ready


# test reversibility
