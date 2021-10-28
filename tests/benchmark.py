import pytest
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

from pmwd.pm import *
from pmwd.test_util import tree_randn_float0_like
from tests.pm_test import gen_ptcl


@pytest.mark.benchmark(min_rounds=1)
@pytest.mark.parametrize('mesh_size', [64, 256])
class TestBenchmarkIntegrate:
    def test_benchmark_integrate(self, benchmark, mesh_size):
        benchmark(self.run_integrate, mesh_size)

    def run_integrate(self, mesh_size):
        mesh_shape = (mesh_size,) * 3
        ptcl_grid_shape = mesh_shape
        disp_std = 3.
        int_dtype = 'i2'
        real_dtype = 'f4'
        ptcl = gen_ptcl(ptcl_grid_shape, disp_std, vel_ratio=0.1,
                        int_dtype=int_dtype, real_dtype=real_dtype)
        state = State(ptcl)
        obsvbl = None
        param = 0.
        time_steps = jnp.full(9, 0.1, dtype=real_dtype)
        dconf = DynamicConfig(time_steps=time_steps)
        sconf = StaticConfig(mesh_shape, chunk_size=1<<21)

        state = integrate(state, obsvbl,
                          param, dconf, sconf)[0]
        state.dm.vel.block_until_ready()  # wait for async ops to complete

    def test_benchmark_integrate_adj(self, benchmark, mesh_size):
        benchmark(self.run_integrate_adj, mesh_size)

    def run_integrate_adj(self, mesh_size):
        mesh_shape = (mesh_size,) * 3
        ptcl_grid_shape = mesh_shape
        disp_std = 3.
        int_dtype = 'i2'
        real_dtype = 'f4'
        ptcl = gen_ptcl(ptcl_grid_shape, disp_std, vel_ratio=0.1,
                        int_dtype=int_dtype, real_dtype=real_dtype)
        state = State(ptcl)
        state_cot = tree_randn_float0_like(state)
        obsvbl_cot = None
        param = 0.
        time_steps = jnp.full(9, 0.1, dtype=real_dtype)
        dconf = DynamicConfig(time_steps=time_steps)
        sconf = StaticConfig(mesh_shape, chunk_size=1<<21)

        state = integrate_adj(state, state_cot, obsvbl_cot,
                              param, dconf, sconf)[0]
        state.dm.vel.block_until_ready()  # wait for async ops to complete
