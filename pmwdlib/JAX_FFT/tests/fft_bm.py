# -*- coding: utf-8 -*-

# from mpi4py import MPI
import os

import jax
import jax.numpy as jnp

from fft_common import Dist, Dir
from cufftmp_jax import cufftmp

def test_main(): 
    nrank = 0
    nsize = 2
    
    jax.experimental.enable_x64(new_val=True)
    os.environ["JAX_ENABLE_X64"] = "True"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (nrank)

    dist_fft = cufftmp
    # dist_fft = xfft

    dist = Dist.create('X')
    input_shape = (4, 6, 8)
    # dtype = jnp.complex64
    dtype = jnp.float64
    key = jax.random.PRNGKey(nrank)
    input1 = jax.random.normal(key, shape=input_shape, dtype=dtype)
    input = jnp.concatenate((input1, input1), dtype=dtype)
    # input = jnp.ones(shape=input_shape, dtype=dtype)
    
    def fwd(x, rank, size, dist):
        # return dist_fft(x, rank, size, dist, Dir.FWD)
        return jnp.fft.rfftn(x)

    def bwd(x, rank, size, dist):
        # return dist_fft(x, rank, size, dist, Dir.INV)
        return jnp.fft.irfftn(x)

    def fwd_bwd_bench(x, rank, size):
        m = fwd(x, rank, size, dist)
        print('++++++++++after fwd++++++++++' + str(rank))
        mm = m/384
        [m1, m2] = jnp.split(mm, [4], axis=0)
        print(m1.shape)
        print(m1)
        print(m2)
        # print(mm)
        
        # jax.debug.visualize_array_sharding(m)
        n = bwd(m, rank, size, dist)
        print('++++++++++after bwd++++++++++' + str(rank))
        print(n.shape)
        print(n.dtype)
        print(n)
        # jax.debug.visualize_array_sharding(n)
        return n

    # Warmup
    x = fwd_bwd_bench(input, nrank, nsize).block_until_ready()
    # x = fwd(input, nrank, nsize, dist).block_until_ready()

    # error = jnp.sum((input - x) ** 2)
    error = jnp.std(input - x)/jnp.std(input)
    print(f"误差: {error:.10f}")
    # print(x)
    
if __name__ == "__main__":
    test_main()