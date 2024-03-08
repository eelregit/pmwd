# -*- coding: utf-8 -*-

from mpi4py import MPI
import os

import jax
import jax.numpy as jnp

from fft_common import Dist, Dir
from cufftmp_jax import cufftmp

def main():
    comm = MPI.COMM_WORLD
    nrank = comm.Get_rank()
    nsize = comm.Get_size()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (nrank + 4)
    os.environ["JAX_ENABLE_X64"] = "True"

    dist_fft = cufftmp
    # dist_fft = xfft
    dist = Dist.create('X')

    def fwd(x, rank, size, dist):
        return dist_fft(x, rank, size, dist, Dir.FWD)

    def bwd(x, rank, size, dist):
        return dist_fft(x, rank, size, dist, Dir.INV)

    def fwd_bwd_bench(x, rank, size):
        m = fwd(x, rank, size, dist)
        print('++++++++++after fwd++++++++++' + str(rank))
        print(m.shape)
        # print(m)
        
        # jax.debug.visualize_array_sharding(m)
        n = bwd(m, rank, size, dist)
        print('++++++++++after bwd++++++++++' + str(rank))
        print(n.shape)
        # print(n)
        # jax.debug.visualize_array_sharding(n)
        return n    

    
    input_shape = (128, 128, 128)
    # dtype = jnp.complex64
    dtype = jnp.float32
    key = jax.random.PRNGKey(nrank)
    input = jax.random.normal(key, shape=input_shape, dtype=dtype)
    # input = jnp.ones(shape=input_shape, dtype=dtype)
    
    # Warmup
    x = fwd_bwd_bench(input, nrank, nsize).block_until_ready()
    # x = fwd(input, nrank, nsize, dist).block_until_ready()

    # error = jnp.sum((input - x) ** 2)
    error = jnp.std(input - x)/jnp.std(input)
    print(f"1st 误差: {error:.20f}")

    input_shape = (64, 64, 64)
    # dtype = jnp.complex64
    dtype = jnp.float64
    key = jax.random.PRNGKey(nrank)
    input = jax.random.normal(key, shape=input_shape, dtype=dtype)
    # input = jnp.ones(shape=input_shape, dtype=dtype)
    
    # Warmup
    x = fwd_bwd_bench(input, nrank, nsize).block_until_ready()
    # x = fwd(input, nrank, nsize, dist).block_until_ready()

    # error = jnp.sum((input - x) ** 2)
    error = jnp.std(input - x)/jnp.std(input)
    print(f"2nd 误差: {error:.20f}")

    # print(x)

def test_main(): 
    nrank = 0
    nsize = 2
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (nrank)

    dist_fft = cufftmp
    # dist_fft = xfft

    dist = Dist.create('X')
    input_shape = (4, 6, 8)
    # dtype = jnp.complex64
    dtype = jnp.float32
    key = jax.random.PRNGKey(nrank)
    # input = jax.random.normal(key, shape=input_shape, dtype=dtype)
    input = jnp.ones(shape=input_shape, dtype=dtype)
    
    def fwd(x, rank, size, dist):
        # return dist_fft(x, rank, size, dist, Dir.FWD)
        return jnp.fft.rfftn(x)

    def bwd(x, rank, size, dist):
        # return dist_fft(x, rank, size, dist, Dir.INV)
        return jnp.fft.irfftn(x)

    def fwd_bwd_bench(x, rank, size):
        m = fwd(x, rank, size, dist)
        print('++++++++++after fwd++++++++++' + str(rank))
        print(m.shape)
        print(m)
        
        # jax.debug.visualize_array_sharding(m)
        n = bwd(m, rank, size, dist)
        print('++++++++++after bwd++++++++++' + str(rank))
        print(n.shape)
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
    main()
    # test_main()