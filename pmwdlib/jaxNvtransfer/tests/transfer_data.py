# -*- coding: utf-8 -*-

from mpi4py import MPI
import os
import time

import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental.pjit import pjit
from jax import block_until_ready

from jaxnvtransfer import nvtransfer

def main():
    comm = MPI.COMM_WORLD
    nrank = comm.Get_rank()
    nsize = comm.Get_size()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (nrank + 6)
    os.environ["JAX_ENABLE_X64"] = "True"

    jit_nvtransfer = nvtransfer
    # jit_nvtransfer = jit(nvtransfer, static_argnums=(1, 2, 3, 4))
    # jit_nvtransfer = block_until_ready(nvtransfer)

    dst_rank = nrank + 1
    dst_rank = dst_rank % nsize   

    src_rank = nrank - 1
    if(src_rank < 0):
        src_rank = nsize - 1    
    # '''
    if(nrank == 0):
        send_size = 0
        send_data = jax.numpy.zeros(1)
    else:
        send_size = 512
        # dtype = jnp.float32
        dtype = jnp.float64
        key = jax.random.PRNGKey(nrank)
        send_data = jax.random.normal(key, shape=(send_size,), dtype=dtype)
        # input = jnp.ones(shape=input_shape, dtype=dtype)  
    '''
    send_size = 101 * (nrank + 5)
    # dtype = jnp.float32
    dtype = jnp.float64
    key = jax.random.PRNGKey(nrank)
    send_data = jax.random.normal(key, shape=(send_size,), dtype=dtype)
    # input = jnp.ones(shape=input_shape, dtype=dtype)   
    '''

    print(f'rank is {nrank}, send size is {send_size}, dst rank is {dst_rank}')
    comm.isend(send_size, dst_rank, tag=11)
    req = comm.irecv(source=src_rank, tag=11)
    recv_size = req.wait()
    print(f'rank is {nrank}, recv size is {recv_size}, src rank is {src_rank}')

    # nvtransfer(send_buf, send_buf_size, recv_buf_size, src_rank, dst_rank)
    start_time = time.time()
    recv_data = jit_nvtransfer(send_data, send_size, recv_size, src_rank, dst_rank)
    end_time = time.time()
    elapsed_time = end_time - start_time
    #print(f"函数 nvtransfer 第一次 运行时间: {elapsed_time} 秒")
    if(nrank == 1):
        print(f'第一次 send size = {send_size}')
        # print(send_data)
        print('send data:', send_data[:5], '...' , send_data[-5:])
    if(nrank == 0):
        print(f'第一次 recv size = {recv_size}')
        # print(recv_data)
        print('recv data:', recv_data[:5], '...' , recv_data[-5:])
    '''       
    # dtype = jnp.float32
    dtype = jnp.float64
    key = jax.random.PRNGKey(nrank+1)
    send_data = jax.random.normal(key, shape=(send_size,), dtype=dtype)
    # input = jnp.ones(shape=input_shape, dtype=dtype)
    #     
    start_time = time.time()
    recv_data = jit_nvtransfer(send_data, send_size, recv_size, src_rank, dst_rank)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"函数 nvtransfer 第二次 运行时间: {elapsed_time} 秒")   
    if(nrank == 0):
        print(f'第二次 send size = {send_size}')
        # print(send_data)
        print('send data:', send_data[:5], '...' , send_data[-5:])
    if(nrank == 1):
        print(f'第二次 recv size = {recv_size}')
        # print(recv_data)
        print('recv data:', recv_data[:5], '...' , recv_data[-5:])

    send_size = 102 * (nrank + 5)
    # dtype = jnp.float32
    dtype = jnp.float64
    key = jax.random.PRNGKey(nrank+2)
    send_data = jax.random.normal(key, shape=(send_size,), dtype=dtype)
    # input = jnp.ones(shape=input_shape, dtype=dtype)

    # print(f'rank is {nrank}, send size is {send_size}, dst rank is {dst_rank}')
    comm.isend(send_size, dst_rank, tag=11)
    req = comm.irecv(source=src_rank, tag=11)
    recv_size = req.wait()   

    start_time = time.time()
    recv_data = jit_nvtransfer(send_data, send_size, recv_size, src_rank, dst_rank)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"函数 nvtransfer 第三次 运行时间: {elapsed_time} 秒")   
    '''   
    '''
    if(nrank == 0):
        print(f'第三次 send size = {send_size}')
        # print(send_data)
        print('send data:', send_data[:5], '...' , send_data[-5:])
    if(nrank == 1):
        print(f'第三次 recv size = {recv_size}')
        # print(recv_data)
        print('recv data:', recv_data[:5], '...' , recv_data[-5:])
    send_size = 103 * (nrank + 5)
    # dtype = jnp.float32
    dtype = jnp.float64
    key = jax.random.PRNGKey(nrank+3)
    send_data = jax.random.normal(key, shape=(send_size,), dtype=dtype)
    # input = jnp.ones(shape=input_shape, dtype=dtype)

    # print(f'rank is {nrank}, send size is {send_size}, dst rank is {dst_rank}')
    comm.isend(send_size, dst_rank, tag=11)
    req = comm.irecv(source=src_rank, tag=11)
    recv_size = req.wait()   

    start_time = time.time()
    recv_data = jit_nvtransfer(send_data, send_size, recv_size, src_rank, dst_rank)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"函数 nvtransfer 第四次 运行时间: {elapsed_time} 秒")      
    '''
    '''
    if(nrank == 0):
        print(f'第四次 send size = {send_size}')
        # print(send_data)
        print('send data:', send_data[:5], '...' , send_data[-5:])
    if(nrank == 1):
        print(f'第四次 recv size = {recv_size}')
        # print(recv_data)
        print('recv data:', recv_data[:5], '...' , recv_data[-5:])
    send_size = 104 * (nrank + 5)
    # dtype = jnp.float32
    dtype = jnp.float64
    key = jax.random.PRNGKey(nrank+4)
    send_data = jax.random.normal(key, shape=(send_size,), dtype=dtype)
    # input = jnp.ones(shape=input_shape, dtype=dtype)

    # print(f'rank is {nrank}, send size is {send_size}, dst rank is {dst_rank}')
    comm.isend(send_size, dst_rank, tag=11)
    req = comm.irecv(source=src_rank, tag=11)
    recv_size = req.wait()   

    start_time = time.time()
    recv_data = jit_nvtransfer(send_data, send_size, recv_size, src_rank, dst_rank)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"函数 nvtransfer 第五次 运行时间: {elapsed_time} 秒")     
    ''' 
    '''
    if(nrank == 0):
        print(f'第五次 send size = {send_size}')
        # print(send_data)
        print('send data:', send_data[:5], '...' , send_data[-5:])
    if(nrank == 1):
        print(f'第五次 recv size = {recv_size}')
        # print(recv_data)        
        print('recv data:', recv_data[:5], '...' , recv_data[-5:])
    '''
if __name__ == "__main__":
    main()
    # NVSHMEM_BOOTSTRAP=MPI mpirun --allow-run-as-root -n 2 python tests/transfer_data.py

