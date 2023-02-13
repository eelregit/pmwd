import numpy as np
import jax.numpy as jnp
from jax import device_put
import pmwd
import time

print("loading")
pmid = jnp.load("/mnt/ceph/users/yinli/scatter_gather_tests/pmid.npy")
mesh = jnp.load("/mnt/ceph/users/yinli/scatter_gather_tests/mesh.npy")
val = jnp.load("/mnt/ceph/users/yinli/scatter_gather_tests/val.npy")
disp = jnp.load("/mnt/ceph/users/yinli/scatter_gather_tests/disp.npy")
print("finish loading")

print("casting")
pmid = pmid.astype(jnp.uint32)
print("finish casting")

cell_size = 1.0
ptcl_spacing = 1.0
offset = tuple((0.0,0.0,0.))

mesh0 = mesh*0;
print(mesh0)
start = time.time()
mesh0 = pmwd.scatter_cuda(pmid, disp, val, mesh0, offset, ptcl_spacing, cell_size).block_until_ready()
mesh0 = pmwd.scatter_cuda(pmid, disp, val, mesh0, offset, ptcl_spacing, cell_size).block_until_ready()
print(time.time() - start)

print("called")

print(mesh0)
