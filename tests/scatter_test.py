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
print(pmid.min())
pmid = pmid.astype(jnp.uint32)
print("finish casting")
print(pmid.shape)
print(pmid.dtype)
print(disp.shape)
print(disp.dtype)
print(val.shape)
print(val.dtype)
print(mesh.shape)
print(mesh.dtype)

cell_size = 1.0
ptcl_spacing = 1.0
offset = tuple((0.0,0.0,0.))

mesh0 = mesh*0;
#print(mesh0)
print(mesh0.unsafe_buffer_pointer())
start = time.time()
for ii in range(1):
    mesh0 = pmwd.scatter_cuda(pmid, disp, val, mesh0, offset, ptcl_spacing, cell_size).block_until_ready()
print(time.time() - start)

print(mesh.max())
print(mesh0.max())
print(mesh.min())
print(mesh0.min())
print("called")

print(pmid.max())
print(pmid.min())
print(disp.min())
print(disp.max())
#print(mesh0)
