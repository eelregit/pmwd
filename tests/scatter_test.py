import numpy as np
import jax.numpy as jnp
from jax import device_put
import pmwd
import time
from pmwd import (
    Configuration,
    Cosmology, SimpleLCDM,
    boltzmann,
    white_noise, linear_modes,
    lpt,
    nbody,
    scatter,
)
from pmwd.scatter import _scatter


print("loading")
pmid = jnp.load("/mnt/ceph/users/yinli/scatter_gather_tests/pmid.npy")
mesh = jnp.load("/mnt/ceph/users/yinli/scatter_gather_tests/mesh.npy")
val = jnp.load("/mnt/ceph/users/yinli/scatter_gather_tests/val.npy")
disp = jnp.load("/mnt/ceph/users/yinli/scatter_gather_tests/disp.npy")
print(mesh.sum())
print(val.sum())
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
print(mesh0.unsafe_buffer_pointer())
start = time.time()
for ii in range(10000):
    mesh0 = pmwd.scatter_cuda(pmid, disp, val, mesh0, offset, ptcl_spacing, cell_size).block_until_ready()
print("time:")
print(time.time() - start)
print(mesh0.sum())
print(mesh0.max())
print(mesh0.min())

ptcl_spacing = 1.  # Lagrangian space Cartesian particle grid spacing, in Mpc/h by default
ptcl_grid_shape = (64,) * 3
conf = Configuration(ptcl_spacing, ptcl_grid_shape, mesh_shape=1)  # 1x mesh shape
mesh_val = mesh0*0

start = time.time()
for ii in range(10000):
    mesh_val = _scatter(pmid, disp, conf, mesh_val, val, 0, cell_size).block_until_ready()
print("time:")
print(time.time() - start)
print(mesh_val.sum())
print(mesh_val.max())
print(mesh_val.min())
print(jnp.std(mesh_val-mesh0))
