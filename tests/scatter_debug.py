import numpy as np
import jax.numpy as jnp
from jax import device_put
import pmwd
import time

np = 1000
pmid = 10*jnp.ones([np,3],dtype=jnp.uint32)
mesh = jnp.zeros([2,2,2],dtype=jnp.float32)
val = jnp.ones([np,],dtype=jnp.float32)
disp = 0.0*jnp.ones([np,3],dtype=jnp.float32)

disp = disp.at[:,0].set(0.7)
disp = disp.at[:,1].set(0.2)
disp = disp.at[:,2].set(0.4)
print("disp:")
print(disp)

print(pmid.shape)
print(disp.shape)
print(val.shape)
print(mesh.shape)

cell_size = 1.0
ptcl_spacing = 1.0
offset = tuple((0.0,0.0,0.))

mesh0 = mesh*0;
start = time.time()
for ii in range(1):
    mesh0 = pmwd.scatter_cuda(pmid, disp, val, mesh0, offset, ptcl_spacing, cell_size).block_until_ready()
print(time.time() - start)

print(mesh0.flatten())
