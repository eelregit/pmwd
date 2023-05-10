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
    Particles,
)
from pmwd.scatter import _scatter
from jax import random
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.9'
key = random.PRNGKey(101)
ngrid = 512
print("loading")
ptcl_spacing = 1.  # Lagrangian space Cartesian particle grid spacing, in Mpc/h by default
ptcl_grid_shape = (ngrid,) * 3
conf = Configuration(ptcl_spacing, ptcl_grid_shape, mesh_shape=1.,float_dtype=jnp.float32)  # 1x mesh shape
ptcl = Particles.gen_grid(conf)
pmid = ptcl.pmid.astype(jnp.uint32)
#pmid = ptcl.pmid
mesh = random.uniform(key, shape=conf.mesh_shape,dtype=jnp.float32,minval=0.0,maxval=2.0)
val = random.uniform(key, shape=(ngrid*ngrid*ngrid,),dtype=jnp.float32,minval=0.0,maxval=2.0)
disp = random.uniform(key, shape=(ngrid*ngrid*ngrid,3),dtype=jnp.float32,minval=0.0,maxval=2.0)
cell_size = 1.
offset = tuple((0.,0.,0.))

mesh0 = mesh*0;
mesh0 = pmwd.scatter_cuda(pmid, disp, val, mesh0, offset, ptcl_grid_shape, ptcl_spacing, cell_size).block_until_ready()
print("cuda")
start = time.time()
for ii in range(10):
    mesh0 = pmwd.scatter_cuda(pmid, disp, val, mesh0, offset, ptcl_grid_shape, ptcl_spacing, cell_size).block_until_ready()
print(time.time() - start)

mesh_val = mesh0*0
mesh_val = _scatter(pmid, disp, conf, mesh_val, val, offset, cell_size).block_until_ready()
print("jax")
start = time.time()
for ii in range(10):
    mesh_val = _scatter(pmid, disp, conf, mesh_val, val, offset, cell_size).block_until_ready()
print(time.time() - start)

print("mesh0 max:",mesh0.max())
print("mesh_val max:",mesh_val.max())
print("mesh0 min:",mesh0.min())
print("mesh_val min:",mesh_val.min())
print("diff std:", jnp.std(mesh_val-mesh0))
