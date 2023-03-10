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
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.99'
key = random.PRNGKey(101)
ngrid = 512
print("loading")
ptcl_spacing = 1.7  # Lagrangian space Cartesian particle grid spacing, in Mpc/h by default
ptcl_grid_shape = (ngrid,) * 3
conf = Configuration(ptcl_spacing, ptcl_grid_shape, mesh_shape=1)  # 1x mesh shape
ptcl = Particles.gen_grid(conf)
pmid = ptcl.pmid
mesh = random.uniform(key, shape=(ngrid,ngrid,ngrid),dtype=jnp.float32,minval=0.0,maxval=2.0)
val = random.uniform(key, shape=(ngrid*ngrid*ngrid,),dtype=jnp.float32,minval=0.0,maxval=2.0)
disp = random.uniform(key, shape=(ngrid*ngrid*ngrid,3),dtype=jnp.float32,minval=0.0,maxval=2.0)
cell_size = ptcl_spacing
offset = tuple((0.,0.,0.))

print("jax simple")
mesh0 = mesh*0;
mesh0 = _scatter(pmid, disp, conf, mesh0, val, offset, None).block_until_ready()
start = time.time()
for ii in range(10):
    mesh0 = _scatter(pmid, disp, conf, mesh0, val, offset, None).block_until_ready()
print(time.time() - start)

print("jax complex")
mesh1 = mesh*0
mesh1 = _scatter(pmid, disp, conf, mesh1, val, offset, cell_size).block_until_ready()
start = time.time()
for ii in range(10):
    mesh1 = _scatter(pmid, disp, conf, mesh1, val, offset, cell_size).block_until_ready()
print(time.time() - start)

print("mesh0 max:",mesh0.max())
print("mesh0 min:",mesh0.min())
print("mesh1 max:",mesh1.max())
print("mesh1 min:",mesh1.min())
print("diff abs max:", jnp.abs(mesh0-mesh1).max())
print("diff std:", jnp.std(mesh0-mesh1))
