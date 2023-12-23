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
    gather,
    Particles,
)
from pmwd.gather import _gather
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
pmid = ptcl.pmid
nbatch = 6
mesh = random.uniform(key, shape=conf.mesh_shape+(nbatch,),dtype=jnp.float32,minval=0.0,maxval=2.0)
val = random.uniform(key, shape=(ngrid*ngrid*ngrid,nbatch),dtype=jnp.float32,minval=0.0,maxval=2.0)
disp = random.uniform(key, shape=(ngrid*ngrid*ngrid,3),dtype=jnp.float32,minval=0.0,maxval=2.0)
cell_size = 0.83
offset = tuple((0.31,0.21,0.13))

val0 = val*0;
val0 = pmwd.gather_cuda(pmid, disp, val0, mesh, offset, ptcl_grid_shape, ptcl_spacing, cell_size).block_until_ready()
print("cuda")
start = time.time()
for ii in range(10):
    val0 = pmwd.gather_cuda(pmid, disp, val0, mesh, offset, ptcl_grid_shape, ptcl_spacing, cell_size).block_until_ready()
print(time.time() - start)

val1 = val*0
val1 = _gather(pmid, disp, conf, mesh, val1, offset, cell_size).block_until_ready()
print("jax")
start = time.time()
for ii in range(10):
    val1 = _gather(pmid, disp, conf, mesh, val1, offset, cell_size).block_until_ready()
print(time.time() - start)

print("val0 max:",val0.max())
print("val1 max:",val1.max())
print("val0 min:",val0.min())
print("val1 min:",val1.min())
print("diff std:", jnp.std(val1-val0))
