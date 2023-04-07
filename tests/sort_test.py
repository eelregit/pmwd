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
nkeys = 8192
keys = random.uniform(key, shape=(nkeys,),dtype=jnp.float32,minval=0.0,maxval=2.0)
sorted_keys = pmwd.sort_keys_cuda(keys)
sorted_index, keys_out = pmwd.argsort_cuda(keys)
print(keys)
print(sorted_index)
print("max error: ",jnp.max(jnp.abs(keys[sorted_index]-keys_out)))
print(keys_out)
print(sorted_keys)
