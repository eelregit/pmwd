import numpy as np
import jax.numpy as jnp
from tqdm.notebook import tqdm
import h5py

import jax

from pmwd import (
    Configuration,
    SimpleLCDM,
    boltzmann,
    white_noise, linear_modes,
    lpt,
    nbody,
    scatter,
)
from pmwd.nbody import nbody_init, nbody_step


ptcl_spacing = 10.
ptcl_grid_shape = (16, 27, 16)
mesh_shape = (32, 54, 32)

conf = Configuration(ptcl_spacing, ptcl_grid_shape, mesh_shape=2,
                     a_nbody_maxstep=1/512)
a_nbody = conf.a_nbody
cosmo = SimpleLCDM(conf)
seed = 0
modes = white_noise(seed, conf)

cosmo = boltzmann(cosmo, conf)
modes = linear_modes(modes, cosmo, conf)
ptcl, obsvbl = lpt(modes, cosmo, conf)

f = h5py.File('data/time_evo.h5', 'w')

def save_snap(f, i, a, ptcl, conf, prefix):
    pos = np.asarray(ptcl.pos())
    dset = f.create_dataset(f'{prefix}_{i}', data=pos)
    dset.attrs.create('a', a)


# forward simulation
ptcl, obsvbl = nbody_init(a_nbody[0], ptcl, obsvbl, cosmo, conf)
save_snap(f, 0, a_nbody[0], ptcl, conf, 'fw')

for i, (a_prev, a_next) in enumerate(tqdm(zip(a_nbody[:-1], a_nbody[1:]), total=len(a_nbody)-1)):
    ptcl, obsvbl = nbody_step(a_prev, a_next, ptcl, obsvbl, cosmo, conf)
    save_snap(f, i+1, a_next, ptcl, conf, 'fw')

# reverse simulation
a_nbody_r = a_nbody[::-1]
save_snap(f, 0, a_nbody_r[0], ptcl, conf, 're')

for i, (a_prev, a_next) in enumerate(tqdm(zip(a_nbody_r[:-1], a_nbody_r[1:]), total=len(a_nbody_r)-1)):
    ptcl, obsvbl = nbody_step(a_prev, a_next, ptcl, obsvbl, cosmo, conf)
    save_snap(f, i+1, a_next, ptcl, conf, 're')

f.close()
