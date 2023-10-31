import matplotlib.pyplot as plt
plt.style.use('adjoint.mplstyle')

from pmwd import (
    Configuration,
    SimpleLCDM,
    boltzmann,
    white_noise, linear_modes,
    lpt,
    nbody,
    scatter,
)
from pmwd.vis_util import simshow


def model(modes, cosmo, conf):
    cosmo = boltzmann(cosmo, conf)
    modes = linear_modes(modes, cosmo, conf)
    ptcl, obsvbl = lpt(modes, cosmo, conf)
    ptcl, obsvbl = nbody(ptcl, obsvbl, cosmo, conf)
    return ptcl, obsvbl


ptcl_spacing = 1.
ptcl_grid_shape = (512,) * 3
conf = Configuration(ptcl_spacing, ptcl_grid_shape, mesh_shape=2)

cosmo = SimpleLCDM(conf)

seed = 0
modes = white_noise(seed, conf)

ptcl, obsvbl = model(modes, cosmo, conf)

dens = scatter(ptcl, conf)
fig, _ = simshow(dens[:16].mean(axis=0), norm='CosmicWebNorm')
fig.savefig('slab.pdf')
plt.close(fig)
