"""pmwd: particle mesh with derivatives"""


import jax
import jax.numpy as jnp

#print(f'{jnp.array(1.)!r}')

#FIXME should use absolute in package __init__.py??
#FIXME somehow JAX automatically enable x64 after the following line?
from pmwd.cosmology import Cosmology, SimpleLCDM, Planck18
#print(f'{jnp.array(2.)!r}, NOTE dtype changes mysteriously now')
from pmwd.background import E2, H_deriv, Omega_m_a, distance_cache, distance
from pmwd.perturbation import (transfer_cache, transfer_fit, transfer,
                               growth_cache, growth,
                               varlin_cache, varlin,
                               linear_power)
from pmwd.solver import Solver
from pmwd.modes import white_noise, linear_modes
#from pmwd.particles import (Particles, ptcl_mass, ptcl_enmesh,  #FIXME conf problem
#                            ptcl_rpos, ptcl_rsd, ptcl_los)
from pmwd.particles import Particles, ptcl_mass
#from pmwd.observables import FIXME
from pmwd.scatter import scatter
from pmwd.gather import gather
from pmwd.gravity import laplace, neg_grad, gravity
from pmwd.lpt import lpt
from pmwd.nbody import nbody
try:
    from pmwd._version import __version__
except ModuleNotFoundError:
    pass  # not installed


#jax.config.update("jax_enable_x64", True)


#move this to some style script for ipynb's, or the worst to every ipynb
#jnp.set_printoptions(precision=3, edgeitems=2, linewidth=128)
