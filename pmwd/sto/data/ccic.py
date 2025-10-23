import jax.numpy as jnp

from pmwd.configuration import Configuration
from pmwd.cosmology import Cosmology
from pmwd.boltzmann import boltzmann
from pmwd.modes import white_noise, linear_modes
from pmwd.lpt import lpt


def gen_cc(sobol, mesh_shape=1, a_snapshots=(1,), a_nbody_num=61,
           so_type=None, so_nodes=None, soft_i=None, a_start=1/16, a_stop=1+1/128,
           cal_boltz=True):
    """Setup conf and cosmo given a sobol."""
    conf = Configuration(
        ptcl_spacing = sobol[0] / 128,
        ptcl_grid_shape = (128,) * 3,
        a_start = a_start,
        a_stop = a_stop,
        float_dtype = jnp.float64,
        mesh_shape = mesh_shape,
        a_snapshots = a_snapshots,
        a_nbody_num = a_nbody_num,
        so_type = so_type,
        so_nodes = so_nodes,
        soft_i = soft_i,
        softening_length = sobol[8],
    )

    cosmo = Cosmology(
        conf = conf,
        A_s_1e9 = sobol[2],
        n_s = sobol[3],
        Omega_m = sobol[4],
        Omega_b = sobol[5],
        Omega_k_ = sobol[6],
        h = sobol[7],
    )
    if cal_boltz:
        cosmo = boltzmann(cosmo, conf)

    return conf, cosmo


def gen_ic(seed, conf, cosmo):
    """Generate the initial condition with lpt for nbody."""
    modes = white_noise(seed, conf)

    modes = linear_modes(modes, cosmo, conf)
    ptcl, obsvbl = lpt(modes, cosmo, conf)

    return ptcl
