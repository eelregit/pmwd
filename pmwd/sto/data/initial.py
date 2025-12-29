import jax.numpy as jnp

from pmwd.configuration import Configuration
from pmwd.cosmology import Cosmology
from pmwd.boltzmann import boltzmann
from pmwd.modes import white_noise, linear_modes
from pmwd.lpt import lpt


def gen_cc(sobol,
           mesh_shape=1,
           a_snapshots=(1,),
           a_nbody_num=61,
           so_type=None,
           so_nodes=None,
           a_start=1/16,
           a_stop=1+1/128,
           float_dtype=jnp.float32,
           cal_boltz=True):
    """Setup conf and cosmo given a scaled Sobol and configurations."""
    conf = Configuration(
        ptcl_spacing = float(sobol[0] / 128),  # np.array -> float
        ptcl_grid_shape = (128,) * 3,
        float_dtype = float_dtype,
        mesh_shape = mesh_shape,
        observe_snapshots = True,
        a_nbody_num = a_nbody_num,
        so_type = so_type,
        so_nodes = so_nodes,
    )

    cosmo = Cosmology(
        conf = conf,
        A_s_1e9 = sobol[2],
        n_s = sobol[3],
        Omega_m = sobol[4],
        Omega_b = sobol[5],
        Omega_k_ = sobol[6],
        h = sobol[7],
        a_start = a_start,
        a_stop = a_stop,
        a_snapshots = a_snapshots,
        softening_length = sobol[8],
    )
    if cal_boltz:
        cosmo = boltzmann(cosmo, conf)

    return conf, cosmo


def gen_ic(seed, conf, cosmo):
    """Generate the initial condition with lpt for nbody."""
    modes = white_noise(seed, conf)
    modes = linear_modes(modes, cosmo, conf)
    ptcl, _ = lpt(modes, cosmo, conf)
    return ptcl
