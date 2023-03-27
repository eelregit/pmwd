import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial
from torch.utils.data import Dataset, DataLoader
import os

from pmwd import (
    Configuration,
    Cosmology,
    boltzmann,
    white_noise,
    linear_modes,
    lpt,
    nbody
)
from pmwd.io_util import read_gadget_hdf5


def scale_Sobol(fn='sobol.txt', ind=slice(None)):
    """Scale the Sobol sequence samples, refer to the Table in the paper."""
    sobol = np.loadtxt(fn)[ind].T
    # functions mapping uniform random samples in [0, 1] to a desired one
    f_uni = lambda x, a, b : a + x * (b - a)
    f_log_uni = lambda x, a, b : np.exp(f_uni(x, np.log(a), np.log(b)))
    def f_log_trap(x, a, b, c):
        # a, b, c are the locations of the first 3 points of the symmetric trapezoid
        h = 1 / (c - a)
        x1 = (b - a) * h / 2
        x2 = (2*c - b - a) * h / 2
        y = np.zeros_like(x)
        m = (x < x1)
        y[m] = a + np.sqrt(2 * (b - a) * x[m] / h)
        m = (x1 <= x) & (x < x2)
        y[m] = x[m] / h + (a + b) / 2
        m = (x2 <= x)
        y[m] = c + b - a - np.sqrt((1 - x[m]) * 2 * (b - a) / h)
        return np.exp(y)

    # 0: box size, log-trapezoidal
    sobol[0] = f_log_trap(sobol[0], np.log(128)+np.log(0.2),
                          np.log(512)+np.log(0.2), np.log(128)+np.log(5))
    # 1: snapshot offset, uniform
    sobol[1] = f_uni(sobol[1], 0, 1/128)
    # 2: A_s_1e9, log-uniform
    sobol[2] = f_log_uni(sobol[2], 1, 4)
    # 3: n_s, log-uniform
    sobol[3] = f_log_uni(sobol[3], 0.75, 1.25)
    # 4: Omega_m, log-uniform
    sobol[4] = f_log_uni(sobol[4], 1/5, 1/2)
    # 5: Omega_b / Omega_m, log-uniform
    sobol[5] = f_log_uni(sobol[5], 1/8, 1/4)
    sobol[5] *= sobol[4]  # get Omega_b
    # 6: Omega_k / (1 - Omega_k), uniform
    sobol[6] = f_uni(sobol[6], -1/3, 1/3)
    sobol[6] = sobol[6] / (1 + sobol[6])  # get Omega_k
    # 7: h, log-uniform
    sobol[7] = f_log_uni(sobol[7], 0.5, 1)
    # 8: softening ratio, log-uniform
    sobol[8] = f_log_uni(sobol[8], 1/50, 1/20)
    sobol[8] *= sobol[0] / 128  # * ptcl_spacing = softening length

    return sobol.T


def gen_ic(i, fn_sobol='sobol.txt', re_sobol=False,
           a_start=1/16, a_stop=1+1/128, a_nbody_maxstep=1/64, mesh_shape=1):
    """Generate the initial condition with lpt for nbody.
    The seed for white noise is simply the Sobol index i.
    """
    sobol = scale_Sobol(fn_sobol, i)  # scaled Sobol parameters at i

    conf = Configuration(
        ptcl_spacing = sobol[0] / 128,
        ptcl_grid_shape = (128,) * 3,
        a_start = a_start,
        a_stop = a_stop,
        float_dtype = jnp.float64,
        growth_dt0 = 1,
        mesh_shape = mesh_shape,
        # TODO number of time steps
        a_nbody_maxstep=a_nbody_maxstep,
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

    seed = i
    modes = white_noise(seed, conf)

    cosmo = boltzmann(cosmo, conf)
    modes = linear_modes(modes, cosmo, conf)
    ptcl, obsvbl = lpt(modes, cosmo, conf)

    ret = (ptcl, cosmo, conf)
    if re_sobol: ret += (sobol,)

    return ret


class G4snapDataset(Dataset):

    def __init__(self, sims_dir, sobols=None, snaps_per_sim=121, sobols_edge=None):
        self.sims_dir = sims_dir
        if sobols is None:
            sobols = np.arange(*sobols_edge)
        self.sobols = sobols
        self.num_sims = len(sobols)
        self.snaps_per_sim = snaps_per_sim

        self.num_snaps = self.num_sims * self.snaps_per_sim

    def __len__(self):
        return self.num_snaps

    def __getitem__(self, idx):
        i_sobol = idx // self.snaps_per_sim
        i_snap = idx % self.snaps_per_sim
        snap_file = os.path.join(self.sims_dir, f'{self.sobols[i_sobol]:03}',
                                 'output', f'snapshot_{i_snap:03}')
        pos, vel, a = read_gadget_hdf5(snap_file)
        return pos, vel, a
