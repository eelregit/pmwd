import os
import numpy as np
import jax.numpy as jnp
from torch.utils.data import Dataset
from joblib import Parallel, delayed

from pmwd.configuration import Configuration
from pmwd.cosmology import Cosmology
from pmwd.boltzmann import boltzmann
from pmwd.modes import white_noise, linear_modes
from pmwd.lpt import lpt
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


def gen_cc(sobol, mesh_shape=1, a_out=1, a_nbody_num=63, so_type=None, so_nodes=None,
           a_start=1/16, a_stop=1+1/128):
    """Setup conf and cosmo given a sobol."""
    conf = Configuration(
        ptcl_spacing = sobol[0] / 128,
        ptcl_grid_shape = (128,) * 3,
        a_start = a_start,
        a_stop = a_stop,
        float_dtype = jnp.float64,
        mesh_shape = mesh_shape,
        a_out = a_out,
        a_nbody_num = a_nbody_num,
        so_type = so_type,
        so_nodes = so_nodes,
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
    cosmo = boltzmann(cosmo, conf)

    return conf, cosmo


def gen_ic(seed, conf, cosmo):
    """Generate the initial condition with lpt for nbody."""
    modes = white_noise(seed, conf)

    modes = linear_modes(modes, cosmo, conf)
    ptcl, obsvbl = lpt(modes, cosmo, conf)

    return ptcl


def read_g4data(sims_dir, sobol_ids, snap_ids, fn_sobol):
    data = {}
    def load_sobol(sidx):
        data[sidx] = {}
        sobol = scale_Sobol(fn_sobol, sidx)
        for snap in snap_ids:
            snap_file = os.path.join(sims_dir, f'{sidx:03}',
                                     'output', f'snapshot_{snap:03}')
            pos, vel, a = read_gadget_hdf5(snap_file)
            data[sidx][snap] = (pos, vel, a, sidx, sobol, snap)
    Parallel(n_jobs=min(8, len(sobol_ids)), prefer='threads', require='sharedmem')(
        delayed(load_sobol)(sidx) for sidx in sobol_ids)
    return data


class G4snapDataset(Dataset):

    def __init__(self, sims_dir, sobol_ids, snap_ids, fn_sobol='sobol.txt'):
        self.sims_dir = sims_dir

        self.sobol_ids = sobol_ids
        self.snap_ids = snap_ids
        self.snaps_per_sim = len(snap_ids)

        self.n_sims = len(sobol_ids)
        self.n_snaps = self.n_sims * self.snaps_per_sim

        self.g4data = read_g4data(sims_dir, self.sobol_ids, self.snap_ids, fn_sobol)

    def __len__(self):
        return self.n_snaps

    def __getitem__(self, idx):
        sobol_id = self.sobol_ids[idx // self.snaps_per_sim]
        snap_id = self.snap_ids[idx % self.snaps_per_sim]

        # TODO generate pmwd parameters and IC here?

        return self.g4data[sobol_id][snap_id]

    def getsnap(self, sidx, snap):
        return self.g4data[sidx][snap]
