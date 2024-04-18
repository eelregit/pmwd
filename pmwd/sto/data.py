import os
import numpy as np
import jax.numpy as jnp
from torch.utils.data import Dataset
from joblib import Parallel, delayed
import h5py

from pmwd.io_util import read_gadget_hdf5
from pmwd.sto.sample import scale_Sobol


def read_g4snap(sims_dir, sobol_ids, snap_ids, fn_sobol):
    data = {}
    def load_sobol(sidx):
        data[sidx] = {}
        sobol = scale_Sobol(fn=fn_sobol, ind=sidx)
        for snap in snap_ids:
            snap_file = os.path.join(sims_dir, f'{sidx:03}',
                                     'output', f'snapshot_{snap:03}')
            pos, vel, a = read_gadget_hdf5(snap_file)
            data[sidx][snap] = (pos, vel, a, sidx, sobol, snap)
    Parallel(n_jobs=min(8, len(sobol_ids)), prefer='threads', require='sharedmem')(
        delayed(load_sobol)(sidx) for sidx in sobol_ids)
    return data


class G4snapDataset(Dataset):
    """Gadget4 dataset with each data sample being a single snapshot."""

    def __init__(self, sims_dir, sobol_ids, snap_ids, fn_sobol='sobol.txt'):

        self.g4data = read_g4snap(sims_dir, sobol_ids, snap_ids, fn_sobol)

        self.sobol_ids = sobol_ids
        self.snap_ids = snap_ids
        self.snaps_per_sim = len(snap_ids)

        self.n_sims = len(sobol_ids)
        self.n_snaps = self.n_sims * self.snaps_per_sim


    def __len__(self):
        return self.n_snaps

    def __getitem__(self, idx):
        sobol_id = self.sobol_ids[idx // self.snaps_per_sim]
        snap_id = self.snap_ids[idx % self.snaps_per_sim]

        return self.g4data[sobol_id][snap_id]

    def getsnap(self, sidx, snap):
        return self.g4data[sidx][snap]


def read_g4sobol(sims_dir, sobol_ids, snap_ids, fn_sobol):
    data = {}
    def load_sobol(sidx):
        sobol = scale_Sobol(fn=fn_sobol, ind=sidx)
        data[sidx] = {
            'sidx': sidx,
            'sobol': sobol,
            'snap_ids': snap_ids,
            'a_snaps': (),
            'snapshots': [],
        }
        for snap in snap_ids:
            snap_file = os.path.join(sims_dir, f'{sidx:03}',
                                     'output', f'snapshot_{snap:03}')
            pos, vel, a = read_gadget_hdf5(snap_file)
            data[sidx]['a_snaps'] += (a,)
            data[sidx]['snapshots'].append((pos, vel))
    Parallel(n_jobs=min(8, len(sobol_ids)), prefer='threads', require='sharedmem')(
        delayed(load_sobol)(sidx) for sidx in sobol_ids)
    return data


def read_gsdata(sims_dir, sobol_ids, snap_ids, fn_sobol):
    data = {}
    def load_sobol(sidx):
        sobol = scale_Sobol(fn=fn_sobol, ind=sidx)
        data[sidx] = {
            'sidx': sidx,
            'sobol': sobol,
            'snap_ids': snap_ids,
        }
        with h5py.File(os.path.join(sims_dir, f'{sidx:03}.hdf5'), 'r') as f:
            data[sidx]['a_snaps'] = tuple(f['a'][snap_ids])
            pos = f['pos'][snap_ids]
            # vel = f['vel'][snap_ids]
            vel = np.full(len(snap_ids), 0.)  # not using vel in loss now, saving mem
        data[sidx]['pv'] = (pos, vel)
    for sidx in sobol_ids:
        load_sobol(sidx)
    return data


class G4sobolDataset(Dataset):
    """Gadget4 dataset with each data sample including all snapshots in a sobol."""

    def __init__(self, sims_dir, sobol_ids, snap_ids, fn_sobol='sobol.txt'):

        # self.g4data = read_g4sobol(sims_dir, sobol_ids, snap_ids, fn_sobol)
        self.g4data = read_gsdata(sims_dir, sobol_ids, snap_ids, fn_sobol)
        self.sobol_ids = sobol_ids

    def __len__(self):
        return len(self.sobol_ids)

    def __getitem__(self, idx):
        return self.g4data[self.sobol_ids[idx]]

    def getsnaps(self, sidx, snaps_idx):
        return (tuple(self.g4data[sidx]['a_snaps'][i] for i in snaps_idx),
                [self.g4data[sidx]['snapshots'][i] for i in snaps_idx],
                self.g4data[sidx]['sobol'],
                self.g4data[sidx]['snap_ids'][snaps_idx])
