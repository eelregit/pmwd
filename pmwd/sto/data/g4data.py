import os
import numpy as np
from torch.utils.data import Dataset
import h5py

from pmwd.sto.data.sample import scale_Sobol


def read_g4sim(sims_dir, sidx, snap_ids, fn_sobol):
    """Load snapshots in a Gadget4 simulation from GS512 dataset."""

    sobol = scale_Sobol(fn=fn_sobol, ind=sidx)
    data = {
        'sidx': sidx,
        'sobol': sobol,
        'snap_ids': snap_ids,
    }
    with h5py.File(os.path.join(sims_dir, f'{sidx:03}.hdf5'), 'r') as f:
        data['a_ic'] = f['a_ic'][()]
        pos_ic = f['pos_ic'][:]
        vel_ic = f['vel_ic'][:]
        data['a_snaps'] = tuple(f['a'][snap_ids])
        pos = f['pos'][snap_ids]
        vel = np.full(len(snap_ids), 0.)  # not using vel in loss now, saving mem
    data['ic'] = (pos_ic, vel_ic)
    data['pv'] = (pos, vel)

    return data


class G4Dataset(Dataset):

    def __init__(self, sims_dir, sobol_ids, snap_ids, fn_sobol):
        self.sims_dir = sims_dir
        self.sobol_ids = sobol_ids
        self.snap_ids = snap_ids
        self.fn_sobol = fn_sobol

    def __len__(self):
        return len(self.sobol_ids)

    def __getitem__(self, idx):
        sidx = self.sobol_ids[idx]
        return read_g4sim(self.sims_dir, sidx, self.snap_ids, self.fn_sobol)


def read_gsdata(sims_dir, sobol_ids, snap_ids, fn_sobol):
    """Load training data from GS512 dataset."""
    data = {}

    for sidx in sobol_ids:
        data[sidx] = read_g4sim(sims_dir, sidx, snap_ids, fn_sobol)

    return data
