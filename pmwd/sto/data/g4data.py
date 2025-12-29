import os
import h5py
import numpy as np
import jax
import jax.numpy as jnp
import grain.python as grain
import collections

from pmwd.sto.data.sample import scale_Sobol


def read_g4sim(sims_dir, sidx, snap_ids, fn_sobol, float_dtype=np.float32):
    """Load snapshots in a given Gadget4 simulation."""
    sobol = scale_Sobol(fn=fn_sobol, ind=sidx)

    with h5py.File(os.path.join(sims_dir, f'{sidx:03}.hdf5'), 'r') as f:
        a_ic = f['a_ic'][()]
        pos_ic = f['pos_ic'][:]
        vel_ic = f['vel_ic'][:].astype(float_dtype)
        # notice that the final output times of Gadget4 are not exactly the same
        # as the desired times as specified in the input outtimes.txt file,
        # therefore here we read the true output times from the output snapshots
        a_snaps = f['a'][snap_ids]
        pos = f['pos'][snap_ids]
        # not using vel in loss now, scalar placeholder for saving mem
        vel = np.full(len(snap_ids), 0., dtype=float_dtype)

    # get disp from pos, using mesh shape = 1, same for all integer meshes
    box_size = jnp.array(sobol[0])
    cell_size = box_size / 128
    def get_disp(pos):
        # get pmid
        pmid_1d = jnp.linspace(0, 128, num=128, endpoint=False)
        pmid_1d = jnp.rint(pmid_1d)
        pmid_1d = pmid_1d.astype(jnp.int16)
        pmid = [pmid_1d] * 3
        pmid = jnp.meshgrid(*pmid, indexing='ij')
        pmid = jnp.stack(pmid, axis=-1).reshape(-1, 3)

        disp = pos - pmid * cell_size
        # wrap around the periodic boundaries, disp: [-L/2, L/2]
        disp -= jnp.rint(disp / box_size) * box_size

        return np.array(disp, dtype=float_dtype)  # back to host

    disp_ic = get_disp(pos_ic)
    disp = get_disp(pos)

    data = {
        'sidx': sidx,
        'sobol': sobol,
        'snap_ids': snap_ids,
        'a_snaps': a_snaps,
        'a_ic': a_ic,
        'ic': (disp_ic, vel_ic),
        'tgts': (disp, vel),
    }

    return data


def read_gsdata(sims_dir, sobol_ids, snap_ids, fn_sobol, float_dtype=np.float32):
    """Load snapshots from multiple Gadget4 simulations."""
    gsdata = {}

    for sidx in sobol_ids:
        gsdata[sidx] = read_g4sim(sims_dir, sidx, snap_ids, fn_sobol, float_dtype)

    return gsdata


class G4DataSource(grain.RandomAccessDataSource):
    def __init__(self, sims_dir, sobol_ids, snap_ids, fn_sobol):
        self.sims_dir = sims_dir
        self.sobol_ids = sobol_ids
        self.snap_ids = snap_ids
        self.fn_sobol = fn_sobol

        # load all data to cpu mem
        self.gsdata = read_gsdata(sims_dir, sobol_ids, snap_ids, fn_sobol)

    def __len__(self):
        return len(self.sobol_ids)

    def __getitem__(self, idx):
        sidx = self.sobol_ids[idx]  # get sobol index

        # fetch a sobol data sample
        # NOTE: We do NOT call jax.device_put here.
        # Grain workers run in separate processes (multiprocessing).
        # Sending GPU arrays across processes is inefficient/error-prone.
        # We return CPU numpy arrays.
        data = self.gsdata[sidx]

        return data


def create_g4data_loader(sims_dir, sobol_ids, snap_ids, fn_sobol, seed=42):
    source = G4DataSource(sims_dir, sobol_ids, snap_ids, fn_sobol)

    sampler = grain.IndexSampler(
        num_records=len(source),
        shuffle=True,
        seed=seed,
    )

    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
    )

    return loader


class PrefetchToDevice:
    """
    Iterator that prefetches data from a Grain loader and pushes it to the GPU.
    """
    def __init__(self, iterator, size=2):
        self.iterator = iterator
        self.size = size

    def __len__(self):
        return len(self.iterator)

    def __iter__(self):
        iterator = iter(self.iterator)
        queue = collections.deque()

        def _push(item):
            item = item.copy()  # shallow copy of dictionary
            # Async transfer to GPU
            item['ic'] = jax.device_put(item['ic'])
            item['tgts'] = jax.device_put(item['tgts'])
            return item

        # Prefill the queue
        for _ in range(self.size):
            try:
                queue.append(_push(next(iterator)))
            except StopIteration:
                break

        # Yield and replenish
        while queue:
            yield queue.popleft()
            try:
                queue.append(_push(next(iterator)))
            except StopIteration:
                pass
