"""Multi-process SO training using jax pmap, one process contains one gpu."""
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
n_procs = int(os.getenv('SLURM_NTASKS'))
procid = int(os.getenv('SLURM_PROCID'))
n_tasks_per_node = int(os.getenv('SLURM_NTASKS_PER_NODE'))
slurm_job_id = os.getenv('SLURM_JOB_ID')
os.environ['CUDA_VISIBLE_DEVICES'] = str(procid % n_tasks_per_node)

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'

import jax
# must be called before any jax functions, incl. jax.devices() etc
jax.distributed.initialize(local_device_ids=[0])

import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import pickle

from pmwd.sto.train import train_epoch, loss_epoch
from pmwd.sto.vis import track_figs
from pmwd.sto.data import G4sobolDataset
from pmwd.sto.hypars import (
    n_epochs, sobol_ids_global, snap_ids, shuffle_epoch,
    learning_rate, get_optimizer, so_params)
from pmwd.sto.post import pmwd_fwd
from pmwd.sto.util import pv2ptcl, global_mean


def printinfo(s, procid=procid, flush=False):
    print(f"[{datetime.now().strftime('%H:%M:%S  %m-%d')}] Proc {procid:>2d}: {s}",
          flush=flush)


def jax_device_sync(verbose=False):
    """Nothing but to sync all devices, with dummy global mean."""
    x = global_mean(jnp.array(procid))
    assert round(2 * x + 1) == n_procs, 'something wrong with global mean'
    if verbose:
        printinfo(f'# global devices: {len(jax.devices())}, device sync successful',
                  flush=True)


def checkpoint(epoch, so_params):
    os.makedirs(f'params/{slurm_job_id}', exist_ok=True)
    with open(fn := f'params/{slurm_job_id}/e{epoch:0>3d}.pickle', 'wb') as f:
        dic = {'so_params': so_params}
        pickle.dump(dic, f)
    printinfo(f'epoch {epoch} done, params saved: {fn}', flush=True)


def track(writer, epoch, scalars, check_sobols, check_snaps, so_params, g4data,
           mesh_shape, n_steps):
    if scalars is not None:
        for k, v in scalars.items():
            writer.add_scalar(k, v, epoch)

    # check a few training sobols and snaps
    # TODO memory issue
    # for sidx in check_sobols:
    #     # get g4 snap and sobol
    #     a_snaps, tgts, sobol, snap_ids = g4data.getsnaps(sidx, check_snaps)

    #     # run pmwd
    #     obsvbl = pmwd_fwd(so_params, sidx, sobol, a_snaps, mesh_shape, n_steps,
    #                       so_type, so_nodes)

    #     # compare
    #     for i in range(len(a_snaps)):
    #         ptcl = obsvbl['snapshots'][i]
    #         ptcl_t = pv2ptcl(*tgts[i], ptcl.pmid, ptcl.conf)
    #         figs = track_figs(ptcl, ptcl_t)
    #         for key, fig in figs.items():
    #             writer.add_figure(f'{key}/sobol_{sidx}/snap_{snap_ids[i]}', fig, epoch)
    #             fig.clf()

    # check the test sobols and snaps


# check global devices
jax_device_sync(verbose=True)

# RNGs with fixed seeds
# numpy: pmwd MC sampling;
np_rng = np.random.default_rng(0)
# torch: dataloader shuffle of sobols
tc_rng = torch.Generator().manual_seed(procid)
# jax: dropout layer
jax_key = jax.random.PRNGKey(0)

# the corresponding sobol ids of training data for current proc
# each proc must have the same number of sobol ids
sobol_ids = np.split(sobol_ids_global, n_procs)[procid]

optimizer = get_optimizer(learning_rate)
opt_state = optimizer.init(so_params)
skd_state = None

# load training data to CPU memory
printinfo(f'loading gadget-4 data, {len(sobol_ids)} sobol ids:\n {sobol_ids}', flush=True)
tic = time.perf_counter()
g4data = G4sobolDataset('gs512', sobol_ids, snap_ids)
g4loader = DataLoader(g4data, batch_size=None, shuffle=shuffle_epoch,
                      generator=tc_rng, num_workers=0, collate_fn=lambda x: x)
printinfo(f'loading {len(sobol_ids)} sobols takes {(time.perf_counter() - tic)/60:.1f} mins',
          flush=True)


jax_device_sync()
if procid == 0:
    print('>> devices synced, start training <<')
    print('time, epoch, sidx, mesh_shape, n_steps, loss', flush=True)
    writer = SummaryWriter(log_dir=f'runs/{slurm_job_id}')


for epoch in range(0, n_epochs+1):

    if epoch == 0:  # evaluate the loss before training, with init so_params
        loss_epoch_mean = loss_epoch(
            procid, epoch, g4loader, so_params, jax_key)
    else:
        (loss_epoch_mean, so_params, opt_state, optimizer, learning_rate, skd_state
        ) = train_epoch(
            procid, epoch, g4loader, so_params, opt_state, optimizer,
            learning_rate, skd_state, jax_key)

    # test on test data
    # also distribute to multiple devices, evaluate and collect the loss

    # checkpoint and track
    if procid == 0:
        checkpoint(epoch, so_params)

        scalars = {
            'loss': loss_epoch_mean,
            'learning rate': learning_rate,
            # TODO add the mean test loss, could plot together with training
            # loss using add_scalars
        }
        check_sobols = sobol_ids
        check_snaps = [0, len(snap_ids)//2, -1]
        mesh_shape_track = 1
        n_steps_track = 100
        track(writer, epoch, scalars, check_sobols, check_snaps, so_params, g4data,
              mesh_shape_track, n_steps_track)


if procid == 0:
    writer.close()
