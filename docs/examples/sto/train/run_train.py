"""Multi-process SO training using jax pmap, one process contains one gpu.
"""
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
n_procs = int(os.getenv('SLURM_NTASKS'))
procid = int(os.getenv('SLURM_PROCID'))
n_tasks_per_node = int(os.getenv('SLURM_NTASKS_PER_NODE'))
os.environ['CUDA_VISIBLE_DEVICES'] = str(procid % n_tasks_per_node)

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import pickle

from pmwd.sto.train import train_step
from pmwd.sto.vis import vis_inspect
from pmwd.sto.data import G4snapDataset
from pmwd.sto.hypars import (
    n_epochs, sobol_ids_global, snap_ids, shuffle_snaps,
    optimizer, so_type, so_nodes, so_params)


def _printinfo(s, flush=False):
    print(f"[{datetime.now().strftime('%H:%M:%S  %m-%d')}] Proc {procid}: {s}",
          flush=flush)


def _checkpoint(epoch, so_params):
    jobid = os.getenv('SLURM_JOB_ID')
    with open(fn := f'params/j{jobid}_e{epoch:0>3d}.pickle', 'wb') as f:
        dic = {'so_params': so_params}
        pickle.dump(dic, f)
    _printinfo(f'epoch {epoch} done, params saved: {fn}', flush=True)


if __name__ == "__main__":

    # must be called before any jax functions, incl. jax.devices() etc
    jax.distributed.initialize(local_device_ids=[0])

    # RNGs with fixed seeds, for same randomness across processes
    np_rng = np.random.default_rng(0)  # for pmwd MC sampling
    tc_rng = torch.Generator().manual_seed(0)  # for dataloader shuffle

    # the corresponding sobol ids of training data for current proc
    sobol_ids = np.array_split(sobol_ids_global, n_procs)[procid]
    _printinfo(f'local sobol ids: {sobol_ids}')

    # keep a copy of the initial params
    so_params_init = so_params

    opt_state = optimizer.init(so_params)

    # load training data
    _printinfo('preparing the data loader')
    g4data = G4snapDataset('g4sims', sobol_ids, snap_ids)
    g4loader = DataLoader(g4data, batch_size=None, shuffle=shuffle_snaps,
                          generator=tc_rng, num_workers=0, collate_fn=lambda x: x)

    _printinfo('training started ...', flush=True)
    if procid == 0:
        print('time, epoch, step, mesh_shape, n_steps, snap_id, loss')
        writer = SummaryWriter()

    tic = time.perf_counter()
    for epoch in range(n_epochs):
        loss_epoch = 0.  # the sum of loss of the whole epoch

        for step, g4snap in enumerate(g4loader):
            pos, vel, a, sidx, sobol, snap_id = g4snap

            # mesh shape, 128 * [1, 2, 3, 4]
            # mesh_shape = np_rng.integers(1, 5) * 128
            # number of time steps, [10, 1000], log-uniform
            # n_steps = np.rint(10**np_rng.uniform(1, 3)).astype(int)
            mesh_shape = 128
            n_steps = 100

            tgt = (pos, vel)
            pmwd_params = (a, sidx, sobol, mesh_shape, n_steps, so_type, so_nodes)
            opt_params = (optimizer, opt_state)
            so_params, loss, opt_state = train_step(tgt, so_params, pmwd_params,
                                                    opt_params)

            loss = float(loss)
            loss_epoch += loss

            # runtime print information
            if procid == 0:
                tt = time.perf_counter() - tic
                tic = time.perf_counter()
                print((f'{tt:.0f} s, {epoch}, {step:>3d}, {mesh_shape:>3d}, ' +
                       f'{n_steps:>4d}, {snap_id:>4d}, {loss:12.3e}'), flush=True)

            # tensorboard log
            if procid == 0:
                global_step = epoch * len(g4loader) + step + 1
                # writer.add_scalar('loss/train/step', loss, global_step)

                # epoch: check a few snapshots
                check_snaps = (snap_ids[i] for i in [0, len(snap_ids)//2, -1])
                if snap_id in check_snaps:
                    # check the status before training
                    if epoch == 0:
                        figs = vis_inspect(tgt, so_params_init, pmwd_params)
                        for key, fig in figs.items():
                            writer.add_figure(f'{key}/epoch/snap_{snap_id}', fig, 0)
                            fig.clf()

                    writer.add_scalar(f'loss/train/epoch/snap_{snap_id}', loss_epoch, epoch+1)
                    figs = vis_inspect(tgt, so_params, pmwd_params)
                    for key, fig in figs.items():
                        writer.add_figure(f'{key}/epoch/snap_{snap_id}', fig, epoch+1)
                        fig.clf()

        if procid == 0:
            writer.add_scalar('loss/train/epoch/mean', loss_epoch/len(g4loader), epoch+1)

            # checkpoint SO params every epoch
            _checkpoint(epoch, so_params)

    if procid == 0:
        writer.close()
