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

from pmwd.sto.so import soft_len
from pmwd.sto.mlp import init_mlp_params
from pmwd.sto.train import train_step
from pmwd.sto.vis import vis_inspect
from pmwd.sto.data import G4snapDataset


def printinfo(s, flush=False):
    print(f"[{datetime.now().strftime('%H:%M:%S  %m-%d')}] Proc {procid}: {s}",
          flush=flush)


if __name__ == "__main__":

    # must be called before any jax functions, incl. jax.devices() etc
    jax.distributed.initialize(local_device_ids=[0])

    # hyper parameters of training
    n_epochs = 150
    learning_rate = 1e-3
    sobol_ids = [0]
    snap_ids = np.arange(0, 121, 3)

    # RNGs with fixed seeds, for same randomness across processes
    np_rng = np.random.default_rng(16807)  # for pmwd MC sampling
    tc_rng = torch.Generator().manual_seed(16807)  # for dataloader shuffle

    # the corresponding sobol ids of training data for current proc
    sobol_ids = np.array_split(sobol_ids, n_procs)[procid]
    printinfo(f'sobol ids: {sobol_ids}')

    # load training data
    printinfo('preparing the data loader')
    g4data = G4snapDataset('g4sims', sobol_ids, snap_ids)
    g4loader = DataLoader(g4data, batch_size=None, shuffle=True, generator=tc_rng,
                          num_workers=0, collate_fn=lambda x: x)

    # structure of the so neural nets
    printinfo('initializing SO parameters & optimizer')

    # SO scheme: h(k_i) * g(k) * [f(k_1) * f(k_2) * f(k_3)]
    # so_type = 3
    # n_input = [soft_len()] * 3  # three nets
    # so_nodes = [[n * 2 // 3, n // 3, 1] for n in n_input]
    # so_params = init_mlp_params(n_input, so_nodes, scheme='last_ws_b1')

    # SO scheme: f(k_i) * g(k_1, k_2, k_3)
    so_type = 2
    n_input = [soft_len(l_fac=3), soft_len()]
    so_nodes = [[n * 2 // 3, n // 3, 1] for n in n_input]
    so_params = init_mlp_params(n_input, so_nodes, scheme='last_ws_b1')

    # keep a copy of the initial params
    so_params_init = so_params

    # mannually turn off nets by setting the corresponding so_nodes to None
    # for i in [0, 2]: so_nodes[i] = None

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(so_params)

    printinfo('training started ...', flush=True)
    if procid == 0:
        print('time, epoch, step, mesh_shape, n_steps, snap_id, loss')
        writer = SummaryWriter()

    tic = time.perf_counter()
    for epoch in range(n_epochs):
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

            # track
            if procid == 0:
                # step
                tt = time.perf_counter() - tic
                tic = time.perf_counter()
                print((f'{tt:.0f} s, {epoch}, {step:>3d}, {mesh_shape:>3d}, ' +
                       f'{n_steps:>4d}, {snap_id:>4d}, {loss:12.3e}'), flush=True)
                global_step = epoch * len(g4loader) + step + 1
                writer.add_scalar('loss/train/step', float(loss), global_step)

                # epoch: last snapshot
                if snap_id == snap_ids[-1]:
                    # check the status before training
                    if epoch == 0:
                        figs = vis_inspect(tgt, so_params_init, pmwd_params)
                        for key, fig in figs.items():
                            writer.add_figure(f'fig/epoch/{key}', fig, 0)
                            fig.clf()

                    writer.add_scalar('loss/train/epoch', float(loss), epoch+1)
                    figs = vis_inspect(tgt, so_params, pmwd_params)
                    for key, fig in figs.items():
                        writer.add_figure(f'fig/epoch/{key}', fig, epoch+1)
                        fig.clf()

        if procid == 0:
            # checkpoint SO params
            jobid = os.getenv('SLURM_JOB_ID')
            with open(fn := f'params/j{jobid}_e{epoch:0>3d}.pickle', 'wb') as f:
                dic = {'so_params': so_params}
                pickle.dump(dic, f)
            printinfo(f'epoch {epoch} done, params saved: {fn}', flush=True)

    if procid == 0:
        writer.close()
