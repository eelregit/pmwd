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

from pmwd.so_util import soft_len, init_mlp_params
from pmwd.train_util import G4snapDataset, train_step, vis_inspect


def printinfo(s, flush=False):
    print(f"[{datetime.now().strftime('%H:%M:%S  %m-%d')}] Proc {procid}: {s}",
          flush=flush)


if __name__ == "__main__":

    # must be called before any jax functions, incl. jax.devices() etc
    jax.distributed.initialize(local_device_ids=[0])

    # hyper parameters of training
    n_epochs = 30
    learning_rate = 1e-3
    # sobol_ids = np.arange(0, 1)
    sobol_ids = [1]

    # RNGs with fixed seeds, for same randomness across processes
    np_rng = np.random.default_rng(16807)  # for pmwd MC sampling
    tc_rng = torch.Generator().manual_seed(16807)  # for dataloader shuffle

    # the corresponding sobol ids of training data for current proc
    sobol_ids = np.array_split(sobol_ids, n_procs)[procid]
    printinfo(f'sobol ids: {sobol_ids}')

    # load training data
    printinfo('preparing the data loader')
    g4data = G4snapDataset('g4sims', sobol_ids=sobol_ids)
    g4loader = DataLoader(g4data, batch_size=None, shuffle=False, generator=tc_rng,
                          num_workers=0, collate_fn=lambda x: x)

    # structure of the so neural nets
    printinfo('initializing SO parameters & optimizer')
    n_input = [soft_len()] * 3  # three nets
    so_nodes = [[n * 2 // 3, n // 3, 1] for n in n_input]
    so_params = init_mlp_params(n_input, so_nodes, scheme='last_w0_b1')
    # keep a copy of the initial params
    so_params_init = so_params

    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(so_params)

    printinfo('training started ...', flush=True)
    if procid == 0:
        print('time, epoch, step, mesh_shape, steps, loss')
        writer = SummaryWriter()

    tic = time.perf_counter()
    for epoch in range(n_epochs):
        for step, g4snap in enumerate(g4loader):
            pos, vel, a, sidx, sobol = g4snap

            # mesh shape, 128 * [1, 2, 3, 4]
            # mesh_shape = np_rng.integers(1, 5) * 128
            # number of time steps, [10, 1000], log-uniform
            # n_steps = np.rint(10**np_rng.uniform(1, 3)).astype(int)
            mesh_shape = 256
            n_steps = 100

            tgt = (pos, vel)
            pmwd_params = (a, sidx, sobol, mesh_shape, n_steps, so_nodes)
            so_params, loss, opt_state = train_step(tgt, so_params, pmwd_params,
                                                    learning_rate, opt_state)

            # step track
            if procid == 0:
                tt = time.perf_counter() - tic
                tic = time.perf_counter()
                print((f'{tt:.0f} s, {epoch}, {step:>3d}, {mesh_shape:>3d}, ' +
                       f'{n_steps:>4d}, {loss:12.3e}'), flush=True)
                global_step = epoch * len(g4loader) + step + 1
                writer.add_scalar('loss/train', float(loss), global_step)

        if procid == 0:
            # check the status before training
            if epoch == 0:
                figs = vis_inspect(tgt, so_params_init, pmwd_params)
                for key, fig in figs.items():
                    writer.add_figure(f'fig/epoch/{key}', fig, 0)
                    fig.clf()

            # epoch track
            figs = vis_inspect(tgt, so_params, pmwd_params)
            for key, fig in figs.items():
                writer.add_figure(f'fig/epoch/{key}', fig, epoch+1)
                fig.clf()

            # checkpoint SO params
            jobid = os.getenv('SLURM_JOB_ID')
            with open(fn := f'params/j{jobid}_e{epoch}.pickle', 'wb') as f:
                dic = {'n_input': n_input,
                       'so_nodes': so_nodes,
                       'so_params': so_params}
                pickle.dump(dic, f)
            printinfo(f'epoch {epoch} done, params saved: {fn}', flush=True)

    if procid == 0:
        writer.close()
