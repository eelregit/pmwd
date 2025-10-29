"""Multi-process training using jax pmap, one process contains one gpu."""
import os

# process and job information
n_procs = int(os.getenv('SLURM_NTASKS'))
procid = int(os.getenv('SLURM_PROCID'))
n_tasks_per_node = int(os.getenv('SLURM_NTASKS_PER_NODE'))
slurm_job_id = os.getenv('SLURM_JOB_ID')

# setup the CUDA device binded to the current proc
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(procid % n_tasks_per_node)

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'

import jax
# must be called before any jax functions, incl. jax.devices() etc
# explicitly set local device, the only visible one
jax.distributed.initialize(local_device_ids=[0])

import jax.numpy as jnp
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import pickle

from pmwd.sto.data.g4data import read_gsdata
from pmwd.sto.train.train import train_epoch, loss_epoch
from pmwd.sto.train.utils import procinfo, device_sync


def checkpoint(epoch, so_params, opt_state, lr, log_id=None, verbose=True):
    """Checkpoint the model parameters and optimizer state."""
    dic = {'so_params': so_params,
           'opt_state': opt_state,
           'lr': lr,}
    dir = f'params/{slurm_job_id}'
    if log_id is not None:
        dir += f'_{log_id}'
    os.makedirs(dir, exist_ok=True)
    with open(fn := f'{dir}/e{epoch:0>3d}.pickle', 'wb') as f:
        pickle.dump(dic, f)
    if verbose:
        procinfo(f'epoch {epoch} done, params saved: {fn}', procid, flush=True)


def track(writer, epoch, scalars):
    """Track the training with tensorboard."""
    if scalars is not None:
        for k, v in scalars.items():
            writer.add_scalar(k, v, epoch)


def setup_train(data_conf):
    """Prepare for training, incl. data loading on host etc."""
    # check global devices
    device_sync(procid, n_procs, verbose=True)

    # the corresponding sobol ids of training data for current proc
    # each proc must have the same number of sobol ids
    sobol_ids = np.split(data_conf['sobol_ids_global'], n_procs)[procid]
    data_conf['sobol_ids'] = sobol_ids

    # load training data to host memory
    procinfo(f'loading gadget-4 data, {len(sobol_ids)} sobol ids: {sobol_ids}',
             procid, flush=True)
    tic = time.perf_counter()
    gsdata = read_gsdata(data_conf['data_dir'], sobol_ids, data_conf['snap_ids'],
                         data_conf['sobol_file'])
    toc = time.perf_counter()
    procinfo(f'loading {len(sobol_ids)} sobols' +
             f' each with {len(data_conf['snap_ids'])} snapshots' +
             f' takes {(toc - tic)/60:.1f} mins', procid, flush=True)

    return gsdata, data_conf


def run_train(n_epochs, gsdata, data_conf, loss_conf, opt_conf, model_conf,
              so_params, opt_state, log_id=None, verbose=True):

    # RNGs with fixed seeds
    # rng for pmwd MC sampling
    np_rng = np.random.default_rng(42)
    # rng for shuffling data samples across epoch
    np_rng_shuffle = np.random.default_rng(42+procid)

    # sync and setup log file directory
    device_sync(procid, n_procs)
    if procid == 0:
        if verbose:
            print('>>> devices synced, start training <<<')
            print('time, epoch, sidx, mesh_shape, n_steps, loss', flush=True)
        log_dir = f'runs/{slurm_job_id}'
        if log_id is not None:
            log_dir += f'_{log_id}'
        writer = SummaryWriter(log_dir=log_dir)

    sobol_ids_epoch = data_conf['sobol_ids'].copy()

    # training loop over epochs
    for epoch in range(0, n_epochs+1):

        # shuffle the data samples across epoch
        if data_conf['shuffle_epoch']:
            np_rng_shuffle.shuffle(sobol_ids_epoch)

        # evaluate the loss before training, with init so_params
        if epoch == 0:
            loss_epoch_mean = loss_epoch(
                procid, epoch, gsdata, sobol_ids_epoch, model_conf,
                so_params, loss_conf, verbose)
        # training for one epoch
        else:
            loss_epoch_mean, so_params, opt_state = train_epoch(
                procid, epoch, gsdata, sobol_ids_epoch, model_conf,
                so_params, opt_conf, opt_state, loss_conf, verbose)

        # checkpoint and track
        if procid == 0:
            checkpoint(epoch, so_params, opt_state, opt_conf['learning_rate'],
                       log_id=log_id, verbose=verbose)
            scalars = {
                'loss': loss_epoch_mean,
                'learning rate': opt_conf['learning_rate'],
            }
            track(writer, epoch, scalars)

    if procid == 0:
        writer.close()


if __name__ == "__main__":

    from pmwd.sto.train.hypars import (
        n_epochs, data_conf, loss_conf, opt_conf, model_conf,
        so_params, opt_state)

    gsdata, data_conf = setup_train(data_conf)

    run_train(n_epochs, gsdata, data_conf, loss_conf, opt_conf, model_conf,
              so_params, opt_state)
