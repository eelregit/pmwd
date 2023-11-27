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
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import pickle
import optax

from pmwd.sto.train import train_epoch, loss_epoch
from pmwd.sto.vis import track_figs
from pmwd.sto.data import read_gsdata
from pmwd.sto.post import pmwd_fwd
from pmwd.sto.util import pv2ptcl, global_mean
from pmwd.sto.hypars import lr_scheduler


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


def checkpoint(epoch, so_params, opt_state, lr, log_id=None, verbose=True):
    dic = {
            'so_params': so_params,
            'opt_state': opt_state,
            'lr': lr,
    }
    dir = f'params/{slurm_job_id}'
    if log_id is not None:
        dir += f'_{log_id}'
    os.makedirs(dir, exist_ok=True)
    with open(fn := f'{dir}/e{epoch:0>3d}.pickle', 'wb') as f:
        pickle.dump(dic, f)
    if verbose:
        printinfo(f'epoch {epoch} done, params saved: {fn}', flush=True)


def track(writer, epoch, scalars, check_sobols, check_snaps,
          so_type, so_nodes, so_params, gsdata, mesh_shape, n_steps):
    if scalars is not None:
        for k, v in scalars.items():
            writer.add_scalar(k, v, epoch)

    # check a few training sobols and snaps
    for sidx in check_sobols:
        # get target snap and sobol
        tgts, a_snaps, sobol, snap_ids = (gsdata[sidx][k] for k in (
                                          'pv', 'a_snaps', 'sobol', 'snap_ids'))
        a_snaps = tuple(a_snaps[i] for i in check_snaps)
        tgts = tuple(t[check_snaps] for t in tgts)
        snap_ids = snap_ids[check_snaps]

        # run pmwd
        obsvbl, cosmo, conf = pmwd_fwd(so_params, sidx, sobol, a_snaps,
                                       mesh_shape, n_steps, so_type, so_nodes)

        # compare
        for i in range(len(a_snaps)):
            ptcl = obsvbl['snaps'][i]
            tgt = tuple(t[i] for t in tgts)
            ptcl_t = pv2ptcl(*tgt, ptcl.pmid, ptcl.conf)
            figs = track_figs(ptcl, ptcl_t, cosmo, conf, a_snaps[i])
            for key, fig in figs.items():
                writer.add_figure(f'{key}/sobol_{sidx}/snap_{snap_ids[i]}', fig, epoch)
                fig.clf()

    # check the test sobols and snaps


def prep_train(sobol_ids_global, snap_ids):
    # check global devices
    jax_device_sync(verbose=True)

    # the corresponding sobol ids of training data for current proc
    # each proc must have the same number of sobol ids
    sobol_ids = np.split(sobol_ids_global, n_procs)[procid]

    # load training data to CPU memory
    printinfo(f'loading gadget-4 data, {len(sobol_ids)} sobol ids: {sobol_ids}', flush=True)
    tic = time.perf_counter()
    gsdata = read_gsdata('gs512', sobol_ids, snap_ids, 'sobol.txt')
    printinfo(f'loading {len(sobol_ids)} sobols takes {(time.perf_counter() - tic)/60:.1f} mins',
              flush=True)

    return sobol_ids, gsdata


def run_train(n_epochs, sobol_ids, gsdata, snap_ids, shuffle_epoch,
              learning_rate, optimizer, opt_state, so_type, so_nodes, so_params,
              ret=False, log_id=None, verbose=True):

    # RNGs with fixed seeds
    # pmwd MC sampling
    np_rng = np.random.default_rng(0)
    # shuffle of data samples across epoch
    np_rng_shuffle = np.random.default_rng(procid)
    # jax: dropout layer
    jax_key = jax.random.PRNGKey(0)

    skd_state = None

    jax_device_sync()
    if procid == 0:
        if verbose:
            print('>> devices synced, start training <<')
            print('time, epoch, sidx, mesh_shape, n_steps, loss', flush=True)
        log_dir = f'runs/{slurm_job_id}'
        if log_id is not None:
            log_dir += f'_{log_id}'
        writer = SummaryWriter(log_dir=log_dir)

    loss_epoch_all = []  # to collect the loss of all epochs
    sobol_ids_epoch = sobol_ids.copy()

    for epoch in range(0, n_epochs+1):

        # shuffle the data samples across epoch
        if shuffle_epoch:
            np_rng_shuffle.shuffle(sobol_ids_epoch)

        # training for one epoch
        if epoch == 0:  # evaluate the loss before training, with init so_params
            loss_epoch_mean = loss_epoch(
                procid, epoch, gsdata, sobol_ids_epoch, so_type, so_nodes, so_params,
                jax_key, verbose)
        else:
            loss_epoch_mean, so_params, opt_state, optimizer = train_epoch(
                procid, epoch, gsdata, sobol_ids_epoch, so_type, so_nodes, so_params,
                opt_state, optimizer, jax_key, verbose)

            # learning rate scheduler
            # learning_rate, skd_state = lr_scheduler(learning_rate, skd_state, loss_epoch_mean)
            # optimizer = get_optimizer(learning_rate)

        # TODO test on test data
        # also distribute to multiple devices, evaluate and collect the loss

        # checkpoint and track
        if procid == 0:
            checkpoint(epoch, so_params, opt_state, learning_rate,
                       log_id=log_id, verbose=verbose)

            scalars = {
                'loss': loss_epoch_mean,
                'learning rate': learning_rate,
                # TODO add the mean test loss, could plot together with training
                # loss using add_scalars
            }
            # the sobols and snaps to track
            check_sobols = sobol_ids[:3]
            check_snaps = [0, len(snap_ids)//2, -1]
            mesh_shape_track = 1
            n_steps_track = 100
            track(writer, epoch, scalars, check_sobols, check_snaps,
                  so_type, so_nodes, so_params, gsdata, mesh_shape_track, n_steps_track)
            # printinfo('logged for tensorboard')

        loss_epoch_all.append(loss_epoch_mean)

    if procid == 0:
        writer.close()

    if ret:
        return loss_epoch_all


if __name__ == "__main__":

    from pmwd.sto.hypars import (
        n_epochs, sobol_ids_global, snap_ids, shuffle_epoch,
        learning_rate, optimizer, opt_state, so_type, so_nodes, so_params)

    sobol_ids, gsdata = prep_train(sobol_ids_global, snap_ids)

    run_train(n_epochs, sobol_ids, gsdata, snap_ids, shuffle_epoch,
              learning_rate, optimizer, opt_state, so_type, so_nodes, so_params)
