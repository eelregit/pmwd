"""Multi-process training using jax pmap, one process contains one gpu."""
import os

# process and job information
n_procs = int(os.getenv('SLURM_NTASKS'))
procid = int(os.getenv('SLURM_PROCID'))
n_procs_per_node = int(os.getenv('SLURM_NTASKS_PER_NODE'))
slurm_job_id = os.getenv('SLURM_JOB_ID')

# setup the CUDA device binded to the current proc
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(procid % n_procs_per_node)

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'

import jax
# must be called before any jax functions, incl. jax.devices() etc
# explicitly set local device, the only visible one
jax.distributed.initialize(local_device_ids=[0])

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import pickle

from pmwd.sto.data.g4data import create_g4data_loader, PrefetchToDevice
from pmwd.sto.train.train import train_epochs, evaluate_loss_epoch
from pmwd.sto.train.utils import procinfo, device_sync
from pmwd.sto.so.mlp import init_mlp_params


def setup_data(data_conf):
    """Prepare data for training, incl. data loading on host etc."""
    # check global devices
    device_sync(procid, n_procs, verbose=True)

    # the corresponding sobol ids of training data for current proc
    # each proc must have the same number of sobol ids
    sobol_ids = np.split(data_conf['sobol_ids_global'], n_procs)[procid]
    data_conf['sobol_ids'] = sobol_ids

    # initialize data loader
    procinfo(f'loading gadget-4 data, {len(sobol_ids)} sobol ids: {sobol_ids}',
             procid, flush=True)
    tic = time.perf_counter()
    data_loader = create_g4data_loader(data_conf['data_dir'],
                                       sobol_ids,
                                       data_conf['snap_ids'],
                                       data_conf['sobol_file'],
                                       seed=42+procid)
    toc = time.perf_counter()
    procinfo(f'loading {len(sobol_ids)} sobols' +
             f' each with {len(data_conf['snap_ids'])} snapshots' +
             f' takes {(toc - tic)/60:.1f} mins', procid, flush=True)

    # wrap with GPU prefetcher
    data_loader = PrefetchToDevice(data_loader, size=2, length=len(sobol_ids))

    return data_loader, data_conf


def setup_state(job_id, epoch_init, n_input, opt_reset=False):
    """Prepare model and opt state for training, init or continue."""
    if job_id is None:  # training from scratch
        so_params = init_mlp_params(n_input, model_conf.so_nodes, scheme='last_w0', last_b=1.)
        opt_state = opt_conf['optimizer'].init(so_params)
    else:  # continue training
        param_fn = f'params/{job_id}/e{epoch_init}.pickle'
        with open(param_fn, 'rb') as f:
            dic = pickle.load(f)
            so_params = dic['so_params']
            if opt_reset:
                opt_state = opt_conf['optimizer'].init(so_params)
            else:
                opt_state = dic['opt_state']

    return so_params, opt_state


def run_train(n_epochs, data_loader, loss_conf, opt_conf, model_conf,
              so_params, opt_state, epoch_init, rng_seed=42, verbose=True):

    # sync and setup log file directory
    device_sync(procid, n_procs)
    writer = None
    if procid == 0:
        if verbose:
            print('>>> devices synced, start training <<<')
            print('time, step, sobol, loss', flush=True)
        log_dir = f'runs/{slurm_job_id}'
        writer = SummaryWriter(log_dir=log_dir)

    rng_key = jax.random.key(rng_seed)

    if epoch_init == 0:
        # evaluate the loss before training, with init so_params
        evaluate_loss_epoch(procid, data_loader, model_conf,
                            so_params, loss_conf, verbose, writer, rng_key)

    train_epochs(procid, n_epochs, data_loader, model_conf,
                 so_params, opt_conf, opt_state, loss_conf,
                 verbose, writer, epoch_init, rng_key)

    if procid == 0:
        writer.close()


if __name__ == "__main__":

    from pmwd.sto.train.hypars import data_conf, loss_conf, opt_conf, model_conf, n_input

    data_loader, data_conf = setup_data(data_conf)

    # setup model param and opt state
    job_id, epoch_init = None, 0
    so_params, opt_state = setup_state(job_id, epoch_init, n_input)

    n_epochs = 300
    run_train(n_epochs, data_loader, loss_conf, opt_conf, model_conf,
              so_params, opt_state, epoch_init)

    print('\n>>> run_train finished <<<\n')
