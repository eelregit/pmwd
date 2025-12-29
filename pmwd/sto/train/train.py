import os
procid = int(os.getenv('SLURM_PROCID'))
slurm_job_id = os.getenv('SLURM_JOB_ID')

import jax
import numpy as np
import optax
import time
import pickle

from pmwd.nbody import nbody
from pmwd.particles import Particles

from pmwd.sto.data.initial import gen_cc
from pmwd.sto.train.loss import loss_func
from pmwd.sto.train.utils import tree_global_mean, procinfo


def obj(tgts, ptcl, so_params, cosmo, conf, loss_conf):
    cosmo = cosmo.replace(so_params=so_params)
    _, obsvbl = nbody(ptcl, None, cosmo, conf)
    loss = loss_func(obsvbl, tgts, conf, loss_conf)
    return loss


def setup_model(ic, model_conf):
    conf, cosmo = gen_cc(model_conf['sobol'],
                         mesh_shape=model_conf['mesh_shape'],
                         a_snapshots=model_conf['a_snaps'],
                         a_nbody_num=model_conf['n_steps'],
                         so_type=model_conf['so_type'],
                         so_nodes=model_conf['so_nodes'],
                         a_start=model_conf['a_ic'],
                         a_stop=model_conf['a_stop'])
    ptcl = Particles.gen_grid(conf)
    ptcl = ptcl.replace(disp=ic[0].astype(conf.float_dtype),
                        vel=ic[1].astype(conf.float_dtype))

    return ptcl, cosmo, conf


def train_step(tgts, ptcl, cosmo, conf, so_params, opt_state, opt_conf, loss_conf):

    # get loss and grad
    obj_valgrad = jax.value_and_grad(obj, argnums=2)
    loss, grad = obj_valgrad(tgts, ptcl, so_params, cosmo, conf, loss_conf)

    # average over global devices
    loss, grad = tree_global_mean((loss, grad))

    # optimize
    updates, opt_state = opt_conf['optimizer'].update(grad, opt_state, so_params)
    so_params = optax.apply_updates(so_params, updates)

    return so_params, opt_state, loss


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


def train_epochs(procid, n_epochs, data_loader, model_conf,
                 so_params, opt_conf, opt_state, loss_conf,
                 verbose, writer, epoch_start=1):
    epoch_size = len(data_loader)
    step_start = epoch_start * epoch_size
    total_steps = n_epochs * epoch_size

    # loop for n_epochs
    loss_epoch = 0.
    epoch = epoch_start
    for step, data in zip(range(step_start, step_start + total_steps), data_loader):
        if procid == 0 and verbose:
            tic = time.perf_counter()

        # data for this step
        sidx, a_ic, a_snaps, sobol = (data[k] for k in
                                        ('sidx', 'a_ic', 'a_snaps', 'sobol'))
        ic, tgts = data['ic'], data['tgts']

        # setup model for this step
        model_conf.update({'a_snaps': a_snaps, 'a_ic': a_ic, 'sobol': sobol})
        ptcl, cosmo, conf = setup_model(ic, model_conf)

        # train for this step
        so_params, opt_state, loss = train_step(
            tgts, ptcl, cosmo, conf, so_params, opt_state, opt_conf, loss_conf)

        loss = np.array(loss)  # move loss back to CPU memory
        loss_epoch += loss

        # step output
        if procid == 0:
            writer.add_scalar('loss', loss, step)
            if verbose:
                toc = time.perf_counter()
                print((f'{toc - tic:.0f} s, {step:>2d}, {sidx:>3d}, ' +
                       f'{loss:12.3e}'), flush=True)

        # epoch output
        if (step + 1) % epoch_size == 0:
            loss_epoch = loss_epoch / epoch_size  # mean loss per step of epoch
            if procid == 0:
                print(f'epoch mean loss: {loss_epoch:12.3e}', flush=True)
                checkpoint(epoch, so_params, opt_state, opt_conf['learning_rate'],
                           verbose=verbose)
            loss_epoch = 0.
            epoch += 1


def evaluate_loss_epoch(procid, data_loader, model_conf,
                        so_params, loss_conf, verbose, writer):
    """Simply evaluate the loss w/o grad."""
    loss_epoch = 0.
    epoch_size = len(data_loader)

    # loop for one epoch
    for step, data in zip(range(epoch_size), data_loader):
        if procid == 0 and verbose:
            tic = time.perf_counter()

        # data for this step
        sidx, a_ic, a_snaps, sobol = (data[k] for k in
                                        ('sidx', 'a_ic', 'a_snaps', 'sobol'))
        ic, tgts = data['ic'], data['tgts']

        # setup model for this step
        model_conf.update({'a_snaps': a_snaps, 'a_ic': a_ic, 'sobol': sobol})
        ptcl, cosmo, conf = setup_model(ic, model_conf)

        # evaluate loss for this step
        loss = obj(tgts, ptcl, so_params, cosmo, conf, loss_conf)
        loss = tree_global_mean(loss)

        loss = np.array(loss)  # move loss back to CPU memory
        loss_epoch += loss

        # step output
        if procid == 0:
            writer.add_scalar('loss', loss, step)
            if verbose:
                toc = time.perf_counter()
                print((f'{toc - tic:.0f} s, {step:>2d}, {sidx:>3d}, ' +
                       f'{loss:12.3e}'), flush=True)

    loss_epoch = loss_epoch / epoch_size  # mean loss per step of epoch
    if procid == 0:
        print(f'epoch mean loss: {loss_epoch:12.3e}', flush=True)
