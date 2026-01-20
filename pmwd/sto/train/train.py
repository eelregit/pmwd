import os
procid = int(os.getenv('SLURM_PROCID'))
slurm_job_id = os.getenv('SLURM_JOB_ID')

import jax
from jax import jit
import numpy as np
import optax
import time
import pickle
from functools import partial

from pmwd.nbody import nbody
from pmwd.particles import Particles

from pmwd.sto.data.initial import gen_cosmo
from pmwd.sto.train.loss import loss_func
from pmwd.sto.train.utils import tree_global_mean, procinfo


def obj(tgts, ptcl, so_params, cosmo, conf, loss_conf):
    cosmo = cosmo.replace(so_params=so_params)
    _, obsvbl = nbody(ptcl, None, cosmo, conf)
    loss = loss_func(obsvbl, tgts, conf, loss_conf)
    return loss


def setup_model(data, model_conf):
    conf = model_conf.replace(ptcl_spacing=data['ptcl_spacing'])

    cosmo = gen_cosmo(conf, data['sobol'], data['a_snaps'])

    ptcl = Particles.gen_grid(conf)
    ptcl = ptcl.replace(disp=data['ic'][0].astype(conf.float_dtype),
                        vel=data['ic'][1].astype(conf.float_dtype))

    return ptcl, cosmo, conf


@partial(jit, static_argnums=(0,))
def optim_step(optimizer, opt_state, grad, params):
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state


def train_step(tgts, ptcl, cosmo, conf, so_params, opt_state, opt_conf, loss_conf):

    # get loss and grad
    obj_valgrad = jax.value_and_grad(obj, argnums=2)
    loss, grad = obj_valgrad(tgts, ptcl, so_params, cosmo, conf, loss_conf)

    # average over global devices
    loss, grad = tree_global_mean((loss, grad))

    # optimize
    so_params, opt_state = optim_step(opt_conf['optimizer'], opt_state, grad, so_params)

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
                 verbose, writer, epoch_init, rng_key):
    epoch_size = len(data_loader)
    step_start = (epoch_init + 1) * epoch_size
    total_steps = n_epochs * epoch_size

    # loop for n_epochs
    loss_epoch = 0.
    epoch = epoch_init
    for step, data in zip(range(step_start, step_start + total_steps), data_loader):
        epoch += 1  # current epoch of training
        if procid == 0 and verbose:
            tic = time.perf_counter()

        # setup model for this step
        ptcl, cosmo, conf = setup_model(data, model_conf)
        rng_key, subkey = jax.random.split(rng_key)
        loss_conf['key'] = subkey

        # train for this step
        so_params, opt_state, loss = train_step(
            data['tgts'], ptcl, cosmo, conf, so_params, opt_state, opt_conf, loss_conf)

        loss = np.array(loss)  # move loss back to CPU memory
        loss_epoch += loss

        # step output
        if procid == 0:
            writer.add_scalar('loss', loss, step)
            if verbose:
                toc = time.perf_counter()
                print((f'{toc - tic:>3.0f} s, {step:>6d}, {data['sidx']:>3d}, ' +
                       f'{loss:16.5e}'), flush=True)

        # epoch output
        if (step + 1) % epoch_size == 0:
            loss_epoch = loss_epoch / epoch_size  # mean loss per step of epoch
            if procid == 0:
                writer.add_scalar('epoch_mean_loss', loss_epoch, epoch)
                print(f'epoch mean loss: {loss_epoch:16.5e}', flush=True)
                checkpoint(epoch, so_params, opt_state, opt_conf['learning_rate'],
                           verbose=verbose)
            loss_epoch = 0.


def evaluate_loss_epoch(procid, data_loader, model_conf,
                        so_params, loss_conf, verbose, writer, rng_key):
    """Simply evaluate the loss w/o grad."""
    loss_epoch = 0.
    epoch_size = len(data_loader)

    # loop for one epoch
    for step, data in zip(range(epoch_size), data_loader):
        if procid == 0 and verbose:
            tic = time.perf_counter()

        # setup model for this step
        ptcl, cosmo, conf = setup_model(data, model_conf)
        rng_key, subkey = jax.random.split(rng_key)
        loss_conf['key'] = subkey

        # evaluate loss for this step
        loss = obj(data['tgts'], ptcl, so_params, cosmo, conf, loss_conf)
        loss = tree_global_mean(loss)

        loss = np.array(loss)  # move loss back to CPU memory
        loss_epoch += loss

        # step output
        if procid == 0:
            writer.add_scalar('loss', loss, step)
            if verbose:
                toc = time.perf_counter()
                print((f'{toc - tic:>3.0f} s, {step:>6d}, {data['sidx']:>3d}, ' +
                       f'{loss:16.5e}'), flush=True)

    # epoch output
    loss_epoch = loss_epoch / epoch_size  # mean loss per step of epoch
    if procid == 0:
        writer.add_scalar('epoch_mean_loss', loss_epoch, 0)
        print(f'epoch mean loss: {loss_epoch:16.5e}', flush=True)
