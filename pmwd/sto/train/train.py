import jax
import jax.numpy as jnp
import numpy as np
import optax
import time

from pmwd.nbody import nbody
from pmwd.particles import Particles

from pmwd.sto.data.initial import gen_cc
from pmwd.sto.train.loss import loss_func
from pmwd.sto.train.utils import tree_global_mean


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
    # initialize ptcl given input (pos, vel) data
    ptcl = Particles.gen_grid(conf)
    ptcl = ptcl.replace(disp=ic[0].astype(conf.float_dtype),
                        vel=ic[1].astype(conf.float_dtype))

    return ptcl, cosmo, conf


def train_step(data_step, so_params, model_conf, opt_conf, opt_state, loss_conf):
    ic, tgts = data_step

    # setup input for model
    ptcl, cosmo, conf = setup_model(ic, model_conf)

    # get loss and grad
    obj_valgrad = jax.value_and_grad(obj, argnums=2)
    loss, grad = obj_valgrad(tgts, ptcl, so_params, cosmo, conf, loss_conf)

    # average over global devices
    loss, grad = tree_global_mean((loss, grad))

    # optimize
    updates, opt_state = opt_conf['optimizer'].update(grad, opt_state, so_params)
    so_params = optax.apply_updates(so_params, updates)

    return so_params, loss, opt_state


def train_epoch(procid, epoch, data_loader, model_conf,
                so_params, opt_conf, opt_state, loss_conf,
                verbose, writer):
    loss_epoch = 0.
    epoch_size = len(data_loader)

    for step, data in enumerate(data_loader):
        if procid == 0 and verbose:
            tic = time.perf_counter()

        sidx, a_ic, a_snaps, sobol = (data[k] for k in
                                        ('sidx', 'a_ic', 'a_snaps', 'sobol'))
        ic, tgts = data['ic'], data['tgts']

        # setup and training for one step
        # notice that the final output times of Gadget4 are not exactly the same
        # as the desired times as specified in outtimes.txt, therefore here we
        # use the final output times from Gadget4 snapshot data
        model_conf.update({'a_snaps': a_snaps, 'a_ic': a_ic, 'sobol': sobol})
        so_params, loss, opt_state = train_step(
            ic, tgts, so_params, model_conf, opt_conf, opt_state, loss_conf)
        loss_epoch += loss

        if procid == 0:
            global_step = epoch * epoch_size + step
            writer.add_scalar('loss', np.array(loss), global_step)
            if verbose:
                toc = time.perf_counter()
                print((f'{toc - tic:.0f} s, {step:>2d}, {sidx:>3d}, ' +
                       f'{loss:12.3e}'), flush=True)


    loss_epoch = loss_epoch / epoch_size  # mean loss per step of epoch

    return loss_epoch, so_params, opt_state


def evaluate_loss_epoch(procid, epoch, data_loader, model_conf,
                        so_params, loss_conf,
                        verbose, writer):
    """Simply evaluate the loss w/o grad."""
    loss_epoch = 0.
    epoch_size = len(data_loader)

    for step, data in enumerate(data_loader):
        if procid == 0 and verbose:
            tic = time.perf_counter()

        sidx, a_ic, a_snaps, sobol = (data[k] for k in
                                        ('sidx', 'a_ic', 'a_snaps', 'sobol'))
        ic, tgts = data['ic'], data['tgts']

        model_conf.update({'a_snaps': a_snaps, 'a_ic': a_ic, 'sobol': sobol})
        ptcl, cosmo, conf = setup_model(ic, model_conf)
        loss = obj(tgts, ptcl, so_params, cosmo, conf, loss_conf)
        loss = tree_global_mean(loss)
        loss_epoch += loss

        if procid == 0:
            global_step = epoch * epoch_size + step
            writer.add_scalar('loss', np.array(loss), global_step)
            if verbose:
                toc = time.perf_counter()
                print((f'{toc - tic:.0f} s, {step:>2d}, {sidx:>3d}, ' +
                       f'{loss:12.3e}'), flush=True)

    loss_epoch = loss_epoch / epoch_size  # mean loss per step of epoch

    return loss_epoch
