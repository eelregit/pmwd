import jax
import jax.numpy as jnp
import optax
from functools import partial
import time

from pmwd.nbody import nbody
from pmwd.particles import Particles

from pmwd.sto.data.ccic import gen_cc
from pmwd.sto.train.loss import loss_func
from pmwd.sto.train.utils import tree_global_mean


def obj(tgts, ptcl, so_params, cosmo, conf, loss_hypars):
    cosmo = cosmo.replace(so_params=so_params)
    _, obsvbl = nbody(ptcl, None, cosmo, conf)
    loss = loss_func(obsvbl, tgts, conf, loss_hypars)
    return loss


def setup_model(pv_ic, model_conf):
    (a_snaps, a_ic, sobol, mesh_shape, n_steps, so_type, so_nodes, soft_i
     ) = model_conf

    conf, cosmo = gen_cc(sobol, mesh_shape=mesh_shape, a_snapshots=a_snaps,
                         a_nbody_num=n_steps, so_type=so_type, so_nodes=so_nodes,
                         soft_i=soft_i, a_start=a_ic)
    # initialize ptcl given input (pos, vel) data
    ptcl = Particles.gen_grid(conf)
    disp = (pv_ic[0] - ptcl.pmid * conf.cell_size).astype(conf.float_dtype)
    ptcl = ptcl.replace(disp=disp, vel=pv_ic[1].astype(conf.float_dtype))

    return ptcl, cosmo, conf


def train_step(data_step, so_params, model_conf, optimizer, opt_state, loss_hypars):
    tgts, pv_ic = data_step

    # setup input for model
    ptcl, cosmo, conf = setup_model(pv_ic, model_conf)

    # get loss and grad
    obj_valgrad = jax.value_and_grad(obj, argnums=2)
    loss, grad = obj_valgrad(tgts, ptcl, so_params, cosmo, conf, loss_hypars)

    # average over global devices
    loss, grad = tree_global_mean((loss, grad))

    # optimize
    updates, opt_state = optimizer.update(grad, opt_state, so_params)
    so_params = optax.apply_updates(so_params, updates)

    return so_params, loss, opt_state


def train_epoch(procid, epoch, gsdata, sobol_ids_epoch, so_type, so_nodes, soft_i,
                so_params, optimizer, opt_state, loss_hypars, verbose):
    loss_epoch = 0.

    for _, sidx in enumerate(sobol_ids_epoch):
        if procid == 0 and verbose:
            tic = time.perf_counter()

        pv_ic, a_ic, tgts, a_snaps, sobol = (gsdata[sidx][k] for k in
                                       ('ic', 'a_ic', 'pv', 'a_snaps', 'sobol'))

        # put ic and loss data of this step to device, could be asynchronous
        data_step = jax.device_put((pv_ic, tgts))

        # training time hypars
        mesh_shape = 1
        n_steps = 61
        loss_hypars['grid_offset'] = 0
        loss_hypars['loss_mesh_shape'] = 3

        # setup and training for one step
        model_conf = (a_snaps, a_ic, sobol, mesh_shape, n_steps,
                      so_type, so_nodes, soft_i)
        so_params, loss, opt_state = train_step(data_step, so_params, model_conf,
                                                optimizer, opt_state, loss_hypars)
        loss_epoch += loss

        if procid == 0 and verbose:
            toc = time.perf_counter()
            print((f'{toc - tic:.0f} s, {epoch}, {sidx:>3d}, {mesh_shape:>3d}, '
                   + f'{n_steps:>4d}, {loss:12.3e}'), flush=True)

    loss_epoch = loss_epoch / len(gsdata)  # mean loss per step of epoch

    return loss_epoch, so_params, opt_state


def loss_epoch(procid, epoch, gsdata, sobol_ids_epoch, so_type, so_nodes, soft_i,
               so_params, loss_hypars, verbose):
    """Simply evaluate the loss w/o grad."""
    loss_epoch = 0.

    for _, sidx in enumerate(sobol_ids_epoch):
        if procid == 0 and verbose:
            tic = time.perf_counter()

        pv_ic, a_ic, tgts, a_snaps, sobol = (gsdata[sidx][k] for k in
                                       ('ic', 'a_ic', 'pv', 'a_snaps', 'sobol'))

        # put ic and loss data of this step to device, could be asynchronous
        data_step = jax.device_put((pv_ic, tgts))

        # training time hypars
        mesh_shape = 1
        n_steps = 61
        loss_hypars['grid_offset'] = 0
        loss_hypars['loss_mesh_shape'] = 3

        model_conf = (a_snaps, a_ic, sobol, mesh_shape, n_steps,
                      so_type, so_nodes, soft_i)
        ptcl, cosmo, conf = setup_model(pv_ic, model_conf)
        loss = obj(tgts, ptcl, so_params, cosmo, conf, loss_hypars)
        loss = tree_global_mean(loss)
        loss_epoch += loss

        if procid == 0 and verbose:
            toc = time.perf_counter()
            print((f'{toc - tic:.0f} s, {epoch}, {sidx:>3d}, {mesh_shape:>3d}, '
                   + f'{n_steps:>4d}, {loss:12.3e}'), flush=True)

    loss_epoch = loss_epoch / len(gsdata)
    return loss_epoch
