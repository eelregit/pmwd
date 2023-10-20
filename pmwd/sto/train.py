import jax
import jax.numpy as jnp
import optax
from functools import partial
import time

from pmwd.nbody import nbody
from pmwd.sto.data import gen_cc, gen_ic
from pmwd.sto.loss import loss_func
from pmwd.sto.hypars import (
    so_type, so_nodes, lr_scheduler, get_optimizer, dropout_rate)
from pmwd.sto.util import global_mean


def pmodel(ptcl_ic, so_params, cosmo, conf):
    cosmo = cosmo.replace(so_params=so_params)
    _, obsvbl = nbody(ptcl_ic, None, cosmo, conf)
    return obsvbl, cosmo


def obj(tgts, ptcl_ic, so_params, cosmo, conf):
    obsvbl, cosmo = pmodel(ptcl_ic, so_params, cosmo, conf)
    loss = loss_func(obsvbl, tgts, conf)
    return loss


def init_pmwd(pmwd_params):
    (a_snaps, sidx, sobol, mesh_shape, n_steps, so_type, so_nodes,
     dropout_rate, dropout_key) = pmwd_params

    # generate ic, cosmo, conf
    conf, cosmo = gen_cc(sobol, mesh_shape=mesh_shape, a_snapshots=a_snaps,
                         a_nbody_num=n_steps, so_type=so_type, so_nodes=so_nodes,
                         dropout_rate=dropout_rate, dropout_key=dropout_key)
    ptcl_ic = gen_ic(sidx, conf, cosmo)

    return ptcl_ic, cosmo, conf


def train_step(tgts, so_params, pmwd_params, opt_params):
    ptcl_ic, cosmo, conf = init_pmwd(pmwd_params)

    # get loss and grad
    obj_valgrad = jax.value_and_grad(obj, argnums=2)
    loss, grad = obj_valgrad(tgts, ptcl_ic, so_params, cosmo, conf)

    # average over global devices
    loss, grad = global_mean((loss, grad))

    # optimize
    optimizer, opt_state = opt_params
    updates, opt_state = optimizer.update(grad, opt_state, so_params)
    so_params = optax.apply_updates(so_params, updates)

    # not necessary, but no harm
    so_params = global_mean(so_params)

    return so_params, loss, opt_state


def train_epoch(procid, epoch, gsdata, sobol_ids_epoch, so_params, opt_state, optimizer,
                learning_rate, skd_state, jax_key):
    loss_epoch = 0.  # the sum of loss of the whole epoch

    tic = time.perf_counter()
    for step, sidx in enumerate(sobol_ids_epoch):
        tgts, a_snaps, sidx, sobol = (gsdata[sidx][k] for k in (
                                      'pv', 'a_snaps', 'sidx', 'sobol'))
        tgts = jax.device_put(tgts)  # could be asynchronous

        # mesh shape, [1, 2, 3, 4]
        # mesh_shape = np_rng.integers(1, 5)
        mesh_shape = 1
        # number of time steps, [10, 1000], log-uniform
        # n_steps = np.rint(10**np_rng.uniform(1, 3)).astype(int)
        n_steps = 100

        jax_key, dropout_key = jax.random.split(jax_key)

        pmwd_params = (a_snaps, sidx, sobol, mesh_shape, n_steps, so_type, so_nodes,
                       dropout_rate, dropout_key)
        opt_params = (optimizer, opt_state)
        so_params, loss, opt_state = train_step(tgts, so_params, pmwd_params,
                                                opt_params)

        loss = float(loss)
        loss_epoch += loss

        # runtime print information
        if procid == 0:
            tt = time.perf_counter() - tic
            tic = time.perf_counter()
            print((f'{tt:.0f} s, {epoch}, {sidx:>3d}, {mesh_shape:>3d}, ' +
                   f'{n_steps:>4d}, {loss:12.3e}'), flush=True)

    # learning rate scheduler
    loss_epoch_mean = loss_epoch / len(gsdata)
    learning_rate, skd_state = lr_scheduler(learning_rate, skd_state, loss_epoch_mean)
    optimizer = get_optimizer(learning_rate)

    return loss_epoch_mean, so_params, opt_state, optimizer, learning_rate, skd_state


def loss_epoch(procid, epoch, gsdata, sobol_ids_epoch, so_params, jax_key):
    """Simply evaluate the loss w/o grad."""
    loss_epoch = 0.  # the sum of loss of the whole epoch

    tic = time.perf_counter()
    for step, sidx in enumerate(sobol_ids_epoch):
        tgts, a_snaps, sidx, sobol = (gsdata[sidx][k] for k in (
                                      'pv', 'a_snaps', 'sidx', 'sobol'))
        tgts = jax.device_put(tgts)  # could be asynchronous

        # mesh shape, [1, 2, 3, 4]
        # mesh_shape = np_rng.integers(1, 5)
        mesh_shape = 1
        # number of time steps, [10, 1000], log-uniform
        # n_steps = np.rint(10**np_rng.uniform(1, 3)).astype(int)
        n_steps = 100

        jax_key, dropout_key = jax.random.split(jax_key)

        pmwd_params = (a_snaps, sidx, sobol, mesh_shape, n_steps, so_type, so_nodes,
                       dropout_rate, dropout_key)

        ptcl_ic, cosmo, conf = init_pmwd(pmwd_params)
        loss = obj(tgts, ptcl_ic, so_params, cosmo, conf)
        loss = global_mean(loss)
        loss_epoch += float(loss)

        # runtime print information
        if procid == 0:
            tt = time.perf_counter() - tic
            tic = time.perf_counter()
            print((f'{tt:.0f} s, {epoch}, {sidx:>3d}, {mesh_shape:>3d}, ' +
                   f'{n_steps:>4d}, {loss:12.3e}'), flush=True)

    loss_epoch_mean = loss_epoch / len(gsdata)
    return loss_epoch_mean
