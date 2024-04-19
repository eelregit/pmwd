import jax
import jax.numpy as jnp
import optax
from functools import partial
import time

from pmwd.nbody import nbody
from pmwd.sto.ccic import gen_cc, gen_ic
from pmwd.sto.loss import loss_func
from pmwd.sto.util import tree_global_mean


def pmodel(ptcl_ic, so_params, cosmo, conf):
    cosmo = cosmo.replace(so_params=so_params)
    _, obsvbl = nbody(ptcl_ic, None, cosmo, conf)
    return obsvbl, cosmo


def obj(tgts, ptcl_ic, so_params, cosmo, conf, loss_pars, loss_mesh_shape):
    obsvbl, cosmo = pmodel(ptcl_ic, so_params, cosmo, conf)
    loss = loss_func(obsvbl, tgts, conf, loss_pars, loss_mesh_shape)
    return loss


def init_pmwd(pmwd_params):
    (a_snaps, sidx, sobol, mesh_shape, n_steps, so_type, so_nodes, soft_i
     ) = pmwd_params

    # generate ic, cosmo, conf
    conf, cosmo = gen_cc(sobol, mesh_shape=mesh_shape, a_snapshots=a_snaps,
                         a_nbody_num=n_steps, so_type=so_type, so_nodes=so_nodes,
                         soft_i=soft_i)
    ptcl_ic = gen_ic(sidx, conf, cosmo)

    return ptcl_ic, cosmo, conf


def train_step(tgts, so_params, pmwd_params, opt_params, loss_pars, loss_mesh_shape):
    ptcl_ic, cosmo, conf = init_pmwd(pmwd_params)

    # get loss and grad
    obj_valgrad = jax.value_and_grad(obj, argnums=2)
    loss, grad = obj_valgrad(tgts, ptcl_ic, so_params, cosmo, conf, loss_pars,
                             loss_mesh_shape)

    # average over global devices
    loss, grad = tree_global_mean((loss, grad))

    # optimize
    optimizer, opt_state = opt_params
    updates, opt_state = optimizer.update(grad, opt_state, so_params)
    so_params = optax.apply_updates(so_params, updates)

    # not necessary, but no harm
    so_params = tree_global_mean(so_params)

    return so_params, loss, opt_state


def train_epoch(procid, epoch, gsdata, sobol_ids_epoch, so_type, so_nodes, soft_i,
                so_params, opt_state, optimizer, loss_pars, verbose):
    loss_epoch = 0.  # the sum of loss of the whole epoch

    if procid == 0:
        tic = time.perf_counter()
    for step, sidx in enumerate(sobol_ids_epoch):
        tgts, a_snaps, sobol = (gsdata[sidx][k] for k in ('pv', 'a_snaps', 'sobol'))
        tgts = jax.device_put(tgts)  # could be asynchronous

        # mesh shape, [1, 2, 3, 4]
        # mesh_shape = np_rng.integers(1, 5)
        mesh_shape = 1
        loss_mesh_shape = 3
        loss_pars['grid_offset'] = 0

        # number of nbody time steps
        n_steps = 61

        pmwd_params = (a_snaps, sidx, sobol, mesh_shape, n_steps, so_type,
                       so_nodes, soft_i)
        opt_params = (optimizer, opt_state)
        so_params, loss, opt_state = train_step(tgts, so_params, pmwd_params,
                                                opt_params, loss_pars, loss_mesh_shape)

        loss = float(loss)
        loss_epoch += loss

        # runtime print information
        if procid == 0 and verbose:
            toc = time.perf_counter()
            print((f'{toc - tic:.0f} s, {epoch}, {sidx:>3d}, {mesh_shape:>3d}, '
                   + f'{n_steps:>4d}, {loss:12.3e}'), flush=True)
            tic = time.perf_counter()

    loss_epoch_mean = loss_epoch / len(gsdata)

    return loss_epoch_mean, so_params, opt_state


def loss_epoch(procid, epoch, gsdata, sobol_ids_epoch, so_type, so_nodes, soft_i,
               so_params, loss_pars, verbose):
    """Simply evaluate the loss w/o grad."""
    loss_epoch = 0.  # the sum of loss of the whole epoch

    tic = time.perf_counter()
    for step, sidx in enumerate(sobol_ids_epoch):
        tgts, a_snaps, sobol = (gsdata[sidx][k] for k in ('pv', 'a_snaps', 'sobol'))
        tgts = jax.device_put(tgts)  # could be asynchronous

        # mesh shape, [1, 2, 3, 4]
        # mesh_shape = np_rng.integers(1, 5)
        mesh_shape = 1
        loss_mesh_shape = 3
        loss_pars['grid_offset'] = 0

        # number of nbody time steps
        n_steps = 61

        pmwd_params = (a_snaps, sidx, sobol, mesh_shape, n_steps, so_type, so_nodes,
                       soft_i)

        ptcl_ic, cosmo, conf = init_pmwd(pmwd_params)
        loss = obj(tgts, ptcl_ic, so_params, cosmo, conf, loss_pars, loss_mesh_shape)
        loss = tree_global_mean(loss)
        loss_epoch += float(loss)

        # runtime print information
        if procid == 0 and verbose:
            tt = time.perf_counter() - tic
            tic = time.perf_counter()
            print((f'{tt:.0f} s, {epoch}, {sidx:>3d}, {mesh_shape:>3d}, ' +
                   f'{n_steps:>4d}, {loss:12.3e}'), flush=True)

    loss_epoch_mean = loss_epoch / len(gsdata)
    return loss_epoch_mean
