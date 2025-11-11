import jax
import jax.numpy as jnp
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


def setup_model(pv_ic, model_conf):
    conf, cosmo = gen_cc(model_conf['sobol'],
                         mesh_shape=model_conf['mesh_shape'],
                         a_snapshots=model_conf['a_snaps'],
                         a_nbody_num=model_conf['n_steps'],
                         so_type=model_conf['so_type'],
                         so_nodes=model_conf['so_nodes'],
                         soft_i=model_conf['soft_i'],
                         a_start=model_conf['a_ic'],
                         a_stop=model_conf['a_stop'])
    # initialize ptcl given input (pos, vel) data
    ptcl = Particles.gen_grid(conf)
    disp = pv_ic[0] - ptcl.pmid * conf.cell_size
    # wrap around the periodic boundaries, disp: [-L/2, L/2]
    box_size = jnp.array(conf.box_size)
    disp -= jnp.rint(disp / box_size) * box_size
    ptcl = ptcl.replace(disp=disp.astype(conf.float_dtype),
                        vel=pv_ic[1].astype(conf.float_dtype))

    return ptcl, cosmo, conf


def train_step(data_step, so_params, model_conf, opt_conf, opt_state, loss_conf):
    tgts, pv_ic = data_step

    # setup input for model
    ptcl, cosmo, conf = setup_model(pv_ic, model_conf)

    # get loss and grad
    obj_valgrad = jax.value_and_grad(obj, argnums=2)
    loss, grad = obj_valgrad(tgts, ptcl, so_params, cosmo, conf, loss_conf)

    # average over global devices
    loss, grad = tree_global_mean((loss, grad))

    # optimize
    updates, opt_state = opt_conf['optimizer'].update(grad, opt_state, so_params)
    so_params = optax.apply_updates(so_params, updates)

    return so_params, loss, opt_state


def train_epoch(procid, epoch, gsdata, sobol_ids_epoch, model_conf,
                so_params, opt_conf, opt_state, loss_conf, verbose):
    loss_epoch = 0.

    for _, sidx in enumerate(sobol_ids_epoch):
        if procid == 0 and verbose:
            tic = time.perf_counter()

        pv_ic, a_ic, tgts, a_snaps, sobol = (gsdata[sidx][k] for k in
                                       ('ic', 'a_ic', 'pv', 'a_snaps', 'sobol'))

        # put ic and loss data of this step to device, could be asynchronous
        data_step = jax.device_put((pv_ic, tgts))

        # setup and training for one step
        # notice that the final output times of Gadget4 are not exactly the same
        # as the desired times as specified in outtimes.txt, therefore here we
        # use the final output times from Gadget4 snapshot data
        model_conf.update({'a_snaps': a_snaps, 'a_ic': a_ic, 'sobol': sobol})
        so_params, loss, opt_state = train_step(
            data_step, so_params, model_conf, opt_conf, opt_state, loss_conf)
        loss_epoch += loss

        if procid == 0 and verbose:
            toc = time.perf_counter()
            print((f'{toc - tic:.0f} s, {epoch}, {sidx:>3d}, {model_conf['mesh_shape']:>3d},' +
                   f' {model_conf['n_steps']:>4d}, {loss:12.3e}'), flush=True)

    loss_epoch = loss_epoch / len(gsdata)  # mean loss per step of epoch

    return loss_epoch, so_params, opt_state


def loss_epoch(procid, epoch, gsdata, sobol_ids_epoch, model_conf,
               so_params, loss_conf, verbose):
    """Simply evaluate the loss w/o grad."""
    loss_epoch = 0.

    for _, sidx in enumerate(sobol_ids_epoch):
        if procid == 0 and verbose:
            tic = time.perf_counter()

        pv_ic, a_ic, tgts, a_snaps, sobol = (gsdata[sidx][k] for k in
                                       ('ic', 'a_ic', 'pv', 'a_snaps', 'sobol'))

        # put ic and loss data of this step to device, could be asynchronous
        pv_ic, tgts = jax.device_put((pv_ic, tgts))

        model_conf.update({'a_snaps': a_snaps, 'a_ic': a_ic, 'sobol': sobol})
        ptcl, cosmo, conf = setup_model(pv_ic, model_conf)
        loss = obj(tgts, ptcl, so_params, cosmo, conf, loss_conf)
        loss = tree_global_mean(loss)
        loss_epoch += loss

        if procid == 0 and verbose:
            toc = time.perf_counter()
            print((f'{toc - tic:.0f} s, {epoch}, {sidx:>3d}, {model_conf['mesh_shape']:>3d},' +
                   f' {model_conf['n_steps']:>4d}, {loss:12.3e}'), flush=True)

    loss_epoch = loss_epoch / len(gsdata)
    return loss_epoch
