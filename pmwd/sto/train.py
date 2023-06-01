import jax
import jax.numpy as jnp
import optax
from functools import partial

from pmwd.nbody import nbody
from pmwd.sto.data import gen_cc, gen_ic
from pmwd.sto.loss import loss_func


def pmodel(ptcl_ic, so_params, cosmo, conf):
    cosmo = cosmo.replace(so_params=so_params)
    _, obsvbl = nbody(ptcl_ic, None, cosmo, conf)
    return obsvbl[0], cosmo


def obj(tgt, ptcl_ic, so_params, cosmo, conf):
    ptcl, cosmo = pmodel(ptcl_ic, so_params, cosmo, conf)
    loss = loss_func(ptcl, tgt, conf)
    return loss


@partial(jax.pmap, axis_name='global', in_axes=(0, None), out_axes=None)
def _global_mean(loss, grad):
    loss = jax.lax.pmean(loss, axis_name='global')
    grad = jax.lax.pmean(grad, axis_name='global')
    return loss, grad


def _init_pmwd(pmwd_params):
    a_out, sidx, sobol, mesh_shape, n_steps, so_type, so_nodes = pmwd_params

    # generate ic, cosmo, conf
    conf, cosmo = gen_cc(sobol, mesh_shape=(mesh_shape,)*3, a_out=a_out,
                         a_nbody_num=n_steps, so_type=so_type, so_nodes=so_nodes)
    ptcl_ic = gen_ic(sidx, conf, cosmo)

    return ptcl_ic, cosmo, conf


def train_step(tgt, so_params, pmwd_params, opt_params):
    ptcl_ic, cosmo, conf = _init_pmwd(pmwd_params)

    # loss and grad
    obj_valgrad = jax.value_and_grad(obj, argnums=2)
    loss, grad = obj_valgrad(tgt, ptcl_ic, so_params, cosmo, conf)

    # average over global devices
    loss = jnp.expand_dims(loss, axis=0)  # for pmap
    loss, grad = _global_mean(loss, grad)

    # optimize
    optimizer, opt_state = opt_params
    updates, opt_state = optimizer.update(grad, opt_state, so_params)
    so_params = optax.apply_updates(so_params, updates)

    return so_params, loss, opt_state
