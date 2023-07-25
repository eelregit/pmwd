import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import optax
from functools import partial

from pmwd.nbody import nbody
from pmwd.sto.data import gen_cc, gen_ic
from pmwd.sto.loss import loss_func


def pmodel(ptcl_ic, so_params, cosmo, conf):
    cosmo = cosmo.replace(so_params=so_params)
    _, obsvbl = nbody(ptcl_ic, None, cosmo, conf)
    return obsvbl, cosmo


def obj(tgts, ptcl_ic, so_params, cosmo, conf):
    obsvbl, cosmo = pmodel(ptcl_ic, so_params, cosmo, conf)
    loss = loss_func(obsvbl, tgts, conf)
    return loss


def global_mean(tree):
    """Global average of a pytree x, i.e. for all leaves."""
    def global_mean_arr(x):
        x = jnp.expand_dims(x, axis=0)  # leading axis for pmap
        x = jax.pmap(lambda x: jax.lax.pmean(x, axis_name='device'),
                     axis_name='device')(x)
        return x[0]  # rm leading axis
    tree = tree_map(global_mean_arr, tree)
    return tree


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
