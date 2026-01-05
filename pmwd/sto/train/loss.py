from functools import partial
from jax import jit, checkpoint, vmap
import jax.numpy as jnp
from jax.lax import scan

from pmwd.particles import Particles
from pmwd.spec_util import powspec
from pmwd.scatter import _scatter


def snap_disp_loss(disp_d, disp_t):
    # log mse over all particles in snapshot
    return jnp.log(jnp.sum(disp_d**2) / jnp.sum(disp_t**2))


@jit
def eval_disp_loss(obsvbl, tgts, box_size):
    disp = obsvbl['snaps'].disp
    disp_t = tgts[0]

    # get the relative disp
    disp_d = disp - disp_t
    # wrap to [-L/2, L/2] for the shorter disp
    # in case e.g. disp = L/2 - d (a small number), disp_t = -L/2 + d
    # -> disp_d = L - 2d, which should be wrapped to 2d
    disp_d -= jnp.rint(disp_d / box_size) * box_size

    # vmap and sum over all snapshots
    loss = jnp.sum(vmap(snap_disp_loss)(disp_d, disp_t))

    return loss


def snap_dens_loss(disp, disp_t, pmid, conf, offset, log_eps):
    # get the density fields
    dens = _scatter(pmid, disp, conf, None, None, offset, None)
    dens_t = _scatter(pmid, disp_t, conf, None, None, offset, None)

    # loss on power spec
    k, P_d, _, _ = powspec(dens - dens_t, 1.)
    k, P_t, _, _ = powspec(dens_t, 1.)
    loss = jnp.sum(jnp.log(P_d / P_t + log_eps)) / len(k)

    return loss


def _eval_dens_loss(conf, offset, log_eps, loss, x):
    snap, disp_t = x
    # accumulate loss of this snapshot
    loss += snap_dens_loss(snap.disp, disp_t, snap.pmid, conf, offset, log_eps)
    return loss, None


@jit
def eval_dens_loss(obsvbl, tgts, conf, offset, log_eps):
    loss = 0.
    # scan over snapshots to accumulate loss
    loss, _ = scan(partial(_eval_dens_loss, conf, offset, log_eps),
                   loss, (obsvbl['snaps'], tgts[0]))
    return loss


def loss_func(obsvbl, tgts, conf, loss_conf):
    """Loss function of the simulated snapshots and target snapshots."""
    loss = jnp.array(0., dtype=conf.float_dtype)
    n_snaps = len(tgts[0])

    if 'disp' in loss_conf['loss_fields']:
        box_size = jnp.array(conf.box_size, dtype=conf.float_dtype)
        loss += eval_disp_loss(obsvbl, tgts, box_size).astype(conf.float_dtype)

    if 'dens' in loss_conf['loss_fields']:
        offset = jnp.array(loss_conf['grid_offset'], dtype=conf.float_dtype)
        log_eps = jnp.array(loss_conf['log_eps'], dtype=conf.float_dtype)
        loss += eval_dens_loss(obsvbl, tgts, conf, offset, log_eps).astype(conf.float_dtype)

    # mean loss per snapshot
    loss /= n_snaps

    return loss
