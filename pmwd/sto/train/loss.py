from functools import partial
from jax import jit, checkpoint, vmap
import jax.numpy as jnp
from jax.lax import scan

from pmwd.particles import Particles
from pmwd.spec_util import powspec
from pmwd.scatter import scatter


def snap_disp_loss(disp_d, disp_t):
    # log mse over all particles in snapshot
    return jnp.log(jnp.sum(disp_d**2) / jnp.sum(disp_t**2))


@jit
def eval_disp_loss(disp, disp_t, box_size):
    # get the relative disp
    disp_d = disp - disp_t
    # wrap to [-L/2, L/2] for the shorter disp
    # in case e.g. disp = L/2 - d (a small number), disp_t = -L/2 + d
    # -> disp_d = L - 2d, which should be wrapped to 2d
    disp_d -= jnp.rint(disp_d / box_size) * box_size

    # vmap and sum over all snapshots
    loss = jnp.sum(vmap(snap_disp_loss)(disp_d, disp_t))

    return loss


def snap_dens_loss(ptcl, ptcl_t, conf, offset, log_eps):
    # get the density fields
    dens = scatter(ptcl, conf, offset=offset)
    dens_t = scatter(ptcl_t, conf, offset=offset)

    # loss on power spec
    k, P_d, _, _ = powspec(dens - dens_t, 1.)
    k, P_t, _, _ = powspec(dens_t, 1.)
    loss = jnp.sum(jnp.log(P_d / P_t + log_eps)) / len(k)

    return loss.astype(conf.float_dtype)


@jit
def eval_dens_loss(conf, offset, log_eps, loss, x):
    tgt, snap = x

    # make target ptcl from pos and vel
    snap_t = Particles(conf, snap.pmid, tgt[0].astype(conf.float_dtype),
                       vel=tgt[1].astype(conf.float_dtype))

    # accumulate loss of this snapshot
    loss += snap_dens_loss(snap, snap_t, conf, offset, log_eps)

    return loss, None


def loss_func(obsvbl, tgts, conf, loss_conf):
    """Loss function of the simulated snapshots and target snapshots."""
    loss = jnp.array(0., dtype=conf.float_dtype)
    n_snaps = len(tgts[0])

    if 'disp' in loss_conf['loss_fields']:
        box_size = jnp.array(conf.box_size, dtype=conf.float_dtype)
        disp = obsvbl['snaps'].disp
        disp_t = tgts[0].astype(conf.float_dtype)
        loss += eval_disp_loss(disp, disp_t, box_size)

    if 'dens' in loss_conf['loss_fields']:

        offset = loss_conf['grid_offset']
        log_eps = loss_conf['log_eps']

        # scan over snapshots to accumulate loss
        loss, _ = scan(partial(eval_dens_loss, conf, offset, log_eps),
                       loss, (tgts, obsvbl['snaps']))

    # mean loss per snapshot
    loss /= n_snaps

    return loss
