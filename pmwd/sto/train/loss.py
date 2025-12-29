from jax import jit, checkpoint
import jax.numpy as jnp
from jax.lax import scan

from pmwd.particles import Particles
from pmwd.spec_util import powspec
from pmwd.scatter import scatter


def loss_ptcl_dens(ptcl, ptcl_t, conf, loss_conf):
    # get the density fields
    dens = scatter(ptcl, conf, offset=loss_conf['grid_offset'])
    dens_t = scatter(ptcl_t, conf, offset=loss_conf['grid_offset'])

    # loss on power spec
    k, P_d, _, _ = powspec(dens - dens_t, 1.)
    k, P_t, _, _ = powspec(dens_t, 1.)
    loss = jnp.sum(jnp.log(P_d / P_t + loss_conf['log_eps'])) / len(k)

    return loss


def loss_ptcl_disp(ptcl, ptcl_t, conf, loss_conf):
    # get the relative disp
    disp_d = ptcl.disp - ptcl_t.disp
    # wrap to [-L/2, L/2] for the shorter disp
    # in case e.g. disp = L/2 - d (a small number), disp_t = -L/2 + d
    # -> disp_d = L - 2d, which should be wrapped to 2d
    box_size = jnp.array(conf.box_size, dtype=conf.float_dtype)
    disp_d -= jnp.rint(disp_d / box_size) * box_size

    # mse loss
    loss = jnp.log(jnp.sum(disp_d**2) / jnp.sum(ptcl_t.disp**2))

    return loss


def loss_ptcl(snap, snap_t, conf, loss_conf):
    loss = 0.

    # displacement
    if 'disp' in loss_conf['loss_fields']:
        loss += loss_ptcl_disp(snap, snap_t, conf, loss_conf)

    # density field
    if 'dens' in loss_conf['loss_fields']:
        loss += loss_ptcl_dens(snap, snap_t, conf, loss_conf)

    return loss


def loss_func(obsvbl, tgts, conf, loss_conf):
    """Loss function of the simulated snapshots and target snapshots."""
    loss = 0.
    n_snaps = len(tgts[0])

    # @checkpoint  # checkpoint for saving memory in backward AD
    def _loss_snap(carry, x):
        loss = carry
        tgt, snap = x

        # make target ptcl from pos and vel
        snap_t = Particles(conf, snap.pmid, tgt[0].astype(conf.float_dtype),
                           vel=tgt[1].astype(conf.float_dtype))

        # accumulate loss of this snapshot
        loss += loss_ptcl(snap, snap_t, conf, loss_conf)
        return loss, None

    # scan over snapshots to accumulate loss
    loss, _ = scan(_loss_snap, loss, (tgts, obsvbl['snaps']))

    # mean loss per snapshot
    loss /= n_snaps

    return loss
