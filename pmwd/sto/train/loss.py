from jax import jit, checkpoint
import jax.numpy as jnp
from jax.lax import scan
from functools import partial

from pmwd.particles import Particles, ptcl_rpos
from pmwd.spec_util import powspec
from pmwd.sto.utils import scatter_dens


def loss_ptcl_dens(ptcl, ptcl_t, conf, loss_conf):
    # get the density fields
    (dens, dens_t), _ = scatter_dens((ptcl, ptcl_t), conf,
                                     loss_conf['loss_mesh_shape'],
                                     offset=loss_conf['grid_offset'])

    # loss on power spec
    k, P_d, _, _ = powspec(dens - dens_t, 1.)
    k, P_t, _, _ = powspec(dens_t, 1.)
    loss = jnp.sum(jnp.log(P_d / P_t + loss_conf['log_eps'])) / len(k)

    return loss


def loss_ptcl_disp(ptcl, ptcl_t, conf, loss_conf):
    # get the disp from particles' grid Lagrangian positions
    disp = ptcl_rpos(ptcl, Particles.gen_grid(conf), conf)
    disp_t = ptcl_rpos(ptcl_t, Particles.gen_grid(conf), conf)

    # get the relative disp
    disp_d = disp - disp_t
    # wrap to [-L/2, L/2] for the shorter disp
    # in case e.g. disp = L/2 - d (a small number), disp_t = -L/2 + d
    # -> disp_d = L - 2d, which should be wrapped to 2d
    box_size = jnp.array(conf.box_size, dtype=conf.float_dtype)
    disp_d -= jnp.rint(disp_d / box_size) * box_size

    # mse loss
    loss = jnp.log(jnp.sum(disp_d**2) / jnp.sum(disp_t**2))

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

    # @checkpoint  # checkpoint for saving memory in backward AD
    def _loss_snap(carry, x):
        loss = carry
        tgt, snap = x

        # make target ptcl from pos and vel
        disp_t = (tgt[0] - snap.pmid * conf.cell_size).astype(conf.float_dtype)
        snap_t = Particles(conf, snap.pmid, disp_t, vel=tgt[1].astype(conf.float_dtype))

        # accumulate loss of this snapshot
        loss += loss_ptcl(snap, snap_t, conf, loss_conf)
        return loss, None

    # scan over snapshots to accumulate loss
    loss, _ = scan(_loss_snap, loss, (tgts, obsvbl['snaps']))

    # mean loss per snapshot
    loss /= len(tgts[0])

    return loss
