from jax import jit, checkpoint
import jax.numpy as jnp
from jax.lax import scan
from functools import partial

from pmwd.particles import Particles, ptcl_rpos
from pmwd.spec_util import powspec
from pmwd.sto.utils import scatter_dens


def loss_mse(f, g, log=True, norm=True, weights=None):
    """MSE between two arrays, with optional modifications."""
    loss = jnp.abs(f - g)**2

    if weights is not None:
        loss *= weights

    loss = jnp.sum(loss)

    if norm:
        loss /= jnp.sum(jnp.abs(g)**2)
    else:
        loss /= len(f)  # simple mean

    if log:
        loss = jnp.log(loss)

    return loss


def loss_power_w(f, g, spacing=1, log=True, w=None, cut_nyq=False):
    # f (model) & g (target) are fields of the same shape in configuration space
    k, P_d, N, bins = powspec(f - g, spacing, w=w, cut_nyq=cut_nyq)
    k, P_g, N, bins = powspec(g, spacing, cut_nyq=cut_nyq)
    loss = (P_d / P_g).sum() / len(k)
    if log:
        loss = jnp.log(loss)
    return loss


def loss_power_ln(f, g, eps, spacing=1, cut_nyq=False):
    k, P_d, N, bins = powspec(f - g, spacing, cut_nyq=cut_nyq)
    k, P_g, N, bins = powspec(g, spacing, cut_nyq=cut_nyq)
    loss = jnp.log(P_d / P_g + eps).sum() / len(k)
    return loss


def loss_ptcl_dens(ptcl, ptcl_t, conf, loss_conf):
    # get the density fields
    (dens, dens_t), cell_size = scatter_dens((ptcl, ptcl_t), conf,
                                             loss_conf['loss_mesh_shape'],
                                             offset=loss_conf['grid_offset'])

    loss = loss_power_ln(dens, dens_t, loss_conf['log_eps'])
    return loss


def loss_ptcl_disp(ptcl, ptcl_t, conf, loss_conf):
    # get the disp from particles' grid Lagrangian positions
    disp, disp_t = (ptcl_rpos(p, Particles.gen_grid(conf), conf) for p in (ptcl, ptcl_t))

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

    @checkpoint  # checkpoint for saving memory in backward AD
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
