from jax import jit
import jax.numpy as jnp
from jax.lax import scan
from functools import partial

from pmwd.particles import Particles, ptcl_rpos
from pmwd.pm_util import rfftnfreq
from pmwd.spec_util import powspec
from pmwd.sto.util import scatter_dens, pv2ptcl


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


def loss_power_ln(f, g, spacing=1, cut_nyq=False):
    k, P_d, N, bins = powspec(f - g, spacing, cut_nyq=cut_nyq)
    k, P_g, N, bins = powspec(g, spacing, cut_nyq=cut_nyq)
    loss = jnp.log(P_d / P_g).sum() / len(k)
    return loss


def loss_snap_disp(ptcl, ptcl_t, conf):
    # get the disp from particles' grid Lagrangian positions
    # may be necessary since we have it divided in the mse
    disp, disp_t = (ptcl_rpos(p, Particles.gen_grid(p.conf), p.conf)
                    for p in (ptcl, ptcl_t))
    # reshape -> make last 3 axes spatial dims
    shape_ = (-1,) + conf.ptcl_grid_shape
    disp = disp.T.reshape(shape_)
    disp_t = disp_t.T.reshape(shape_)

    loss = loss_mse(disp, disp_t)
    return loss


def loss_snap_dens(ptcl, ptcl_t, conf, loss_mesh_shape):
    # get the density fields
    (dens, dens_t), cell_size = scatter_dens((ptcl, ptcl_t), conf, loss_mesh_shape)

    # loss = loss_power_w(dens, dens_t)
    loss = loss_power_ln(dens, dens_t)
    return loss


@jit
def loss_snap(ptcl, ptcl_t, a, conf, loss_mesh_shape=1):
    loss = 0.
    # displacement
    loss += loss_snap_disp(ptcl, ptcl_t, conf)
    # density field
    loss += loss_snap_dens(ptcl, ptcl_t, conf, loss_mesh_shape)
    # divided by the number of nbody steps to this snap
    # loss /= (a - conf.a_start) // conf.a_nbody_step + 1
    return loss


def loss_func(obsvbl, tgts, conf):
    loss = 0.

    # TODO adapt the structure of tgts for scan
    # def f_loss(carry, x):
    #     loss = carry
    #     tgt, a_snap, snap = x
    #     ptcl_t = pv2ptcl(*tgt, ptcl.pmid, ptcl.conf)
    #     loss += loss_snap(ptcl, ptcl_t, a_snap, conf)
    #     return loss, None

    # loss = scan(f_loss, loss, (tgts, obsvbl['a_snaps'], obsvbl['snaps']))[0]
    # loss /= len(tgts)

    for i, (tgt, a_snap) in enumerate(zip(tgts, obsvbl['a_snaps'])):
        ptcl = obsvbl['snaps'][i]
        ptcl_t = pv2ptcl(*tgt, ptcl.pmid, ptcl.conf)  # get the target ptcl
        loss += loss_snap(ptcl, ptcl_t, a_snap, conf)
    loss /= len(tgts)  # mean loss per snapshot

    return loss
