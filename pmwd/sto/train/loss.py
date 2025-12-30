from jax import jit, checkpoint, vmap
import jax.numpy as jnp
from jax.lax import scan

from pmwd.particles import Particles
from pmwd.spec_util import powspec
from pmwd.scatter import scatter


def eval_dens_loss(ptcl, ptcl_t, conf, offset, log_eps):
    # get the density fields
    dens = scatter(ptcl, conf, offset=offset)
    dens_t = scatter(ptcl_t, conf, offset=offset)

    # loss on power spec
    k, P_d, _, _ = powspec(dens - dens_t, 1.)
    k, P_t, _, _ = powspec(dens_t, 1.)
    loss = jnp.sum(jnp.log(P_d / P_t + log_eps)) / len(k)

    return loss.astype(conf.float_dtype)


@jit
def eval_disp_loss(disp, disp_t, box_size):
    # get the relative disp
    disp_d = disp - disp_t
    # wrap to [-L/2, L/2] for the shorter disp
    # in case e.g. disp = L/2 - d (a small number), disp_t = -L/2 + d
    # -> disp_d = L - 2d, which should be wrapped to 2d
    disp_d -= jnp.rint(disp_d / box_size) * box_size

    # log mse over all particles in snapshot
    def _disp_loss_snap(_disp_d, _disp_t):
        return jnp.log(jnp.sum(_disp_d**2) / jnp.sum(_disp_t**2))

    # sum over all snapshots
    loss = jnp.sum(vmap(_disp_loss_snap)(disp_d, disp_t))

    return loss


def loss_func(obsvbl, tgts, conf, loss_conf):
    """Loss function of the simulated snapshots and target snapshots."""
    loss = 0.
    n_snaps = len(tgts[0])

    if 'disp' in loss_conf['loss_fields']:
        box_size = jnp.array(conf.box_size, dtype=conf.float_dtype)
        disp = obsvbl['snaps'].disp
        disp_t = tgts[0].astype(conf.float_dtype)
        loss += eval_disp_loss(disp, disp_t, box_size)

    if 'dens' in loss_conf['loss_fields']:

        offset = loss_conf['grid_offset']
        log_eps = loss_conf['log_eps']

        def _snap_dens_loss(carry, x):
            loss = carry
            tgt, snap = x

            # make target ptcl from pos and vel
            snap_t = Particles(conf, snap.pmid, tgt[0].astype(conf.float_dtype),
                               vel=tgt[1].astype(conf.float_dtype))

            # accumulate loss of this snapshot
            loss += eval_dens_loss(snap, snap_t, conf, offset, log_eps)
            return loss, None

        # scan over snapshots to accumulate loss
        loss, _ = scan(_snap_dens_loss, loss, (tgts, obsvbl['snaps']))

    # mean loss per snapshot
    loss /= n_snaps

    return loss
