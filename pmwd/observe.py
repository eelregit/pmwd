from jax import vjp
import jax.numpy as jnp
from jax.lax import cond, scan
from jax.tree_util import tree_map
from functools import partial

from pmwd.interp_util import itp_snap
from pmwd.particles import Particles


def _isclose(a1, a2, rtol=0, atol=1e-6):
    return jnp.isclose(a1, a2, rtol=rtol, atol=atol)


def init_obsvbl(ptcl, cosmo, conf):
    # a dict to carry all observables and related useful information
    obsvbl = {}

    if conf.observe_snapshots:
        obsvbl['a_snaps'] = cosmo.a_snapshots
        # all output snapshots, at times given by cosmo.a_snapshots
        obsvbl['snaps'] = [Particles(ptcl.conf,
                                     ptcl.pmid,
                                     jnp.zeros_like(ptcl.disp),
                                     vel=jnp.zeros_like(ptcl.vel))
                           ] * len(cosmo.a_snapshots)
        # transposed pytree with leading axis for scan
        obsvbl['snaps'] = tree_map(lambda *xs: jnp.stack(xs), *obsvbl['snaps'])

        # the nbody (a_prev, a_next] step for each interpolated snapshot
        # used in observe to determine the time for interpolation
        idx = jnp.searchsorted(conf.a_nbody, cosmo.a_snapshots, side='left')
        obsvbl['itp_a_step'] = jnp.array((conf.a_nbody[idx-1], conf.a_nbody[idx])).T

    return obsvbl


def _obs_itp_update(order, ptcl, cosmo, a_step, a_snap, i, obsvbl):
    disp_itp, vel_itp = itp_snap(order, ptcl.disp, ptcl.vel,
                                 a_step[0], a_step[1], a_snap, cosmo)
    obsvbl['snaps'] = obsvbl['snaps'].replace(
        disp=obsvbl['snaps'].disp.at[i].add(disp_itp),
        vel=obsvbl['snaps'].vel.at[i].add(vel_itp))
        # NOTE i is traced instead of static
        # JAX's in-place update syntax supports dynamic indexing (Tracers)
    return obsvbl


def _identity(_): return _


def _obs_itp_snap(a, ptcl, cosmo, obsvbl, x):
    i, a_snap, a_step = x

    obsvbl = cond(_isclose(a_step[0], a),
                  partial(_obs_itp_update, 'prev', ptcl, cosmo, a_step, a_snap, i),
                  _identity, obsvbl)

    obsvbl = cond(_isclose(a_step[1], a),
                  partial(_obs_itp_update, 'next', ptcl, cosmo, a_step, a_snap, i),
                  _identity, obsvbl)

    return obsvbl, None


def observe(a, ptcl, obsvbl, cosmo, conf):
    if conf.observe_snapshots:
        obsvbl, _ = scan(partial(_obs_itp_snap, a, ptcl, cosmo), obsvbl,
                         (jnp.arange(len(cosmo.a_snapshots)), obsvbl['a_snaps'], obsvbl['itp_a_step']))

    return obsvbl


def observe_adj(a, ptcl, ptcl_cot, obsvbl, obsvbl_cot, cosmo, cosmo_cot, conf):
    _, observe_vjp = vjp(observe, a, ptcl, obsvbl, cosmo, conf)
    _, ptcl_cot_obs, _, cosmo_cot_obs, _ = observe_vjp(obsvbl_cot)

    disp_cot = ptcl_cot.disp + ptcl_cot_obs.disp
    vel_cot = ptcl_cot.vel + ptcl_cot_obs.vel
    ptcl_cot = ptcl_cot.replace(disp=disp_cot, vel=vel_cot)

    cosmo_cot += cosmo_cot_obs

    return ptcl_cot, cosmo_cot
