from jax import vjp
import jax.numpy as jnp
from jax.lax import cond, scan
from jax.tree_util import tree_map

from pmwd.interp_util import itp_prev, itp_next, itp_prev_adj, itp_next_adj
from pmwd.particles import Particles


def _isclose(a1, a2, rtol=0, atol=1e-6):
    return jnp.isclose(a1, a2, rtol=rtol, atol=atol)


def create_obsvbl(ptcl, conf):
    # a dict to carry all observables and related useful information
    obsvbl = {}

    if conf.a_snapshots is not None:
        obsvbl['a_snaps'] = jnp.array(conf.a_snapshots)
        # all output snapshots, at times given by conf.a_snapshots
        obsvbl['snaps'] = [Particles(ptcl.conf, ptcl.pmid, jnp.zeros_like(ptcl.disp),
                           vel=jnp.zeros_like(ptcl.vel))] * len(conf.a_snapshots)
        # transposed pytree with leading axis for scan
        obsvbl['snaps'] = tree_map(lambda *xs: jnp.stack(xs), *obsvbl['snaps'])

        # the nbody (a_prev, a_next] step for each interpolated snapshot
        # used in observe to determine the time for interpolation
        idx = jnp.searchsorted(conf.a_nbody, jnp.array(conf.a_snapshots), side='left')
        obsvbl['itp_a_step'] = jnp.array((conf.a_nbody[idx-1], conf.a_nbody[idx])).T

    return obsvbl


def observe(a, ptcl, obsvbl, cosmo, conf):

    def obs_interp(obsvbl, x):
        i, a_snap, a_step = x

        def _obs_prev(obsvbl):
            disp, vel = itp_prev(ptcl, a_step[0], a_step[1], a_snap, cosmo)
            obsvbl['snaps'] = obsvbl['snaps'].replace(
                disp=obsvbl['snaps'].disp.at[i].add(disp),
                vel=obsvbl['snaps'].vel.at[i].add(vel))
            return obsvbl

        def _obs_next(obsvbl):
            disp, vel = itp_next(ptcl, a_step[0], a_step[1], a_snap, cosmo)
            obsvbl['snaps'] = obsvbl['snaps'].replace(
                disp=obsvbl['snaps'].disp.at[i].add(disp),
                vel=obsvbl['snaps'].vel.at[i].add(vel))
            return obsvbl

        obsvbl = cond(_isclose(a_step[0], a), _obs_prev, lambda _: _,
                      obsvbl)

        obsvbl = cond(_isclose(a_step[1], a), _obs_next, lambda _: _,
                      obsvbl)

        return obsvbl, None

    if conf.a_snapshots is not None:
        obsvbl, _ = scan(obs_interp, obsvbl, (jnp.arange(len(conf.a_snapshots)),
                                              obsvbl['a_snaps'], obsvbl['itp_a_step']))

    return obsvbl


def observe_adj(a, ptcl, ptcl_cot, obsvbl, obsvbl_cot, cosmo, cosmo_cot, conf):
    _, observe_vjp = vjp(observe, a, ptcl, obsvbl, cosmo, conf)
    _, ptcl_cot_obs, _, cosmo_cot_obs, _ = observe_vjp(obsvbl_cot)

    disp_cot = ptcl_cot.disp + ptcl_cot_obs.disp
    vel_cot = ptcl_cot.vel + ptcl_cot_obs.vel
    ptcl_cot = ptcl_cot.replace(disp=disp_cot, vel=vel_cot)

    cosmo_cot += cosmo_cot_obs

    return ptcl_cot, cosmo_cot


# def observe_adj(a, ptcl, ptcl_cot, obsvbl, obsvbl_cot, cosmo, cosmo_cot, conf):

#     def obs_interp_adj(carry, x):
#         ptcl_cot, cosmo_cot = carry
#         a_snap, a_step, snap_cot = x

#         ptcl_cot, cosmo_cot = cond(_isclose(a_step[1], a), itp_next_adj,
#                                    lambda *args: (ptcl_cot, cosmo_cot),
#                                    ptcl_cot, cosmo_cot, snap_cot, ptcl,
#                                    a_step[0], a_step[1], a_snap, cosmo)

#         ptcl_cot, cosmo_cot = cond(_isclose(a_step[0], a), itp_prev_adj,
#                                    lambda *args: (ptcl_cot, cosmo_cot),
#                                    ptcl_cot, cosmo_cot, snap_cot, ptcl,
#                                    a_step[0], a_step[1], a_snap, cosmo)

#         return (ptcl_cot, cosmo_cot), None

#     if conf.a_snapshots is not None:
#         (ptcl_cot, cosmo_cot), _ = scan(obs_interp_adj, (ptcl_cot, cosmo_cot),
#                                         (obsvbl['a_snaps'], obsvbl['itp_a_step'],
#                                          obsvbl_cot['snaps']))

#     return ptcl_cot, cosmo_cot
