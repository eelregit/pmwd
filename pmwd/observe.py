import jax.numpy as jnp
from jax.lax import cond, scan
from jax.tree_util import tree_map

from pmwd.interp_util import itp_prev, itp_next, itp_prev_adj, itp_next_adj
from pmwd.particles import Particles


def observe(a_prev, a_next, ptcl, obsvbl, cosmo, conf):

    def obs_interp(obsvbl, i):
        a_snap = obsvbl['a_snaps'][i]
        a_step = obsvbl['itp_a_step'][i]

        def _obs_prev(obsvbl):
            disp, vel = itp_prev(ptcl, a_step[0], a_step[1], a_snap, cosmo)
            obsvbl['snaps'] = obsvbl['snaps'].replace(
                disp=obsvbl['snaps'].disp.at[i].set(disp),
                vel=obsvbl['snaps'].vel.at[i].set(vel))
            return obsvbl

        obsvbl = cond(jnp.isclose(a_step[0], a_next), _obs_prev, lambda _: _,
                      obsvbl)

        def _obs_next(obsvbl):
            disp, vel = itp_next(ptcl, a_step[0], a_step[1], a_snap, cosmo)
            obsvbl['snaps'] = obsvbl['snaps'].replace(
                disp=obsvbl['snaps'].disp.at[i].add(disp),  # add itp next part
                vel=obsvbl['snaps'].vel.at[i].add(vel))
            return obsvbl

        obsvbl = cond(jnp.isclose(a_step[1], a_next), _obs_next, lambda _: _,
                      obsvbl)

        return obsvbl, None

    obsvbl = scan(obs_interp, obsvbl, jnp.arange(len(conf.a_snapshots)))[0]

    return obsvbl


def observe_init(a, ptcl, obsvbl, cosmo, conf):
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


def observe_adj(a_prev, a_next, ptcl, ptcl_cot, obsvbl, obsvbl_cot, cosmo, cosmo_cot, conf):

    def itp_cond_adj(carry, x):
        ptcl_cot, cosmo_cot = carry
        a_snap, a_step, snap_cot = x

        ptcl_cot, cosmo_cot = cond(jnp.isclose(a_step[1], a_next), itp_next_adj,
                                   lambda *args: (ptcl_cot, cosmo_cot),
                                   ptcl_cot, cosmo_cot, snap_cot, ptcl,
                                   a_step[0], a_step[1], a_snap, cosmo)

        ptcl_cot, cosmo_cot = cond(jnp.isclose(a_step[0], a_next), itp_prev_adj,
                                   lambda *args: (ptcl_cot, cosmo_cot),
                                   ptcl_cot, cosmo_cot, snap_cot, ptcl,
                                   a_step[0], a_step[1], a_snap, cosmo)

        return (ptcl_cot, cosmo_cot), None

    if conf.a_snapshots is not None:
        ptcl_cot, cosmo_cot = scan(itp_cond_adj, (ptcl_cot, cosmo_cot),
                                   (obsvbl['a_snaps'], obsvbl['itp_a_step'],
                                   obsvbl_cot['snaps']))[0]

    return ptcl_cot, cosmo_cot


def observe_adj_init(a, ptcl, ptcl_cot, obsvbl, obsvbl_cot, cosmo, cosmo_cot, conf):

    def itp_cond_adj(carry, x):
        ptcl_cot, cosmo_cot = carry
        a_snap, a_step, snap_cot = x
        ptcl_cot, cosmo_cot = cond(jnp.isclose(a_step[1], a), itp_next_adj,
                                   lambda *args: (ptcl_cot, cosmo_cot),
                                   ptcl_cot, cosmo_cot, snap_cot, ptcl,
                                   a_step[0], a_step[1], a_snap, cosmo)
        return (ptcl_cot, cosmo_cot), None

    if conf.a_snapshots is not None:
        # check if the last ptcl is used in interpolation
        ptcl_cot, cosmo_cot = scan(itp_cond_adj, (ptcl_cot, cosmo_cot),
                                   (obsvbl['a_snaps'], obsvbl['itp_a_step'],
                                   obsvbl_cot['snaps']))[0]

    return ptcl_cot, cosmo_cot
