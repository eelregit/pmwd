"""Some util functions for post-training tests and validation."""
import pickle
import matplotlib.pyplot as plt

from pmwd.nbody import nbody
from pmwd.sto.train import pmodel, init_pmwd
from pmwd.sto.data import G4snapDataset
from pmwd.sto.ccic import gen_cc, gen_ic
from pmwd.sto.mlp import mlp_size
from pmwd.sto.util import power_tfcc, scatter_dens, pv2ptcl, load_soparams, tree_unstack


def pmwd_fwd(so_params, sidx, sobol, a_snaps, mesh_shape, n_steps, so_type,
             so_nodes, soft_i):
    """End-to-end forward run of pmwd with SO."""
    conf, cosmo = gen_cc(sobol, mesh_shape=mesh_shape, a_snapshots=a_snaps,
                         a_nbody_num=n_steps, so_type=so_type, so_nodes=so_nodes,
                         soft_i=soft_i)
    ptcl_ic = gen_ic(sidx, conf, cosmo)
    cosmo = cosmo.replace(so_params=so_params)
    _, obsvbl = nbody(ptcl_ic, None, cosmo, conf)
    return obsvbl, cosmo, conf


def test_snap(tgt, pmwd_params, so_params, vis_mesh_shape, vis_cut_nyq):
    ptcl_ic, cosmo, conf = init_pmwd(pmwd_params)
    a = pmwd_params[0][0]

    # run pmwd w/ and w/o optimization
    ptcl = pmodel(ptcl_ic, so_params, cosmo, conf)[0]['snaps']
    ptcl = tree_unstack(ptcl)[0]
    conf = conf.replace(so_type=None)
    ptcl_o = pmodel(ptcl_ic, so_params, cosmo, conf)[0]['snaps']
    ptcl_o = tree_unstack(ptcl_o)[0]

    ptcl_t = pv2ptcl(*tgt, ptcl.pmid, ptcl.conf)

    (dens, dens_o, dens_t), cell_size = scatter_dens(
                                    (ptcl, ptcl_o, ptcl_t), conf, vis_mesh_shape)

    # compare the tf and cc
    k, tf, cc = power_tfcc(dens, dens_t, cell_size, cut_nyq=vis_cut_nyq)
    k, tf_o, cc_o = power_tfcc(dens_o, dens_t, cell_size, cut_nyq=vis_cut_nyq)

    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.6), tight_layout=True)
    ax.plot(k, tf, c='tab:blue', label=r'$T$, w/ SO')
    ax.plot(k, cc, c='tab:orange', label=r'$r$, w/ SO')
    ax.plot(k, tf_o, ls='--', c='tab:blue', label=r'$T$, w/o SO')
    ax.plot(k, cc_o, ls='--', c='tab:orange', label=r'$r$, w/o SO')
    ax.set_xlabel(r'$k$ [$h$/Mpc]')
    ax.set_xscale('log')
    ax.set_xlim(k[0], k[-1])
    ax.set_ylim(0.4, 1.5)
    ax.grid(c='grey', alpha=0.5, ls=':')
    ax.legend(ncols=2, frameon=False)

    return fig, ax


def test_so(so_params, sobol_ids, snap_ids, g4sims_dir='../g4sims',
            mesh_shape=1, n_steps=100, so_type='NN', vis_mesh_shape=1,
            vis_cut_nyq=False):
    # load the g4data
    print('loading gadget4 data')
    g4data = G4snapDataset(g4sims_dir, sobol_ids, snap_ids)

    # trained so_params
    print('preparing so parameters')
    so_params, n_input, so_nodes = load_soparams(so_params)

    # compare
    print('generating the figures')
    figs = []
    for sidx in sobol_ids:
        for snap in snap_ids:
            pos, vel, a, sidx, sobol, snap_id = g4data.getsnap(sidx, snap)
            tgt = (pos, vel)
            pmwd_params = ((a,), sidx, sobol, mesh_shape, n_steps, so_type, so_nodes, None, None)
            fig, ax = test_snap(tgt, pmwd_params, so_params, vis_mesh_shape, vis_cut_nyq)
            ax.set_title(f'Sobol: {sidx}')
            figs.append(fig)

    return figs
