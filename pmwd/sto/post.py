"""Some util functions for post-training tests and validation."""
import pickle
import matplotlib.pyplot as plt

from pmwd.sto.train import pmodel, init_pmwd
from pmwd.sto.data import G4snapDataset
from pmwd.sto.mlp import mlp_size
from pmwd.sto.util import power_tfcc, scatter_dens, pv2ptcl


def test_snap(tgt, pmwd_params, so_params, vis_mesh_shape):
    ptcl_ic, cosmo, conf = init_pmwd(pmwd_params)

    # run pmwd w/ and w/o optimization
    ptcl, _ = pmodel(ptcl_ic, so_params, cosmo, conf)
    conf = conf.replace(so_type=None)
    ptcl_o, _ = pmodel(ptcl_ic, so_params, cosmo, conf)

    ptcl_t = pv2ptcl(*tgt, ptcl.pmid, ptcl.conf)

    (dens, dens_o, dens_t), (vis_mesh_shape, cell_size) = scatter_dens(
                                    (ptcl, ptcl_o, ptcl_t), conf, vis_mesh_shape)

    # compare the tf and cc
    k, tf, cc = power_tfcc(dens, dens_t, cell_size)
    k, tf_o, cc_o = power_tfcc(dens_o, dens_t, cell_size)

    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.6), tight_layout=True)
    ax.plot(k, tf, c='tab:blue', label=r'$T$, w/ SO')
    ax.plot(k, cc, c='tab:orange', label=r'$r$, w/ SO')
    ax.plot(k, tf_o, ls='--', c='tab:blue', label=r'$T$, w/o SO')
    ax.plot(k, cc_o, ls='--', c='tab:orange', label=r'$r$, w/o SO')
    ax.set_xlabel(r'$k$')
    ax.set_xscale('log')
    ax.set_xlim(k[0], k[-1])
    ax.set_ylim(0.5, 1.5)
    ax.grid(c='grey', alpha=0.5, ls=':')
    ax.legend()

    return fig


def test_so(so_params, sobol_ids, snap_ids, g4sims_dir='../g4sims',
            mesh_shape=128, n_steps=100, so_type=2, vis_mesh_shape=1):
    # load the g4data
    print('loading gadget4 data')
    g4data = G4snapDataset(g4sims_dir, sobol_ids, snap_ids)

    # trained so_params
    print('preparing so parameters')
    if isinstance(so_params, str):
        with open(so_params, 'rb') as f:
            so_params = pickle.load(f)['so_params']
    n_input, so_nodes = mlp_size(so_params)

    # compare
    print('generating the figures')
    figs = []
    for sidx in sobol_ids:
        for snap in snap_ids:
            pos, vel, a, sidx, sobol, snap_id = g4data.getsnap(sidx, snap)
            tgt = (pos, vel)
            pmwd_params = (a, sidx, sobol, mesh_shape, n_steps, so_type, so_nodes)
            figs.append(test_snap(tgt, pmwd_params, so_params, vis_mesh_shape))

    return figs
