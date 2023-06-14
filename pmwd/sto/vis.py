import jax.numpy as jnp
import matplotlib.pyplot as plt

from pmwd.nbody import nbody
from pmwd.particles import Particles
from pmwd.vis_util import simshow, CosmicWebNorm
from pmwd.pm_util import rfftnfreq
from pmwd.sto.so import sotheta, sonn_bc
from pmwd.sto.train import init_pmwd, pmodel
from pmwd.sto.util import scatter_dens, power_tfcc


def plt_tf(k, tf, ylim=(-1.1, 1.1)):
    """Plot the transfer function."""
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.6), tight_layout=True)
    ax.plot(k, tf - 1)
    ax.set_ylabel(r'$T-1$')
    ax.set_yscale('symlog', linthresh=0.01)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r'$k$')
    ax.set_xscale('log')
    ax.set_xlim(k[0], k[-1])
    # ax.axhline(y=0, ls='--', c='grey')
    ax.grid(c='grey', alpha=0.5, ls=':')
    return fig


def plt_cc(k, cc, ylim=(-0.011, 1.1)):
    """Plot the correlation coefficient."""
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.6), tight_layout=True)
    ax.plot(k, 1 - cc**2)
    ax.set_ylabel(r'$1-r^2$')
    ax.set_yscale('symlog', linthresh=0.01)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r'$k$')
    ax.set_xscale('log')
    ax.set_xlim(k[0], k[-1])
    # ax.axhline(y=0, ls='--', c='grey')
    ax.grid(c='grey', alpha=0.5, ls=':')
    return fig


def plt_sofunc(nid, k, cosmo, conf):
    """Plot the SO function given k. nid: 0:f, 1:g, 2:h."""
    nid_dic = {0: 'f', 1: 'g', 2: 'h'}

    theta = sotheta(cosmo, conf, conf.a_out)
    sout = sonn_bc(k, theta, cosmo, conf, nid)

    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.6), tight_layout=True)
    ax.plot(k, sout)
    ax.axhline(y=1, ls='--', c='grey')
    if nid == 1:
        ax.set_xscale('log')
    else:
        ax.set_xscale('symlog')
    ax.set_xlabel(r'$k$')
    ax.set_title(f'{nid_dic[nid]} net')

    return fig


def vis_inspect(tgt, so_params, pmwd_params, vis_mesh_shape=1,
                p_power=True, p_sofunc=False, p_slab=False):
    # run pmwd with given params
    ptcl_ic, cosmo, conf = init_pmwd(pmwd_params)
    ptcl, cosmo = pmodel(ptcl_ic, so_params, cosmo, conf)

    nyquist = jnp.pi / conf.cell_size

    # get the target ptcl
    pos_t, vel_t = tgt
    disp_t = pos_t - ptcl.pmid * conf.cell_size
    ptcl_t = Particles(conf, ptcl.pmid, disp_t, vel_t)

    figs = {}

    # plot power spectra
    if p_power:
        (dens, dens_t), (vis_mesh_shape, cell_size) = scatter_dens(
                                            (ptcl, ptcl_t), conf, vis_mesh_shape)
        k, tf, cc = power_tfcc(dens, dens_t, cell_size)
        figs['tf'] = plt_tf(k, tf)
        figs['cc'] = plt_cc(k, cc)

    # plot SO functions
    # k sample points to evaluate the functions
    if p_sofunc:
        kvec = rfftnfreq(conf.mesh_shape, conf.cell_size, dtype=conf.float_dtype)
        k_1d = jnp.sort(kvec[0].ravel())
        k_min = kvec[0].ravel()[1]
        k_max = jnp.sqrt(3 * jnp.abs(k_1d).max()**2)
        k_3d = jnp.logspace(jnp.log10(k_min), jnp.log10(k_max), 1000)
        for nid, n, k in zip([0, 1, 2], ['f', 'g', 'h'], [k_1d, k_3d, k_1d]):
            if conf.so_nodes[nid] is not None:
                figs[f'{n}_net'] = plt_sofunc(nid, k, cosmo, conf)

    # plot the density slab
    if p_slab:
        norm = CosmicWebNorm(dens_t)
        slab = int(dens.shape[0] * 0.2)
        figs['dens_target'] = simshow(dens_t[:slab].mean(axis=0), norm=norm)[0]
        figs['dens_target'].tight_layout()
        figs['dens'] = simshow(dens[:slab].mean(axis=0), norm=norm)[0]
        figs['dens'].tight_layout()

    return figs
