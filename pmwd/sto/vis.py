import jax.numpy as jnp
import matplotlib.pyplot as plt

from pmwd.nbody import nbody
from pmwd.particles import Particles
from pmwd.spec_util import powspec
from pmwd.vis_util import simshow, CosmicWebNorm
from pmwd.pm_util import rfftnfreq
from pmwd.sto.so import sotheta, sonn_bc
from pmwd.sto.train import _init_pmwd
from pmwd.sto.util import ptcl2dens


def plt_power(dens, dens_t, cell_size):
    """Plot power spectra related."""
    # estimate power spectra
    k, ps, N = powspec(dens, cell_size)
    ps = ps.real
    k, ps_t, N = powspec(dens_t, cell_size)
    ps_t = ps_t.real
    k, ps_cross, N = powspec(dens, cell_size, g=dens_t)
    ps_cross = ps_cross.real

    # the transfer function and correlation coefficient
    tf = jnp.sqrt(ps / ps_t)
    cc = ps_cross / jnp.sqrt(ps * ps_t)

    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.6), tight_layout=True)
    ax.plot(k, tf, label=r'trans. func.')
    ax.plot(k, cc, label=r'corr. coef.')
    ax.axhline(y=1, ls='--', c='grey')
    ax.set_xscale('log')
    ax.set_xlabel(r'$k$')
    ax.set_xlim(k[0], k[-1])
    ax.set_ylim(0.7, 1.3)
    ax.legend()

    return fig


def plt_sofuncs(nid, k, cosmo, conf):
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


def vis_inspect(tgt, so_params, pmwd_params, mesh_shape=3):
    # run pmwd with given params
    ptcl_ic, cosmo, conf = _init_pmwd(pmwd_params)
    cosmo = cosmo.replace(so_params=so_params)
    _, obsvbl = nbody(ptcl_ic, None, cosmo, conf)
    ptcl = obsvbl[0]

    # get the target ptcl
    pos_t, vel_t = tgt
    disp_t = pos_t - ptcl.pmid * conf.cell_size
    ptcl_t = Particles(conf, ptcl.pmid, disp_t, vel_t)

    figs = {}

    # plot power spectra
    (dens, dens_t), (mesh_shape, cell_size) = ptcl2dens(
                                               (ptcl, ptcl_t), conf, mesh_shape)
    kvec_dens = rfftnfreq(mesh_shape, cell_size, dtype=conf.float_dtype)
    figs['power'] = plt_power(dens, dens_t, cell_size)

    # plot SO functions
    # k sample points to evaluate the functions
    kvec = rfftnfreq(conf.mesh_shape, conf.cell_size, dtype=conf.float_dtype)
    k_1d = jnp.sort(kvec[0].ravel())
    k_min = kvec[0].ravel()[1]
    k_max = jnp.sqrt(3 * jnp.abs(k_1d).max()**2)
    k_3d = jnp.logspace(jnp.log10(k_min), jnp.log10(k_max), 1000)
    for nid, n, k in zip([0, 1, 2], ['f', 'g', 'h'], [k_1d, k_3d, k_1d]):
        if conf.so_nodes[nid] is not None:
            figs[f'{n}_net'] = plt_sofuncs(nid, k, cosmo, conf)

    # plot the density slab
    norm = CosmicWebNorm(dens_t)
    figs['dens_target'] = simshow(dens_t[:16].mean(axis=0), norm=norm)[0]
    figs['dens_target'].tight_layout()
    figs['dens'] = simshow(dens[:16].mean(axis=0), norm=norm)[0]
    figs['dens'].tight_layout()


    return figs