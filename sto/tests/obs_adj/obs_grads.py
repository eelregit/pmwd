import os
import sys

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.lax import scan
import matplotlib.pyplot as plt

from pmwd import (
    Configuration,
    SimpleLCDM,
    boltzmann,
    white_noise, linear_modes,
    lpt,
    nbody,
    scatter,
)
from pmwd.vis_util import simshow


def gen_ic(modes, cosmo, conf):
    modes = linear_modes(modes, cosmo, conf)
    ptcl, _ = lpt(modes, cosmo, conf)
    return ptcl


def model(ptcl, cosmo, conf):
    ptcl, obsvbl = nbody(ptcl, None, cosmo, conf)  # obsvbl init in nbody
    def _scatter(_, x):
        ptcl = x
        dens = scatter(ptcl, conf)
        return None, dens
    # make dens for all obsvbl snapshots
    _, dens = scan(_scatter, None, obsvbl['snaps'])
    return dens


def obj(tgt_dens, ptcl, cosmo, conf):
    dens = model(ptcl, cosmo, conf)
    return (dens - tgt_dens).var()

obj_grad = jax.grad(obj, argnums=(1, 2), allow_int=True)


ptcl_spacing = 1.
ptcl_grid_shape = (32,) * 3
conf = Configuration(ptcl_spacing, ptcl_grid_shape, mesh_shape=2,
                     a_start=1/16, a_stop=2/16, a_nbody_num=1,
                     a_snapshots=(1.5/16,))  # set observable snapshots

cosmo = SimpleLCDM(conf)
cosmo = boltzmann(cosmo, conf)


# control the target dens variation
fname = 'dens.npy'
if not os.path.exists(fname):
    seed = 0  # seed for target
    modes = white_noise(seed, conf, real=True)
    ptcl = gen_ic(modes, cosmo, conf)
    dens = model(ptcl, cosmo, conf)  # target density
    jnp.save(fname, dens)
dens = jnp.load(fname)


# control the input modes variation
fname = 'modes.npy'
if not os.path.exists(fname):
    seed = 1
    modes = white_noise(seed, conf, real=True)
    jnp.save(fname, modes)
modes = jnp.load(fname)
ptcl = gen_ic(modes, cosmo, conf)


n = 3
fname_am = 'grads_am{}.npy'  # adjoint mode gradients
fname_ad = 'grads_ad{}.npy'  # AD mode gradients

if not os.path.exists(fname_am.format(0)):  # adjoint gradients
    print('#### adjoint method ####')
    for i in range(n):
        ptcl_cot, cosmo_cot = obj_grad(dens, ptcl, cosmo, conf)
        jnp.save(fname_am.format(i), ptcl_cot.disp.ravel())
        print(cosmo_cot.Omega_m)
elif not os.path.exists(fname_ad.format(0)):  # AD gradients
    # HACK for AD: commenting out custom_vjp and defvjp on scatter, gather, and nbody
    print('#### AD ####')
    for i in range(n):
        ptcl_cot, cosmo_cot = obj_grad(dens, ptcl, cosmo, conf)
        jnp.save(fname_ad.format(i), ptcl_cot.disp.ravel())
        print(cosmo_cot.Omega_m)
else:  # making plots
    gam = np.stack([np.load(fname_am.format(i)) for i in range(n)], axis=0)
    gad = np.stack([np.load(fname_ad.format(i)) for i in range(n)], axis=0)

    from matplotlib.colors import SymLogNorm, LogNorm
    plt.style.use('adjoint.mplstyle')

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    bins = np.linspace(gam.min(), gam.max(), num=100, endpoint=True)
    ax.hist2d(gam.ravel(), gad.ravel(), bins=[bins, bins], cmap='binary', norm=LogNorm())
    ax.set_yticks(ax.get_xticks())
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(bins[0], bins[-1])
    ax.set_xlabel('AD grad')
    ax.set_ylabel('adjoint grad')
    fig.savefig('cots_cmp.pdf')
    plt.close(fig)

    def diffpair(g0, g1):
        g0 = g0.reshape(n, np.prod(ptcl_grid_shape) * 3)
        g1 = g1.reshape(n, np.prod(ptcl_grid_shape) * 3)
        gd = np.zeros((n * (n-1) // 2, g0.shape[1]), dtype=g0.dtype)
        for i in range(n):
            for j in range(i):
                ind  = i * (i-1) // 2 + j
                gd[ind] = g1[j] - g0[i]
        return gd.ravel()

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    # bins = np.linspace(-6.4e-4, 6.4e-4, num=129, endpoint=True)
    kwargs = dict(bins=100, histtype='step', joinstyle='round', capstyle='round')
    gd = diffpair(gam, gam)
    print('adj-adj:', gd.std())
    ax.hist(gd, color='tab:blue', ls='-', lw=1, alpha=0.5, label='adj-adj', **kwargs)
    gd = diffpair(gam, gad)
    print('adj-AD: ', gd.std())
    ax.hist(gd, color='k', ls='-', lw=1, alpha=0.5, label='adj-AD', **kwargs)
    gd = diffpair(gad, gad)
    print('AD-AD:  ', gd.std())
    ax.hist(gd, color='tab:orange', ls='-', lw=1, alpha=0.5, label='AD-AD', **kwargs)
    # ax.set_xlim(bins[0], bins[-1])
    # ax.set_ylim(2e4, 5e9)
    ax.set_yscale('log')
    ax.ticklabel_format(axis='x', scilimits=(0, 0))
    ax.tick_params(axis='y', which='minor', left=False, right=False)
    ax.set_xlabel('grad diff')
    ax.set_ylabel('counts')
    ax.legend(loc=2, handlelength=0.5, handletextpad=0.5)
    fig.savefig('cots_diff.pdf')
    plt.close(fig)
