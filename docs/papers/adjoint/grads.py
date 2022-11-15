import os

import numpy as np
import jax
import jax.numpy as jnp
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


def model(modes, cosmo, conf):
    cosmo = boltzmann(cosmo, conf)
    modes = linear_modes(modes, cosmo, conf)
    ptcl, obsvbl = lpt(modes, cosmo, conf)
    ptcl, obsvbl = nbody(ptcl, obsvbl, cosmo, conf)
    dens = scatter(ptcl, conf)
    return dens


def obj(tgt_dens, modes, cosmo, conf):
    cosmo = boltzmann(cosmo, conf)
    modes = linear_modes(modes, cosmo, conf)
    ptcl, obsvbl = lpt(modes, cosmo, conf)
    ptcl, obsvbl = nbody(ptcl, obsvbl, cosmo, conf)
    dens = scatter(ptcl, conf)
    return (dens - tgt_dens).var()

obj_grad = jax.grad(obj, argnums=(1, 2))


ptcl_spacing = 1.
ptcl_grid_shape = (128,) * 3
conf = Configuration(ptcl_spacing, ptcl_grid_shape, mesh_shape=2,
                     a_start=1/16, a_nbody_maxstep=1/16)

cosmo = SimpleLCDM(conf)


# control the target dens variation
fname = 'dens.npy'
if not os.path.exists(fname):
    seed = 0  # seed for target
    modes = white_noise(seed, conf)

    dens = model(modes, cosmo, conf)  # target density
    jnp.save(fname, dens)
dens = jnp.load(fname)


# control the input modes variation
fname = 'modes.npy'
if not os.path.exists(fname):
    seed = 1
    modes = white_noise(seed, conf, real=True)
    jnp.save(fname, modes)
modes = jnp.load(fname)


n = 8
fname_am = 'grads_am{}.npy'  # adjoint mode gradients
fname_ad = 'grads_ad{}.npy'  # AD mode gradients

if not os.path.exists(fname_am.format(0)):  # adjoint gradients
    print('#### adjoint method ####')
    for i in range(n):
        modes_grad, cosmo_grad = obj_grad(dens, modes, cosmo, conf)
        jnp.save(fname_am.format(i), modes_grad)
        print(cosmo_grad)
elif not os.path.exists(fname_ad.format(0)):  # AD gradients
    # HACK for AD: commenting out custom_vjp and defvjp on scatter, gather, and nbody
    print('#### AD ####')
    for i in range(n):
        modes_grad, cosmo_grad = obj_grad(dens, modes, cosmo, conf)
        jnp.save(fname_ad.format(i), modes_grad)
        print(cosmo_grad)
else:  # making plots
    gam = np.stack([np.load(fname_am.format(i)) for i in range(n)], axis=0)
    gad = np.stack([np.load(fname_ad.format(i)) for i in range(n)], axis=0)

    from matplotlib.colors import SymLogNorm, LogNorm
    plt.style.use('font.mplstyle')

    fig, _ = simshow(gam[0, 32], cmap='RdBu_r',
                     norm=SymLogNorm(0.01, vmin=-0.1, vmax=0.1), colorbar=False)
    fig.savefig('grads.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    bins = np.linspace(-0.32, 0.32, num=65, endpoint=True)
    ax.hist2d(gam.ravel(), gad.ravel(), bins=[bins, bins], cmap='binary', norm=LogNorm())
    ax.set_yticks(ax.get_xticks())
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(bins[0], bins[-1])
    ax.set_xlabel('AD grad')
    ax.set_ylabel('adjoint grad')
    fig.savefig('grads_cmp.pdf')
    plt.close(fig)

    def diffpair(g0, g1):
        g0 = g0.reshape(n, np.prod(ptcl_grid_shape))
        g1 = g1.reshape(n, np.prod(ptcl_grid_shape))
        gd = np.zeros((n * (n-1) // 2, g0.shape[1]), dtype=g0.dtype)
        for i in range(n):
            for j in range(i):
                ind  = i * (i-1) // 2 + j
                gd[ind] = g1[j] - g0[i]
        return gd.ravel()

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    bins = np.linspace(-6.4e-4, 6.4e-4, num=129, endpoint=True)
    kwargs = dict(bins=bins, histtype='step', joinstyle='round',  capstyle='round')
    gd = diffpair(gam, gam)
    print('adj-adj:', gd.std())
    ax.hist(gd, color='gray', ls='-', lw=1, label='adj-adj', **kwargs)
    gd = diffpair(gam, gad)
    print('adj-AD: ', gd.std())
    ax.hist(gd, color='k', ls='-', lw=1, label='adj-AD', **kwargs)
    gd = diffpair(gad, gad)
    print('AD-AD:  ', gd.std())
    ax.hist(gd, color='gray', ls=':', lw=1.5, label='AD-AD', **kwargs)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(5e2, 5e7)
    ax.set_yscale('log')
    ax.ticklabel_format(axis='x', scilimits=(0, 0))
    ax.tick_params(axis='y', which='minor', left=False, right=False)
    ax.set_xlabel('grad diff')
    ax.set_ylabel('counts')
    ax.legend(loc=2, handlelength=0.5, handletextpad=0.5)
    fig.savefig('grads_diff.pdf')
    plt.close(fig)
