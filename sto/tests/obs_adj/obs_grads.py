import os

import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import scan

from pmwd import (
    Configuration,
    Cosmology,
    Particles,
    boltzmann,
    white_noise, linear_modes,
    lpt,
    nbody,
    scatter,
)
from pmwd.sto.so.so import soft_len
from pmwd.sto.so.mlp import init_mlp_params


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


# setup the model
ptcl_spacing = 1.
ptcl_grid_shape = (16,) * 3
# SO neural net parameters
so_type = 'NN'
soft_i = 'soft_v2'
n_input = [soft_len(soft_i, 'g'), soft_len(soft_i, 'f')]
so_nodes = [[8, 1], [8, 1]]
so_params = init_mlp_params(n_input, so_nodes, scheme='last_ws')

conf = Configuration(ptcl_spacing, ptcl_grid_shape, mesh_shape=2,
                     a_start=1/16, a_stop=1, a_nbody_num=15, a_snapshots=(0.7, 0.8, 0.9),
                     so_type=so_type, so_nodes=so_nodes, soft_i=soft_i, softening_length=0.01)

cosmo = Cosmology(conf, A_s_1e9=2.0, n_s=0.96, Omega_m=0.3, Omega_b=0.05, h=0.7,
                  so_params=so_params)
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


# control the initial condition variation
fname = 'ptcl.npy'
if not os.path.exists(fname):
    seed = 1
    modes = white_noise(seed, conf, real=True)
    ptcl = gen_ic(modes, cosmo, conf)
    jnp.save(fname, jnp.array([ptcl.disp, ptcl.vel]))
disp, vel = jnp.load(fname)
ptcl = Particles.gen_grid(conf, vel=True)
ptcl = ptcl.replace(disp=disp, vel=vel)


n = 3

fname_am = 'grads/{}_grad_am{}.npy'  # adjoint mode gradients
fname_ad = 'grads/{}_grad_ad{}.npy'  # AD mode gradients

if not os.path.exists(fname_am.format('ptcl', 0)):
    print('#### adjoint method ####')
    for i in range(n):
        ptcl_cot, cosmo_cot = obj_grad(dens, ptcl, cosmo, conf)
        ptcl_grad = jnp.array([ptcl_cot.disp.ravel(), ptcl_cot.vel.ravel()])
        jnp.save(fname_am.format('ptcl', i), ptcl_grad)
        sonn_grad = jnp.concatenate([x.ravel() for x in jax.tree.leaves(cosmo_cot.so_params)])
        jnp.save(fname_am.format('sonn', i), sonn_grad)
elif not os.path.exists(fname_ad.format('ptcl', 0)):
    # HACK for AD: commenting out custom_vjp and defvjp on scatter, gather, and nbody
    print('#### AD ####')
    for i in range(n):
        ptcl_cot, cosmo_cot = obj_grad(dens, ptcl, cosmo, conf)
        ptcl_grad = jnp.array([ptcl_cot.disp.ravel(), ptcl_cot.vel.ravel()])
        jnp.save(fname_ad.format('ptcl', i), ptcl_grad)
        sonn_grad = jnp.concatenate([x.ravel() for x in jax.tree.leaves(cosmo_cot.so_params)])
        jnp.save(fname_ad.format('ptcl', i), sonn_grad)
else:  # making plots
    var = 'sonn'
    gam = np.stack([np.load(fname_am.format(var, i)) for i in range(n)], axis=0)
    gad = np.stack([np.load(fname_ad.format(var, i)) for i in range(n)], axis=0)

    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm, LogNorm
    plt.style.use('adjoint.mplstyle')

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    bins = np.linspace(gam.min(), gam.max(), num=100, endpoint=True)
    # bins = np.linspace(-1e-9, 1e-9, num=101, endpoint=True)
    ax.hist2d(gam.ravel(), gad.ravel(), bins=[bins, bins], cmap='binary', norm=LogNorm())
    ax.set_yticks(ax.get_xticks())
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(bins[0], bins[-1])
    ax.set_xlabel('AD grad')
    ax.set_ylabel('adjoint grad')
    fig.savefig(f'{var}_grad_cmp.pdf')
    plt.close(fig)

    def diffpair(g0, g1):
        g0 = g0.reshape(n, -1)
        g1 = g1.reshape(n, -1)
        gd = np.zeros((n * (n-1) // 2, g0.shape[1]), dtype=g0.dtype)
        for i in range(n):
            for j in range(i):
                ind  = i * (i-1) // 2 + j
                gd[ind] = g1[j] - g0[i]
        return gd.ravel()

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    bins = np.linspace(-5e-4, 5e-4, num=101, endpoint=True)
    kwargs = dict(bins=bins, histtype='step', joinstyle='round', capstyle='round')
    gd = diffpair(gam, gam)
    print('adj-adj:', gd.std())
    # ax.hist(gd, color='tab:blue', ls='-', lw=1, alpha=0.5, label='adj-adj', **kwargs)
    gd = diffpair(gam, gad)
    print('adj-AD: ', gd.std())
    ax.hist(gd, color='k', ls='-', lw=1, alpha=0.5, label='adj-AD', **kwargs)
    gd = diffpair(gad, gad)
    print('AD-AD:  ', gd.std())
    # ax.hist(gd, color='tab:orange', ls='-', lw=1, alpha=0.5, label='AD-AD', **kwargs)
    ax.set_xlim(bins[0], bins[-1])
    # ax.set_ylim(2e4, 5e9)
    ax.set_yscale('log')
    ax.ticklabel_format(axis='x', scilimits=(0, 0))
    ax.tick_params(axis='y', which='minor', left=False, right=False)
    ax.set_xlabel('grad diff')
    ax.set_ylabel('counts')
    ax.legend(loc=2, handlelength=0.5, handletextpad=0.5)
    fig.savefig(f'{var}_grad_diff.pdf')
    plt.close(fig)
