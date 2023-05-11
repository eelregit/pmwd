import os

import numpy as np
import matplotlib.pyplot as plt

from pmwd.sto.train_util import scale_Sobol


def gen_sobol(filename, d=9, m=9, extra=9, seed=55868, seed_max=65536):
    from scipy.stats.qmc import Sobol, discrepancy

    nicer_seed = seed
    if seed is None:
        disc_min = np.inf
        for s in range(seed_max):
            sampler = Sobol(d, scramble=True, seed=s)  # d is the dimensionality
            sample = sampler.random_base2(m)  # m is the log2 of the number of samples
            disc = discrepancy(sample, method='MD')
            if disc < disc_min:
                nicer_seed = s
                disc_min = disc
        print(f'0 <= seed = {nicer_seed} < {seed_max}, minimizes mixture discrepancy = '
                f'{disc_min}')
        # nicer_seed = 55868, mixture discrepancy = 0.016109347957680598

    sampler = Sobol(d, scramble=True, seed=nicer_seed)
    sample = sampler.random(n=2**m + extra)  # extra is the additional testing samples
    np.savetxt(filename, sample)


def plt_proj(filename, max_rows=None, max_cols=None):
    usecols = range(max_cols) if isinstance(max_cols, int) else max_cols
    sample = np.loadtxt(filename, usecols=usecols, max_rows=max_rows)

    n, d = sample.shape

    axsize = 0.8
    fig, axes = plt.subplots(
        nrows=d,
        ncols=d,
        sharex=True,
        sharey=True,
        squeeze=False,
        subplot_kw={
            'box_aspect': 1,
            'xlim': [0, 1],
            'ylim': [0, 1],
            'xticks': [],
            'yticks': [],
        },
        gridspec_kw={
            'top': 1,
            'left': 0,
            'right': 1,
            'bottom': 0,
            'wspace': 0,
            'hspace': 0,
        },
        figsize=(d * axsize,) * 2,
    )

    for i in range(d):
        for j in range(i):
            axes[i, j].scatter(* sample.T[[j, i]],
                               s=2, marker='o', alpha=0.75, linewidth=0)

        axes[i, i].hist(sample[:, i], bins='sqrt', range=(0, 1),
                        density=True, cumulative=True, histtype='step')

        for j in range(i+1, d):
            axes[i, j].remove()

    filename = os.path.splitext(filename)[0] + str(n) + '.pdf'
    fig.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def plt_scaled(filename):
    sample = scale_Sobol(filename).T
    assert sample.shape[0] == 9

    fig, axes = plt.subplots(nrows=3, ncols=3, sharey=True, figsize=(9, 9),
                             subplot_kw={'box_aspect': 1})

    params = np.array([
        ('box size in Mpc', 'log', 25.6, 2560),
        ('snapshot offset $\Delta a$', 'linear', 0, 1/128),
        (r'$A_\mathrm{s} \times 10^9$', 'log', 1, 4),
        ('$n_\mathrm{s}$', 'log', 0.75, 1.25),
        ('$\Omega_\mathrm{m}$', 'log', 1/5, 1/2),
        ('$\Omega_\mathrm{b}$', 'log', 1/40, 1/8),
        ('$\Omega_k$', 'linear', -1/2, 1/4),
        ('$h$', 'log', 0.5, 1),
        ('softening ratio', 'log', 1/50, 1/20),
    ], dtype=[('xlabel', '<U39'), ('xscale', '<U6'), ('xmin', 'f8'), ('xmax', 'f8')])

    sample = sample.reshape(3, 3, -1)
    params = params.reshape(3, 3)

    for i in range(3):
        for j in range(3):
            xlabel, xscale, xmin, xmax = params[i, j]

            if xscale == 'log':
                bins = np.logspace(np.log10(xmin), np.log10(xmax), num=17)
            elif xscale == 'linear':
                bins = np.linspace(xmin, xmax, num=17)
            else:
                raise ValueError

            axes[i, j].hist(sample[i, j], bins=bins, histtype='step')
            axes[i, j].set_xlabel(xlabel)
            axes[i, j].set_xscale(xscale)

    filename = os.path.splitext(filename)[0] + '_scaled.pdf'
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    filename = 'sobol.txt'

    if not os.path.exists(filename):
        gen_sobol(filename)

    plt.style.use('font.mplstyle')

    plt_proj(filename, max_rows=8)
    plt_proj(filename, max_rows=64)
    plt_proj(filename, max_rows=512)

    plt_scaled(filename)
