import os
import sys

import numpy as np
import matplotlib.pyplot as plt


def plt_proj(filename='sobol.txt'):
    sample = np.loadtxt(filename)

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

        axes[i, i].hist(sample[:, i], bins='sqrt',
                        density=True, cumulative=True, histtype='step')

        for j in range(i+1, d):
            axes[i, j].remove()

    filename = os.path.splitext(filename)[0] + '.pdf'
    fig.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


if __name__ == '__main__':
    plt_proj(* sys.argv[1:])
