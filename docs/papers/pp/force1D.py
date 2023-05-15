import numpy as np
import matplotlib.pyplot as plt


def force1D_sum(N, x=None, kernel=lambda k: 1, error=False):
    """Partial summed 1D force kernel, with cell size l=1."""
    n = np.arange(1, (N-1)//2 + 1)
    n = n[:, None]
    k = 2 * np.pi / N * n
    if x is None:
        x = np.arange(N)
    force = -2 * (kernel(k) / (k * N) * np.sin(k * x)).sum(axis=0)
    if error:
        force -= force1D(N, x=x)
    return force


def force1D_pm(N, d=None, x_src=0.5, kernel=lambda k: 1, error=False):
    """PM 1D force kernel, with cell size l=1."""
    freq_period = 2 * np.pi
    nyquist = np.pi
    eps = nyquist * np.finfo(np.float64).eps

    k = np.fft.rfftfreq(N) * freq_period

    # 1D scatter
    dens = np.zeros(N)
    i_src = np.floor(x_src).astype(int)
    t_src = x_src - i_src
    dens[i_src % N] = 1 - t_src
    dens[(i_src+1) % N] = t_src

    dens = np.fft.rfft(dens)
    pot = np.where(k != 0, - dens / k**2, 0)
    grad = np.where(np.abs(np.abs(k) - nyquist) <= eps, 0, -1j * k * pot)
    grad *= kernel(k)
    grad = np.fft.irfft(grad, n=N)

    # 1D gather
    if d is None:
        d = np.arange(N, dtype=float)
    x = x_src + d
    i = np.floor(x).astype(int)
    t = x - i
    grad = (1-t) * grad[i % N] + t * grad[(i+1) % N]

    if error:
        grad -= force1D(N, x=d)

    return grad


def force1D(N, x=None):
    """Analytic 1D force kernel, with cell size l=1."""
    if x is None:
        x = np.arange(N)
    return x / N - 0.5


def force1D_corr(k):
    """1D force correction kernel, with cell size l=1."""
    return np.where(k != 0, 0.5 * k / np.tan(0.5 * k), 1)


plt.style.use('font.mplstyle')

rows = ['1D force', 'force error']
cols = [16, 15] #, 126, 125]

fig, axes = plt.subplots(
    nrows=len(rows),
    ncols=len(cols),
    sharex='col',
    sharey='row',
    figsize=(2 * len(cols), 2 * len(rows)),
    gridspec_kw={
        'top': 1,
        'left': 0,
        'right': 1,
        'bottom': 0,
        'wspace': 0,
        'hspace': 0,
    },
)

def plt_panel(ax, N, error, check_pm, legend=False, refine=16):
    x = np.linspace(0, N, num=N * refine, endpoint=False)
    xm = np.arange(N)

    ax.plot(x, force1D_sum(N, x=x, error=error), 'C0--', lw=2, alpha=0.7)
    ax.scatter(xm, force1D_sum(N, error=error), c='C0', marker='s', s=6, lw=1, zorder=2, label='partial sum' if legend else None)
    if check_pm:
        ax.scatter(xm, force1D_pm(N, x_src=0, error=error), c='C0', marker='D', s=6, lw=1, zorder=2, label='particle mesh' if legend else None)
        ax.plot(x, force1D_pm(N, d=x, kernel=force1D_corr, error=error), 'C2:', lw=1.5)
        ax.scatter(xm, force1D_pm(N, kernel=force1D_corr, error=error), c='C2', marker='D', s=9, lw=1, zorder=2, label='PM corrected' if legend else None)
    ax.plot(x, force1D_sum(N, x=x, kernel=force1D_corr, error=error), 'C1-', lw=1)
    ax.scatter(xm, force1D_sum(N, kernel=force1D_corr, error=error), c='C1', marker='o', s=6, lw=1, zorder=2, label='corrected' if legend else None)
    if legend:
        ax.legend()
    ax.set_xlim(0, N)

for i, ylabel in enumerate(rows):
    for j, N in enumerate(cols):
        plt_panel(axes[i, j], N, error=i==1, check_pm=False, legend=i==0 and j==0)
        if i == 0:
            axes[i, j].set_ylim(-0.57, 0.57)
            axes[i, j].set_title(f'$N={N}$')
        else:
            axes[i, j].set_ylim(-0.7, 0.7)
            axes[i, j].set_yscale('symlog', linthresh=1e-3, linscale=0.5)
            axes[i, j].set_xlabel('$x/l$')
        if j == 0:
            axes[i, j].set_ylabel(ylabel)
fig.savefig('force1D.pdf')
plt.close(fig)
