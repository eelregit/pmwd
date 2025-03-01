from functools import partial
import math

from jax import jit
import jax.numpy as jnp

from pmwd.pm_util import fftfreq, fftfwd


def _getbins(grid_shape, bins, cut_nyq):
    kfun = 1 / max(grid_shape)
    knyq = 0.5
    kmax = knyq * math.sqrt(3)

    # first bin only contains the DC mode, and thus the "1 + " in front of bnum and bcut
    if isinstance(bins, (int, float)):
        bins *= kfun
        bnum = 1 + math.ceil(kmax / bins)
        bcut = 1 + math.ceil(knyq / bins) if cut_nyq else bnum
        bins *= jnp.arange(bnum)
        right = True

    elif isinstance(bins, complex):
        kmaxable = all(s % 2 == 0 for s in grid_shape)  # extra bin just in case
        bnum = 1 + math.ceil(math.log2(kmax / kfun) / bins.imag) + kmaxable
        bcut = 1 + math.ceil(math.log2(knyq / kfun) / bins.imag) if cut_nyq else bnum
        bins = kfun * 2 ** (bins.imag * jnp.arange(bnum))
        right = False

    elif isinstance(bins, tuple):
        if bins[0] != 0:
            raise ValueError(f'{bins=} must starts from 0')

        bnum = len(bins)
        if cut_nyq:
            for bcut, edge in enumerate(bins, start=1):
                if edge >= knyq:
                    break
        else:
            bcut = bnum
        bins = jnp.asarray(bins)
        right = True

    else:
        raise ValueError(f'{bins=} not supported')

    return bnum, bcut, bins, right


@partial(jit, static_argnames=('bins', 'cut_zero', 'cut_nyq', 'dtype', 'int_dtype'))
def powspec(f, spacing, bins=1j/3, g=None, deconv=None, cut_zero=True, cut_nyq=True,
            dtype=jnp.float64, int_dtype=jnp.uint32):
    """Compute auto or cross power spectrum in 3D averaged in spherical bins.

    Parameters
    ----------
    f : ArrayLike
        The field, with the last 3 axes for FFT and the other reduced by sum of squares.
    spacing : float
        Field grid spacing.
    bins : float, complex, or tuple, optional
        (Angular) wavenumber bins. A real number sets the linear bin width in unit of
        the smallest fundamental in 3D (right-edge inclusive and starting from 0); an
        imaginary number sets the log bin width in octave (left-edge inclusive and
        starting from the smallest fundamental in 3D); and a tuple sets the bin edges
        directly (increasing, right-edge inclusive, and must starts from 0) in unit of
        2x the Nyquist frequency (such that ``cut_nyq`` cuts beyond 0.5).
    g : ArrayLike, optional
        Another field of the same shape for cross correlation.
    deconv : int, optional
        Power of sinc factors to deconvolve in the power spectrum.
    cut_zero : bool, optional
        Whether to discard the bin containing the k=0 or DC mode.
    cut_nyq : bool, optional
        Whether to discard the bins beyond the Nyquist frequency.
    dtype : DTypeLike, optional
        Float dtype for the wavenumber and power spectrum.
    int_dtype : DTypeLike, optional
        Integer dtype for the number of modes.

    Returns
    -------
    k : jax.Array
        Wavenumber.
    P : jax.Array
        Power spectrum.
    N : jax.Array
        Number of modes.
    bins : jax.Array
        Wavenumber bins.

    """
    f = jnp.asarray(f)

    if g is not None and f.shape != jnp.shape(g):
        raise ValueError(f'shape mismatch: {f.shape} != {jnp.shape(g)}')
    grid_shape = f.shape[-3:]

    bnum, bcut, bins, right = _getbins(grid_shape, bins, cut_nyq)

    last_three = range(-3, 0)
    f = fftfwd(f, axes=last_three)
    if g is None:
        P = f.real**2 + f.imag**2
    else:
        g = jnp.asarray(g)
        g = fftfwd(g, axes=last_three)
        P = f * g.conj()

    if P.ndim > 3:
        P = P.sum(tuple(range(P.ndim-3)))

    kvec = fftfreq(grid_shape, None, dtype=P.real.dtype)
    k = jnp.sqrt(sum(k**2 for k in kvec))

    if deconv is not None:
        P = math.prod((jnp.sinc(k) ** -deconv for k in kvec), start=P)  # numpy sinc has pi

    #N = jnp.full_like(P, 2, dtype=jnp.uint8)
    N = jnp.full_like(P, 2, dtype=int_dtype)  # FIXME after google/jax/issues/18440
    N = N.at[..., 0].set(1)
    if grid_shape[-1] % 2 == 0:
        N = N.at[..., -1].set(1)

    k = k.ravel()
    P = P.ravel()
    N = N.ravel()
    b = jnp.digitize(k, bins, right=right)
    k = (k * N).astype(dtype)
    P = (P * N).astype(dtype)  # FIXME after google/jax/issues/18440
    k = jnp.bincount(b, weights=k, length=bnum)  # only k=0 goes to b=0
    P = jnp.bincount(b, weights=P, length=bnum)
    N = jnp.bincount(b, weights=N, length=bnum)

    k = k[cut_zero:bcut]
    P = P[cut_zero:bcut]
    N = N[cut_zero:bcut]
    bins = bins[:bcut]

    k /= N
    P /= N

    k *= 2 * math.pi / spacing
    bins *= 2 * math.pi / spacing
    P *= spacing**3 / math.prod(grid_shape)

    return k, P, N, bins
