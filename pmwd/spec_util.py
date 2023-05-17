import math
from functools import reduce
from operator import mul

import jax.numpy as jnp

from pmwd.pm_util import rfftnfreq


def powspec(f, spacing, bins=1j/3, g=None, deconv=0, cut_zero=True, cut_nyq=True,
            int_dtype=jnp.uint32):
    """Compute auto or cross power spectrum in 3D averaged in spherical bins.

    Parameters
    ----------
    f : array_like
        The field, with the last 3 axes for FFT and the other summed over.
    spacing : float
        Field grid spacing.
    bins : float, complex, or 1D array_like, optional
        Wavenumber bins. A real number sets the linear spaced bin width in unit of the
        smallest fundamental in 3D (right edge inclusive starting from zero); an
        imaginary number sets the log spaced bin width in octave (left edge inclusive
        starting from the smallest fundamental in 3D); and an array sets the bin edges
        directly (right edge inclusive and must starting from zero).
    g : array_like, optional
        Another field of the same shape for cross correlation.
    deconv : int, optional
        Power of sinc factors to deconvolve in the power spectrum.
    cut_zero : bool, optional
        Whether to discard the bin containing the zero or DC mode.
    cut_nyq : bool, optional
        Whether to discard the bins beyond the Nyquist.
    int_dtype : dtype_like, optional
        Integer dtype for the number of modes.

    Returns
    -------
    k : jax.numpy.ndarray
        Wavenumber.
    P : jax.numpy.ndarray
        Power spectrum.
    N : jax.numpy.ndarray
        Number of modes.
    bins : jax.numpy.ndarray
        Wavenumber bins.

    """
    f = jnp.asarray(f)
    grid_shape = f.shape[-3:]

    if g is not None and f.shape != jnp.shape(g):
        raise ValueError(f'shape mismatch: {f.shape} != {jnp.shape(g)}')

    last_three = range(-3, 0)
    f = jnp.fft.rfftn(f, axes=last_three)

    if g is None:
        P = f.real**2 + f.imag**2
    else:
        g = jnp.asarray(g)
        g = jnp.fft.rfftn(g, axes=last_three)

        P = f * g.conj()

    if P.ndim > 3:
        P = P.sum(tuple(range(P.ndim-3)))

    kvec = rfftnfreq(grid_shape, None, dtype=P.real.dtype)
    k = jnp.sqrt(sum(k**2 for k in kvec))
    kfun = 1 / max(grid_shape)
    knyq = 0.5
    kmax = knyq * math.sqrt(3)

    if deconv != 0:
        P = reduce(mul, (jnp.sinc(k) ** -deconv for k in kvec), P)  # numpy sinc has pi

    N = jnp.full_like(P, 2, dtype=int_dtype)
    N = N.at[..., 0].set(1)
    if grid_shape[-1] % 2 == 0:
        N = N.at[..., -1].set(1)

    k = k.ravel()
    P = P.ravel()
    N = N.ravel()

    if isinstance(bins, (int, float)):
        bins *= kfun
        bin_num = math.ceil(kmax / bins)
        bins *= jnp.arange(bin_num + 1)
        right = True
    elif isinstance(bins, complex):
        kmaxable = all(s % 2 == 0 for s in grid_shape)
        bin_num = math.ceil(math.log2(kmax / kfun) / bins.imag) + kmaxable
        bins = kfun * 2 ** (bins.imag * jnp.arange(bin_num + 1))
        right = False
    else:
        bin_num = len(bins) - 1
        bins = jnp.asarray(bins)
        bins *= spacing / (2 * jnp.pi)  # convert to 2π spacing
        right = True

    b = jnp.digitize(k, bins, right=right)
    k *= N
    P *= N
    k = jnp.bincount(b, weights=k, length=1+bin_num)  # k=0 goes to b=0
    P = jnp.bincount(b, weights=P, length=1+bin_num)
    N = jnp.bincount(b, weights=N, length=1+bin_num)

    bmax = jnp.digitize(knyq if cut_nyq else kmax, bins, right=True)
    k = k[cut_zero:bmax+1]
    P = P[cut_zero:bmax+1]
    N = N[cut_zero:bmax+1]
    bins = bins[:bmax+1]

    k /= N
    P /= N

    k *= 2 * jnp.pi / spacing
    bins *= 2 * jnp.pi / spacing
    P *= spacing**3 / reduce(mul, grid_shape)

    return k, P, N, bins
