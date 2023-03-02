import jax.numpy as jnp

from pmwd.pm_util import rfftnfreq


def powspec(f, spacing, dk=None, g=None, int_dtype=jnp.uint32):
    """Compute auto or cross power spectrum in 3D in spherical bins.

    Parameters
    ----------
    f : array_like
        The field, with the last 3 axes for FFT and the other summed over.
    spacing : float
        Field grid spacing.
    dk : float, optional
        Wavenumber bin width. Default is the largest fundamental frequency.
    g : array_like, optional
        Another field of the same shape for cross correlation.
    int_dtype : dtype_like, optional
        Integer dtype for the number of modes.

    Returns
    -------
    k : jax.numpy.ndarray
        Wavenumber, excluding the DC mode in the first bin and stopping before or at the
        Nyquist frequency in the last one.
    P : jax.numpy.ndarray
        Power spectrum.
    N : jax.numpy.ndarray
        Number of modes.

    """
    grid_shape = f.shape[-3:]
    grid_size = reduce(mul, grid_shape)

    if dk is None:
        dk = 1 / min(s for s in grid_shape)

    if g is not None and f.shape != g.shape:
        raise ValueError(f'shape mismatch: {f.shape} != {g.shape}')

    last_three = range(-3, 0)
    f = jnp.fft.rfftn(f, axes=last_three)

    if g is None:
        P = f.real**2 + f.imag**2
    else:
        g = jnp.fft.rfftn(g, axes=last_three)

        P = f * g.conj()

    if P.ndim > 3:
        P = P.sum(tuple(range(P.ndim-3)))

    kvec = rfftnfreq(grid_shape, None, dtype=P.real.dtype)
    k = jnp.sqrt(sum(k**2 for k in kvec))

    N = jnp.full_like(P, 2, dtype=int_dtype)
    N = N.at[..., 0].set(1)
    if grid_shape[-1] % 2 == 0:
        N = N.at[..., -1].set(1)

    k = k.ravel()
    P = P.ravel()
    N = N.ravel()

    b = jnp.ceil(k / dk).astype(int_dtype)
    k *= N
    P *= N
    k = jnp.bincount(b, weights=k)
    P = jnp.bincount(b, weights=P)
    N = jnp.bincount(b, weights=N)

    bmax = jnp.floor(0.5 / dk).astype(int_dtype)
    k = k[1:1+bmax]
    P = P[1:1+bmax]
    N = N[1:1+bmax]

    k /= N
    P /= N

    k *= 2 * jnp.pi / spacing
    P *= spacing**3 / grid_size

    return k, P, N
