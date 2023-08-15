from functools import partial
import math

from jax import jit, ensure_compile_time_eval
import jax.numpy as jnp
from jax import custom_vjp, jit, ensure_compile_time_eval

from pmwd.pm_util import fftfreq, fftfwd


@partial(jit, static_argnames=('bins', 'cut_zero', 'cut_nyq', 'dtype', 'int_dtype'))
def powspec(f, spacing, bins=1j/3, g=None, deconv=None, cut_zero=True, cut_nyq=True,
            dtype=jnp.float_, int_dtype=jnp.uint32):
    """Compute auto or cross power spectrum in 3D averaged in spherical bins.

    Parameters
    ----------
    f : ArrayLike
        The field, with the last 3 axes for FFT and the other reduced by sum of squares.
    spacing : float
        Field grid spacing.
    bins : float, complex, or tuple, optional
        (Angular) wavenumber bins. A real number sets the linear bin width in unit of
        the smallest fundamental in 3D (right edge inclusive starting from zero); an
        imaginary number sets the log bin width in octave (left edge inclusive starting
        from the smallest fundamental in 3D); and a tuple sets the bin edges directly
        (right edge inclusive and must starting from zero).
    g : ArrayLike, optional
        Another field of the same shape for cross correlation.
    deconv : int, optional
        Power of sinc factors to deconvolve in the power spectrum.
    cut_zero : bool, optional
        Whether to discard the bin containing the zero or DC mode.
    cut_nyq : bool, optional
        Whether to discard the bins beyond the Nyquist, only for linear or log ``bins``.
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

    with ensure_compile_time_eval():
        kfun = 1 / max(grid_shape)
        knyq = 0.5
        kmax = knyq * math.sqrt(3)
        if isinstance(bins, (int, float)):
            bins *= kfun
            bin_num = math.ceil(kmax / bins)
            bins *= jnp.arange(1 + bin_num)
            right = True
            bcut = jnp.digitize(knyq if cut_nyq else kmax, bins, right=right).item() + 1
        elif isinstance(bins, complex):
            kmaxable = all(s % 2 == 0 for s in grid_shape)  # extra bin just in case
            bin_num = math.ceil(math.log2(kmax / kfun) / bins.imag) + kmaxable
            bins = kfun * 2 ** (bins.imag * jnp.arange(1 + bin_num))
            right = False
            bcut = jnp.digitize(knyq if cut_nyq else kmax, bins, right=right).item() + 1
        else:
            bin_num = len(bins) - 1
            bins = jnp.asarray(bins)
            bins *= spacing / (2 * jnp.pi)  # convert to 2π spacing
            right = True
            bcut = bin_num + 1  # no trim otherwise hard to jit

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
    k = jnp.bincount(b, weights=k, length=1+bin_num)  # only k=0 goes to b=0
    P = jnp.bincount(b, weights=P, length=1+bin_num)
    N = jnp.bincount(b, weights=N, length=1+bin_num)

    k = k[cut_zero:bcut]
    P = P[cut_zero:bcut]
    N = N[cut_zero:bcut]
    bins = bins[:bcut]

    k /= N
    P /= N

    k *= 2 * jnp.pi / spacing
    bins *= 2 * jnp.pi / spacing
    P *= spacing**3 / math.prod(grid_shape)

    return k, P, N, bins
