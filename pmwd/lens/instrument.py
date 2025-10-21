from functools import partial
import math

import jax
from jax.typing import ArrayLike, DTypeLike
import jax.numpy as jnp

from pmwd.tree_util import (Tree, pytree_dataclass, fxd_field, aux_field, issubdtype_of,
                            asarray_of)
from pmwd.pm_util import fftlen
from pmwd.lens.util import _canonicalize_array


@pytree_dataclass
class Instrument(Tree):
    """Astronomical instrument and relevant properties.

    Parameters
    ----------
    dtype : DTypeLike, optional
    P : ArrayLike
        Point spread function (PSF).
    sigma : ArrayLike
        Noise level.
    fft_conv_thld : int, optional
        Threshold PSF size, for using FFT convolution above its square.

    """

    dtype: DTypeLike = aux_field(default=jnp.float32,
                                 validate=(jnp.dtype, issubdtype_of(jnp.floating)))

    P: ArrayLike = fxd_field(validate=(asarray_of(field='dtype'),
                                       partial(jnp.clip, min=0)), repr=True)
    sigma: ArrayLike = fxd_field(validate=(asarray_of(field='dtype'),
                                           partial(jnp.clip, min=0)), repr=True)

    # resolution
    # interpolation???

    fft_conv_thld: int = aux_field(default=13)

    # instead of file_name, should implement serialization instead
    #file_name: str = aux_field(optional=True)

    def astype(self, dtype):
        """Return a new object with pytree children casted to `dtype`."""
        return self.replace(dtype=dtype)

    #TODO necessary to implement scipy.signal.oaconvolve?
    #     https://en.wikipedia.org/wiki/Overlap-add_method
    #     only if this is bottleneck (with AMR quadtree)
    @staticmethod
    def convolve(I, instr, mode='valid', method=None):
        """Convolution with PSF. Also see documentation of `convolve`.

        Parameters
        ----------
        mode : 'full', 'same', or 'valid', optional
            Convolution mode to pass to `jax.scipy.signal.convolve`.
        method : 'direct', 'fft', or None, optional
            Convolution method to pass to `jax.scipy.signal.convolve`. Default is to
            select based on ``instr.fft_conv_thld``.

        """
        if I.ndim != 2:
            raise ValueError(f'only supporting 2D but {I.ndim = }')

        if method == 'direct':
            return jax.scipy.signal.convolve(I, instr.P, mode=mode, method=method)

        # next_fast_len not implemented in jax.scipy.signal.fftconvolve yet:
        # https://github.com/jax-ml/jax/discussions/15200
        # https://github.com/jax-ml/jax/blob/jax-v0.9.0.1/jax/_src/scipy/signal.py#L121
        # so we pad the PSF for the same effect
        pad = tuple(fftlen(s1 + s2 - 1) - (s1 + s2 - 1)
                    for s1, s2 in zip(I.shape, instr.P.shape))
        if method is None:
            fft_size = math.prod(s + p for s, p in zip(instr.P.shape, pad))
            method = 'direct' if fft_size < instr.fft_conv_thld ** 2 else 'fft'
        P = instr.P
        if method == 'fft':
            pad = tuple((p//2, p - p//2) for p in pad)
            P = jnp.pad(instr.P, pad, mode='constant', constant_values=0)

        I = jax.scipy.signal.convolve(I, P, mode=mode, method=method)

        return I


def convolve(I, instr, *args, **kwargs):
    """Convolution with PSF.

    Evaluate ``type(instr).convolve`` on image `I`. Also see documentation there.

    Parameters
    ----------
    I : ArrayLike
        Original 2D image.
    instr : Instrument
    *args, **kwargs
        Additional arguments to pass to ``type(instr).convolve``.

    Returns
    -------
    I : ArrayLike
        Convolved 2D image.

    """
    I = _canonicalize_array(I)
    instr = instr.astype(I.dtype)
    I = type(instr).convolve(I, instr, *args, **kwargs)
    return _canonicalize_array(I, dtype=instr.dtype)
