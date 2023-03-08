from functools import partial

from jax import jit, checkpoint, custom_vjp
from jax import random
import jax.numpy as jnp

from pmwd.boltzmann import linear_power
from pmwd.pm_util import rfftnfreq


#TODO follow pmesh to fill the modes in Fourier space
@partial(jit, static_argnames=('real', 'unit_abs', 'negate'))
def white_noise(seed, conf, real=False, unit_abs=False, negate=False):
    """White noise Fourier or real modes.

    Parameters
    ----------
    seed : int
        Seed for the pseudo-random number generator.
    conf : Configuration
    real : bool, optional
        Whether to return real or Fourier modes.
    unit_abs : bool, optional
        Whether to set the absolute values to 1.
    negate : bool, optional
        Whether to reverse the signs (180° phase flips).

    Returns
    -------
    modes : jax.numpy.ndarray of conf.float_dtype
        White noise modes.

    """
    key = random.PRNGKey(seed)

    # sample linear modes on Lagrangian particle grid
    modes = random.normal(key, shape=conf.ptcl_grid_shape, dtype=conf.float_dtype)

    if real and not unit_abs and not negate:
        return modes

    modes = jnp.fft.rfftn(modes, norm='ortho')

    if unit_abs:
        modes /= jnp.abs(modes)

    if negate:
        modes = -modes

    if real:
        modes = jnp.fft.irfftn(modes, s=conf.ptcl_grid_shape, norm='ortho')

    return modes


@custom_vjp
def _safe_sqrt(x):
    return jnp.sqrt(x)

def _safe_sqrt_fwd(x):
    y = _safe_sqrt(x)
    return y, y

def _safe_sqrt_bwd(y, y_cot):
    x_cot = jnp.where(y != 0, 0.5 / y * y_cot, 0)
    return (x_cot,)

_safe_sqrt.defvjp(_safe_sqrt_fwd, _safe_sqrt_bwd)


@jit
@checkpoint
def linear_modes(modes, cosmo, conf, a=None):
    """Linear matter overdensity Fourier modes.

    Parameters
    ----------
    modes : jax.numpy.ndarray
        Fourier or real modes with white noise prior.
    cosmo : Cosmology
    conf : Configuration
    a : float or None, optional
        Scale factors. Default (None) is to not scale the output modes by growth.

    Returns
    -------
    modes : jax.numpy.ndarray of conf.float_dtype
        Linear matter overdensity Fourier modes in [L^3].

    Notes
    -----

    .. math::

        \delta(\mathbf{k}) = \sqrt{V P_\mathrm{lin}(k)} \omega(\mathbf{k})

    """
    kvec = rfftnfreq(conf.ptcl_grid_shape, conf.ptcl_spacing, dtype=conf.float_dtype)
    k = jnp.sqrt(sum(k**2 for k in kvec))

    if a is not None:
        a = jnp.asarray(a, dtype=conf.float_dtype)

    Plin = linear_power(k, a, cosmo, conf)

    if jnp.isrealobj(modes):
        modes = jnp.fft.rfftn(modes, norm='ortho')

    modes *= _safe_sqrt(Plin * conf.box_vol)

    return modes
