from functools import partial

from jax import jit, checkpoint, custom_vjp
from jax import random
import jax.numpy as jnp

from pmwd.boltzmann import linear_power, linear_transfer
from pmwd.pm_util import fftfreq, fftfwd, fftinv


#TODO follow pmesh to fill the modes in Fourier space
@partial(jit, static_argnames=('real', 'unit_abs'))
def white_noise(seed, conf, real=False, unit_abs=False):
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

    Returns
    -------
    modes : jax.Array of conf.float_dtype
        White noise Fourier or real modes, both dimensionless with zero mean and unit
        variance.

    """
    key = random.PRNGKey(seed)

    # sample linear modes on Lagrangian particle grid
    modes = random.normal(key, shape=conf.ptcl_grid_shape, dtype=conf.float_dtype)

    if real and not unit_abs:
        return modes

    modes = fftfwd(modes, norm='ortho')

    if unit_abs:
        modes /= jnp.abs(modes)

    if real:
        modes = fftinv(modes, shape=conf.ptcl_grid_shape, norm='ortho')

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


@partial(jit, static_argnums=4)
# @partial(checkpoint, static_argnums=4)
def linear_modes(modes, cosmo, conf, a=None, real=False):
    """Linear matter overdensity Fourier or real modes.

    Parameters
    ----------
    modes : jax.Array
        Fourier or real modes with white noise prior.
    cosmo : Cosmology
    conf : Configuration
    a : float or None, optional
        Scale factors. Default (None) is to not scale the output modes by growth.
    real : bool, optional
        Whether to return real or Fourier modes.

    Returns
    -------
    modes : jax.Array of conf.float_dtype
        Linear matter overdensity Fourier or real modes, in [L^3] or dimensionless,
        respectively.

    Notes
    -----

    .. math::

        \delta(\mathbf{k}) = \sqrt{V P_\mathrm{lin}(k)} \omega(\mathbf{k})

    """
    kvec = fftfreq(conf.ptcl_grid_shape, conf.ptcl_spacing, dtype=conf.float_dtype)
    k = jnp.sqrt(sum(k**2 for k in kvec))

    if a is not None:
        a = jnp.asarray(a, dtype=conf.float_dtype)

    if jnp.isrealobj(modes):
        modes = fftfwd(modes, norm='ortho')
        
    if cosmo.f_nl_loc is not None:
        Tlin = linear_transfer(k, a, cosmo, conf)*k*k
        Pprim = 2*jnp.pi**2. * cosmo.A_s * (k/cosmo.k_pivot)**(cosmo.n_s-1.)\
                    * k**(-3.)
        
        modes *= _safe_sqrt(Pprim / conf.box_vol)
        modes = modes.at[0,0,0].set(0.+0.j)
        
        # TF: To generate non-Gaussian primordial field without aliassing effects, we generate the square of the field at a higher grid size
        # TF: When squaring the field in real space, the generated higher frequency modes can be accomodated on the larger grid and don't 'fold back' over the relevant modes.
        modes_NG = jnp.zeros(shape=(conf.ptcl_grid_shape[0]*2,conf.ptcl_grid_shape[1]*2,conf.ptcl_grid_shape[2] + 1),dtype=modes.dtype)
        # TF: We fill the higher resolution box only halfway with the previously generated modes (note factor of 8 for 2**3 times more gridpoints):
        modes_NG = modes_NG.at[conf.ptcl_grid_shape[0]-conf.ptcl_grid_shape[0]//2:conf.ptcl_grid_shape[0]+conf.ptcl_grid_shape[0]//2,conf.ptcl_grid_shape[1]-conf.ptcl_grid_shape[1]//2:conf.ptcl_grid_shape[1]+conf.ptcl_grid_shape[1]//2,:conf.ptcl_grid_shape[2]//2+1].set(jnp.fft.fftshift(modes*jnp.sqrt(8),axes=[0,1]))
        modes_NG = jnp.fft.ifftshift(modes_NG,axes=[0,1])
        # TF: Move to real space, square and back to Fourier space
        modes_NG = fftfwd(fftinv(modes_NG, norm='ortho')**2., norm='ortho') 
        # TF: After squaring, downsample back to the target resolution in Fourier space
        modes_NG = jnp.fft.fftshift(modes_NG,axes=[0,1])
        modes_NG = modes_NG[conf.ptcl_grid_shape[0]-conf.ptcl_grid_shape[0]//2:conf.ptcl_grid_shape[0]+conf.ptcl_grid_shape[0]//2,conf.ptcl_grid_shape[1]-conf.ptcl_grid_shape[1]//2:conf.ptcl_grid_shape[1]+conf.ptcl_grid_shape[1]//2,:conf.ptcl_grid_shape[2]//2+1]/jnp.sqrt(8)
        modes_NG = jnp.fft.ifftshift(modes_NG,axes=[0,1])

        # TF: And now to real space again to do the addition in the proper way
        modes = fftinv(modes, norm='ortho')
        modes_NG = fftinv(modes_NG, norm='ortho')
        
        # TF: add the non-guassian field, factor of 3/5 is because we are generating \zeta and f_nl is defined for \Phi
        modes = modes + 3/5 * cosmo.f_nl_loc * jnp.sqrt(conf.ptcl_num) * (modes_NG - jnp.mean(modes_NG))

        # TF: apply transfer function
        modes = fftfwd(modes, norm='ortho')
        modes *= Tlin * conf.box_vol
    else:
        Plin = linear_power(k, a, cosmo, conf)
        modes *= _safe_sqrt(Plin * conf.box_vol)

    if real:
        modes = fftinv(modes, shape=conf.ptcl_grid_shape, norm=conf.ptcl_spacing)

    return modes
