from abc import ABC, abstractmethod
from operator import itemgetter
from functools import partial

import jax
from jax.typing import ArrayLike, DTypeLike
import jax.numpy as jnp
from jax.tree_util import tree_map

from pmwd.tree_util import (Tree, TanMixin, pytree_dataclass, aux_field, issubdtype_of,
                            asarray_of)
from pmwd.lens.util import (lens_dyn_field, lens_dyn_2d_field, lens_fxd_field,
                            _canonicalize_ang_pos, _canonicalize_array)


@pytree_dataclass
class Lenses(TanMixin, Tree, ABC):
    """Lenses for strong gravitational lensing."""

    dtype: DTypeLike = aux_field(default=jnp.float64,
                                 validate=(jnp.dtype, issubdtype_of(jnp.floating)))

    x: ArrayLike = lens_dyn_2d_field()
    a: ArrayLike = lens_fxd_field(repr=True)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, key):
        return tree_map(itemgetter(key), self)  #FIXME itemgetter -> itembetter, broadcastability friendly

    def astype(self, dtype):
        """Return a new object with pytree children casted to `dtype`."""
        return self.replace(dtype=dtype)

    @property
    def z(self):
        """Redshifts."""
        return 1 / a - 1

    @staticmethod
    @abstractmethod
    def potential(x, lens):
        pass


def potential(x, lens):
    """Lensing potentials for sources at infinity.

    Evaluate ``type(lens).potential`` on rays `x`. Also see documentation there.

    Parameters
    ----------
    x : ArrayLike of shape (..., 2)
        Ray angular positions in :math:`A`.
    lens : Lenses

    Returns
    -------
    psi : jax.Array of x.dtype and shape (...,)
        Lensing potentials in :math:`A^2`.

    """
    x = _canonicalize_ang_pos(x)
    lens = lens.astype(x.dtype)
    psi = type(lens).potential(x, lens)
    return _canonicalize_array(psi, shape=x.shape[:-1], dtype=x.dtype)


@jax.grad
def _deflect_one_ray(x, lens):
    return potential(x, lens).reshape(())  # .item() leads to ConcretizationTypeError


@partial(jax.vmap, in_axes=(0, None), out_axes=0)
def _deflect(x, lens):
    return _deflect_one_ray(x, lens)


def deflect(x, lens, always_grad=False):
    """Deflection angles for sources at infinity.

    Evaluate ``type(lens).deflect`` if it exists, otherwise the lensing potential
    gradient ``jax.grad(type(lens).potential)``, on rays `x`. Also see documentation
    there.

    Parameters
    ----------
    x : ArrayLike of shape (..., 2)
        Ray angular positions in :math:`A`.
    lens : Lenses
    always_grad : bool, optional
        Whether to ignore the deflection angle implementation even if it exists and
        always use the lensing potential gradient.

    Returns
    -------
    alpha_hat : jax.Array of x.dtype and shape (..., 2)
        Deflection angles in :math:`A`.

    """
    x = _canonicalize_ang_pos(x)
    lens = lens.astype(x.dtype)

    if hasattr(type(lens), 'deflect') and not always_grad:
        alpha_hat = type(lens).deflect(x, lens)
    else:
        alpha_hat = _deflect(x.reshape(-1, 2), lens)
        alpha_hat = alpha_hat.reshape(x.shape)

    return _canonicalize_array(alpha_hat, shape=x.shape, dtype=x.dtype)


def _log(z):
    """why can this be faster than jnp.log(jnp.abs(z)) + 1j * jnp.angle(z)?"""
    return jnp.log(z.real**2 + z.imag**2) / 2 + 1j * jnp.arctan2(z.imag, z.real)


#FIXME change theta to PA the standard one, rotating semi-major axis to the north!!!
@pytree_dataclass
class dPIELenses(Lenses):
    r"""Lenses with dual pseudo isothermal elliptical mass distribution.

    Parameters
    ----------
    dtype : DTypeLike
    x : ArrayLike of shape (num_lenses, 2) or (2,)
        Angular positions in :math:`A`.
    a : ArrayLike
        Scale factors.
    lnE_0 : ArrayLike
        Natural log of asymptotic Einstein radii for sources at infinity, in :math:`A`.
        They share the same unit as the lens radii and the ray positions, and thus are
        better than, e.g., velocity dispersions, as the normalization parameter.
    lnc : ArrayLike
        Natural log of core radii in :math:`A`.
    lns : ArrayLike
        Natural log of scale radii in :math:`A`.
    q : ArrayLike
        Axis ratios of the minor axes to the major axes.
    theta : ArrayLike
        Position angles in radians.

    References
    ----------
    .. _Kassiola and Kovner 1993, Elliptic mass distributions versus elliptic potentials in gravitational lenses:
        https://doi.org/10.1086/173325
    .. _Elíasdóttir et al. 2007, Where is the matter in the merging cluster Abell 2218? Appendix:
        https://arxiv.org/abs/0710.5636

    Notes
    -----

    .. math::

        TODO

    """
    #FIXME above & below for changing theta to PA standard

    lnE_0: ArrayLike = lens_dyn_field()
    lnc: ArrayLike = lens_dyn_field()
    lns: ArrayLike = lens_dyn_field()
    q: ArrayLike = lens_dyn_field()
    theta: ArrayLike = lens_dyn_field()

    @property
    def E_0(self):
        """Asymptotic Einstein radii for sources at infinity, in :math:`A`."""
        return jnp.exp(self.lnE_0)

    @property
    def c(self):
        """Core radii in :math:`A`."""
        return jnp.exp(self.lnc)

    @property
    def s(self):
        """Scale radii in :math:`A`."""
        return jnp.exp(self.lns)

    @property
    def eps(self):
        r"""Ellipticity :math:`\epsilon \triangleq (a-b)/(a+b)`."""
        return (1 - self.q) / (1 + self.q)

    @staticmethod
    def sigma(lens, cosmo):
        r"""Velocity dispersions in :math:`L/T`.

        Notes
        -----
        .. math::

            E_0 = 6\pi \frac{D_\mathrm{LS}}{D_\mathrm{S}} \frac{sigma^2}{c^2},
            E_0 = 4\pi \frac{D_\mathrm{LS}}{D_\mathrm{S}} \frac{sigma^2}{c^2},

        with :math:`D` being the angular diameter distance :math:`d_\mathrm{A}`, and
        :math:`D_\mathrm{LS} / D_\mathrm{S} \to 1`.

        """
        # TODO depending on velocity dispersion convention
        return jnp.sqrt(cosmo.A * cosmo.c**2 / (6*jnp.pi) * lens.E_0)
        #return jnp.sqrt(cosmo.A * cosmo.c**2 / (4*jnp.pi) * lens.E_0)

    @staticmethod
    def _Phi(misc, omega):
        x, rM2, sinh_2eta, cosh_2eta = misc
        rM2 /= omega ** 2  # Mahalanobis distance in unit of omega
        sinh_2zeta = jnp.sqrt(rM2)
        cosh_2zeta = jnp.sqrt(1 + rM2)
        cosh_2eta_plus_2zeta = cosh_2eta * cosh_2zeta + sinh_2eta * sinh_2zeta

        z1 = _log((cosh_2eta + 1) / (cosh_2eta + cosh_2zeta))
        z2 = _log((cosh_2eta_plus_2zeta + 1) / (cosh_2eta + cosh_2zeta))
        Kconj = sinh_2eta * z1 + sinh_2zeta * z2
        Phi = (x.conj() * Kconj).imag / jnp.sqrt(rM2)

        return Phi

    #FIXME worth changing the PIEMD eps convention to the mass conserving one???
    @staticmethod
    def potential(x, lens):
        """dPIE lensing potentials. Also see documentation of `potential`.

        Adapted from Eqs 4.1.5-4.1.8 of Kassiola & Kovner 1993, but optimized away all
        trigonometric and hyperbolic function calls.

        """
        x = x[..., jnp.newaxis, :] - lens.x  # shape = (..., num_lenses, 2)
        x = x[..., 0] + 1j * x[..., 1]
        x *= jnp.exp(- 1j * lens.theta)
        r2 = x.real**2 + x.imag**2
        # Mahalanobis distance
        #rM2 = (x.real / (1 + lens.eps))**2 + (x.imag / (1 - lens.eps))**2
        rM2 = ((1 + lens.q) / 2 * x.real)**2 + ((1 + 1/lens.q) / 2 * x.imag)**2
        sinh_2eta = jnp.sqrt(lens.eps * rM2) / r2 * (2j * x)
        cosh_2eta = ((1 - lens.eps**2) * rM2 / r2
                     - 1j * ((1/lens.q - lens.q) * x.real * x.imag / r2))
        misc = x, rM2, sinh_2eta, cosh_2eta

        Phi = dPIELenses._Phi(misc, lens.c) - dPIELenses._Phi(misc, lens.s)
        # TODO where the hell does s/(s-c) come from?
        Phi *= (lens.s / (lens.s - lens.c)
                * lens.E_0 * (1 - lens.eps**2) / (2 * jnp.sqrt(lens.eps)))

        return Phi.sum(axis=-1)  # sum over lenses  #FIXME fully general case

    @staticmethod
    def _i_I(misc, omega):
        x, rM2, lens = misc
        i_I = _log(
            ((x.real * lens.q) + 1j * (x.imag / lens.q
                - 2 * jnp.sqrt(lens.eps * (omega**2 + rM2))))
            / (x.real + 1j * (x.imag - 2 * (omega * jnp.sqrt(lens.eps))))
        )
        return i_I

    @staticmethod
    def deflect(x, lens):
        """dPIE deflection angles. Also see documentation of `deflect`.

        Adapted from Eq 4.1.2 of Kassiola & Kovner 1993, but using :math:`i I` instead
        of :math:`I^*`, so that the 2nd rotation are flipped and the x-y components are
        swapped.

        """
        x = x[..., jnp.newaxis, :] - lens.x  # shape = (..., num_lenses, 2)
        x = x[..., 0] + 1j * x[..., 1]
        x *= jnp.exp(- 1j * lens.theta)
        # Mahalanobis distance
        #rM2 = (x.real / (1 + lens.eps))**2 + (x.imag / (1 - lens.eps))**2
        rM2 = ((1 + lens.q) / 2 * x.real)**2 + ((1 + 1/lens.q) / 2 * x.imag)**2
        misc = x, rM2, lens

        alpha_hat = dPIELenses._i_I(misc, lens.c) - dPIELenses._i_I(misc, lens.s)
        # TODO where the hell does s/(s-c) come from?
        alpha_hat *= ((lens.eps**2 - 1) * lens.E_0 / (2 * jnp.sqrt(lens.eps))
                      * jnp.exp(- 1j * lens.theta))

        alpha_hat = alpha_hat.sum(axis=-1)  # sum over lenses  #FIXME fully general case
        return jnp.stack((alpha_hat.imag, alpha_hat.real), axis=-1)
