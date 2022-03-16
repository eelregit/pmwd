from dataclasses import field
from functools import partial
from operator import add
from pprint import pformat
from typing import Optional

from jax import value_and_grad
import jax.numpy as jnp
from jax.tree_util import tree_map

from pmwd.conf import Configuration
from pmwd.dataclasses import pytree_dataclass


@partial(pytree_dataclass, aux_fields="conf", frozen=True)
class Cosmology:
    """Cosmological and configuration parameters, "immutable" as a frozen dataclass.
    Cosmological parameters are traced by JAX while configuration parameters are not.

    Parameters
    ----------
    conf : Configuration
        Configuration parameters, not traced by JAX.
    A_s_1e9 : float
        Primordial scalar power spectrum amplitude, multiplied by 1e9.
    n_s : float
        Primordial scalar power spectrum spectral index.
    Omega_m : float
        Total matter density parameter today.
    Omega_b : float
        Baryonic matter density parameter today.
    Omega_k : float
        Spatial curvature density parameter today.
    w_0 : float
        First order term of dark energy equation.
    w_a : float
        Second order term of dark energy equation of state.
    h : float
        Hubble constant in unit of 100 [km/s/Mpc].

    """

    conf: Configuration

    A_s_1e9: float
    n_s: float
    Omega_m: float
    Omega_b: float
    Omega_k: float
    w_0: float
    w_a: float
    h: float

    transfer: Optional[jnp.ndarray] = field(default=None, repr=False, compare=False)

    growth: Optional[jnp.ndarray] = field(default=None, repr=False, compare=False)

    def __str__(self):
        return pformat(self, indent=4, width=1)  # for python >= 3.10

    def __add__(self, other):
        return tree_map(add, self, other)

    def __mul__(self, other):
        return tree_map(lambda x: x * other, self)

    def __rmul__(self, other):
        return self.__mul__(other)

    @property
    def k_pivot(self):
        """Primordial scalar power spectrum pivot scale in [1/L].

        Notes
        -----
        Pivot scale is defined h-less unit, so needs h to convert its unit to [1/L].

        """
        return self.conf.k_pivot_Mpc / (self.h * self.conf.Mpc_SI) * self.conf.L

    @property
    def Omega_c(self):
        """Cold dark matter density parameter today."""
        return self.Omega_m - self.Omega_b

    @property
    def Omega_de(self):
        """Dark energy density parameter today."""
        return 1. - (self.Omega_m + self.Omega_k)


SimpleLCDM = partial(
    Cosmology,
    A_s_1e9=2.0,
    n_s=0.96,
    Omega_m=0.3,
    Omega_b=0.05,
    Omega_k=0.0,
    w_0=-1.0,
    w_a=0.0,
    h=0.7,
)
SimpleLCDM.__doc__ = "Simple ΛCDM cosmology, for convenience and subject to change."

Planck18 = partial(
    Cosmology,
    A_s_1e9=2.105,
    n_s=0.9665,
    Omega_m=0.3111,
    Omega_b=0.04897,
    Omega_k=0.0,
    w_0=-1.0,
    w_a=0.0,
    h=0.6766,
)
Planck18.__doc__ = "Planck 2018 cosmology, arXiv:1807.06209 Table 2 last column."


def E2(a, cosmo):
    r"""Squared Hubble parameter time scaling factors, :math:`E^2`, at given scale
    factors.

    Parameters
    ----------
    a : array_like
        Scale factors.
    cosmo : Cosmology

    Returns
    -------
    E2 : jax.numpy.ndarray
        Squared Hubble parameter time scaling factors.

    Notes
    -----
    The squared Hubble parameter

    .. math::

        H^2(a) = H_0^2 E^2(a),

    has time scaling

    .. math::

        E^2(a) = \Omega_\mathrm{m} a^{-3} + \Omega_\mathrm{k} a^{-2}
                 + \Omega_\mathrm{de} a^{-3 (1 + w_0 + w_a)} e^{-3 w_a (1 - a)}.

    """
    a = jnp.asarray(a)
    de_a = (a**(-3.0 * (1.0 + cosmo.w_0 + cosmo.w_a))
           * jnp.exp(-3.0 * cosmo.w_a * (1.0 - a)))
    return cosmo.Omega_m * a**-3 + cosmo.Omega_k * a**-2 + cosmo.Omega_de * de_a


@partial(jnp.vectorize, excluded=(1,))
def H_deriv(a, cosmo):
    r"""Hubble parameter derivatives, :math:`\mathrm{d}\ln H / \mathrm{d}\ln a`, at
    given scale factors.

    Parameters
    ----------
    a : array_like
        Scale factors.
    cosmo : Cosmology

    Returns
    -------
    dlnH_dlna : jax.numpy.ndarray
        Hubble parameter derivatives.

    """
    E2_value, E2_grad = value_and_grad(E2)(a, cosmo)
    dlnH_dlna = 0.5 * a * E2_grad / E2_value
    return dlnH_dlna


def Omega_m_a(a, cosmo):
    r"""Matter density parameters, :math:`\Omega_\mathrm{m}(a)`, at given scale factors.

    Parameters
    ----------
    a : array_like
        Scale factors.
    cosmo : Cosmology

    Returns
    -------
    Omega : jax.numpy.ndarray
        Matter density parameters.

    Notes
    -----

    .. math::

        \Omega_\mathrm{m}(a) = \frac{\Omega_\mathrm{m} a^{-3}}{E^2(a)}

    """
    a = jnp.asarray(a)
    return cosmo.Omega_m / (a**3 * E2(a, cosmo))
