from dataclasses import field
from functools import partial
from operator import add, sub
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
    Omega_k : float or None, optional
        Spatial curvature density parameter today. If None (default), Omega_k is 0.
    w_0 : float or None, optional
        Dark energy equation of state (0th order) parameter. If None (default), w is -1.
    w_a : float or None, optional
        Dark energy equation of state (linear) parameter. If None (default), w_a is 0.
    h : float
        Hubble constant in unit of 100 [km/s/Mpc].

    Notes
    -----
    For (extension) parameters with None as default values, one needs to set them
    explicitly to some values to receive their gradients.

    """

    conf: Configuration = field(repr=False)

    A_s_1e9: float
    n_s: float
    Omega_m: float
    Omega_b: float
    h: float

    Omega_k: Optional[float] = None
    w_0: Optional[float] = None
    w_a: Optional[float] = None

    transfer: Optional[jnp.ndarray] = field(default=None, repr=False, compare=False)

    growth: Optional[jnp.ndarray] = field(default=None, repr=False, compare=False)

    def __str__(self):
        return pformat(self, indent=4, width=1)  # for python >= 3.10

    def __add__(self, other):
        return tree_map(add, self, other)

    def __sub__(self, other):
        return tree_map(sub, self, other)

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
        Omk = self.Omega_m if self.Omega_k is None else self.Omega_m + self.Omega_k
        return 1 - Omk

    @property
    def ptcl_mass(self):
        """Particle mass in [M]."""
        return self.conf.rho_crit * self.Omega_m * self.conf.ptcl_cell_vol


SimpleLCDM = partial(
    Cosmology,
    A_s_1e9=2.0,
    n_s=0.96,
    Omega_m=0.3,
    Omega_b=0.05,
    h=0.7,
)
SimpleLCDM.__doc__ = "Simple ΛCDM cosmology, for convenience and subject to change."

Planck18 = partial(
    Cosmology,
    A_s_1e9=2.105,
    n_s=0.9665,
    Omega_m=0.3111,
    Omega_b=0.04897,
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
    Oka2 = 0 if cosmo.Omega_k is None else cosmo.Omega_k * a**-2
    w_0 = -1 if cosmo.w_0 is None else cosmo.w_0
    w_a = 0 if cosmo.w_a is None else cosmo.w_a
    de_a = (a**(-3 * (1 + w_0 + w_a))
           * jnp.exp(-3 * w_a * (1 - a)))
    return cosmo.Omega_m * a**-3 + Oka2 + cosmo.Omega_de * de_a


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
