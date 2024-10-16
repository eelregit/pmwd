from dataclasses import field
from functools import partial
from operator import add, sub
from typing import ClassVar, Optional

from jax import Array, value_and_grad
from jax.typing import ArrayLike
import jax.numpy as jnp
from jax.tree_util import tree_map

from pmwd.tree_util import pytree_dataclass
from pmwd.configuration import Configuration


@partial(pytree_dataclass, aux_fields="conf", frozen=True)
class Cosmology:
    """Cosmological and configuration parameters, "immutable" as a frozen dataclass.

    Cosmological parameters with trailing underscores ("foo_") can be set to None, in
    which case they take some fixed values (set by class variable "foo_fixed") and will
    not receive gradients. They should be accessed through corresponding properties
    named without the trailing underscores ("foo").

    Linear operators (addition, subtraction, and scalar multiplication) are defined for
    Cosmology tangent and cotangent vectors.

    Float parameters are converted to JAX arrays of conf.cosmo_dtype at instantiation,
    to avoid possible JAX weak type problems.

    Parameters
    ----------
    conf : Configuration
        Configuration parameters.
    A_s_1e9 : float ArrayLike
        Primordial scalar power spectrum amplitude, multiplied by 1e9.
    n_s : float ArrayLike
        Primordial scalar power spectrum spectral index.
    Omega_m : float ArrayLike
        Total matter density parameter today.
    Omega_b : float ArrayLike
        Baryonic matter density parameter today.
    Omega_k_ : None or float ArrayLike, optional
        Spatial curvature density parameter today. Default is None.
    w_0_ : None or float ArrayLike, optional
        Dark energy equation of state constant parameter. Default is None.
    w_a_ : None or float ArrayLike, optional
        Dark energy equation of state linear parameter. Default is None.
    h : float ArrayLike
        Hubble constant in unit of 100 [km/s/Mpc].

    """

    conf: Configuration = field(repr=False)

    A_s_1e9: ArrayLike
    n_s: ArrayLike
    Omega_m: ArrayLike
    Omega_b: ArrayLike
    h: ArrayLike

    Omega_k_: Optional[ArrayLike] = None
    Omega_k_fixed: ClassVar[float] = 0
    w_0_: Optional[ArrayLike] = None
    w_0_fixed: ClassVar[float] = -1
    w_a_: Optional[ArrayLike] = None
    w_a_fixed: ClassVar[float] = 0

    transfer: Optional[Array] = field(default=None, compare=False)

    growth: Optional[Array] = field(default=None, compare=False)

    varlin: Optional[Array] = field(default=None, compare=False)

    def __post_init__(self):
        if self._is_transforming():
            return

        dtype = self.conf.cosmo_dtype
        for name, value in self.named_children():
            value = tree_map(lambda x: jnp.asarray(x, dtype=dtype), value)
            object.__setattr__(self, name, value)

    def __add__(self, other):
        return tree_map(add, self, other)

    def __sub__(self, other):
        return tree_map(sub, self, other)

    def __mul__(self, other):
        return tree_map(lambda x: x * other, self)

    def __rmul__(self, other):
        return self.__mul__(other)

    @classmethod
    def from_sigma8(cls, conf, sigma8, *args, **kwargs):
        """Construct cosmology with sigma8 instead of A_s."""
        from pmwd.boltzmann import boltzmann

        cosmo = cls(conf, 1, *args, **kwargs)
        cosmo = boltzmann(cosmo, conf)

        A_s_1e9 = (sigma8 / cosmo.sigma8)**2

        return cls(conf, A_s_1e9, *args, **kwargs)

    def astype(self, dtype):
        """Cast parameters to dtype by changing conf.cosmo_dtype."""
        conf = self.conf.replace(cosmo_dtype=dtype)
        return self.replace(conf=conf)  # calls __post_init__

    @property
    def k_pivot(self):
        """Primordial scalar power spectrum pivot scale in [1/L].

        Pivot scale is defined h-less unit, so needs h to convert its unit to [1/L].

        """
        return self.conf.k_pivot_Mpc / (self.h * self.conf.Mpc_SI) * self.conf.L

    @property
    def A_s(self):
        """Primordial scalar power spectrum amplitude."""
        return self.A_s_1e9 * 1e-9

    @property
    def Omega_c(self):
        """Cold dark matter density parameter today."""
        return self.Omega_m - self.Omega_b

    @property
    def Omega_k(self):
        """Spatial curvature density parameter today."""
        return self.Omega_k_fixed if self.Omega_k_ is None else self.Omega_k_

    @property
    def Omega_de(self):
        """Dark energy density parameter today."""
        return 1 - (self.Omega_m + self.Omega_k)

    @property
    def w_0(self):
        """Dark energy equation of state constant parameter."""
        return self.w_0_fixed if self.w_0_ is None else self.w_0_

    @property
    def w_a(self):
        """Dark energy equation of state linear parameter."""
        return self.w_a_fixed if self.w_a_ is None else self.w_a_

    @property
    def sigma8(self):
        """Linear matter rms overdensity within a tophat sphere of 8 Mpc/h radius at a=1."""
        from pmwd.boltzmann import varlin
        R = 8 * self.conf.Mpc_SI / self.conf.L
        return jnp.sqrt(varlin(R, 1., self, self.conf))

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
    a : ArrayLike
        Scale factors.
    cosmo : Cosmology

    Returns
    -------
    E2 : jax.Array of cosmo.conf.cosmo_dtype
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
    a = jnp.asarray(a, dtype=cosmo.conf.cosmo_dtype)

    de_a = a**(-3 * (1 + cosmo.w_0 + cosmo.w_a)) * jnp.exp(-3 * cosmo.w_a * (1 - a))
    return cosmo.Omega_m * a**-3 + cosmo.Omega_k * a**-2 + cosmo.Omega_de * de_a


@partial(jnp.vectorize, excluded=(1,))
def H_deriv(a, cosmo):
    r"""Hubble parameter derivatives, :math:`\mathrm{d}\ln H / \mathrm{d}\ln a`, at
    given scale factors.

    Parameters
    ----------
    a : ArrayLike
        Scale factors.
    cosmo : Cosmology

    Returns
    -------
    dlnH_dlna : jax.Array of cosmo.conf.cosmo_dtype
        Hubble parameter derivatives.

    """
    a = jnp.asarray(a, dtype=cosmo.conf.cosmo_dtype)

    E2_value, E2_grad = value_and_grad(E2)(a, cosmo)
    return 0.5 * a * E2_grad / E2_value


def Omega_m_a(a, cosmo):
    r"""Matter density parameters, :math:`\Omega_\mathrm{m}(a)`, at given scale factors.

    Parameters
    ----------
    a : ArrayLike
        Scale factors.
    cosmo : Cosmology

    Returns
    -------
    Omega : jax.Array of cosmo.conf.cosmo_dtype
        Matter density parameters.

    Notes
    -----

    .. math::

        \Omega_\mathrm{m}(a) = \frac{\Omega_\mathrm{m} a^{-3}}{E^2(a)}

    """
    a = jnp.asarray(a, dtype=cosmo.conf.cosmo_dtype)

    return cosmo.Omega_m / (a**3 * E2(a, cosmo))
