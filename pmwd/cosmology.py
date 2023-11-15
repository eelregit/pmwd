from dataclasses import field
from functools import partial
import math
from operator import add, sub
from typing import ClassVar, Optional, Tuple, Union

from jax import Array, ensure_compile_time_eval, value_and_grad
from jax.typing import ArrayLike, DTypeLike
import jax.numpy as jnp
from jax.tree_util import tree_map
from mcfit import TophatVar

from pmwd import boltzmann
from pmwd.tree_util import pytree_dataclass


@partial(pytree_dataclass,
         aux_fields=["A_s_1e9", "n_s", "Omega_m", "Omega_b", "h",
                     "T_cmb_", "Omega_k_", "w_0_", "w_a_",
                     "transfer", "growth", "varlin"],
         aux_invert=True,
         frozen=True)
class Cosmology:
    r"""Cosmological parameters and related configurations.

    Cosmological parameters with trailing underscores ("foo_") can be set to None, in
    which case they take some fixed values (set by class variable "foo_fixed") and will
    not receive gradients. They should be accessed through corresponding properties
    named without the trailing underscores ("foo").

    Linear operators (addition, subtraction, and scalar multiplication) are defined for
    Cosmology tangent and cotangent vectors.

    Float parameters are converted to JAX arrays at instantiation, to avoid possible JAX
    weak type problems.

    Parameters
    ----------
    A_s_1e9 : float ArrayLike
        Primordial scalar power spectrum amplitude :math:`A_\mathrm{s} \times 10^9`.
    n_s : float ArrayLike
        Primordial scalar power spectrum spectral index :math:`n_\mathrm{s}`.
    Omega_m : float ArrayLike
        Total matter density parameter today :math:`\Omega_\mathrm{m}`.
    Omega_b : float ArrayLike
        Baryonic matter density parameter today :math:`\Omega_\mathrm{b}`.
    h : float ArrayLike
        Hubble constant in unit of 100 km/s/Mpc :math:`h`.
    T_cmb_ : None or float ArrayLike, optional
        CMB temperature in Kelvin today :math:`T_\mathrm{CMB}`. Default is None.
    Omega_k_ : None or float ArrayLike, optional
        Spatial curvature density parameter today :math:`Omega_k`. Default is None.
    w_0_ : None or float ArrayLike, optional
        Dark energy equation of state constant parameter :math:`w_0`. Default is None.
    w_a_ : None or float ArrayLike, optional
        Dark energy equation of state linear parameter :math:`w_a`. Default is None.
    transfer_fit : bool, optional
        Whether to use Eisenstein & Hu fit to transfer function. Default is True
        (subject to change when False is implemented).
    transfer_fit_nowiggle : bool, optional
        Whether to use non-oscillatory transfer function fit.
    transfer_lgk_min : float, optional
        Minimum transfer function wavenumber in :math:`1/L` in log10.
    transfer_lgk_max : float, optional
        Maximum transfer function wavenumber in :math:`1/L` in log10.
    transfer_lgk_maxstep : float, optional
        Maximum transfer function wavenumber step size in :math:`1/L` in log10. It
        determines the number of wavenumbers ``transfer_k_num``, the actual step size
        ``transfer_lgk_step``, and the wavenumbers ``transfer_k``.
    growth_rtol : float, optional
        Relative tolerance for solving the growth ODEs.
    growth_atol : float, optional
        Absolute tolerance for solving the growth ODEs.
    growth_inistep: float, None, or 2-tuple of float or None, optional
        The initial step size for solving the growth ODEs. If None, use estimation. If a
        tuple, use the two step sizes for forward and reverse integrations,
        respectively.
    growth_lga_min : float, optional
        Minimum growth function scale factor in log10.
    growth_lga_max : float, optional
        Maximum growth function scale factor in log10.
    growth_lga_maxstep : float, optional
        Maximum growth function scale factor step size in log10. It determines the
        number of scale factors ``growth_a_num``, the actual step size
        ``growth_lga_step``, and the scale factors ``growth_a``.
    dtype : DTypeLike, optional
        Parameter float dtype.

    Class Variables
    ---------------
    T_cmb_fixed : float
        Fixed value if ``T_cmb_`` is not specified.
    Omega_k_fixed : float
        Fixed value if ``Omega_k_`` is not specified.
    w_0_fixed : float
        Fixed value if ``w_0_`` is not specified.
    w_a_fixed : float
        Fixed value if ``w_a_`` is not specified.
    M_sun_SI : float
        Solar mass :math:`M_\odot` in kg.
    Mpc_SI : float
        Mpc in m.
    H_0_SI : float
        Hubble constant :math:`H_0` in :math:`h`/s.
    c_SI : int
        Speed of light :math:`c` in m/s.
    G_SI : float
        Gravitational constant :math:`G` in m:math:`^3`/kg/s:math:`^2`
    M : float
        Mass unit :math:`M` defined in kg/:math:`h`. Default is :math:`10^{10}
        M_\odot/h`.
    L : float
        Length unit :math:`L` defined in m/:math:`h`. Default is Mpc/:math:`h`.
    T : float
        Time unit :math:`T` defined in s/:math:`h`. Default is Hubble time :math:`1/H_0
        \sim 10^{10}` years/:math:`h \sim` age of the Universe. So the default velocity
        unit is :math:`L/T =` 100 km/s.
    k_pivot_Mpc : float
        Primordial scalar power spectrum pivot scale :math:`k_\mathrm{pivot}` in 1/Mpc.

    """

    # TODO keyword-only for python>=3.10

    A_s_1e9: ArrayLike
    n_s: ArrayLike
    Omega_m: ArrayLike
    Omega_b: ArrayLike
    h: ArrayLike

    T_cmb_: Optional[ArrayLike] = None
    T_cmb_fixed: ClassVar[float] = 2.7255  # Fixsen 2009, arXiv:0911.1955

    Omega_k_: Optional[ArrayLike] = None
    Omega_k_fixed: ClassVar[float] = 0
    w_0_: Optional[ArrayLike] = None
    w_0_fixed: ClassVar[float] = -1
    w_a_: Optional[ArrayLike] = None
    w_a_fixed: ClassVar[float] = 0

    # constants in SI units
    M_sun_SI: ClassVar[float] = 1.98847e30
    Mpc_SI: ClassVar[float] = 3.0856775815e22
    H_0_SI: ClassVar[float] = 1e5 / Mpc_SI
    c_SI: ClassVar[int] = 299792458
    G_SI: ClassVar[float] = 6.67430e-11

    # Units
    M: ClassVar[float] = 1e10 * M_sun_SI
    L: ClassVar[float] = Mpc_SI
    T: ClassVar[float] = 1 / H_0_SI

    k_pivot_Mpc: ClassVar[float] = 0.05

    transfer_fit: bool = True
    transfer_fit_nowiggle: bool = False
    transfer_lgk_min: float = -4
    transfer_lgk_max: float = 3
    transfer_lgk_maxstep: float = 1/128
    transfer: Optional[Array] = field(default=None, compare=False)

    growth_rtol: Optional[float] = None
    growth_atol: Optional[float] = None
    growth_inistep: Union[float, None,
                          Tuple[Optional[float], Optional[float]]] = (1, None)
    growth_lga_min: float = -3
    growth_lga_max: float = 1
    growth_lga_maxstep: float = 1/128
    growth: Optional[Array] = field(default=None, compare=False)

    varlin: Optional[Array] = field(default=None, compare=False)

    dtype: DTypeLike = jnp.float64

    def __post_init__(self):
        if self._is_transforming():
            return

        object.__setattr__(self, 'dtype', jnp.dtype(self.dtype))
        if not jnp.issubdtype(self.dtype, jnp.floating):
            raise ValueError('dtype must be floating point numbers')

        for name, value in self.named_children():
            value = tree_map(partial(jnp.asarray, dtype=self.dtype), value)
            object.__setattr__(self, name, value)

        with ensure_compile_time_eval():
            object.__setattr__(
                self,
                'var_tophat',
                TophatVar(self.transfer_k[1:], lowring=True, backend='jax'),
            )

        # ~ 1.5e-8 for float64, 3.5e-4 for float32
        growth_tol = math.sqrt(jnp.finfo(self.dtype).eps)
        if self.growth_rtol is None:
            object.__setattr__(self, 'growth_rtol', growth_tol)
        if self.growth_atol is None:
            object.__setattr__(self, 'growth_atol', growth_tol)

    def __add__(self, other):
        return tree_map(add, self, other)

    def __sub__(self, other):
        return tree_map(sub, self, other)

    def __mul__(self, other):
        return tree_map(lambda x: x * other, self)

    def __rmul__(self, other):
        return self.__mul__(other)

    @classmethod
    def from_sigma8(cls, sigma8, *args, **kwargs):
        r"""Construct cosmology with :math:`\sigma_8` instead of :math:`A_s`."""
        cosmo = cls(1, *args, **kwargs)
        cosmo = cosmo.prime()  #FIXME set distance=False

        A_s_1e9 = (sigma8 / cosmo.sigma8)**2

        return cls(A_s_1e9, *args, **kwargs)

    def prime(self, transfer=True, growth=True, varlin=True):
        """Tabulate and cache the transfer and growth functions, etc.

        Parameters
        ----------
        transfer : bool or None, optional
            Whether to compute the transfer function, leave it as is, or set it to None.
        growth : bool or None, optional
            Whether to compute the growth functions, leave it as is, or set it to None.
        varlin : bool or None, optional
            Whether to compute the linear matter overdensity variance, leave it as is,
            or set it to None.

        Returns
        -------
        cosmo : Cosmology
            A new instance containing transfer and growth tables, etc.

        """
        cosmo = self

        if transfer:
            cosmo = boltzmann.transfer_tab(cosmo)
        elif transfer is None:
            cosmo = cosmo.replace(transfer=None)

        if growth:
            cosmo = boltzmann.growth_tab(cosmo)
        elif growth is None:
            cosmo = cosmo.replace(growth=None)

        if varlin:
            cosmo = boltzmann.varlin_tab(cosmo)
        elif varlin is None:
            cosmo = cosmo.replace(varlin=None)

        return cosmo

    def astype(self, dtype):
        """Cast parameters to dtype."""
        return self.replace(dtype=dtype)  # calls __init__ and then __post_init__

    @property
    def k_pivot(self):
        r"""Primordial scalar power spectrum pivot scale :math:`k_\mathrm{pivot}` in :math:`1/L`."""
        return self.k_pivot_Mpc / (self.h * self.Mpc_SI) * self.L

    @property
    def A_s(self):
        r"""Primordial scalar power spectrum amplitude :math:`A_\mathrm{s}`."""
        return self.A_s_1e9 * 1e-9

    @property
    def Omega_c(self):
        r"""Cold dark matter density parameter today :math:`\Omega_\mathrm{c}`."""
        return self.Omega_m - self.Omega_b

    @property
    def T_cmb(self):
        r"""CMB temperature in Kelvin today :math:`T_\mathrm{CMB}."""
        return self.T_cmb_fixed if self.T_cmb_ is None else self.T_cmb_

    @property
    def Omega_k(self):
        r"""Spatial curvature density parameter today :math:`\Omega_k`."""
        return self.Omega_k_fixed if self.Omega_k_ is None else self.Omega_k_

    @property
    def Omega_de(self):
        r"""Dark energy density parameter today :math:`\Omega_\mathrm{de}`."""
        return 1 - (self.Omega_m + self.Omega_k)

    @property
    def w_0(self):
        """Dark energy equation of state constant parameter :math:`w_0`."""
        return self.w_0_fixed if self.w_0_ is None else self.w_0_

    @property
    def w_a(self):
        """Dark energy equation of state linear parameter :math:`w_a`."""
        return self.w_a_fixed if self.w_a_ is None else self.w_a_

    @property
    def sigma8(self):
        r"""Linear matter rms overdensity within a tophat sphere of 8 Mpc/:math:`h` radius today :math:`\sigma_8`."""
        R = 8 * self.Mpc_SI / self.L
        return jnp.sqrt(boltzmann.varlin(R, 1, self))

    @property
    def H_0(self):
        """Hubble constant :math:`H_0` in :math:`1/T`."""
        return self.H_0_SI * self.T

    @property
    def c(self):
        """Speed of light :math:`c` in :math:`L/T`."""
        return self.c_SI * self.T / self.L

    @property
    def G(self):
        """Gravitational constant :math:`G` in :math:`L^3 / M / T^2`."""
        return self.G_SI * self.M * self.T**2 / self.L**3

    @property
    def rho_crit(self):
        r"""Critical density :math:`\rho_\mathrm{crit}` in :math:`M / L^3`."""
        return 3 * self.H_0**2 / (8 * jnp.pi * self.G)

    @property
    def transfer_k_num(self):
        """Number of transfer function wavenumbers, including a leading 0."""
        return 1 + math.ceil((self.transfer_lgk_max - self.transfer_lgk_min)
                             / self.transfer_lgk_maxstep) + 1

    @property
    def transfer_lgk_step(self):
        """Transfer function wavenumber step size in :math:`1/L` in log10."""
        return ((self.transfer_lgk_max - self.transfer_lgk_min)
                / (self.transfer_k_num - 2))

    @property
    def transfer_k(self):
        """Transfer function wavenumbers in :math:`1/L`."""
        k = jnp.logspace(self.transfer_lgk_min, self.transfer_lgk_max,
                         num=self.transfer_k_num - 1, dtype=self.dtype)
        return jnp.concatenate((jnp.array([0]), k))

    @property
    def growth_a_num(self):
        """Number of growth function scale factors, including a leading 0."""
        return 1 + math.ceil((self.growth_lga_max - self.growth_lga_min)
                             / self.growth_lga_maxstep) + 1

    @property
    def growth_lga_step(self):
        """Growth function scale factor step size in log10."""
        return ((self.growth_lga_max - self.growth_lga_min)
                / (self.growth_a_num - 2))

    @property
    def growth_a(self):
        """Growth function scale factors."""
        a = jnp.logspace(self.growth_lga_min, self.growth_lga_max,
                         num=self.growth_a_num - 1, dtype=self.dtype)
        return jnp.concatenate((jnp.array([0]), a))

    @property
    def varlin_R(self):
        """Radii of tophat spheres in :math:`L` for linear matter overdensity variance."""
        return self.var_tophat.y


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
    r"""Squared Hubble parameter time scaling factors, :math:`E^2`.

    Parameters
    ----------
    a : ArrayLike
        Scale factors.
    cosmo : Cosmology

    Returns
    -------
    E2 : jax.Array of cosmo.dtype
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
    a = jnp.asarray(a, dtype=cosmo.dtype)

    de_a = a**(-3 * (1 + cosmo.w_0 + cosmo.w_a)) * jnp.exp(-3 * cosmo.w_a * (1 - a))
    return cosmo.Omega_m * a**-3 + cosmo.Omega_k * a**-2 + cosmo.Omega_de * de_a


@partial(jnp.vectorize, excluded=(1,))
def H_deriv(a, cosmo):
    r"""Hubble parameter derivatives, :math:`\mathrm{d}\ln H / \mathrm{d}\ln a`.

    Parameters
    ----------
    a : ArrayLike
        Scale factors.
    cosmo : Cosmology

    Returns
    -------
    dlnH_dlna : jax.Array of cosmo.dtype
        Hubble parameter derivatives.

    """
    a = jnp.asarray(a, dtype=cosmo.dtype)

    E2_value, E2_grad = value_and_grad(E2)(a, cosmo)
    return 0.5 * a * E2_grad / E2_value


def Omega_m_a(a, cosmo):
    r"""Matter density parameters, :math:`\Omega_\mathrm{m}(a)`.

    Parameters
    ----------
    a : ArrayLike
        Scale factors.
    cosmo : Cosmology

    Returns
    -------
    Omega : jax.Array of cosmo.dtype
        Matter density parameters.

    Notes
    -----

    .. math::

        \Omega_\mathrm{m}(a) = \frac{\Omega_\mathrm{m} a^{-3}}{E^2(a)}

    """
    a = jnp.asarray(a, dtype=cosmo.dtype)

    return cosmo.Omega_m / (a**3 * E2(a, cosmo))
