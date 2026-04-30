from functools import partial
import math
from types import MappingProxyType

from jax import Array, ensure_compile_time_eval
from jax.typing import ArrayLike, DTypeLike
import jax.numpy as jnp
import scipy.special
from mcfit import mcfit, TophatVar

from pmwd.constants import Constants
from pmwd.background import distance_cache
from pmwd.perturbation import transfer_cache, growth_cache, varlin_cache, varlin
from pmwd.tree_util import (Tree, TanMixin, pytree_dataclass, dyn_field, fxd_field,
                            aux_field, issubdtype_of, asarray_of)


cosmo_dyn_field = partial(dyn_field, validate=asarray_of(field='dtype'))
cosmo_dyn_field.__doc__ = '`tree_util.dyn_field` with `Cosmology.dtype` casting.'
cosmo_fxd_field = partial(fxd_field, validate=asarray_of(field='dtype'))
cosmo_fxd_field.__doc__ = '`tree_util.fxd_field` with `Cosmology.dtype` casting.'


# FIXME is float32 enough for cosmology? especially parameter gradients?


def _eps2tol(dtype):
    return math.sqrt(jnp.finfo(dtype).eps)


#FIXME search: can I return within "with"?
def _init_var_tophat(self):
    with ensure_compile_time_eval():
        return TophatVar(self.transfer_k[1:], lowring=True, backend='jax')


@pytree_dataclass
class Cosmology(TanMixin, Tree):
    r"""Cosmological parameters and configurations.

    Parameters
    ----------
    dtype : DTypeLike, optional
        Parameter float dtype.
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
    m_nu : float ArrayLike, optional
        Neutrino masses :math:`m_\nu` in eV/:math:`c^2`, massless by default.
    T_cmb : float ArrayLike, optional
        CMB temperature in Kelvin today :math:`T_\mathrm{CMB}`.
    N_eff : float ArrayLike, optional
        Effective number of relativistic neutrino species :math:`N_\mathrm{eff}".
    Omega_K : float ArrayLike, optional
        Spatial curvature density parameter today :math:`Omega_K`
    w_0 : float ArrayLike, optional
        Dark energy equation of state constant parameter :math:`w_0`.
    w_a : float ArrayLike, optional
        Dark energy equation of state linear parameter :math:`w_a`.
    k_pivot_Mpc : float ArrayLike, optional
        Primordial scalar power spectrum pivot scale :math:`k_\mathrm{pivot}` in 1/Mpc.
    const : Constants, optional
        Physical constants in SI units.
    M : float ArrayLike, optional
        Mass unit :math:`M` in kg/:math:`h`. Default is :math:`10^{10} M_\odot/h`.
    L : float ArrayLike, optional
        Length unit :math:`L` in m/:math:`h`. Default is Mpc/:math:`h`.
    T : float ArrayLike, optional
        Time unit :math:`T` in s/:math:`h`. Default is Hubble time :math:`1/H_0 \sim
        10^{10}` years/:math:`h \sim` age of the Universe. So the default velocity unit
        is :math:`L/T =` 100 km/s.
    A : float ArrayLike, optional
        Angular unit in radians. Default is arcsec.
    distance_lga_min : float, optional
        Minimum distance scale factor in log10.
    distance_lga_max : float, optional
        Maximum distance scale factor in log10.
    distance_lga_maxstep : float, optional
        Maximum distance scale factor step size in log10. It determines the number of
        scale factors `distance_a_num`, the actual step size `distance_lga_step`, and
        the scale factors `distance_a`.
    #FIXME use transfer_function: Callable = aux_field(...) instead
    #transfer_fit : bool, optional
    #    Whether to use Eisenstein & Hu fit to transfer function. Default is True
    #    (subject to change when False is implemented).
    #transfer_fit_nowiggle : bool, optional
    #    Whether to use non-oscillatory transfer function fit.
    transfer_lgk_min : float, optional
        Minimum transfer function wavenumber in :math:`1/L` in log10.
    transfer_lgk_max : float, optional
        Maximum transfer function wavenumber in :math:`1/L` in log10.
    transfer_lgk_maxstep : float, optional
        Maximum transfer function wavenumber step size in :math:`1/L` in log10. It
        determines the number of wavenumbers `transfer_k_num`, the actual step size
        `transfer_lgk_step`, and the wavenumbers `transfer_k`.
    growth_rtol : float, optional
        Relative tolerance for solving the growth ODEs. Default is sqrt of `dtype`
        `jax.numpy.finfo.eps`, i.e., :math:`1.5 \times 10^{-8}` for float64 and
        :math:`3.5 \times 10^{-4}` for float32.
    growth_atol : float, optional
        Absolute tolerance for solving the growth ODEs. Default is sqrt of `dtype`
        `jax.numpy.finfo.eps`, i.e., :math:`1.5 \times 10^{-8}` for float64 and
        :math:`3.5 \times 10^{-4}` for float32.
    growth_inistep: float, None, or 2-tuple of them, optional
        The initial step size for solving the growth ODEs. If None, use estimation. If a
        tuple, use the two step sizes for forward and reverse integrations,
        respectively.
    growth_lga_min : float, optional
        Minimum growth function scale factor in log10.
    growth_lga_max : float, optional
        Maximum growth function scale factor in log10.
    growth_lga_maxstep : float, optional
        Maximum growth function scale factor step size in log10. It determines the
        number of scale factors `growth_a_num`, the actual step size `growth_lga_step`,
        and the scale factors `growth_a`.

    """

    dtype: DTypeLike = aux_field(default=float,
                                 validate=(jnp.dtype, issubdtype_of(jnp.floating)))

    A_s_1e9: ArrayLike = cosmo_dyn_field()
    n_s: ArrayLike = cosmo_dyn_field()
    Omega_m: ArrayLike = cosmo_dyn_field()
    Omega_b: ArrayLike = cosmo_dyn_field()
    h: ArrayLike = cosmo_dyn_field()
    m_nu: ArrayLike = dyn_field(optional=True, validate=(asarray_of(field='dtype'),
                                                         jnp.ravel))

    T_cmb: ArrayLike = fxd_field(default=2.7255)  # Fixsen 2009, arXiv:0911.1955
    N_eff: ArrayLike = fxd_field(default=3.044)
    Omega_K: ArrayLike = fxd_field(default=0.)
    w_0: ArrayLike = fxd_field(default=-1.)
    w_a: ArrayLike = fxd_field(default=0.)
    k_pivot_Mpc: ArrayLike = fxd_field(default=0.05)

    const: Constants = dyn_field(depend=lambda self: Constants(), repr=False)

    M: ArrayLike = fxd_field(depend=lambda self: 1e10 * self.const.M_sun)
    L: ArrayLike = fxd_field(depend=lambda self: self.const.Mpc)
    T: ArrayLike = fxd_field(depend=lambda self: 1 / self.const.H_0)
    A: ArrayLike = fxd_field(default=jnp.pi/(180*3600))

    distance_lga_min: float = aux_field(default=-3)
    distance_lga_max: float = aux_field(default=1)
    distance_lga_maxstep: float = aux_field(default=1/128)
    distance: Array | None = cosmo_dyn_field(cache=distance_cache, compare=False)

    transfer_fit: bool = aux_field(default=True)
    transfer_fit_nowiggle: bool = aux_field(default=False)
    transfer_lgk_min: float = aux_field(default=-4)
    transfer_lgk_max: float = aux_field(default=3)
    transfer_lgk_maxstep: float = aux_field(default=1/128)
    transfer: Array | None = cosmo_dyn_field(cache=transfer_cache, compare=False)

    growth_rtol: float = aux_field(depend=lambda self: _eps2tol(self.dtype))
    growth_atol: float = aux_field(depend=lambda self: _eps2tol(self.dtype))
    growth_inistep: (float | None
                     | tuple[float|None, float|None]) = aux_field(default=(1, 1))  # FIXME (1, None) used to work? but now also causes nan in sigma_8 gradients
    growth_lga_min: float = aux_field(default=-3)
    growth_lga_max: float = aux_field(default=1)
    growth_lga_maxstep: float = aux_field(default=1/128)
    growth: Array | None = cosmo_dyn_field(cache=growth_cache, compare=False)

    varlin: Array | None = cosmo_dyn_field(cache=varlin_cache, compare=False)

    #FIXME although mcfit.mcfit is hashable but maybe this can be more functional
    _var_tophat: mcfit = aux_field(depend=_init_var_tophat)

    @classmethod
    def from_sigma_8(cls, sigma_8, *args, **kwargs):
        r"""Construct cosmology with :math:`\sigma_8` instead of :math:`A_s`."""
        cosmo = cls(1, *args, **kwargs)
        cosmo = cosmo.cache_purge(transfer=True, growth=True, varlin=True)

        A_s_1e9 = (sigma_8 / cosmo.sigma_8)**2

        return cls(A_s_1e9, *args, **kwargs)

    def astype(self, dtype):
        """Return a new object with pytree children casted to `dtype`."""
        return self.replace(dtype=dtype)

    @property
    def H_0(self):
        """Hubble constant :math:`H_0` in :math:`1/T`."""
        return self.const.H_0 * self.T

    @property
    def c(self):
        """Speed of light :math:`c` in :math:`L/T`."""
        return self.const.c * self.T / self.L

    @property
    def G(self):
        """Gravitational constant :math:`G` in :math:`L^3 / M / T^2`."""
        return self.const.G * self.M * self.T**2 / self.L**3

    @property
    def d_H(self):
        """Hubble distance :math:`d_H = c / H_0` in :math:`L`."""
        return self.c / self.H_0

    @property
    def rho_crit(self):
        r"""Critical density :math:`\rho_\mathrm{crit}` in :math:`M / L^3`."""
        return 3 * self.H_0**2 / (8 * jnp.pi * self.G)

    @property
    def k_pivot(self):
        r"""Primordial scalar power spectrum pivot scale :math:`k_\mathrm{pivot}` in :math:`1/L`."""
        return self.k_pivot_Mpc / (self.h * self.const.Mpc) * self.L

    @property
    def A_s(self):
        r"""Primordial scalar power spectrum amplitude :math:`A_\mathrm{s}`."""
        return self.A_s_1e9 * 1e-9

    @property
    def T_nu(self):
        r"""Neutrino temperature in Kelvin today :math:`T_\nu \approx
        \Bigl(\frac{4}{11}\Bigr)^{1/3} \Bigl(\frac{N_\mathrm{eff}}{3}\Bigr)^{1/4}
        T_\mathrm{CMB}`."""
        return math.cbrt(4/11) * jnp.sqrt(jnp.sqrt(self.N_eff / 3)) * self.T_cmb

    @property
    def M_nu(self):
        r"""Sum of neutrino masses :math:`M_\nu = \sum m_\nu` in eV/:math:`c^2`."""
        if self.m_nu is None:
            return 0
        return self.m_nu.sum()

    @property
    def Omega_nu(self):
        r"""Massive neutrino density parameter today :math:`\Omega_\nu`, or 0 if
        massless."""
        if self.m_nu is None:
            return 0
        return self.omega_nu * self.h**-2

    @property
    def Omega_cb(self):
        r"""Cold dark and baryonic matter density parameter today
        :math:`\Omega_\mathrm{cb}`."""
        return self.Omega_m - self.Omega_nu

    @property
    def Omega_c(self):
        r"""Cold dark matter density parameter today :math:`\Omega_\mathrm{c}`."""
        return self.Omega_cb - self.Omega_b

    @property
    def omega_m(self):
        r"""Total matter *physical* density parameter today :math:`\omega_\mathrm{m} =
        \Omega_\mathrm{m} h^2`."""
        return self.Omega_m * self.h**2

    @property
    def omega_nu(self):
        r"""Massive neutrino *physical* density parameter today :math:`\omega_\nu =
        \Omega_\nu h^2`, or 0 if massless."""
        if self.m_nu is None:
            return 0
        coeff = (4 / jnp.pi * scipy.special.zeta(3)).item()
        return (coeff * self.const.G / self.const.H_0**2
                * (self.const.k * self.T_nu / self.const.hbar / self.const.c) ** 3
                * self.M_nu * self.const.e / self.const.c**2)

    @property
    def omega_cb(self):
        r"""Cold dark and baryonic matter *physical* density parameter today
        :math:`\omega_\mathrm{cb} = \Omega_mathrm{cb} h^2`."""
        return self.Omega_cb * self.h**2

    @property
    def omega_b(self):
        r"""Baryonic matter *physical* density parameter today :math:`\omega_\mathrm{b}
        = \Omega_\mathrm{b} h^2`."""
        return self.Omega_b * self.h**2

    @property
    def omega_c(self):
        r"""Cold dark matter *physical* density parameter today :math:`\omega_\mathrm{c}
        = \Omega_\mathrm{c} h^2`."""
        return self.Omega_c * self.h**2

    @property
    def f_nu(self):
        r"""Massive neutrino density fraction :math:`f_\nu = \Omega_\nu /
        \Omega_\matherm{m}`, or 0 if massless."""
        if self.m_nu is None:
            return 0
        return self.Omega_nu / self.Omega_m

    @property
    def f_cb(self):
        r"""Cold dark and baryonic matter density fraction :math:`f_\mathrm{cb} =
        \Omega_mathrm{cb} / \Omega_\matherm{m}`."""
        return self.Omega_cb / self.Omega_m

    @property
    def f_b(self):
        r"""Baryonic matter density fraction :math:`f_\mathrm{b} = \Omega_\mathrm{b} /
        \Omega_\matherm{m}`."""
        return self.Omega_b / self.Omega_m

    @property
    def f_c(self):
        r"""Cold dark matter density fraction :math:`f_\mathrm{c} = \Omega_\mathrm{c} /
        \Omega_\matherm{m}`."""
        return self.Omega_c / self.Omega_m

    @property
    def K(self):
        """Spatial Gaussian curvature :math:`K` in :math:`1/L^2`."""
        return - self.Omega_K / self.d_H**2

    @property
    def Omega_de(self):
        r"""Dark energy density parameter today :math:`\Omega_\mathrm{de}`."""
        return 1 - (self.Omega_m + self.Omega_K)

    @property
    def sigma_8(self):
        r"""Linear matter rms overdensity within a tophat sphere of 8 Mpc/:math:`h`
        radius today :math:`\sigma_8`."""
        R = 8 * self.const.Mpc / self.L
        return jnp.sqrt(varlin(R, 1, self))

    @property
    def distance_a_num(self):
        """Number of distance scale factors, including a leading 0."""
        return 1 + math.ceil((self.distance_lga_max - self.distance_lga_min)
                             / self.distance_lga_maxstep) + 1

    @property
    def distance_lga_step(self):
        """Distance scale factor step size in log10."""
        return ((self.distance_lga_max - self.distance_lga_min)
                / (self.distance_a_num - 2))

    @property
    def distance_a(self):
        """Distance scale factors, starting from 0."""
        a = jnp.logspace(self.distance_lga_min, self.distance_lga_max,
                         num=self.distance_a_num - 1, dtype=self.dtype)
        return jnp.concatenate((jnp.array([0]), a))

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
        """Transfer function wavenumbers in :math:`1/L`, starting from 0."""
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
        """Radii of tophat spheres in :math:`L` for linear matter overdensity variance,
        determined by `transfer_k` and the FFTLog algorithm."""
        return self._var_tophat.y


# Simple ΛCDM cosmology, for convenience and subject to change
simple_LCDM = MappingProxyType(dict(
    A_s_1e9=2.0,
    n_s=0.96,
    Omega_m=0.3,
    Omega_b=0.05,
    h=0.7,
))

# Planck 2018 cosmology, arXiv:1807.06209 Table 2 last column
Planck_18 = MappingProxyType(dict(
    A_s_1e9=2.105,
    n_s=0.9665,
    Omega_m=0.3111,
    Omega_b=0.04897,
    h=0.6766,
    m_nu=[0.06],
))
