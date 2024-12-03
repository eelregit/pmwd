from functools import partial
import math

from jax import Array, ensure_compile_time_eval
from jax.typing import ArrayLike, DTypeLike
import jax.numpy as jnp
from mcfit import mcfit, TophatVar

from pmwd import background, perturbation
from pmwd.tree_util import DataTree, pytree_dataclass, dyn_field, fxd_field, aux_field, issubdtype_of, asarray_of


cosmo_dyn_field = partial(dyn_field, validate=asarray_of(field='dtype'))
cosmo_dyn_field.__doc__ = 'Like `tree_util.dyn_field` with `Cosmology.dtype` casting.'
cosmo_fxd_field = partial(fxd_field, validate=asarray_of(field='dtype'))
cosmo_fxd_field.__doc__ = 'Like `tree_util.fxd_field` with `Cosmology.dtype` casting.'


# FIXME is float32 enough for cosmology? especially parameter gradients?


def _eps2tol(dtype):
    return math.sqrt(jnp.finfo(dtype).eps)


#FIXME search: can I return within "with"?
def _init_var_tophat(self):
    with ensure_compile_time_eval():
        return TophatVar(self.transfer_k[1:], lowring=True, backend='jax')


@pytree_dataclass
class Cosmology(DataTree):
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
    T_cmb : float ArrayLike, optional
        CMB temperature in Kelvin today :math:`T_\mathrm{CMB}`.
    Omega_K : float ArrayLike, optional
        Spatial curvature density parameter today :math:`Omega_K`
    w_0 : float ArrayLike, optional
        Dark energy equation of state constant parameter :math:`w_0`.
    w_a : float ArrayLike, optional
        Dark energy equation of state linear parameter :math:`w_a`.
    k_pivot_Mpc : float ArrayLike, optional
        Primordial scalar power spectrum pivot scale :math:`k_\mathrm{pivot}` in 1/Mpc.
    M_sun_SI : float ArrayLike, optional
        Solar mass :math:`M_\odot` in kg.
    Mpc_SI : float ArrayLike, optional
        Mpc in m.
    H_0_SI : float ArrayLike, optional
        Hubble constant :math:`H_0` in :math:`h`/s.
    c_SI : float ArrayLike, optional
        Speed of light :math:`c` in m/s.
    G_SI : float ArrayLike, optional
        Gravitational constant :math:`G` in m:math:`^3`/kg/s:math:`^2`
    M : float ArrayLike, optional
        Mass unit :math:`M` defined in kg/:math:`h`. Default is :math:`10^{10}
        M_\odot/h`.
    L : float ArrayLike, optional
        Length unit :math:`L` defined in m/:math:`h`. Default is Mpc/:math:`h`.
    T : float ArrayLike, optional
        Time unit :math:`T` defined in s/:math:`h`. Default is Hubble time :math:`1/H_0
        \sim 10^{10}` years/:math:`h \sim` age of the Universe. So the default velocity
        unit is :math:`L/T =` 100 km/s.
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

    # TODO keyword-only for python>=3.10

    dtype: DTypeLike = aux_field(default=float,
                                 validate=(jnp.dtype, issubdtype_of(jnp.floating)))

    A_s_1e9: ArrayLike = cosmo_dyn_field()
    n_s: ArrayLike = cosmo_dyn_field()
    Omega_m: ArrayLike = cosmo_dyn_field()
    Omega_b: ArrayLike = cosmo_dyn_field()
    h: ArrayLike = cosmo_dyn_field()

    T_cmb: ArrayLike = cosmo_fxd_field(default=2.7255)  # Fixsen 2009, arXiv:0911.1955
    Omega_K: ArrayLike = cosmo_fxd_field(default=0)
    w_0: ArrayLike = cosmo_fxd_field(default=-1)
    w_a: ArrayLike = cosmo_fxd_field(default=0)
    k_pivot_Mpc: ArrayLike = cosmo_fxd_field(default=0.05)

    # constants in SI units
    M_sun_SI: ArrayLike = cosmo_fxd_field(default=1.98847e30)
    Mpc_SI: ArrayLike = cosmo_fxd_field(default=3.0856775815e22)
    H_0_SI: ArrayLike = cosmo_fxd_field(default_function=lambda self: 1e5 / self.Mpc_SI)
    c_SI: ArrayLike = cosmo_fxd_field(default=299792458)
    G_SI: ArrayLike = cosmo_fxd_field(default=6.67430e-11)

    # units in SI units
    M: ArrayLike = cosmo_fxd_field(default_function=lambda self: 1e10 * self.M_sun_SI)
    L: ArrayLike = cosmo_fxd_field(default_function=lambda self: self.Mpc_SI)
    T: ArrayLike = cosmo_fxd_field(default_function=lambda self: 1 / self.H_0_SI)

    distance_lga_min: float = aux_field(default=-3)
    distance_lga_max: float = aux_field(default=1)
    distance_lga_maxstep: float = aux_field(default=1/128)
    distance: Array | None = cosmo_dyn_field(cache=background.distance_cache,
                                             compare=False)

    transfer_fit: bool = aux_field(default=True)
    transfer_fit_nowiggle: bool = aux_field(default=False)
    transfer_lgk_min: float = aux_field(default=-4)
    transfer_lgk_max: float = aux_field(default=3)
    transfer_lgk_maxstep: float = aux_field(default=1/128)
    transfer: Array | None = cosmo_dyn_field(cache=perturbation.transfer_cache,
                                             compare=False)

    growth_rtol: float = aux_field(default_function=lambda self: _eps2tol(self.dtype))
    growth_atol: float = aux_field(default_function=lambda self: _eps2tol(self.dtype))
    growth_inistep: (float | None
                     | tuple[float|None, float|None]) = aux_field(default=(1, 1))  # FIXME (1, None) used to work? but now also causes nan in sigma_8 gradients
    growth_lga_min: float = aux_field(default=-3)
    growth_lga_max: float = aux_field(default=1)
    growth_lga_maxstep: float = aux_field(default=1/128)
    growth: Array | None = cosmo_dyn_field(cache=perturbation.growth_cache,
                                           compare=False)

    varlin: Array | None = cosmo_dyn_field(cache=perturbation.varlin_cache,
                                           compare=False)

    #FIXME although mcfit.mcfit is hashable but maybe this can be more functional
    _var_tophat: mcfit = aux_field(default_function=_init_var_tophat)

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
    def K(self):
        """Spatial Gaussian curvature :math:`K` in :math:`1/L^2`."""
        return - self.Omega_K / self.d_H**2

    @property
    def Omega_de(self):
        r"""Dark energy density parameter today :math:`\Omega_\mathrm{de}`."""
        return 1 - (self.Omega_m + self.Omega_K)

    @property
    def sigma_8(self):
        r"""Linear matter rms overdensity within a tophat sphere of 8 Mpc/:math:`h` radius today :math:`\sigma_8`."""
        R = 8 * self.Mpc_SI / self.L
        return jnp.sqrt(perturbation.varlin(R, 1, self))

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
        """Distance scale factors."""
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
        """Radii of tophat spheres in :math:`L` for linear matter overdensity variance,
        determined by `transfer_k` and the FFTLog algorithm."""
        return self._var_tophat.y


SimpleLCDM = partial(
    Cosmology,
    A_s_1e9=2.0,
    n_s=0.96,
    Omega_m=0.3,
    Omega_b=0.05,
    h=0.7,
)
SimpleLCDM.__doc__ = 'Simple ΛCDM cosmology, for convenience and subject to change.'

Planck18 = partial(
    Cosmology,
    A_s_1e9=2.105,
    n_s=0.9665,
    Omega_m=0.3111,
    Omega_b=0.04897,
    h=0.6766,
)
Planck18.__doc__ = 'Planck 2018 cosmology, arXiv:1807.06209 Table 2 last column.'
