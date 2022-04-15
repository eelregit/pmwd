from functools import partial
from math import ceil
from typing import ClassVar, Optional

from numpy.typing import DTypeLike
import jax
jax.config.update("jax_enable_x64", True)
from jax import ensure_compile_time_eval
import jax.numpy as jnp

from pmwd.tree_util import pytree_dataclass


@partial(pytree_dataclass, aux_fields=Ellipsis, frozen=True)
class Configuration:
    """Configuration parameters, "immutable" as a frozen dataclass.

    Parameters
    ----------
    cell_size : float
        Mesh cell size in [L].
    mesh_shape : tuple of ints
        Mesh shape in ``len(mesh_shape)`` dimensions.
    ptcl_grid_shape : tuple of ints, optional
        Lagrangian particle grid shape. Default is ``mesh_shape``. Particle and mesh
        grids must have the same aspect ratio.
    cosmo_dtype : dtype_like, optional
        Float dtype for Cosmology. Default is float64.
    pmid_dtype : dtype_like, optional
        Signed integer dtype for particle pmid. Default is int16.
    float_dtype : dtype_like, optional
        Float dtype for other particle and mesh quantities. Default is float32.
    k_pivot_Mpc : float, optional
        Primordial scalar power spectrum pivot scale in 1/Mpc.
    T_cmb : float, optional
        CMB temperature in K.
    M : float, optional
        Mass unit defined in kg/h. Default is 1e10 M☉/h.
    L : float, optional
        Length unit defined in m/h. Default is Mpc/h.
    T : float, optional
        Time unit defined in s/h. Default is Hubble time 1/H_0 ~ 1e10 years/h ~ age of
        the Universe.
    transfer_fit : bool, optional
        Whether to use Eisenstein & Hu fit to transfer function. Default is True
        (subject to change when False is implemented).
    transfer_fit_nowiggle : bool, optional
        Whether to use non-oscillatory transfer function fit. Default is False.
    transfer_size : int, optional
        Transfer function table size. Wavenumbers ``transfer_k`` are log spaced spanning
        the full range of mesh scales, from the (minimum) fundamental frequency to the
        (space diagonal) Nyquist frequency.
    growth_rtol : float, optional
        Relative tolerance for solving the growth ODEs.
    growth_atol : float, optional
        Absolute tolerance for solving the growth ODEs.
    lpt_order : int in {1, 2}, optional
        LPT order. TODO: add 3rd order.
    a_start : float, optional
        LPT scale factor and N-body starting time.
    a_stop : float, optional
        N-body stopping time (scale factor).
    a_lpt_maxstep : float, optional
        Maximum scale factor LPT light cone step size. It determines the number of
        steps ``a_lpt_num``, the actual step size ``a_lpt_step``, and the steps
        ``a_lpt``.
    a_nbody_maxstep : float, optional
        Maximum scale factor N-body time integration step size. It determines the
        number of steps ``a_nbody_num``, the actual step size ``a_nbody_step``, and the
        steps ``a_nbody``.
    chunk_size : int, optional
        Chunk size of particles for scatter and gather. Default is 2**24.

    Raises
    ------
    ValueError
        Incorrect or inconsistent parameter values.

    """

    cell_size: float
    #max_disp_to_box_size_ratio: float  # shortest axis  # TODO recall what was this?
    mesh_shape: tuple[int, ...]

    ptcl_grid_shape: Optional[tuple[int, ...]] = None

    cosmo_dtype: DTypeLike = jnp.dtype(jnp.float64)
    pmid_dtype: DTypeLike = jnp.dtype(jnp.int16)
    float_dtype: DTypeLike = jnp.dtype(jnp.float32)

    k_pivot_Mpc: float = 0.05

    T_cmb: float = 2.726

    # constants in SI units, as class variables
    M_sun_SI: ClassVar[float] = 1.98847e30  # solar mass in kg
    Mpc_SI: ClassVar[float] = 3.0856775815e22  # Mpc in m
    H_0_SI: ClassVar[float] = 1e5 / Mpc_SI  # Hubble constant in h/s
    c_SI: ClassVar[int] = 299792458  # speed of light in m/s
    G_SI: ClassVar[float] = 6.67430e-11  # Gravitational constant in m^3/kg/s^2

    # Units
    M: float = 1e10 * M_sun_SI
    L: float = Mpc_SI
    T: float = 1 / H_0_SI

    transfer_fit: bool = True
    transfer_fit_nowiggle: bool = False
    transfer_size: int = 1024

    growth_rtol: Optional[float] = None
    growth_atol: Optional[float] = None

    lpt_order: int = 2

    a_start: float = 1/64
    a_stop: float = 1.
    a_lpt_maxstep: float = 1/128
    a_nbody_maxstep: float = 1/64

    chunk_size: int = 1<<24

    def __post_init__(self):
        if self.ptcl_grid_shape is None:
            object.__setattr__(self, 'ptcl_grid_shape', self.mesh_shape)
        elif len(self.ptcl_grid_shape) != len(self.mesh_shape):
            raise ValueError('particle and mesh grid dimensions differ')
        elif any(sp > sm for sp, sm in zip(self.ptcl_grid_shape, self.mesh_shape)):
            raise ValueError('particle grid cannot be larger than mesh grid')
        elif any(self.ptcl_grid_shape[0] * sm != self.mesh_shape[0] * sp
                 for sm, sp in zip(self.mesh_shape[1:], self.ptcl_grid_shape[1:])):
            raise ValueError('particle and mesh grid aspect ratios differ')

        if not jnp.issubdtype(self.cosmo_dtype, jnp.floating):
            raise ValueError('cosmo_dtype must be floating point numbers')
        if not jnp.issubdtype(self.pmid_dtype, jnp.signedinteger):
            raise ValueError('pmid_dtype for pmid must be signed integers')
        if not jnp.issubdtype(self.float_dtype, jnp.floating):
            raise ValueError('float_dtype must be floating point numbers')

        # ~ 1.5e-8 for float64, 3.5e-4 for float32
        with ensure_compile_time_eval():
            growth_tol = jnp.sqrt(jnp.finfo(self.cosmo_dtype).eps).item()
        if self.growth_rtol is None:
            object.__setattr__(self, 'growth_rtol', growth_tol)
        if self.growth_atol is None:
            object.__setattr__(self, 'growth_atol', growth_tol)

    @property
    def dim(self):
        """Spatial dimension."""
        return len(self.mesh_shape)

    @property
    def cell_vol(self):
        """Mesh cell volume in [L^dim]."""
        return self.cell_size ** self.dim

    @property
    def box_size(self):
        """Simulation box size tuple in [L]."""
        return tuple(self.cell_size * s for s in self.mesh_shape)

    @property
    def box_vol(self):
        """Simulation box volume in [L^dim]."""
        with jax.ensure_compile_time_eval():
            return jnp.array(self.box_size).prod().item()

    @property
    def mesh_size(self):
        """Number of mesh grid points."""
        with jax.ensure_compile_time_eval():
            return jnp.array(self.mesh_shape).prod().item()

    @property
    def ptcl_num(self):
        """Number of particles."""
        with jax.ensure_compile_time_eval():
            return jnp.array(self.ptcl_grid_shape).prod().item()

    @property
    def ptcl_spacing(self):
        """Lagrangian particle grid cell size in [L]."""
        return self.cell_size * self.mesh_shape[0] / self.ptcl_grid_shape[0]

    @property
    def ptcl_cell_vol(self):
        """Lagrangian particle grid cell volume in [L^dim]."""
        return self.ptcl_spacing ** self.dim

    @property
    def V(self):
        """Velocity unit as [L/T]. Default is 100 km/s."""
        return self.L / self.T

    @property
    def H_0(self):
        """Hubble constant H_0 in [1/T]."""
        return self.H_0_SI * self.T

    @property
    def c(self):
        """Speed of light in [L/T]."""
        return self.c_SI / self.V

    @property
    def G(self):
        """Gravitational constant in [L^3 / M / T^2]."""
        return self.G_SI * self.M / (self.L * self.V**2)

    @property
    def rho_crit(self):
        """Critical density in [M / L^3]."""
        return 3. * self.H_0**2 / (8. * jnp.pi * self.G)

    @property
    def a_lpt_num(self):
        """Number of LPT light cone scale factor steps, excluding a_start."""
        return ceil(self.a_start / self.a_lpt_maxstep)

    @property
    def a_lpt_step(self):
        """LPT light cone scale factor step size."""
        return self.a_start / self.a_lpt_num

    @property
    def a_nbody_num(self):
        """Number of N-body time integration scale factor steps, excluding a_start."""
        return ceil((self.a_stop - self.a_start) / self.a_nbody_maxstep)

    @property
    def a_nbody_step(self):
        """N-body time integration scale factor step size."""
        return (self.a_stop - self.a_start) / self.a_nbody_num

    @property
    def transfer_k(self):
        """Transfer function wavenumbers, from minimum fundamental to diagonal """
        """Nyquist frequencies."""
        log10_k_min = jnp.log10(2. * jnp.pi / self.box_size.max())
        log10_k_max = jnp.log10(jnp.sqrt(self.dim) * jnp.pi / self.cell_size)
        return jnp.logspace(log10_k_min, log10_k_max, num=self.transfer_size,
                            dtype=self.float_dtype)

    @property
    def growth_a(self):
        """Growth function scale factors, for both LPT and N-body."""
        growth_a_lpt = jnp.linspace(0, self.a_start, num=self.a_lpt_num,
                                    endpoint=False, dtype=self.cosmo_dtype)
        growth_a_nbody = jnp.linspace(self.a_start, self.a_stop, num=1+self.a_nbody_num,
                                      dtype=self.cosmo_dtype)
        return jnp.concatenate((growth_a_lpt, growth_a_nbody))

    @property
    def a_lpt(self):
        """LPT light cone scale factor steps, including a_start."""
        return jnp.linspace(0, self.a_start, num=self.a_lpt_num+1,
                            dtype=self.float_dtype)

    @property
    def a_nbody(self):
        """N-body time integration scale factor steps, including a_start."""
        return jnp.linspace(self.a_start, self.a_stop, num=1+self.a_nbody_num,
                            dtype=self.float_dtype)
