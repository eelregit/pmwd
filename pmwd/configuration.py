from functools import partial
from math import ceil
from typing import ClassVar, Optional, Tuple, Union

from numpy.typing import DTypeLike
import jax
jax.config.update("jax_enable_x64", True)
from jax import ensure_compile_time_eval
import jax.numpy as jnp
jnp.set_printoptions(precision=3, edgeitems=2, linewidth=128)

from pmwd.tree_util import pytree_dataclass
from pmwd.pm_util import next_fft_len


@partial(pytree_dataclass, aux_fields=Ellipsis, frozen=True)
class Configuration:
    """Configuration parameters, "immutable" as a frozen dataclass.

    Parameters
    ----------
    ptcl_spacing : float
        Lagrangian particle grid cell size in [L].
    ptcl_grid_shape : tuple of int
        Lagrangian particle grid shape, in ``len(ptcl_grid_shape)`` spatial dimensions.
    mesh_shape : int, float, or tuple of int, optional
        Mesh shape. If an int or float, it is used as the 1D mesh to particle grid shape
        ratio, to determine the mesh shape from that of the particle grid. The mesh grid
        cannot be smaller than the particle grid (int or float values must not be
        smaller than 1) and the two grids must have the same aspect ratio.
    cosmo_dtype : dtype_like, optional
        Float dtype for Cosmology and Configuration.
    pmid_dtype : dtype_like, optional
        Signed integer dtype for particle or mesh grid indices.
    float_dtype : dtype_like, optional
        Float dtype for other particle and mesh quantities.
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
        Whether to use non-oscillatory transfer function fit.
    transfer_size : int, optional
        Transfer function table size. Its wavenumbers ``transfer_k`` are log spaced
        spanning the full range of particle grid scales.
    growth_rtol : float, optional
        Relative tolerance for solving the growth ODEs.
    growth_atol : float, optional
        Absolute tolerance for solving the growth ODEs.
    lpt_order : int, optional
        LPT order, with 1 for Zel'dovich approximation, 2 for 2LPT, and 3 for 3LPT.
    lpt_padded_shape : int, float, or tuple of int, optional
        LPT grid shape with padding, to avoid aliasing for ``lpt_order >= 2``. See
        ``mesh_shape``.
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
    chunk_len : int, optional
        Chunk length to split particles in batches in scatter and gather to save memory.

    Raises
    ------
    ValueError
        Incorrect or inconsistent parameter values.

    """

    ptcl_spacing: float
    ptcl_grid_shape: Tuple[int, ...]  # tuple[int, ...] for python >= 3.9 (PEP 585)

    mesh_shape: Union[int, float, Tuple[int, ...]] = 1

    cosmo_dtype: DTypeLike = jnp.float64
    pmid_dtype: DTypeLike = jnp.int16
    float_dtype: DTypeLike = jnp.float32

    k_pivot_Mpc: float = 0.05

    T_cmb: float = 2.7255  # Fixsen 2009, arXiv:0911.1955

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
    lpt_padded_shape: Union[int, float, Tuple[int, ...]] = 1

    a_start: float = 1/64
    a_stop: float = 1.
    a_lpt_maxstep: float = 1/128
    a_nbody_maxstep: float = 1/64

    chunk_len: int = 2**24

    def __post_init__(self):
        if self._is_transforming():
            return

        self._set_fft_shape('mesh_shape', self.mesh_shape)
        self._set_fft_shape('lpt_padded_shape', self.lpt_padded_shape)

        object.__setattr__(self, 'cosmo_dtype', jnp.dtype(self.cosmo_dtype))
        object.__setattr__(self, 'pmid_dtype', jnp.dtype(self.pmid_dtype))
        object.__setattr__(self, 'float_dtype', jnp.dtype(self.float_dtype))
        if not jnp.issubdtype(self.cosmo_dtype, jnp.floating):
            raise ValueError('cosmo_dtype must be floating point numbers: '
                             f'{self.cosmo_dtype}')
        if not jnp.issubdtype(self.pmid_dtype, jnp.signedinteger):
            raise ValueError('pmid_dtype must be signed integers: {self.pmid_dtype}')
        if not jnp.issubdtype(self.float_dtype, jnp.floating):
            raise ValueError('float_dtype must be floating point numbers: '
                             f'{self.float_dtype}')

        # ~ 1.5e-8 for float64, 3.5e-4 for float32
        with ensure_compile_time_eval():
            growth_tol = jnp.sqrt(jnp.finfo(self.cosmo_dtype).eps).item()
        if self.growth_rtol is None:
            object.__setattr__(self, 'growth_rtol', growth_tol)
        if self.growth_atol is None:
            object.__setattr__(self, 'growth_atol', growth_tol)

        dtype = self.cosmo_dtype
        for name, value in self.named_children():
            value = tree_map(lambda x: jnp.asarray(x, dtype=dtype), value)
            object.__setattr__(self, name, value)

    def _set_fft_shape(self, name, shape):
        # try to find a good FFT shape, but without guarantee
        if isinstance(shape, (int, float)):
            shape = tuple(next_fft_len(round(sp * shape))
                          for sp in self.ptcl_grid_shape)
            shape = max(s / sp for s, sp in zip(shape, self.ptcl_grid_shape))
            shape = tuple(next_fft_len(round(sp * shape))
                          for sp in self.ptcl_grid_shape)
            object.__setattr__(self, name, shape)

        if len(shape) != len(self.ptcl_grid_shape):
            raise ValueError(f'{name} and ptcl_grid_shape dimensions differ: '
                             f'{shape}, {self.ptcl_grid_shape}')

        if any(s < sp for s, sp in zip(shape, self.ptcl_grid_shape)):
            raise ValueError(f'{name} cannot be smaller than ptcl_grid_shape: '
                             f'{shape}, {self.ptcl_grid_shape}')

        if any(self.ptcl_grid_shape[0] * s != shape[0] * sp
               for s, sp in zip(shape[1:], self.ptcl_grid_shape[1:])):
            raise ValueError(f'{name} and ptcl_grid_shape aspect ratios differ: '
                             f'{shape}, {self.ptcl_grid_shape}')

    @property
    def dim(self):
        """Spatial dimension."""
        return len(self.ptcl_grid_shape)

    @property
    def ptcl_cell_vol(self):
        """Lagrangian particle grid cell volume in [L^dim]."""
        return self.ptcl_spacing ** self.dim

    @property
    def ptcl_num(self):
        """Number of particles."""
        with jax.ensure_compile_time_eval():
            return jnp.array(self.ptcl_grid_shape).prod().item()

    @property
    def box_size(self):
        """Simulation box size tuple in [L]."""
        return tuple(self.ptcl_spacing * s for s in self.ptcl_grid_shape)

    @property
    def box_vol(self):
        """Simulation box volume in [L^dim]."""
        with jax.ensure_compile_time_eval():
            return jnp.array(self.box_size).prod().item()

    @property
    def cell_size(self):
        """Mesh cell size in [L]."""
        return self.ptcl_spacing * self.ptcl_grid_shape[0] / self.mesh_shape[0]

    @property
    def cell_vol(self):
        """Mesh cell volume in [L^dim]."""
        return self.cell_size ** self.dim

    @property
    def mesh_size(self):
        """Number of mesh grid points."""
        with jax.ensure_compile_time_eval():
            return jnp.array(self.mesh_shape).prod().item()

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
        """Number of LPT light cone scale factor steps, excluding ``a_start``."""
        return ceil(self.a_start / self.a_lpt_maxstep)

    @property
    def a_lpt_step(self):
        """LPT light cone scale factor step size."""
        return self.a_start / self.a_lpt_num

    @property
    def a_nbody_num(self):
        """Number of N-body time integration scale factor steps, excluding """
        """``a_start``."""
        return ceil((self.a_stop - self.a_start) / self.a_nbody_maxstep)

    @property
    def a_nbody_step(self):
        """N-body time integration scale factor step size."""
        return (self.a_stop - self.a_start) / self.a_nbody_num

    @property
    def a_lpt(self):
        """LPT light cone scale factor steps, including ``a_start``."""
        return jnp.linspace(0, self.a_start, num=self.a_lpt_num+1,
                            dtype=self.cosmo_dtype)

    @property
    def a_nbody(self):
        """N-body time integration scale factor steps, including ``a_start``."""
        return jnp.linspace(self.a_start, self.a_stop, num=1+self.a_nbody_num,
                            dtype=self.cosmo_dtype)

    @property
    def growth_a(self):
        """Growth function scale factors, for both LPT and N-body."""
        return jnp.concatenate((self.a_lpt, self.a_nbody[1:]))

    #TODO transfer_size -> transfer_lgk_maxstep
    @property
    def transfer_k(self):
        """Transfer function wavenumbers in [1/L], from the minimum fundamental """
        """wavenumber to the space diagonal Nyquist wavenumber of the particle grid."""
        log10_k_min = jnp.log10(2. * jnp.pi / self.box_size.max())
        log10_k_max = jnp.log10(jnp.sqrt(self.dim) * jnp.pi / self.ptcl_spacing)
        return jnp.logspace(log10_k_min, log10_k_max, num=self.transfer_size,
                            dtype=self.float_dtype)
