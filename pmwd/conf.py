from functools import partial
from pprint import pformat
from typing import ClassVar, Optional

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from pmwd.dataclasses import pytree_dataclass


# FIXME dtype whereever jnp


@partial(pytree_dataclass, aux_fields=Ellipsis, frozen=True)
class Configuration:
    """Configuration parameters, "immutable" as a frozen dataclass, not traced by JAX.

    Parameters
    ----------
    seed : int
        Seed for pseudo-random number generator.
    cell_size : float
        Mesh cell size in [L].
    mesh_shape : tuple of ints
        Mesh shape in ``len(mesh_shape)`` dimensions.
    ptcl_sample : int, optional
        Particle pre-initial conditions as a uniform grid that samples the mesh at this
        interval.
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
    transfer_size : int, optional
        Transfer function table size. Log spaced wavenumbers ``transfer_k`` is generated
        to span the full range of mesh scales, from the (minimum) fundamental frequency
        to the (space diagonal) Nyquist frequency.
    growth_lpt_size : int (>= 1), optional
        Growth function table size for LPT. Linearly spaced scale factors are generated
        from 0 to ``a_start``, and then concatenated with ``a_steps`` to make the growth
        function scale factors ``growth_a``.
    growth_dtype : jax.numpy.dtype, optional
        Float dtype for growth functions and derivatives. Default is float64.
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
    a_num : int, optional
        Number of N-body time integration steps, linearly spaced scale factors.
    chunk_size : int, optional
        Chunk size of particles for scatter and gather. Default is 2**24.
    int_dtype : jax.numpy.dtype, optional
        Integer dtype for particle pmid. Default is int16.
    float_dtype : jax.numpy.dtype, optional
        Float dtype for other particle and mesh attributes. Default is float32.

    Raises
    ------
    ValueError
        Incorrect or inconsistent parameter values.

    """

    seed: int

    cell_size: float
    #max_disp_to_box_size_ratio: float  # shortest axis  # TODO recall what was this?
    mesh_shape: tuple[int, ...]

    ptcl_sample: int = 2

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

    transfer_size: int = 1024

    growth_lpt_size: int = 128
    growth_dtype: jnp.dtype = jnp.dtype(jnp.float64)
    growth_rtol: Optional[float] = None
    growth_atol: Optional[float] = None

    lpt_order: int = 2

    a_start: float = 0.02
    a_stop: float = 1.
    a_num: int = 50

    chunk_size: int = 1<<24

    int_dtype: jnp.dtype = jnp.dtype(jnp.int16)
    float_dtype: jnp.dtype = jnp.dtype(jnp.float32)

    def __post_init__(self):
        if self.growth_rtol is None:
            object.__setattr__(self, 'growth_rtol',
                jnp.sqrt(jnp.finfo(self.growth_dtype).eps).item())  # ~ 1.5e-8 for float64
        if self.growth_atol is None:
            object.__setattr__(self, 'growth_atol',
                jnp.sqrt(jnp.finfo(self.growth_dtype).eps).item())  # ~ 1.5e-8 for float64

        if any(s % self.ptcl_sample != 0 for s in self.mesh_shape):
            raise ValueError('particle grid cannot fit on mesh grid')

        if self.growth_lpt_size <= 0:
            raise ValueError('LPT growth table size must >= 1')

    def __str__(self):
        return pformat(self, indent=4, width=1)  # for python >= 3.10

    @property
    def dim(self):
        """Spatial dimensionality."""
        return len(self.mesh_shape)

    @property
    def cell_vol(self):
        """Mesh cell volume in [L^dim]."""
        return self.cell_size ** self.dim

    @property
    def box_size(self):
        """Simulation box size tuple in [L]."""
        return self.cell_size * jnp.array(self.mesh_shape, self.float_dtype)

    @property
    def box_vol(self):
        """Simulation box volume in [L^dim]."""
        return self.box_size.prod()

    @property
    def transfer_k(self):
        """Transfer function wavenumbers, from the minimum fundamental frequency to the diagonal Nyquist frequency."""
        log10_k_min = jnp.log10(2. * jnp.pi / self.box_size.max())
        log10_k_max = jnp.log10(jnp.sqrt(self.dim) * jnp.pi / self.cell_size)
        return jnp.logspace(log10_k_min, log10_k_max, num=self.transfer_size,
                            dtype=self.float_dtype)

    @property
    def growth_a(self):
        """Growth function scale factors."""
        growth_a_lpt = jnp.linspace(0., self.a_start, num=self.growth_lpt_size,
                                    endpoint=False, dtype=self.growth_dtype)
        growth_a_nbody = jnp.linspace(self.a_start, self.a_stop, self.a_num,
                                      dtype=self.growth_dtype)
        return jnp.concatenate((growth_a_lpt, growth_a_nbody))

    @property
    def a_steps(self):
        """N-body time integration steps, linearly spaced scale factors."""
        return jnp.linspace(self.a_start, self.a_stop, self.a_num,
                            dtype=self.float_dtype)

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
