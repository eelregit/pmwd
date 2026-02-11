from jax.typing import ArrayLike
import jax.numpy as jnp

from pmwd.tree_util import Tree, pytree_dataclass, fxd_field


@pytree_dataclass
class Constants(Tree):
    r"""Physical constants in SI units, some are exact.

    Parameters
    ----------
    c : float ArrayLike, optional
        Speed of light :math:`c` in m/s.
    G : float ArrayLike, optional
        Gravitational constant :math:`G` in m:math:`^3`/kg/s:math:`^2`
    hbar : float ArrayLike, optional
        Reduced Planck constant :math:`\hbar` in J:math:`\cdot`s.
    e : float ArrayLike, optional
        Elementary charge :math:`e` in C.
    k : float ArrayLike, optional
        Boltzmann constant :math:`k` in J/K.
    M_sun : float ArrayLike, optional
        Solar mass :math:`M_\odot` in kg.
    Mpc : float ArrayLike, optional
        Mpc in m.
    H_0 : float ArrayLike, optional
        Hubble constant :math:`H_0` in :math:`h`/s.

    """

    c: ArrayLike = fxd_field(default=2.99792458e8, repr=True)  # exact
    G: ArrayLike = fxd_field(default=6.67430e-11, repr=True)
    hbar: ArrayLike = fxd_field(default=6.62607015e-34 / (2*jnp.pi), repr=True)  # exact
    e: ArrayLike = fxd_field(default=1.602176634e-19, repr=True)  # exact
    k: ArrayLike = fxd_field(default=1.380649e-23, repr=True)  # exact

    M_sun: ArrayLike = fxd_field(default=1.98847e30, repr=True)
    Mpc: ArrayLike = fxd_field(default=3.0856775815e22, repr=True)
    H_0: ArrayLike = fxd_field(depend=lambda self: 1e5 / self.Mpc, repr=True)
