from abc import ABC, abstractmethod
from operator import itemgetter

from jax.typing import ArrayLike, DTypeLike
import jax.numpy as jnp
from jax.scipy.special import gamma
from jax.tree_util import tree_map

from pmwd.tree_util import (Tree, TanMixin, pytree_dataclass, aux_field, issubdtype_of,
                            asarray_of)
from pmwd.lens.util import (lens_dyn_field, lens_dyn_2d_field, lens_dyn_pa_field,
                            lens_fxd_field, _canonicalize_ang_pos, _canonicalize_array)


@pytree_dataclass
class Sources(TanMixin, Tree, ABC):
    """Sources for strong gravitational lensing."""

    dtype: DTypeLike = aux_field(default=jnp.float32,
                                 validate=(jnp.dtype, issubdtype_of(jnp.floating)))

    x: ArrayLike = lens_dyn_2d_field()
    a: ArrayLike = lens_fxd_field(repr=True)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, key):
        return tree_map(itemgetter(key), self)  #FIXME itemgetter -> itembetter, broadcastability friendly

    def astype(self, dtype):
        """Return a new object with pytree children casted to `dtype`."""
        return self.replace(dtype=dtype)

    @property
    def z(self):
        """Redshifts."""
        return 1/a - 1

    @staticmethod
    @abstractmethod
    def profile(x, src):
        pass


def profile(x, src):
    """Profile of surface brightnesses.

    Evaluate ``type(src).profile`` on rays `x`. Also see documentation there.

    Parameters
    ----------
    x : ArrayLike of shape (..., num_sources, 2)
        Ray angular positions, per source, in :math:`A`.
    src : Sources

    Returns
    -------
    I : jax.Array of x.dtype and shape (...,)
        Surface brightnesses, per :math:`A^2`.

    """
    x = _canonicalize_ang_pos(x)
    src = src.astype(x.dtype)
    I = type(src).profile(x, src)
    return _canonicalize_array(I, shape=x.shape[:-2], dtype=x.dtype)


@pytree_dataclass
class SersicSources(Sources):
    r"""Sources with Sérsic profile.

    Parameters
    ----------
    dtype : DTypeLike
    x : ArrayLike of shape (num_sources, 2) or (2,)
        Angular positions in :math:`A`.
    a : ArrayLike
        Scale factors.
    F : ArrayLike
        Intrinsic fluxes.
    R_e : ArrayLike
        Effective Radii in :math:`A`.
    n : ArrayLike
        Sérsic indices.
    q : ArrayLike
        Axis ratios of the minor axes to the major axes.
    theta : ArrayLike
        Position angles in radians.
    soften : ArrayLike, optional
        Softening length in :math:`A`, for possible high central concentration with high
        `n`. No softening by default.

    References
    ----------
    .. _L. Ciotti and G. Bertin 1999, Analytical properties of the :math:`R^{1/m}` luminosity law:
        https://arxiv.org/abs/astro-ph/9911078
    .. _Graham and Driver 2005, A concise reference to (projected) Sérsic :math:`R^{1/n}` quantities, including concentration, profile slopes, Petrosian indices, and Kron magnitudes:
        https://arxiv.org/abs/astro-ph/0503176

    Notes
    -----

    .. math::

        F &= I_\mathrm{e} R_\mathrm{e}^2 \frac{2\pi n e^{b_n}}{{b_n}^{2n}} \Gamma(2n), \\
        I(R) &= I_e \exp\Bigl[ - b_n \Bigl( \bigl(\frac{R}{R_e}\bigr)^{1/n} - 1 \Bigr) \Bigr], \\
        R^2 &= \frac{b \Delta x')^2}{a} + \frac{a \Delta y')^2}{b}.

    where :math:`\Delta x'` and :math:`\Delta y'` are rotated by :math:`- \theta`.

    """
    #FIXME above & below for changing theta to PA standard

    F: ArrayLike = lens_dyn_field()
    R_e: ArrayLike = lens_dyn_field()
    n: ArrayLike = lens_dyn_field()
    q: ArrayLike = lens_dyn_field()
    theta: ArrayLike = lens_dyn_pa_field()

    soften: ArrayLike = lens_fxd_field(optional=True)

    @property
    def b_n(self):
        r"""Sérsic :math:`b_n`, determined by :math:`2\gamma(2n, b_n) = \Gamma(2n).

        Asymptotic expansion is used for :math:`n > 0.36` and NaN otherwise.
        smaller ones.

        """
        n = self.n
        u = 1 / n
        asymp = 2*n - 1/3 + u * (4/405 + u * (46/25515 + u * (131/1148175 + u
                                                              * (2194697/30690717750))))
        #poly = 0.01945 + n * (-0.8902 + n * (10.95 + n * (-19.67 + n * 13.43)))
        return jnp.where(n > 0.36, asymp, jnp.nan)
        #return asymp

    @property
    def I_e(self):
        r"""Surface brightnesses at effective radii."""
        return self.F * self.b_n**(2*self.n) / (self.R_e**2 * (2*jnp.pi*self.n)
                                                * jnp.exp(self.b_n) * gamma(2*self.n))

    @staticmethod
    def _I(R2, src):
        R2 = R2.clip(min=src.soften**2)
        return src.I_e * jnp.exp(- src.b_n * ((R2 / src.R_e**2)**(0.5/src.n) - 1))

    @staticmethod
    def profile(x, src):
        """Sérsic profiles. Also see documentation of `profile`."""
        x = x - src.x  # shape = (..., num_sources, 2)
        x = x[..., 0] + 1j * x[..., 1]
        x *= jnp.exp(- 1j * src.theta)
        rM2 = x.real**2 * src.q + x.imag**2 / src.q  # Mahalanobis distance

        I = SersicSources._I(rM2, src)

        return I.sum(axis=-1)  # sum over sources
