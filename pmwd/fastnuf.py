from functools import partial
from pathlib import Path
from typing import ClassVar

from jax import Array, vmap
from jax.typing import ArrayLike, DTypeLike
import jax.numpy as jnp
import jax.scipy.special
import numpy as np
import scipy.special

from pmwd.background import distance
from pmwd.perturbation import transfer, growth
from pmwd.special import SiCi
from pmwd.tree_util import (Tree, pytree_dataclass, dyn_field, fxd_field, aux_field,
                            issubdtype_of, asarray_of, astype_of)


_dyn_field = partial(dyn_field, validate=asarray_of(field='dtype'))
_dyn_field.__doc__ = '`tree_util.dyn_field` for FAST-νf.'
_fxd_field = partial(fxd_field, validate=asarray_of(field='dtype'))
_fxd_field.__doc__ = '`tree_util.fxd_field` for FAST-νf.'


@pytree_dataclass
class CosmicEnuII(Tree):
    r"""Cosmic-Enu-II adapted from `repo <https://codeberg.org/upadhye/Cosmic-Enu-II>`_.

    This emulates :math:`delta_\mathrm{m}(a, k)` and :math:`R_\nu(a, k, u)`. See also
    `Cosmic-Enu-II paper <https://arxiv.org/abs/2511.09945>`_ and `Cosmic-Enu paper
    <https://arxiv.org/abs/2311.11240>`_.

    """

    url: ClassVar = 'https://doi.org/10.5281/zenodo.????????'  # TODO

    dtype: DTypeLike = aux_field(default=jnp.float64,
                                 validate=(jnp.dtype, issubdtype_of(jnp.floating)))

    Na: int = aux_field(repr=True)
    Nk: int = aux_field(repr=True)
    Nu: int = aux_field(optional=True, repr=True)
    Npc: int = aux_field(repr=True)
    Npar: int = aux_field(repr=True)
    Nsim: int = aux_field(repr=True)

    a: ArrayLike = _fxd_field()  # scale factors, of shape (Na,)
    k: ArrayLike = _fxd_field()  # wavenumbers, of shape (Nk,)
    um_eV: ArrayLike = _fxd_field(optional=True)  # flow momenta in eV, of shape (Nu,)
                                                  # "um" means Mν/3 times u

    # See 2311.11240 for equations
    std: ArrayLike = _fxd_field()  # \sigma^* in (1), of shape ()
    mean_i: ArrayLike = _fxd_field()  # \mu^* in (1), of shape (Na*Nk*Nu,)
    phi_ic: ArrayLike = _fxd_field()  # (1), of shape (Na*Nk*Nu, Npc)

    lambda_c: ArrayLike = _fxd_field()  # \lambda_U in (9) & (13), of shape (Npc,)
    beta_cp: ArrayLike = _fxd_field()  # (7), of shape (Npc, Npar)
    Cstar_sp: ArrayLike = _fxd_field()  # C^* in (13), of shape (Nsim, Npar)
    Rw_cs: ArrayLike = _fxd_field()  # [R^*]^{-1} w^* in (11), of shape (Npc, Nsim)

    def astype(self, dtype):
        """Return a new object with pytree children casted to `dtype`."""
        return self.replace(dtype=dtype)

    @classmethod
    def load(cls, file, **kwargs):
        """Load Cosmic-Enu-II data from npz files."""
        try:
            with jnp.load(file) as f:
                N = dict(
                    Na = f['a'].shape[0],
                    Nk = f['k'].shape[0],
                    Nu = f['um_eV'].shape[0] if 'um_eV' in f else None,
                    Npc = f['lambda_c'].shape[0],
                    Npar = f['beta_cp'].shape[1],
                    Nsim = f['Cstar_sp'].shape[0],
                )
                return cls(**N, **f, **kwargs)
        except FileNotFoundError as e:
            p = Path(file)
            raise FileNotFoundError(f'download {p.name} from {cls.url} and/or '
                                    f'change the path from {p.parent}') from e

    def __call__(self, cosmo):
        """Interpolate the Gaussian process. See 2311.11240 for equations."""
        C = jnp.array([
            (cosmo.omega_m - 0.12) / (0.155 - 0.12),
            (cosmo.omega_b - 0.0215) / (0.0235 - 0.0215),
            (cosmo.omega_nu - 0.00017) / (0.01 - 0.00017),
            (cosmo.sigma_8 - 0.7) / (0.9 - 0.7),
            (cosmo.h - 0.55) / (0.85 - 0.55),
            (cosmo.n_s - 0.85) / (1.05 - 0.85),
            (cosmo.w_0 + 1.3) / (1.3 - 0.7),
            ((- cosmo.w_0 - cosmo.w_a) ** (1/4) - 0.3) / (1.29 - 0.3),
        ], cosmo.dtype)

        # (7), (13), (11), & (10)
        r = jnp.exp(- (self.beta_cp[:, None] * (C - self.Cstar_sp) ** 2).sum(axis=2))
        r /= self.lambda_c[:, None]
        w = (r * self.Rw_cs).sum(axis=1)

        P = self.mean_i + self.std * (w * self.phi_ic).sum(axis=1)
        shape = (self.Na, self.Nk)
        if self.Nu is not None:
            shape += (self.Nu,)
        P = P.reshape(shape)

        return P


def fastnuf_cache(fnf, cosmo):
    r"""FAST-νf transfer function :math:`\delta_\nu / \delta_\mathrm{cb}` table."""
    enu_d, enu_R = fnf.enu_d, fnf.enu_R
    cosmo = cosmo.astype(fnf.dtype)
    # a, k, m_nu, and glq are all 1D
    a, k, m_nu = fnf.transfer_a, fnf.transfer_k, cosmo.m_nu

    if m_nu is None:
        raise ValueError('FAST-νf neutrinos must be massive')

    lnd = enu_d(cosmo)  # ln(delta_m / (D T))
    lnR = enu_R(cosmo)  # ln(R_nu)

    # NOTE a-interp differ from 2511.09945 (40-41)
    lnd = _interp_a(jnp.log(a), jnp.log(enu_d.a), lnd)
    d_m = jnp.exp(lnd) * growth(a[:, None], cosmo) * transfer(k, cosmo)

    # rescale DO to XO
    m_nu_degenerate = jnp.full_like(m_nu, 1 / 3, shape=(3,)) * m_nu.sum()
    cosmo_degenerate = cosmo.replace(m_nu=m_nu_degenerate)
    d_m *= (sigma_nu(k, a[:, None], cosmo)
            / sigma_nu(k, a[:, None], cosmo_degenerate)
            * _mass_weighted_xi(k, a[:, None], fnf, cosmo)
            / _mass_weighted_xi(k, a[:, None], fnf, cosmo_degenerate))

    # integrate interpolated delta_m(a, k) for delta_nu(a, k, species, u)
    u = u_glq(fnf.glq_x, cosmo)  # m_nu.shape + (glq_n,)
    d_nu = _fastnuf_integ(u, k, a, d_m, cosmo)  # a.shape + k.shape + u.shape

    # interpolate and extrapolate R_nu for u, and pad for a
    um_eV = u / cosmo.c * m_nu.mean()  # assumption by Cosmic-Enu-II
    lnR = _interp_u(_asinh_u(um_eV, enu_R), _asinh_u(enu_R.um_eV, enu_R), lnR)
    R = jnp.exp(lnR)
    ones = jnp.ones_like(R, shape=(a.shape[0] - R.shape[0],) + R.shape[1:])
    R = jnp.concatenate([ones, R], axis=0)

    # nonlinear enhancement ratio
    d_nu *= R

    # average over species and flows to get delta_nu(a, k)
    d_nu = _fermi_dirac_glq(d_nu, fnf, m_nu)  # of shape a.shape
    d_cb = (d_m - cosmo.f_nu * d_nu) / cosmo.f_cb

    T = cosmo.f_cb * d_nu / (d_m - cosmo.f_nu * d_nu)
    return jax.scipy.special.logit(T)


@partial(vmap, in_axes=(None, None, 1), out_axes=1)
def _interp_a(x, xp, fp):
    return jnp.interp(x, xp, fp, left='extrapolate', right='extrapolate')

@partial(vmap, in_axes=(None, None, 0), out_axes=0)
def _interp_k(x, xp, fp):
    return jnp.interp(x, xp, fp, left='extrapolate', right='extrapolate')

@partial(vmap, in_axes=(None, None, 0), out_axes=0)
def _interp_u(x, xp, fp):
    return _interp_k(x, xp, fp)

def _asinh_u(um_eV, enu_R):
    return jnp.arcsinh(um_eV / enu_R.um_eV[1])


def k_fs(u, a, cosmo):
    return jnp.sqrt(1.5 * cosmo.Omega_m * a) * cosmo.H_0 / u


def u_nu(cosmo):
    """``m_nu.shape``"""
    coeff = np.sqrt(3 / 2 * scipy.special.zeta(3) / np.log(2)).item()
    const = cosmo.const
    return coeff * const.k * cosmo.T_nu * cosmo.c / cosmo.m_nu / const.e


def u_glq(x, cosmo):
    """``m_nu.shape + (glq_n,)``"""
    m_nu = cosmo.m_nu[..., None]
    const = cosmo.const
    return x * const.k * cosmo.T_nu * cosmo.c / m_nu / const.e


def a_nu(cosmo):
    """``m_nu.shape``"""
    coeff = (7 / 180 * jnp.pi**4 / scipy.special.zeta(3)).item()
    const = cosmo.const
    return coeff * const.k * cosmo.T_nu / cosmo.m_nu / const.e


def Q_nu(cosmo):
    """``m_nu.shape``"""
    return 1.25 * (1 - jnp.sqrt(1 - 0.96 * cosmo.m_nu / cosmo.M_nu * cosmo.f_nu))


def sigma_nu(k, a, cosmo):
    Q_, a_, u_ = Q_nu(cosmo), a_nu(cosmo), u_nu(cosmo)
    k_nr = k_fs(u_, a_, cosmo)
    a, k = a[..., None], k[..., None]
    return 1 - jnp.sum(Q_ * jnp.log(a / a_) * k**2 / (5.5*k_nr**2 + 1.8*k*k_nr + k**2),
                       axis=-1)


def xi_fs(u, k, a, cosmo):
    return 1 / (1 + (k / k_fs(u, a, cosmo)) ** 2)


def _fermi_dirac_glq(f, fnf, m_nu):
    """input ``f.shape = (...,) + m_nu.shape + (glq_n,)``, output ``(...,)``"""
    nu_w = m_nu / m_nu.sum()
    u_w = fnf.glq_w * fnf.glq_x**2 / (jnp.exp(-fnf.glq_x) + 1)
    coeff = (1.5 * scipy.special.zeta(3)).item() ** -1
    return coeff * jnp.sum(jnp.sum(u_w * f, axis=-1) * nu_w, axis=-1)


def _mass_weighted_xi(k, a, fnf, cosmo):
    a, k = a[..., None, None], k[..., None, None]
    u = u_glq(fnf.glq_x, cosmo)  # m_nu.shape + (glq_n,)
    xi = xi_fs(u, k, a, cosmo)  # (a.size, k.size) + m_nu.shape + (glq_n,)
    return cosmo.omega_cb + cosmo.omega_nu * _fermi_dirac_glq(xi, fnf, cosmo.m_nu)


def _fastnuf_integ(u, k, a, d_m, cosmo):
    """from 2D u, 1D k, 1D a, and 2D d_m to 4D d_nu"""
    # replace late-time s with a matter dominated one till a=∞
    a_md = 0.1  # some matter dominated time
    s_md = - 2 * cosmo.d_H / jnp.sqrt(cosmo.Omega_m * a_md)  # from a=∞ to a_md
    s = s_md - distance(a, cosmo, type='super', a_ref=a_md)

    # piecewise cubic polynomial interpolation coefficients
    # NOTE extra minus sign compared to 2511.09945
    # g.shape = a.shape + (4,) + k.shape
    #g = _cubic_interp_coeff(- s / cosmo.d_H,
    #                        ((- s / cosmo.d_H) ** 4 * a)[:, None] * d_m)
    # piecewise linear polynomial interpolation coefficients
    # NOTE extra minus sign compared to 2511.09945
    # g.shape = a.shape + (2,) + k.shape
    g = _linear_interp_coeff(- s / cosmo.d_H,
                             ((- s / cosmo.d_H) ** 4 * a)[:, None] * d_m)

    # analytic integration by trigonometric integrals, for which float32 is not enough
    s, k = s[..., None, None, None], k[..., None, None]
    y = - s * k * (u / cosmo.c)  # a.shape + k.shape + m_nu.shape + (glq_n,)

    #SC = jnp.stack(sum([_SC(y, n) for n in range(4)], start=()), axis=1)
    SC = jnp.stack(sum([_SC(y, n) for n in range(2, 4)], start=()), axis=1)
    SC = SC.reshape(SC.shape[0], SC.shape[1] // 2, 2, *SC.shape[2:])  # 2 axes inserted
    SC = SC.at[1:].subtract(SC[:-1])  # SC[0] correct by definition
    #SC *= (u / cosmo.H_0 * k) ** (jnp.arange(4) - 1)[:, None, None, None, None]
    SC *= (u / cosmo.H_0 * k) ** (jnp.arange(2, 4) - 1)[:, None, None, None, None]
    SC *= g[:, ::-1, None, :, None, None]
    SC = SC.sum(axis=1).cumsum(axis=0)
    SC = SC[:, 0] * jnp.cos(y) - SC[:, 1] * jnp.sin(y)

    return 1.5 * cosmo.Omega_m * SC


def _cubic_interp_coeff(x, f):
    """Coefficients of piecewise cubic polynomial interpolation, along ``axis=0`` of 1D
    `x` and 2D `f`, returned padded and with an extra `axis=1` of length 4."""
    N, dtype = x.shape[0], f.dtype
    assert f.shape[0] == N, "interpolation length must match"

    ind = jnp.arange(0, N-3)[:, None] + jnp.arange(4)
    x, f = jnp.float64(x[ind]), jnp.float64(f[ind])  # of shape (N-3, 4) & (N-3, 4, -1)
    V = x[..., None] ** jnp.arange(4)  # Vandermonde matrix, of shape (N-3, 4, 4)
    c = jnp.linalg.solve(V, f).astype(dtype)  # of shape (N-3, 4, -1)

    pad_ind = jnp.clip(jnp.arange(N) - 2, 0, N-4)
    return c[pad_ind]  # of shape (N, 4, -1)


def _linear_interp_coeff(x, f):
    """Coefficients of piecewise linear polynomial interpolation, along ``axis=0`` of 1D
    `x` and 2D `f`, returned padded and with an extra `axis=1` of length 2.

    Monomial basis leads to ill-conditioned Vandermonde matrices. Linear interpolation
    has better condition numbers (10²~10³) than the cubic case (10⁶~10¹⁰)

    """
    N = x.shape[0]
    assert f.shape[0] == N, "interpolation length must match"

    ind = jnp.arange(0, N-1)[:, None] + jnp.arange(2)
    x, f = x[ind], f[ind]  # of shape (N-1, 2) & (N-1, 2, -1)
    V = x[..., None] ** jnp.arange(2)  # Vandermonde matrix, of shape (N-1, 2, 2)
    c = jnp.linalg.solve(V, f)  # of shape (N-1, 2, -1)

    pad_ind = jnp.clip(jnp.arange(N) - 1, 0, N-2)
    return c[pad_ind]  # of shape (N, 2, -1)


def _SC(x, n):
    r"""Generalized trigonometric integrals by coupled recurrence relations:

    ..math::

        S_n(x) &\triangleq \int_x^\infty \frac{\sin t}{t^{n+1}} \mathrm{d}t,
            = \frac{\sin x}{n x^n} + \frac{C_{n-1}(x)}n, \\
        C_n(x) &\triangleq \int_x^\infty \frac{\cos t}{t^{n+1}} \mathrm{d}t,
            = \frac{\cos x}{n x^n} - \frac{S_{n-1}(x)}n, \\
        S_0(x) &= \frac\pi2 - \mathrm{Si}(x),
        C_0(x) = - \mathrm{Ci}(x).

    NOTE these differ from the I's in 2511.09945.

    """
    if n == 0:
        Si, Ci = SiCi(x)
        return 0.5 * jnp.pi - Si, - Ci
    S_, C_ = _SC(x, n-1)
    S = (jnp.sin(x) / x**n + C_) / n
    C = (jnp.cos(x) / x**n - S_) / n
    return S, C


def fastnuf(k, a, fnf):
    r"""Interpolate the FAST-νf transfer function :math:`T \triangleq \delta_\nu /
    \delta_\mathrm{cb}`, and output in shape ``a.shape + k.shape``."""
    if fnf.transfer is None:
        raise ValueError('transfer table is empty: run FASTnuf.cache first')

    k = jnp.asarray(k)
    a = jnp.asarray(a)

    # ravel a to 1D first for vmapped _interp_a
    logitT = _interp_a(jnp.log(a).ravel(), jnp.log(fnf.transfer_a), fnf.transfer)
    logitT = _interp_k(jnp.log(k), jnp.log(fnf.transfer_k), logitT)
    logitT = logitT.reshape(a.shape + k.shape)  # restore a shape

    return jax.scipy.special.expit(logitT)


@pytree_dataclass
class FASTnuf(Tree):
    r"""FAST-νf adapted from `Cosmic-Enu-II repo
    <https://codeberg.org/upadhye/Cosmic-Enu-II>`_ and `FAST-νf repo
    <https://codeberg.org/upadhye/FASTnuf>`_.

    See also `FAST-νf paper <https://arxiv.org/abs/2511.09945>`_ and `pmnumwd paper
    <https://arxiv.org/abs/260X.XXXXX>`_.

    Precompute the natural log of transfer function, :math:`\ln(\delta_\nu /
    \delta_\mathrm{cb})`, to be interpolated later.

    """

    dtype: DTypeLike = aux_field(default=jnp.float64,
                                 validate=(jnp.dtype, issubdtype_of(jnp.floating)))

    enu_d: CosmicEnuII = dyn_field(default=CosmicEnuII.load('CosmicEnuII_delta_m.npz'),
                                   validate=astype_of(field='dtype'))
    enu_R: CosmicEnuII = dyn_field(default=CosmicEnuII.load('CosmicEnuII_R_nu.npz'),
                                   validate=astype_of(field='dtype'))

    # Gauss-Laguerre quadrature
    glq_n: int = aux_field(default=50, repr=True)
    glq_x: Array = _fxd_field(
        depend=lambda self: np.polynomial.laguerre.laggauss(self.glq_n)[0],
        compare=False)
    glq_w: Array = _fxd_field(
        depend=lambda self: np.polynomial.laguerre.laggauss(self.glq_n)[1],
        compare=False)

    transfer: Array | None = _dyn_field(cache=fastnuf_cache, compare=False)

    def astype(self, dtype):
        """Return a new object with pytree children casted to `dtype`."""
        return self.replace(dtype=dtype)

    @property
    def transfer_a(self):
        """Transfer function scale factors."""
        a_fill = jnp.arange(0.01, 0.25, 0.01, dtype=self.enu_d.a.dtype)
        return jnp.concatenate((self.enu_d.a[:1], a_fill, self.enu_d.a[1:]))

    @property
    def transfer_k(self):
        """Transfer function wavenumbers in :math:`1/L`."""
        return self.enu_d.k

    def __call__(self, k, a):
        return fastnuf(k, a, self)
