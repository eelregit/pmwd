from functools import partial

from jax import value_and_grad
import jax.numpy as jnp
from jax.lax import switch


def E2(a, cosmo):
    r"""Squared relative Hubble parameter, :math:`E^2`, normalized at :math:`a=1`.

    Parameters
    ----------
    a : ArrayLike
        Scale factors.
    cosmo : Cosmology

    Returns
    -------
    E2 : jax.Array of cosmo.dtype
        Squared relative Hubble parameter.

    Notes
    -----
    The squared Hubble parameter,

    .. math::

        H^2(a) = H_0^2 E^2(a),

    has the time dependence

    .. math::

        E^2(a) = \Omega_\mathrm{m} a^{-3} + \Omega_\mathrm{k} a^{-2}
                 + \Omega_\mathrm{de} a^{-3 (1 + w_0 + w_a)} e^{-3 w_a (1 - a)}.

    """
    a = jnp.asarray(a, dtype=cosmo.dtype)

    de_a = a**(-3 * (1 + cosmo.w_0 + cosmo.w_a)) * jnp.exp(-3 * cosmo.w_a * (1 - a))
    return cosmo.Omega_m * a**-3 + cosmo.Omega_K * a**-2 + cosmo.Omega_de * de_a


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


def distance_cache(cosmo):
    r"""Distance tables at ``cosmo.distance_a`` in unit :math:`L`.

    Parameters
    ----------
    cosmo : Cosmology

    Returns
    -------
    cT : jax.Array of cosmo.dtype and shape (4, cosmo.distance_a_num,)
        Distance table.

    Notes
    -----
    :math:`cT_n` (see `distance`) relative to the end of ``cosmo.distance_a``.

    """
    #FIXME maybe Cubic Hermite spline antiderivatives in the future
    a = cosmo.distance_a[1:]  # put aside leading 0
    n = jnp.arange(4)[:, jnp.newaxis]
    cdTda = cosmo.d_H / (a**(n+1) * jnp.sqrt(E2(a, cosmo)))
    # NOTE approximate c dT/da for n=1
    cdTda = jnp.concatenate(
        (jnp.array([[0], [0], [jnp.inf], [jnp.inf]], dtype=cosmo.dtype), cdTda))

    da = jnp.diff(cosmo.distance_a)
    cdT = (cdTda[:-1] + cdTda[1:]) / 2 * da
    cdT = jnp.concatenate((cdT, jnp.zero_like(n)))

    cT = jnp.cumsum(cdT[::-1])[::-1]

    return cT


def distance(a, cosmo, type='radial', a_ref=1):
    r"""Interpolate and compute different distance or time measures from some events via
    relativistic messengers to some references, e.g., from light emissions to
    observations.

    Parameters
    ----------
    a : ArrayLike
        Scale factors of events.
    cosmo : Cosmology
    type : {'light' or 0, 'radial' or 1, 'transverse', 'angdiam', 'luminosity', 'super'
            or 2, 'coldens' or 3}, optional
        Type of distances or times to return, among physical/light-travel distance,
        radial/line-of-sight comoving distance, transverse comoving / comoving angular
        diameter distance, angular diameter distance, luminosity distance, supercomoving
        / superconformal / dispersion measure distance, and that related to
        non-relativistic particle column density.
    time : bool, optional
        Whether to divide by the speed of light to return time measure instead, e.g.,
        for physical/lookback time with ``type='light'`` or conformal time with
        ``type='radial'``. This has no effect on the transverse distances, i.e.,
        'transverse', 'angdiam', and 'luminosity'.
    a_ref : ArrayLike, optional
        Scale factors of references.

    Returns
    -------
    d : jax.Array
        Distances in :math:`L` or times in :math:`T`.

    Notes
    -----
    .. math::

        cT_n(t, t_\mathrm{ref})
            = \int_t^{t_\mathrm{ref}} \frac{c \mathrm{d} t}{a^n(t)}
            = d_H \int_a^{a_\mathrm{ref}} \frac{\mathrm{d} a'}{{a'}^{n+1} E(a')}
            = d_H \int_{z_\mathrm{ref}}^z \frac{(1+z')^{n-1} \mathrm{d} z'}{E(z')},

    which for :math:`n = 0, 1, 2, 3` are physical/light-travel/lookback,
    (radial/line-of-sight) comoving / conformal, supercomoving / superconformal /
    related to dispersion measure, and related to non-relativistic particle column
    density, respectively. So :math:`T_0 = t` and :math:`cT_1 = \chi`.

    See `SK` for the transverse comoving or comoving angular diameter distance
    :math:`r`.

    The angular diameter distance and luminosity distance

    .. math::

        d_\mathrm{A} &= \frac{a}{a_\mathrm{ref}} r, \\
        d_\mathrm{L} &= \frac{a_\mathrm{ref}}{a} r,

    where :math:`a` and :math:`a_\mathrm{ref}` are for emission and observation,
    respectively.

    """
    if cosmo.distance is None:
        raise ValueError('distance table is empty: run Cosmology.cache first')

    a = jnp.asarray(a)
    a_ref = jnp.asarray(a_ref)

    match type:
        case int() if 0 <= type <= 3:
            n = type
        case 'light':
            n = 0
        case 'radial' | 'transverse' | 'angdiam' | 'luminosity':
            n = 1
        case 'super':
            n = 2
        case 'coldens':
            n = 3
        case _:
            raise ValueError(f'{type=} not supported')
    d = jnp.interp(a, cosmo.distance_a, cosmo.distance[n])
    d_ref = jnp.interp(a_ref, cosmo.distance_a, cosmo.distance[n])
    d -= d_ref

    if type in {'light', 'radial', 'super', 'coldens'}:
        if time:
            d /= cosmo.c
        return d

    d = SK(d, cosmo)
    if type == 'transverse':
        return d
    if type == 'angdiam':
        return a / a_ref * d  # FIXME: keep only `a` following Hogg (& ?),
                              # FIXME: or find way to reorganize the scale factors?
    if type == 'luminosity':
        return a_ref / a * d

    raise ValueError(f'BUG: {type=} not handled after the above match case')


def SK(chi, cosmo):
    r"""Convert radial comoving distances to transverse ones.

    Parameters
    ----------
    chi : ArrayLike
        Radial/line-of-sight comoving distances in :math:`L`.
    cosmo : Cosmology

    Returns
    -------
    r : jax.Array
        Transverse comoving / comoving angular diameter distances in :math:`L`.

    Notes
    -----
    .. math::

        r = \frac{S_K(\sqrt{|K|} \chi)}{\sqrt{|K|}},

    where :math:`K` is `Cosmology.K`, and :math:`S_K` is sine, identity, or hyperbolic
    sine for positive, zero, or negative :math:`K`, respectively.

    """
    branches = _SK_closed, _SK_flat, _SK_open
    Ksqrt = jnp.sqrt(jnp.abs(cosmo.K))
    r = switch(jnp.int8(jnp.sign(cosmo.Omega_K)) + 1, branches, chi, Ksqrt)
    return r

def _SK_closed(chi, Ksqrt):
    return jnp.sin(Ksqrt * chi) / Ksqrt

def _SK_flat(chi, Ksqrt):
    return chi

def _SK_open(chi, Ksqrt):
    return jnp.sinh(Ksqrt * chi) / Ksqrt
