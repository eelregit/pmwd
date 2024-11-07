from functools import partial

from jax import value_and_grad
import jax.numpy as jnp
from jax.lax import switch


def E2(a, cosmo):
    """Squared relative Hubble parameter, :math:`E^2`, normalized at :math:`a=1`.

    Parameters
    ----------
    a : ArrayLike
        Scale factors.
    cosmo : Cosmology

    Returns
    -------
    E2 : jax.Array of cosmo.Omega_m.dtype
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
    a = jnp.asarray(a, dtype=cosmo.Omega_m.dtype)

    de_a = a**(-3 * (1 + cosmo.w_0 + cosmo.w_a)) * jnp.exp(-3 * cosmo.w_a * (1 - a))
    return cosmo.Omega_m * a**-3 + cosmo.Omega_K * a**-2 + cosmo.Omega_de * de_a


@partial(jnp.vectorize, excluded=(1,))
def H_deriv(a, cosmo):
    """Hubble parameter derivatives, :math:`\mathrm{d}\ln H / \mathrm{d}\ln a`.

    Parameters
    ----------
    a : ArrayLike
        Scale factors.
    cosmo : Cosmology

    Returns
    -------
    dlnH_dlna : jax.Array of cosmo.Omega_m.dtype
        Hubble parameter derivatives.

    """
    a = jnp.asarray(a, dtype=cosmo.Omega_m.dtype)

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
    Omega : jax.Array of cosmo.Omega_m.dtype
        Matter density parameters.

    Notes
    -----

    .. math::

        \Omega_\mathrm{m}(a) = \frac{\Omega_\mathrm{m} a^{-3}}{E^2(a)}

    """
    a = jnp.asarray(a, dtype=cosmo.Omega_m.dtype)

    return cosmo.Omega_m / (a**3 * E2(a, cosmo))


def distance_cache(cosmo):
    r"""Cache the comoving and physical distance tables at ``cosmo.distance_a``.

    Parameters
    ----------
    cosmo : Cosmology

    Returns
    -------
    cosmo : Cosmology
        A new instance containing a distance table, in unit :math:`L`, shape
        ``(2, cosmo.distance_a_num,)``, and precision ``cosmo.Omega_m.dtype``.

    Notes
    -----
    The comoving horizon in the conformal time :math:`\eta`

    .. math::

        c \eta = \int_0^t \frac{c \mathrm{d} t}{a(t)}
               = d_H \int_0^a \frac{\mathrm{d} a'}{a'^2 E(a')}
               = d_H \int_z^\infty \frac{\mathrm{d} z'}{E(z')}.

    The light-travel distance in the age or physical time :math:`t`

    .. math::

        ct = \int_0^t c \mathrm{d} t
           = d_H \int_0^a \frac{\mathrm{d} a'}{a' E(a')}
           = d_H \int_z^\infty \frac{\mathrm{d} z'}{(1+z') E(z')}.

    """
    #FIXME in the future use jax.scipy.integrate.cumulative_trapezoid or Cubic Hermite spline antiderivatives
    a = cosmo.distance_a[1:]
    da = jnp.diff(cosmo.distance_a, prepend=0)

    cdetada = cosmo.d_H / (a**2 * jnp.sqrt(E2(a, cosmo)))
    cdetada = jnp.concatenate((jnp.array([0, 0]), cdetada))
    cdeta = (cdetada[:-1] + cdetada[1:]) / 2 * da
    ceta = jnp.cumsum(cdeta)

    cdtda = cosmo.d_H / (a * jnp.sqrt(E2(a, cosmo)))
    cdtda = jnp.concatenate((jnp.array([0, 0]), cdtda))
    cdt = (cdtda[:-1] + cdtda[1:]) / 2 * da
    ct = jnp.cumsum(cdt)

    distance = jnp.stack((ceta, ct), axis=0)

    #return cosmo.replace(distance=distance)
    return distance


def _SK_closed(chi, Ksqrt):
    return jnp.sin(Ksqrt * chi) / Ksqrt

def _SK_flat(chi, Ksqrt):
    return chi

def _SK_open(chi, Ksqrt):
    return jnp.sinh(Ksqrt * chi) / Ksqrt


def distance(a_e, cosmo, type='radial', a_o=1):
    r"""Interpolate the distances and compute different distance or time measures
    between emissions and observations.

    Parameters
    ----------
    a_e : ArrayLike
        Scale factors at emission.
    cosmo : Cosmology
    type : str in {'radial', 'transverse', 'angdiam', 'luminosity', 'light', 'conformal', 'lookback'}, optional
        Type of distances or times to return, among radial comoving distance, transverse
        comoving distance, angular diameter distance, luminosity distance, light-travel
        distance, conformal time, and lookback time.
    a_o : ArrayLike, optional
        Scale factors at observation.

    Returns
    -------
    d : jax.Array
        Distances in :math:`L` or times in :math:`T`.

    Notes
    -----
    The line-of-sight or radial comoving distance, related to the conformal time
    :math:`\eta`

    .. math::

        \chi = c \eta
             = \int_{t_\mathrm{e}}^{t_\mathrm{o}} \frac{c \mathrm{d} t}{a(t)}
             = d_H \int_{a_\mathrm{e}}^{a_\mathrm{o}} \frac{\mathrm{d} a'}{a'^2 E(a'}
             = d_H \int_{z_\mathrm{o}}^{z_\mathrm{e}} \frac{\mathrm{d} z'}{E(z')}.

    The transverse comoving or comoving angular diameter distance

    .. math::

        r = \frac{S_K(\sqrt{|K|} \chi)}{\sqrt{|K|}},

    where :math:`S_K` is sin, identity, or sinh for positive, zero, or negative
    :math:`K`, respectively.

    The angular diameter distance and luminosity distance

    .. math::

        d_\mathrm{A} &= \frac{a_\mathrm{e}}{a_\mathrm{o}} r, \\
        d_L &= \frac{a_\mathrm{o}}{a_\mathrm{e}} r.

    The light-travel distance in the lookback or physical time :math:`t`

    .. math::

        ct = \int_{t_\mathrm{e}}^{t_\mathrm{o}} c \mathrm{d} t
           = d_H \int_{a_\mathrm{e}}^{a_\mathrm{o}} \frac{\mathrm{d} a'}{a' E(a'}
           = d_H \int_{z_\mathrm{o}}^{z_\mathrm{e}} \frac{\mathrm{d} z'}{(1+z') E(z')}.

    """
    if cosmo.distance is None:
        raise ValueError('distance table is empty: run Cosmology.cache or distance_cache first')

    a_e = jnp.asarray(a_e)
    a_o = jnp.asarray(a_o)

    phys = 1 if type in {'light', 'lookback'} else 0
    d_o = jnp.interp(a_o, cosmo.distance_a, cosmo.distance[phys])
    d_e = jnp.interp(a_e, cosmo.distance_a, cosmo.distance[phys])
    d = d_o - d_e

    if type in {'lookback', 'conformal'}:
        d /= cosmo.c

    if type in {'radial', 'light', 'lookback', 'conformal'}:
        return d

    branches = _SK_closed, _SK_flat, _SK_open
    Ksqrt = jnp.sqrt(jnp.abs(cosmo.K))
    d = switch(jnp.int8(jnp.sign(cosmo.Omega_K)) + 1, branches, d, Ksqrt)
    if type == 'transverse':
        return d
    if type == 'angdiam':
        return a_e / a_o * d
    if type == 'luminosity':
        return a_o / a_e * d

    raise ValueError(f'{type=} not supported')
