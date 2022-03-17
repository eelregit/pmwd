from dataclasses import replace
from functools import partial

from jax import jit, ensure_compile_time_eval
import jax.numpy as jnp
from jax.experimental.ode import odeint

from pmwd.cosmology import H_deriv, Omega_m_a


# TODO maybe need to checkpoint EH for memory?
# TODO Wayne's website
def transfer_EH(k, cosmo):
    """Eisenstein & Hu matter transfer function at given wavenumbers.

    Parameters
    ----------
    k: array_like
        Wavenumbers in [1/L].
    cosmo: Cosmology

    Returns
    -------
    T : jax.numpy.ndarray
        Matter transfer function.

    .. _Transfer Function:
        http://background.uchicago.edu/~whu/transfer/transferpage.html

    """
    conf = cosmo.conf

    k = jnp.asarray(k)

    k = k * cosmo.h / conf.L * conf.Mpc_SI  # unit conversion to [1/Mpc]

    #############################################
    # Quantities computed from 1998:EisensteinHu
    # Provides : - k_eq   : scale of the particle horizon at equality epoch
    #            - z_eq   : redshift of equality epoch
    #            - R_eq   : ratio of the baryon to photon momentum density at z_eq
    #            - z_d    : redshift of drag epoch
    #            - R_d    : ratio of the baryon to photon momentum density at z_d
    #            - sh_d   : sound horizon at drag epoch
    #            - k_silk : Silk damping scale
    T2_cmb_norm = (conf.T_cmb / 2.7)**2
    h2 = cosmo.h**2
    w_m = cosmo.Omega_m * h2
    w_b = cosmo.Omega_b * h2
    f_b = cosmo.Omega_b / cosmo.Omega_m
    f_c = (cosmo.Omega_m - cosmo.Omega_b) / cosmo.Omega_m  # TODO Omega_c or neutrinos

    k_eq = 7.46e-2 * w_m / T2_cmb_norm  # Eq. (3)
    z_eq = 2.50e4 * w_m / T2_cmb_norm**2  # Eq. (2)

    # z drag from Eq. (4)
    b1 = 0.313 * w_m**-0.419 * (1.0 + 0.607 * w_m**0.674)
    b2 = 0.238 * w_m**0.223
    z_d = 1291.0 * w_m**0.251 / (1.0 + 0.659 * w_m**0.828) * (1.0 + b1 * w_b**b2)

    # Ratio of the baryon to photon momentum density at z_d Eq. (5)
    R_d = 31.5 * w_b / T2_cmb_norm**2 * (1.0e3 / z_d)  # TODO what's the 0 in 1.0e3?
    # Ratio of the baryon to photon momentum density at z_eq Eq. (5)
    R_eq = 31.5 * w_b / T2_cmb_norm**2 * (1.0e3 / z_eq)
    # Sound horizon at drag epoch Eq. (6)
    sh_d = (
        2.0
        / (3.0 * k_eq)
        * jnp.sqrt(6.0 / R_eq)
        * jnp.log((jnp.sqrt(1.0 + R_d) + jnp.sqrt(R_eq + R_d)) / (1.0 + jnp.sqrt(R_eq)))
    )
    # Eq. (7)
    k_silk = 1.6 * w_b**0.52 * w_m**0.73 * (1.0 + (10.4 * w_m)**-0.95)
    #############################################

    type = "eisenhu_osc"  # FIXME HACK
    if type == "eisenhu":
        alpha_gamma = (
            1.0
            - 0.328 * jnp.log(431.0 * w_m) * f_b
            + 0.38 * jnp.log(22.3 * w_m) * f_b**2
        )
        gamma_eff = (
            w_m
            * (alpha_gamma + (1.0 - alpha_gamma) / (1.0 + (0.43 * k * sh_d)**4))
        )  # TODO w_m vs Om*h as shape parameter

        q = k * T2_cmb_norm / gamma_eff

        # EH98 (29) TODO Eq. vs EH98
        L = jnp.log(2.0 * jnp.e + 1.8 * q)
        C = 14.2 + 731.0 / (1.0 + 62.5 * q)
        T = L / (L + C * q**2)

    elif type == "eisenhu_osc":
        # Cold dark matter transfer function

        # EH98 (11, 12)
        a1 = (46.9 * w_m)**0.670 * (1.0 + (32.1 * w_m)**-0.532)
        a2 = (12.0 * w_m)**0.424 * (1.0 + (45.0 * w_m)**-0.582)
        alpha_c = a1**-f_b * a2**-f_b**3
        b1 = 0.944 / (1.0 + (458.0 * w_m)**-0.708)
        b2 = (0.395 * w_m)**-0.0266
        beta_c = 1.0 + b1 * (f_c**b2 - 1.0)
        beta_c = 1.0 / beta_c

        # EH98 (19)
        def T_tilde(k, alpha, beta):
            # EH98 (10)
            q = k / (13.41 * k_eq)
            L = jnp.log(jnp.e + 1.8 * beta * q)
            C = 14.2 / alpha + 386.0 / (1.0 + 69.9 * q**1.08)
            T0 = L / (L + C * q**2)
            return T0

        # EH98 (17, 18)
        f = 1.0 / (1.0 + (k * sh_d / 5.4)**4)
        T_c = f * T_tilde(k, 1.0, beta_c) + (1.0 - f) * T_tilde(k, alpha_c, beta_c)

        # Baryon transfer function
        # EH98 (19, 14, 21)
        y = (1.0 + z_eq) / (1.0 + z_d)
        x = jnp.sqrt(1.0 + y)
        G_EH98 = y * (-6.0 * x + (2.0 + 3.0 * y) * jnp.log((x + 1.0) / (x - 1.0)))
        alpha_b = 2.07 * k_eq * sh_d * (1.0 + R_d)**-0.75 * G_EH98

        beta_node = 8.41 * w_m**0.435
        tilde_s = sh_d / jnp.cbrt(1.0 + (beta_node / (k * sh_d))**3)

        beta_b = 0.5 + f_b + (3.0 - 2.0 * f_b) * jnp.sqrt(1.0 + (17.2 * w_m)**2)

        T_b = (
            T_tilde(k, 1.0, 1.0) / (1.0 + (k * sh_d / 5.2)**2)
            + alpha_b
            / (1.0 + (beta_b / (k * sh_d))**3)
            * jnp.exp(-(k / k_silk)**1.4)
        ) * jnp.sinc(k * tilde_s / jnp.pi)

        # Total transfer function
        T = f_c * T_c + f_b * T_b
    else:
        raise NotImplementedError

    return T


def growth_integ(cosmo):
    """Intergrate and tabulate (LPT) growth functions and derivatives at given scale
    factors.

    Parameters
    ----------
    cosmo : Cosmology

    Returns
    -------
    cosmo : Cosmology
        A new instance containing a growth table, or the input one if it already exists.
        The growth table has the shape ``(num_lpt_order, num_derivatives,
        num_scale_factors)`` and ``cosmo.conf.growth_dtype``.

    Notes
    -----

    TODO: ODE math

    """
    if cosmo.growth is not None:
        return cosmo

    conf = cosmo.conf

    with ensure_compile_time_eval():
        eps = jnp.finfo(conf.growth_dtype).eps
        growth_a_ic = 0.5 * jnp.cbrt(eps).item()  # ~ 3e-6 for float64, 2e-3 for float32
        if growth_a_ic >= conf.a_start / conf.growth_lpt_size:
            growth_a_ic = 0.1 * conf.a_start / conf.growth_lpt_size

    a = conf.growth_a
    lna = jnp.log(a.at[0].set(growth_a_ic))

    num_order, num_deriv, num_a = 2, 3, len(a)

    # TODO add lpt_order support
    # G and lna can either be at a single time, or have leading time axes
    def ode(G, lna, cosmo):
        a = jnp.exp(lna)
        H_fac = H_deriv(a, cosmo)
        Omega_fac = 1.5 * Omega_m_a(a, cosmo)
        G_1, Gp_1, G_2, Gp_2 = jnp.split(G, num_order * (num_deriv-1), axis=-1)
        Gpp_1 = (-3. - H_fac + Omega_fac) * G_1 + (-4. - H_fac) * Gp_1
        Gpp_2 = Omega_fac * G_1**2 + (-8. - 2.*H_fac + Omega_fac) * G_2 + (-6. - H_fac) * Gp_2
        return jnp.concatenate((Gp_1, Gpp_1, Gp_2, Gpp_2), axis=-1)

    G_ic = jnp.array((1, 0, 3/7, 0), dtype=conf.growth_dtype)

    G = odeint(ode, G_ic, lna, cosmo, rtol=conf.growth_rtol, atol=conf.growth_atol)

    G_deriv = ode(G, lna[:, jnp.newaxis], cosmo)

    G = G.reshape(num_a, num_order, num_deriv-1)
    G_deriv = G_deriv.reshape(num_a, num_order, num_deriv-1)
    G = jnp.concatenate((G, G_deriv[..., -1:]), axis=2)
    G = jnp.moveaxis(G, 0, 2)

    # D_m /a^m = G
    # D_m'/a^m = G + G'
    # D_m"/a^m = G + 2G' + G"
    growth = jnp.stack((
        G[:, 0],
        G[:, 0] + G[:, 1],
        G[:, 0] + 2.*G[:, 1] + G[:, 2],
    ), axis=1)

    return replace(cosmo, growth=growth)


# TODO add 3rd order, which has two factors, so `order` probably need to support str
@partial(jit, static_argnames=('order', 'deriv'))
def growth(a, cosmo, order=1, deriv=0):
    """Evaluate interpolation of (LPT) growth function or derivative, the n-th
    derivatives of the m-th order growth function :math:`\mathrm{d}^n D_m /
    \mathrm{d}\ln^n a`, at given scale factors. Growth functions are normalized at the
    matter dominated era instead of today.

    Parameters
    ----------
    a : array_like
        Scale factors.
    cosmo : Cosmology
    order : int in {1, 2}, optional
        Order of growth function.
    deriv : int in {0, 1, 2}, optional
        Order of growth function derivatives.

    Returns
    -------
    D : jax.numpy.ndarray of cosmo.conf.growth_dtype
        Growth functions or derivatives.

    Raises
    ------
    ValueError
        If ``cosmo.growth`` table is empty.

    """
    if cosmo.growth is None:
        raise ValueError('Growth table is empty. Call growth_integ or boltzmann first.')

    conf = cosmo.conf

    a = jnp.asarray(a, dtype=conf.growth_dtype)

    D = a**order * jnp.interp(a, conf.growth_a, cosmo.growth[order-1][deriv])

    return D


@jit
def boltzmann(cosmo):
    """Solve Einstein-Boltzmann equations and precompute transfer and growth functions.

    Parameters
    ----------
    cosmo : Cosmology

    Returns
    -------
    cosmo : Cosmology
        A new instance containing transfer (TODO) and growth tables, or the input one if
        it already exists.

    Notes
    -----
    Eisenstein-Hu approximation to transfer function is used instead for now.

    """
    #cosmo = transfer_integ(cosmo)  # TODO
    cosmo = growth_integ(cosmo)
    return cosmo


def linear_power(k, a, cosmo):
    r"""Linear matter power spectrum at given wavenumbers and scale factor.

    Parameters
    ----------
    k : array_like
        Wavenumbers in [1/L].
    a : array_like or None
        Scale factors. If None, output is not scaled by growth.
    cosmo : Cosmology

    Returns
    -------
    Plin: jax.numpy.ndarray of cosmo.conf.float_dtype
        Linear matter power spectrum.

    Notes
    -----

    .. math::

        \frac{k^3}{2\pi^2} P_\mathrm{lin}(k, a)
        = \frac{4}{25} A_\mathrm{s}
            \Bigl( \frac{k}{k_\mathrm{pivot} \Bigr)^{n_\mathrm{s} - 1}
            T^2(k)
            \Bigl( \frac{c k}{H_0} \Bigr)^4
            \Bigl( \frac{D(a)}{\Omega_\mathrm{m}} \Bigr)

    """
    conf = cosmo.conf

    k = jnp.asarray(k, dtype=conf.float_dtype)
    T = transfer_EH(k, cosmo)

    D = 1.
    if a is not None:
        D = growth(a, cosmo)
        D = D.astype(conf.float_dtype)

    Plin = (
        0.32e-9 * cosmo.A_s_1e9 * cosmo.k_pivot * (k / cosmo.k_pivot)**cosmo.n_s
        * (jnp.pi * (conf.c / conf.H_0)**2 / cosmo.Omega_m * T)**2
        * D**2
    )

    return Plin
