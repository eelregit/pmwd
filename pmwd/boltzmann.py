from jax import jit, custom_vjp, ensure_compile_time_eval
import jax.numpy as jnp
from jax.experimental.ode import odeint

from pmwd.cosmology import H_deriv, Omega_m_a
from pmwd.growth_integrals import growth_integ, growth_integ_rk4, growth_integ_mlp, Growth_MLP
growth_fn = Growth_MLP()

@jit
def transfer_integ(cosmo, conf):
    if conf.transfer_fit:
        return cosmo
    else:
        raise NotImplementedError('TODO')


# TODO maybe need to checkpoint EH for memory?
# TODO Wayne's website: neutrino no wiggle case
def transfer_fit(k, cosmo, conf):
    """Eisenstein & Hu fit of matter transfer function at given wavenumbers.

    Parameters
    ----------
    k : array_like
        Wavenumbers in [1/L].
    cosmo : Cosmology
    conf : Configuration

    Returns
    -------
    T : jax.numpy.ndarray of conf.float_dtype
        Matter transfer function.

    .. _Transfer Function:
        http://background.uchicago.edu/~whu/transfer/transferpage.html

    """
    k = jnp.asarray(k, dtype=conf.float_dtype)
    cosmo = cosmo.astype(conf.float_dtype)

    k = k * cosmo.h / conf.L * conf.Mpc_SI  # unit conversion to [1/Mpc]

    T2_cmb_norm = (conf.T_cmb / 2.7)**2
    h2 = cosmo.h**2
    w_m = cosmo.Omega_m * h2
    w_b = cosmo.Omega_b * h2
    f_b = cosmo.Omega_b / cosmo.Omega_m
    f_c = cosmo.Omega_c / cosmo.Omega_m  # TODO neutrinos?

    z_eq = 2.50e4 * w_m / T2_cmb_norm**2
    k_eq = 7.46e-2 * w_m / T2_cmb_norm

    b1 = 0.313 * w_m**-0.419 * (1 + 0.607 * w_m**0.674)
    b2 = 0.238 * w_m**0.223
    z_d = 1291 * w_m**0.251 / (1 + 0.659 * w_m**0.828) * (1 + b1 * w_b**b2)

    R_d = 31.5 * w_b / T2_cmb_norm**2 * (1e3 / z_d)
    R_eq = 31.5 * w_b / T2_cmb_norm**2 * (1e3 / z_eq)
    s = (
        2 / (3 * k_eq) * jnp.sqrt(6 / R_eq)
        * jnp.log((jnp.sqrt(1 + R_d) + jnp.sqrt(R_eq + R_d)) / (1 + jnp.sqrt(R_eq)))
    )
    k_silk = 1.6 * w_b**0.52 * w_m**0.73 * (1 + (10.4 * w_m)**-0.95)

    if conf.transfer_fit_nowiggle:
        alpha_gamma = (1 - 0.328 * jnp.log(431 * w_m) * f_b
                       + 0.38 * jnp.log(22.3 * w_m) * f_b**2)
        gamma_eff_ratio = alpha_gamma + (1 - alpha_gamma) / (1 + (0.43 * k * s)**4)

        q_eff = k / (13.41 * k_eq * gamma_eff_ratio)

        L0 = jnp.log(2 * jnp.e + 1.8 * q_eff)
        C0 = 14.2 + 731 / (1 + 62.5 * q_eff)
        T0 = L0 / (L0 + C0 * q_eff**2)

        return T0

    a1 = (46.9 * w_m)**0.670 * (1 + (32.1 * w_m)**-0.532)
    a2 = (12.0 * w_m)**0.424 * (1 + (45.0 * w_m)**-0.582)
    alpha_c = a1**-f_b * a2**-f_b**3
    b1 = 0.944 / (1 + (458 * w_m)**-0.708)
    b2 = (0.395 * w_m)**-0.0266
    beta_c = 1 / (1 + b1 * (f_c**b2 - 1))

    def T0_tilde(k, alpha_c, beta_c):
        q = k / (13.41 * k_eq)
        L = jnp.log(jnp.e + 1.8 * beta_c * q)
        C = 14.2 / alpha_c + 386 / (1 + 69.9 * q**1.08)
        T0 = L / (L + C * q**2)
        return T0

    f = 1 / (1 + (k * s / 5.4)**4)
    T_c = f * T0_tilde(k, 1, beta_c) + (1 - f) * T0_tilde(k, alpha_c, beta_c)

    y = (1 + z_eq) / (1 + z_d)
    x = jnp.sqrt(1 + y)
    G = y * (-6 * x + (2 + 3 * y) * jnp.log((x + 1) / (x - 1)))
    alpha_b = 2.07 * k_eq * s * (1 + R_d)**-0.75 * G

    beta_node = 8.41 * w_m**0.435
    beta_b = 0.5 + f_b + (3 - 2 * f_b) * jnp.sqrt(1 + (17.2 * w_m)**2)

    T_b = (
        T0_tilde(k, 1, 1) / (1 + (k * s / 5.2)**2)
        + alpha_b * (k * s)**3 / (beta_b**3 + (k * s)**3) * jnp.exp(-(k / k_silk)**1.4)
    ) * jnp.sinc((k * s)**2 / (jnp.pi * jnp.cbrt(beta_node**3 + (k * s)**3)))

    T = f_c * T_c + f_b * T_b

    return T


def transfer(k, cosmo, conf):
    """Evaluate interpolation or Eisenstein & Hu fit of matter transfer function at
    given wavenumbers.

    Parameters
    ----------
    k : array_like
        Wavenumbers in [1/L].
    cosmo : Cosmology
    conf : Configuration

    Returns
    -------
    T : jax.numpy.ndarray
        Matter transfer function.

    """
    if conf.transfer_fit:
        return transfer_fit(k, cosmo, conf)
    else:
        raise NotImplementedError('TODO')



# TODO 3rd order has two factors, so `order` probably need to support str
def growth(a, cosmo, conf, order=1, deriv=0):
    """Evaluate interpolation of (LPT) growth function or derivative, the n-th
    derivatives of the m-th order growth function :math:`\mathrm{d}^n D_m /
    \mathrm{d}\ln^n a`, at given scale factors. Growth functions are normalized at the
    matter dominated era instead of today.

    Parameters
    ----------
    a : array_like
        Scale factors.
    cosmo : Cosmology
    conf : Configuration
    order : int in {1, 2}, optional
        Order of growth function.
    deriv : int in {0, 1, 2}, optional
        Order of growth function derivatives.

    Returns
    -------
    D : jax.numpy.ndarray of conf.cosmo_dtype
        Growth functions or derivatives.

    Raises
    ------
    ValueError
        If ``cosmo.growth`` table is empty.

    """
    if cosmo.growth is None:
        raise ValueError('Growth table is empty. Call growth_integ or boltzmann first.')
    
    a = jnp.asarray(a, dtype=conf.cosmo_dtype)
    if conf.growth_mode == 'mlp':
        D = jnp.interp(a, conf.growth_a, cosmo.growth[order-1][deriv])
    else:
        D = a**order * jnp.interp(a, conf.growth_a, cosmo.growth[order-1][deriv])
    
    # if conf.growth_mode == 'mlp':
    #     D = growth_fn(jnp.array([cosmo.Omega_m]), a, order, deriv)
    # else:
    #     a = jnp.asarray(a, dtype=conf.cosmo_dtype)
    #     D = a**order * jnp.interp(a, conf.growth_a, cosmo.growth[order-1][deriv])
    return D


def boltzmann(cosmo, conf):
    """Solve Einstein-Boltzmann equations and precompute transfer and growth functions.

    Parameters
    ----------
    cosmo : Cosmology
    conf : Configuration

    Returns
    -------
    cosmo : Cosmology
        A new instance containing transfer and growth tables, or the input one if they
        already exists.

    """
    if conf.growth_mode == 'adaptive':
        cosmo = growth_integ(cosmo, conf)
    elif conf.growth_mode == 'rk4':
        cosmo = growth_integ_rk4(cosmo, conf)
    elif conf.growth_mode == 'mlp':
        cosmo = growth_integ_mlp(cosmo, conf)

    cosmo = transfer_integ(cosmo, conf)
    return cosmo


@custom_vjp
def _safe_power(x1, x2):
    """Safe power function for x1==0 and 0<x2<1. x2 must be a scalar."""
    return x1 ** x2

def _safe_power_fwd(x1, x2):
    y = _safe_power(x1, x2)
    return y, (x1, x2, y)

def _safe_power_bwd(res, y_cot):
    x1, x2, y = res

    x1_cot = jnp.where(x1 != 0, x2 * y / x1 * y_cot, 0)

    lnx1 = jnp.where(x1 != 0, jnp.log(x1), 0)
    x2_cot = (lnx1 * y * y_cot).sum()

    return x1_cot, x2_cot

_safe_power.defvjp(_safe_power_fwd, _safe_power_bwd)


def linear_power(k, a, cosmo, conf):
    r"""Linear matter power spectrum in [L^3] at given wavenumbers and scale factors.

    Parameters
    ----------
    k : array_like
        Wavenumbers in [1/L].
    a : array_like or None
        Scale factors. If None, output is not scaled by growth.
    cosmo : Cosmology
    conf : Configuration

    Returns
    -------
    Plin : jax.numpy.ndarray of conf.float_dtype
        Linear matter power spectrum in [L^3].

    Raises
    ------
    ValueError
        If not in 3D.

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
    if conf.dim != 3:
        raise ValueError(f'dim={conf.dim} not supported')

    k = jnp.asarray(k, dtype=conf.float_dtype)
    T = transfer(k, cosmo, conf)

    D = 1
    if a is not None:
        D = growth(a, cosmo, conf)
        D = D.astype(conf.float_dtype)

    Plin = (
        0.32 * cosmo.A_s * cosmo.k_pivot * _safe_power(k / cosmo.k_pivot, cosmo.n_s)
        * (jnp.pi * (conf.c / conf.H_0)**2 / cosmo.Omega_m * T)**2
        * D**2
    )

    return Plin
