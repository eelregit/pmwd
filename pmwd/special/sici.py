# Adapted from The Fullerton Function Library netlib.org/fn, by Wayne Fullerton
# and GSL specfunc/sinint.c
# NOTE "single precision" fn Chebyshev approximations seem good enough and are used

from math import sqrt
from functools import partial

from jax import jit, custom_jvp
import jax.numpy as jnp
from jax.lax import scan, select, select_n


# Chebyshev series evaluation for -1 <= x <= 1
# https://www.netlib.org/fn/csevl.f & https://www.netlib.org/fn/dcsevl.f
# See https://en.wikipedia.org/wiki/Clenshaw_algorithm
# & R. Broucke 1973, Algorithm 446: Ten subroutines for the manipulation of Chebyshev series
def _cheb_eval(cs, x):
    # Clenshaw algorithm reverse recurrence
    def rev_recur(carry, a):
        twox, bip1, bip2 = carry
        bi = a + twox * bip1 - bip2  # a = cs[i]
        carry = twox, bi, bip1
        return carry, None

    carry = 2 * x, jnp.zeros_like(x), jnp.zeros_like(x)  # twox, bip1, bip2
    carry, _ = scan(rev_recur, carry, cs[:0:-1])
    _, b1, b2 = carry

    return 0.5 * cs[0] + x * b1 - b2  # NOTE only half the first coef is summed


# for x >= 4
# https://www.netlib.org/fn/r9sifg.f & https://www.netlib.org/fn/d9sifg.f
def _fg_asymp(x):
    if x.dtype not in (jnp.float32, jnp.float64):
        raise ValueError(f'{x.dtype=} not supported')

    # f Chebyshev series for [0.02, 0.0625]
    _f0_cs = jnp.array([
        -0.1191081969051363610,
        -0.0247823144996236248,
        +0.0011910281453357821,
        -0.0000927027714388562,
        +0.0000093373141568271,
        -0.0000011058287820557,
        +0.0000001464772071460,
        -0.0000000210694496288,
        +0.0000000032293492367,
        -0.0000000005206529618,
        +0.0000000000874878885,
        -0.0000000000152176187,
        +0.0000000000027257192,
        -0.0000000000005007053,
        +0.0000000000000940241,
        -0.0000000000000180014,
        +0.0000000000000035063,
        -0.0000000000000006935,
        +0.0000000000000001391,
        -0.0000000000000000282,
    ], dtype=x.dtype)

    # f Chebyshev series for [0, 0.02]
    _f1_cs = jnp.array([
        -0.0348409253897013234,
        -0.0166842205677959686,
        +0.0006752901241237738,
        -0.0000535066622544701,
        +0.0000062693421779007,
        -0.0000009526638801991,
        +0.0000001745629224251,
        -0.0000000368795403065,
        +0.0000000087202677705,
        -0.0000000022601970392,
        +0.0000000006324624977,
        -0.0000000001888911889,
        +0.0000000000596774674,
        -0.0000000000198044313,
        +0.0000000000068641396,
        -0.0000000000024731020,
        +0.0000000000009226360,
        -0.0000000000003552364,
        +0.0000000000001407606,
        -0.0000000000000572623,
        +0.0000000000000238654,
        -0.0000000000000101714,
        +0.0000000000000044259,
        -0.0000000000000019634,
        +0.0000000000000008868,
        -0.0000000000000004074,
        +0.0000000000000001901,
        -0.0000000000000000900,
        +0.0000000000000000432,
    ], dtype=x.dtype)

    # g Chebyshev series for [0.02, 0.0625]
    _g0_cs = jnp.array([
        -0.3040578798253495954,
        -0.0566890984597120588,
        +0.0039046158173275644,
        -0.0003746075959202261,
        +0.0000435431556559844,
        -0.0000057417294453025,
        +0.0000008282552104503,
        -0.0000001278245892595,
        +0.0000000207978352949,
        -0.0000000035313205922,
        +0.0000000006210824236,
        -0.0000000001125215474,
        +0.0000000000209088918,
        -0.0000000000039715832,
        +0.0000000000007690431,
        -0.0000000000001514697,
        +0.0000000000000302892,
        -0.0000000000000061400,
        +0.0000000000000012601,
        -0.0000000000000002615,
        +0.0000000000000000548,
    ], dtype=x.dtype)

    # g Chebyshev series for [0, 0.02], single precision
    _g1_cs = jnp.array([
        -0.0967329367532432218,
        -0.0452077907957459871,
        +0.0028190005352706523,
        -0.0002899167740759160,
        +0.0000407444664601121,
        -0.0000071056382192354,
        +0.0000014534723163019,
        -0.0000003364116512503,
        +0.0000000859774367886,
        -0.0000000238437656302,
        +0.0000000070831906340,
        -0.0000000022318068154,
        +0.0000000007401087359,
        -0.0000000002567171162,
        +0.0000000000926707021,
        -0.0000000000346693311,
        +0.0000000000133950573,
        -0.0000000000053290754,
        +0.0000000000021775312,
        -0.0000000000009118621,
        +0.0000000000003905864,
        -0.0000000000001708459,
        +0.0000000000000762015,
        -0.0000000000000346151,
        +0.0000000000000159996,
        -0.0000000000000075213,
        +0.0000000000000035970,
        -0.0000000000000017530,
        +0.0000000000000008738,
        -0.0000000000000004487,
        +0.0000000000000002397,
        -0.0000000000000001347,
        +0.0000000000000000801,
        -0.0000000000000000501,
    ], dtype=x.dtype)

    xbnd = sqrt(50)
    xbig = 1 / sqrt(jnp.finfo(x).eps)

    f0 = (1 + _cheb_eval(_f0_cs, (x**-2 - 0.04125) / 0.02125)) * x**-1
    g0 = (1 + _cheb_eval(_g0_cs, (x**-2 - 0.04125) / 0.02125)) * x**-2
    f1 = (1 + _cheb_eval(_f1_cs, 100 * x**-2 - 1)) * x**-1
    g1 = (1 + _cheb_eval(_g1_cs, 100 * x**-2 - 1)) * x**-2
    f2 = x**-1
    g2 = x**-2

    which = jnp.searchsorted(jnp.array([xbnd, xbig], dtype=x.dtype), x,
                             method='compare_all')
    f = select_n(which, f0, f1, f2)
    g = select_n(which, g0, g1, g2)

    return f, g


# https://www.netlib.org/fn/si.f & https://www.netlib.org/fn/dsi.f
# https://www.netlib.org/fn/ci.f & https://www.netlib.org/fn/dci.f
@partial(custom_jvp, nondiff_argnames='Cin')
def _SiCi(x, Cin=False):
    r"""Sine and cosine integrals.

    Parameters
    ----------
    x : ArrayLike
        Real arguments.
    cin : bool, optional
        Whether to return Cin as the cosine integrals.

    Returns
    -------
    si : jax.Array
        Sine integrals.
    ci or cin : jax.Array
        Cosine integrals.

    References
    ----------
    .. _The Fullerton Function Library by Wayne Fullerton:
        https://www.netlib.org/fn/
    .. _GNU Scientific Library:
        https://cgit.git.savannah.gnu.org/cgit/gsl.git/tree/specfunc/sinint.c
    .. _Clenshaw summation:
        https://en.wikipedia.org/wiki/Clenshaw_algorithm

    Notes
    -----
    .. math::

        \mathrm{Si}(x) &= \int_0^x \frac{\sin t}t \mathrm{d}t, \\
        \mathrm{Ci}(x) &= - \int_x^\infty \frac{\cos t}t \mathrm{d}t \\
            &= \gamma + \ln x - \mathrm{Cin}(x),
        \mathrm{Cin}(x) &= \int_0^x \frac{1 - \cos t}t \mathrm{d}t.

    where :math:`gamma \approx 0.57721566490` is the Euler–Mascheroni constant.

    """
    x = jnp.asarray(x)
    if x.dtype not in (jnp.float32, jnp.float64):
        raise ValueError(f'{x.dtype=} not supported')

    # Si Chebyshev series for [0, 16]
    _si_cs = jnp.array([
        -0.1315646598184841929,
        -0.2776578526973601892,
        +0.0354414054866659180,
        -0.0025631631447933978,
        +0.0001162365390497009,
        -0.0000035904327241606,
        +0.0000000802342123706,
        -0.0000000013562997693,
        +0.0000000000179440722,
        -0.0000000000001908387,
        +0.0000000000000016670,
        -0.0000000000000000122,
    ], dtype=x.dtype)

    # Ci Chebyshev series for [0, 16]
    _ci_cs = jnp.array([
        -0.34004281856055363156,
        -1.03302166401177456807,
        +0.19388222659917082877,
        -0.01918260436019865894,
        +0.00110789252584784967,
        -0.00004157234558247209,
        +0.00000109278524300229,
        -0.00000002123285954183,
        +0.00000000031733482164,
        -0.00000000000376141548,
        +0.00000000000003622653,
        -0.00000000000000028912,
        +0.00000000000000000194,
    ], dtype=x.dtype)

    si0 = x * (0.75 + _cheb_eval(_si_cs, 0.125 * x**2 - 1))
    ci0 = - 0.5 + _cheb_eval(_ci_cs, 0.125 * x**2 - 1)
    if Cin:
        ci0 = jnp.euler_gamma - ci0
    else:
        ci0 += jnp.log(x)

    f, g = _fg_asymp(x)
    si1 = jnp.sign(x) * (0.5 * jnp.pi) - f * jnp.cos(x) - g * jnp.sin(x)
    ci1 = f * jnp.sin(x) - g * jnp.cos(x)
    if Cin:
        ci1 = jnp.euler_gamma + jnp.log(x) - ci1

    si = select(jnp.fabs(x) <= 4, si0, si1)
    ci = select(jnp.fabs(x) <= 4, ci0, ci1)

    return si, ci

@_SiCi.defjvp
def _SiCi_jvp(Cin, primals, tangents):
    (x,), (x_tan,) = primals, tangents
    si, ci = SiCi(x)
    si_tan = jnp.sin(x) * x_tan / x
    if Cin:
        ci_tan = (1 - jnp.cos(x)) * x_tan / x
    else:
        ci_tan = jnp.cos(x) * x_tan / x
    return (si, ci), (si_tan, ci_tan)

SiCi = jit(_SiCi, static_argnames='Cin')
