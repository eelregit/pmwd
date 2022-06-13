from itertools import permutations, combinations

from jax import jit, checkpoint, ensure_compile_time_eval
import jax.numpy as jnp

from pmwd.particles import Particles
from pmwd.cosmology import E2
from pmwd.boltzmann import growth
from pmwd.gravity import laplace, neg_grad
from pmwd.pm_util import rfftnfreq

def _strain(kvec, i, j, pot, conf):
    """LPT strain component sourced by scalar potential only.

     The Nyquist planes are not zeroed when i == j.

    .. _Notes on FFT-based differentiation:
        https://math.mit.edu/~stevenj/fft-deriv.pdf

    """
    k_i, k_j = kvec[i], kvec[j]

    nyquist = jnp.pi / conf.ptcl_spacing
    eps = nyquist * jnp.finfo(conf.float_dtype).eps

    #TODO test if more accurate
    if i != j:
        k_i = jnp.where(jnp.abs(jnp.abs(k_i) - nyquist) <= eps, 0, k_i)
        k_j = jnp.where(jnp.abs(jnp.abs(k_j) - nyquist) <= eps, 0, k_j)

    strain = -k_i * k_j * pot

    strain = jnp.fft.irfftn(strain, s=conf.ptcl_grid_shape)
    strain = strain.astype(conf.float_dtype)  # no jnp.complex32

    return strain


def _L(kvec, pot_m, pot_n, conf):
    m_eq_n = pot_n is None
    if m_eq_n:
        pot_n = pot_m

    L = jnp.zeros(conf.ptcl_grid_shape, dtype=conf.float_dtype)

    for i in range(conf.dim):
        strain_m = _strain(kvec, i, i, pot_m, conf)

        for j in range(conf.dim-1, i, -1):
            strain_n = _strain(kvec, j, j, pot_n, conf)

            L += strain_m * strain_n

        if not m_eq_n:
            for j in range(i-1, -1, -1):
                strain_n = _strain(kvec, j, j, pot_n, conf)

                L += strain_m * strain_n

    if not m_eq_n:
        L *= 0.5

    # Assuming strain sourced by scalar potential only, symmetric about ``i`` and ``j``,
    # for lpt_order <=3, i.e., m, n <= 2
    for i in range(conf.dim-1):
        for j in range(i+1, conf.dim):
            strain_m = _strain(kvec, i, j, pot_m, conf)

            strain_n = strain_m
            if not m_eq_n:
                strain_n = _strain(kvec, j, i, pot_n, conf)

            L -= strain_m * strain_n

    return L


def levi_civita(indices):
    """Levi-Civita symbol in n-D.

    Parameters
    ----------
    indices : array_like

    Returns
    -------
    epsilon : int
        Levi-Civita symbol value.

    """
    indices = jnp.asarray(indices)

    dim = len(indices)
    lohi = tuple(combinations(range(dim), r=2))
    lohi = jnp.array(lohi).T
    lo, hi = lohi[0], lohi[1]  # https://github.com/google/jax/issues/1583

    epsilon = jnp.sign(indices[hi] - indices[lo]).prod()

    return epsilon.item()


def _M(kvec, pot, conf):
    M = jnp.zeros(conf.ptcl_grid_shape, dtype=conf.float_dtype)

    for indices in permutations(range(conf.dim), r=3):
        i, j, k = indices
        strain_0i = _strain(kvec, 0, i, pot, conf)
        strain_1j = _strain(kvec, 1, j, pot, conf)
        strain_2k = _strain(kvec, 2, k, pot, conf)

        with ensure_compile_time_eval():
            epsilon = levi_civita(indices)
        M += epsilon * strain_0i * strain_1j * strain_2k

    return M

def _V(kvec, pot_1, pot_2, axis, conf):
    V = jnp.zeros(conf.ptcl_grid_shape, dtype=conf.float_dtype)

    i = (axis + 1) % 3
    j = (axis + 2) % 3
    for k in range(conf.dim) :
        strain_1 = _strain(kvec, i, k, pot_1, conf)
        strain_2 = _strain(kvec, j, k, pot_2, conf)
        V += strain_1 * strain_2
        strain_1 = _strain(kvec, j, k, pot_1, conf)
        strain_2 = _strain(kvec, i, k, pot_2, conf)
        V -= strain_1 * strain_2

    return V

def next_fast_len(n):
    """Find the next fast size to FFT.

    .. _scipy fftpack next_fast_len:
        https://github.com/scipy/scipy/blob/v1.8.0/scipy/fftpack/_helper.py
    .. _cupy scipy fft next_fast_len:
        https://github.com/cupy/cupy/blob/v10.4.0/cupyx/scipy/fft/_helper.py
    .. _JuliaDSP DSP.jl util:
        https://github.com/JuliaDSP/DSP.jl/blob/v0.7.5/src/util.jl#L109-L116
    .. _JuliaLang julia combinatorics:
        https://github.com/JuliaLang/julia/blob/v1.7.2/base/combinatorics.jl#L299-L316
    .. _nextprod-py:
        https://github.com/fasiha/nextprod-py
    """
    raise NotImplementedError


@jit
@checkpoint
def lpt(modes, cosmo, conf):
    """Lagrangian perturbation theory at ``conf.lpt_order``.

    Parameters
    ----------
    modes : jax.numpy.ndarray
        Linear matter overdensity Fourier modes in [L^3].
    cosmo : Cosmology
    conf : Configuration

    Returns
    -------
    ptcl : Particles
    obsvbl : Observables

    Raises
    ------
    ValueError
        If ``conf.dim`` or ``conf.lpt_order`` is not supported.

    """
    if conf.dim not in (1, 2, 3):
        raise ValueError(f'dim={conf.dim} not supported')
    if conf.lpt_order not in (0, 1, 2, 3):
        raise ValueError(f'lpt_order={conf.lpt_order} not supported')

    modes /= conf.ptcl_cell_vol  # remove volume factor first for convenience

    kvec = rfftnfreq(conf.ptcl_grid_shape, conf.ptcl_spacing, dtype=conf.float_dtype)

    a = conf.a_start
    ptcl = Particles.gen_grid(conf)

    scalar_pots = []
    vector_pot = []

    if conf.lpt_order > 0:
        src_1 = modes

        pot_1 = laplace(kvec, src_1, cosmo)

        scalar_pots.append(pot_1)

    if conf.lpt_order > 1:
        src_2 = _L(kvec, pot_1, None, conf)

        src_2 = jnp.fft.rfftn(src_2)

        pot_2 = laplace(kvec, src_2, cosmo)

        scalar_pots.append(pot_2)

    if conf.lpt_order > 2:

        src_3_a = _L(kvec, pot_1, pot_2, conf)

        src_3_a = jnp.fft.rfftn(src_3_a)

        pot_3_a = laplace(kvec, src_3_a, cosmo)

        scalar_pots.append(pot_3_a)

        src_3_b = -2. * _M(kvec, pot_1, conf)

        src_3_b = jnp.fft.rfftn(src_3_b)

        pot_3_b = pot_3_a + laplace(kvec, src_3_b, cosmo)

        scalar_pots.append(pot_3_b)

        for i in range(conf.dim) :
            vector_src = _V(kvec, pot_1, pot_2, i, conf)

            vector_src = jnp.fft.rfftn(vector_src)

            vector_src = laplace(kvec, vector_src, cosmo)

            vector_pot.append(vector_src)

    for order in range(1,conf.lpt_order+1) :

        D = growth(a, cosmo, conf, order=order)
        dD_dlna = growth(a, cosmo, conf, order=order, deriv=1)
        a2HDp = a**2 * jnp.sqrt(E2(a, cosmo)) * dD_dlna
        D = D.astype(conf.float_dtype)
        dD_dlna = dD_dlna.astype(conf.float_dtype)
        a2HDp = a2HDp.astype(conf.float_dtype)

        for i, k in enumerate(kvec):
            grad = neg_grad(k, scalar_pots[order-1], conf.ptcl_spacing)

            grad = jnp.fft.irfftn(grad, s=conf.ptcl_grid_shape)
            grad = grad.astype(conf.float_dtype)  # no jnp.complex32

            grad = grad.ravel()

            disp = ptcl.disp.at[:, i].add(D * grad)
            vel = ptcl.vel.at[:, i].add(a2HDp * grad)
            ptcl = ptcl.replace(disp=disp, vel=vel)

        if order == 3 :
            D = growth(a, cosmo, conf, order=order+1)
            dD_dlna = growth(a, cosmo, conf, order=order+1, deriv=1)
            a2HDp = a**2 * jnp.sqrt(E2(a, cosmo)) * dD_dlna
            D = D.astype(conf.float_dtype)
            dD_dlna = dD_dlna.astype(conf.float_dtype)
            a2HDp = a2HDp.astype(conf.float_dtype)

            for i, k in enumerate(kvec):
                grad = neg_grad(k, scalar_pots[order], conf.ptcl_spacing)

                grad = jnp.fft.irfftn(grad, s=conf.ptcl_grid_shape)
                grad = grad.astype(conf.float_dtype)  # no jnp.complex32

                grad = grad.ravel()

                disp = ptcl.disp.at[:, i].add(D * grad)
                vel = ptcl.vel.at[:, i].add(a2HDp * grad)
                ptcl = ptcl.replace(disp=disp, vel=vel)

            D = growth(a, cosmo, conf, order=order+2)
            dD_dlna = growth(a, cosmo, conf, order=order+2, deriv=1)
            a2HDp = a**2 * jnp.sqrt(E2(a, cosmo)) * dD_dlna
            D = D.astype(conf.float_dtype)
            dD_dlna = dD_dlna.astype(conf.float_dtype)
            a2HDp = a2HDp.astype(conf.float_dtype)

            for j in permutations(range(conf.dim), r=3) :
                with ensure_compile_time_eval():
                    sgn = -levi_civita(j)

                grad = neg_grad(kvec[j[1]], vector_pot[j[2]], conf.ptcl_spacing)

                grad = jnp.fft.irfftn(grad, s=conf.ptcl_grid_shape)
                grad = grad.astype(conf.float_dtype)  # no jnp.complex32

                grad = grad.ravel()

                disp = ptcl.disp.at[:, j[0]].add(sgn * D * grad)
                vel = ptcl.vel.at[:, j[0]].add(sgn * a2HDp * grad)
                ptcl = ptcl.replace(disp=disp, vel=vel)

    return ptcl, None
