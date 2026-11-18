"""Hybrid effective field theory (HEFT) operators on periodic cubic grids.

Adapted from Haojie Xu's JAX HEFT implementation: forward-normalized FFTs,
de-aliased quadratic weights, and weighted CIC assignment. Lengths follow
Configuration units (Mpc/h by default); wavenumbers use inverse lengths.
Configuration.float_dtype selects float32 or float64 field precision.
Float32 accumulation can limit weak-signal and cancellation-sensitive statistics.

For JIT, keep configuration and keyword options static. Input fields, particle
displacements, and cosmological parameters support reverse-mode differentiation.
Paths using native pmwd custom VJPs do not support general forward-mode JVPs.
"""

from operator import index

import jax.numpy as jnp

from pmwd.boltzmann import boltzmann, growth
from pmwd.lpt import lpt
from pmwd.modes import linear_modes
from pmwd.nbody import nbody
from pmwd.particles import Particles
from pmwd.scatter import scatter


def _check_conf(conf):
    shape = conf.ptcl_grid_shape
    if len(shape) != 3 or len(set(shape)) != 1 or shape[0] < 2:
        raise ValueError('HEFT requires a cubic 3D particle grid with side >= 2')
    if len(set(conf.mesh_shape)) != 1 or conf.ptcl_spacing <= 0:
        raise ValueError('HEFT requires a cubic periodic box with positive spacing')
    if conf.float_dtype not in (jnp.dtype('float32'), jnp.dtype('float64')):
        raise ValueError('HEFT requires float32 or float64 precision')


def _check_modes(modes, conf, check_dtype=True):
    _check_conf(conf)
    n = conf.ptcl_grid_shape[0]
    if modes.shape != (n, n, n // 2 + 1) or not jnp.iscomplexobj(modes):
        raise ValueError('linear_field must be a cubic complex rFFT array')
    if check_dtype and modes.real.dtype != conf.float_dtype:
        raise ValueError('linear_field precision must match conf.float_dtype')


def _kvec(n, box_size, dtype):
    k = (2 * jnp.pi * jnp.fft.fftfreq(n, d=box_size / n)).astype(dtype)
    kr = (2 * jnp.pi * jnp.fft.rfftfreq(n, d=box_size / n)).astype(dtype)
    return k[:, None, None], k[None, :, None], kr[None, None, :]


def _real(modes):
    return jnp.fft.irfftn(modes, s=(modes.shape[0],) * 3, norm='forward')


def _fftlen(n):
    """Next 2/3/5-smooth size; shape arithmetic is static."""
    while True:
        rest = n
        for prime in (2, 3, 5):
            while rest % prime == 0:
                rest //= prime
        if rest == 1:
            return n
        n += 1


def _pad(modes):
    """Pad above the quadratic alias boundary, splitting Nyquist planes."""
    n = modes.shape[0]
    h = n // 2
    padded = _fftlen(3 * h + 1)
    if n % 2 == 0:
        modes = modes.at[..., h].multiply(0.5)
        modes = modes.at[:, h, :].multiply(0.5)
        modes = modes.at[h, :, :].multiply(0.5)
    modes = jnp.pad(modes, ((0, 0), (0, 0),
                           (0, padded // 2 + 1 - modes.shape[2])))
    for axis in (1, 0):
        a = jnp.moveaxis(modes, axis, 0)
        zeros = jnp.zeros((padded - n - (n % 2 == 0),) + a.shape[1:], a.dtype)
        start = h if n % 2 == 0 else h + 1
        modes = jnp.moveaxis(jnp.concatenate((a[:h + 1], zeros, a[start:])),
                            0, axis)
    return modes


def _trim(modes, n):
    """Project onto the retained cube and merge both Nyquist signs."""
    padded, h = modes.shape[0], n // 2
    if n % 2 == 0:
        opposite = (-jnp.arange(padded)) % padded
        # Negative kz is implicit: F(kx, ky, -kz) = conj(F(-kx, -ky, kz)).
        nyquist = modes[..., h] + jnp.conj(
            modes[opposite[:, None], opposite[None, :], h])
        modes = jnp.concatenate((modes[..., :h], nyquist[..., None]), axis=2)
    else:
        modes = modes[..., :h + 1]
    for axis in (1, 0):
        a = jnp.moveaxis(modes, axis, 0)
        if n % 2 == 0:
            nyquist = a[h:h + 1] + a[padded - h:padded - h + 1]
            a = jnp.concatenate((a[:h], nyquist, a[padded - h + 1:]))
        else:
            a = jnp.concatenate((a[:h + 1], a[-h:]))
        modes = jnp.moveaxis(a, 0, axis)
    return modes


def _center_project(field, n):
    modes = jnp.fft.rfftn(field - field.mean(), norm='forward')
    return _real(_trim(modes, n).at[0, 0, 0].set(0))


def heft_weights(linear_field, conf, *, smoothing='gaussian', kd=0.75,
                 radius=5., pivot=1.):
    """Return Lagrangian weights ``[d, L, d2, s2]``, shape ``(4, N, N, N)``.

    ``linear_field`` is the unsmoothed complex rFFT of linear density at the
    reference epoch, with ``norm='forward'`` and real dtype ``conf.float_dtype``.
    Convert native pmwd modes with :func:`linear_field_from_modes`. The zero
    mode is discarded.

    ``smoothing`` selects a field-amplitude window: ``'gaussian'`` uses
    exp(-k**2/(2*kd**2)), ``'csst'`` uses
    exp(-radius**2*max(k-pivot, 0)**2/4), and ``'none'`` uses unity.
    Here k is the wavenumber magnitude; kd and pivot are inverse lengths,
    and radius is a length, in Configuration units.

    L is the Laplacian of d. d2 and s2 are the de-aliased, mean-subtracted
    products d**2 and sum_ij(K_ij**2), where
    K_ij = (k_i*k_j/k**2 - delta_ij/3)*d(k) is the traceless tidal tensor.
    Spectral padding splits Nyquist modes and merges them on projection.
    """
    linear_field = jnp.asarray(linear_field)
    _check_modes(linear_field, conf)
    n = conf.ptcl_grid_shape[0]
    box_size = conf.box_size[0]
    d = linear_field.at[0, 0, 0].set(0)
    kvec = _kvec(n, box_size, d.real.dtype)
    if smoothing == 'gaussian':
        if kd <= 0:
            raise ValueError('kd must be positive')
        for k in kvec:
            d = d * jnp.exp(-0.5 * (k / kd)**2)
    elif smoothing == 'csst':
        if radius < 0 or pivot < 0:
            raise ValueError('radius and pivot must be nonnegative')
        k = jnp.sqrt(sum(k**2 for k in kvec))
        d = d * jnp.exp(-0.25 * radius**2 * jnp.maximum(k - pivot, 0)**2)
    elif smoothing != 'none':
        raise ValueError("smoothing must be 'gaussian', 'csst', or 'none'")

    density = _real(d)
    laplacian = _real(-sum(k**2 for k in kvec) * d)
    padded = _pad(d)
    quadratic = _center_project(_real(padded)**2, n)

    kvec = _kvec(padded.shape[0], box_size, d.real.dtype)
    k2 = sum(k**2 for k in kvec)
    potential = (-padded / jnp.where(k2 == 0, 1, k2)).at[0, 0, 0].set(0)
    tidal = jnp.zeros((padded.shape[0],) * 3, dtype=d.real.dtype)
    for i in range(3):
        for j in range(i, 3):
            kernel = k2 / 3 - kvec[i]**2 if i == j else -kvec[i] * kvec[j]
            # All input support lies strictly below the padded Nyquist planes.
            component = _real((kernel * potential).at[0, 0, 0].set(0))
            tidal = tidal + (1 if i == j else 2) * component**2
    tidal = _center_project(tidal, n)
    return jnp.stack((density, laplacian, quadratic, tidal))


def heft_operators(linear_field, particles, conf, *, output_grid=None,
                   smoothing='gaussian', kd=0.75, radius=5., pivot=1.,
                   offset=None):
    """Return CIC-advected fields ``[O1, Od, OL, Od2, Os2]``.

    Particle weights ``[1, d, L, d2, s2]`` use :func:`heft_weights` and the
    global normalization (M/N)**3. Unit weights give O1 = rho/mean(rho) - 1;
    the other channels deposit their signed weights without further division.

    ``particles`` must use the supplied Configuration and contain all N**3
    particles in their original C-order Lagrangian association (z fastest).
    Restore this order if rows were rearranged; it cannot be inferred from
    evolved positions. Displacement dtype must match ``conf.float_dtype``.
    Positions are ``pmid * conf.cell_size + disp``.

    ``output_grid`` sets the cubic side M and defaults to the force-mesh side.
    The result has shape (5, M, M, M). ``offset`` is the physical location of
    output node 0, as a scalar or length-3 tuple. Its default is half an output
    cell along each axis; use 0 for nodes at integer cell coordinates.
    """
    _check_conf(conf)
    if not isinstance(particles, Particles) or particles.conf != conf:
        raise ValueError('particles must use the supplied Configuration')
    expected = (conf.ptcl_num, 3)
    if particles.pmid.shape != expected or particles.disp.shape != expected:
        raise ValueError('particles must contain the complete Lagrangian grid')
    if particles.disp.dtype != conf.float_dtype:
        raise ValueError('displacement precision must match conf.float_dtype')
    m = conf.mesh_shape[0] if output_grid is None else index(output_grid)
    if m < 2:
        raise ValueError('output_grid must be an integer >= 2')
    cell_size = conf.box_size[0] / m
    offset = cell_size / 2 if offset is None else offset
    if jnp.asarray(offset).shape not in ((), (3,)):
        raise ValueError('offset must be scalar or length 3')

    weights = heft_weights(linear_field, conf, smoothing=smoothing, kd=kd,
                           radius=radius, pivot=pivot)
    weights = jnp.concatenate((jnp.ones_like(weights[:1]), weights))
    values = weights.reshape(5, -1).T * (m / conf.ptcl_grid_shape[0])**3
    mesh = jnp.zeros((m, m, m, 5), dtype=conf.float_dtype)
    mesh = scatter(particles, conf, mesh=mesh, val=values,
                   offset=offset, cell_size=cell_size)
    return jnp.moveaxis(mesh, -1, 0).at[0].add(-1)


def linear_field_from_modes(modes, cosmo, conf, *, a_ref=1., modes_a=None):
    """Convert native pmwd Fourier modes to a HEFT forward-normalized rFFT.

    By default, ``modes`` comes from ``linear_modes(..., a=None)``:
    ``delta_forward(a_ref) = modes * growth(a_ref) / conf.box_vol``.
    For modes already scaled at an epoch, set ``modes_a``; the factor becomes
    growth(a_ref)/growth(modes_a)/box_vol. Growth at 1 is not assumed to be unity.

    Both epochs must lie in (0, conf.a_stop], within the growth table populated
    by boltzmann. The input complex dtype is preserved.
    """
    modes = jnp.asarray(modes)
    _check_modes(modes, conf, check_dtype=False)
    if cosmo.conf != conf:
        raise ValueError('cosmology must use the supplied Configuration')
    for epoch in (a_ref, modes_a):
        if epoch is not None and not 0 < epoch <= conf.a_stop:
            raise ValueError('reference/modes epoch must be in (0, conf.a_stop]')
    factor = growth(a_ref, cosmo, conf) / conf.box_vol
    if modes_a is not None:
        factor = factor / growth(modes_a, cosmo, conf)
    return modes * factor.astype(modes.real.dtype)


def simulate_heft(white_noise, cosmo, conf, *, a_ref=1., output_grid=None,
                  smoothing='gaussian', kd=0.75, radius=5., pivot=1.,
                  offset=None):
    """Compute five HEFT fields from white noise and cosmological parameters.

    ``white_noise`` is a real or Fourier array from pmwd.modes.white_noise.
    Native linear theory, LPT, and N-body evolution produce particles at
    ``conf.a_stop``. The linear weights refer to ``a_ref``; its default is 1,
    and :func:`linear_field_from_modes` requires 0 < a_ref <= conf.a_stop.

    Unsmoothed linear modes drive the dynamics. Smoothing applies only to
    the HEFT weights, whose precision matches ``conf.float_dtype``. Other
    keyword options and the returned channel order follow :func:`heft_operators`.
    Derivatives include both changing weights and particle trajectories.
    """
    _check_conf(conf)
    white_noise = jnp.asarray(white_noise)
    expected = (conf.ptcl_grid_shape if jnp.isrealobj(white_noise) else
                conf.ptcl_grid_shape[:2] + (conf.ptcl_grid_shape[2] // 2 + 1,))
    if white_noise.shape != expected:
        raise ValueError('white_noise shape must match the particle grid')
    if cosmo.conf != conf:
        raise ValueError('cosmology must use the supplied Configuration')
    if not 0 < a_ref <= conf.a_stop:
        raise ValueError('a_ref must be in (0, conf.a_stop]')
    cosmo = boltzmann(cosmo, conf, varlin=False)
    modes = linear_modes(white_noise, cosmo, conf)
    particles, observables = lpt(modes, cosmo, conf)
    particles, _ = nbody(particles, observables, cosmo, conf)
    field = linear_field_from_modes(modes, cosmo, conf, a_ref=a_ref)
    field = field.astype(jnp.promote_types(conf.float_dtype, jnp.complex64))
    return heft_operators(field, particles, conf, output_grid=output_grid,
                          smoothing=smoothing, kd=kd, radius=radius, pivot=pivot,
                          offset=offset)

