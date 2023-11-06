import jax.numpy as jnp


def _chunk_split(ptcl_num, chunk_size, *arrays):
    """Split and reshape particle arrays into chunks and remainders, with the remainders
    preceding the chunks. 0D ones are duplicated as full arrays in the chunks."""
    chunk_size = ptcl_num if chunk_size is None else min(chunk_size, ptcl_num)
    remainder_size = ptcl_num % chunk_size
    chunk_num = ptcl_num // chunk_size

    remainder = None
    chunks = arrays
    if remainder_size:
        remainder = [x[:remainder_size] if x.ndim != 0 else x for x in arrays]
        chunks = [x[remainder_size:] if x.ndim != 0 else x for x in arrays]

    # `scan` triggers errors in scatter and gather without the `full`
    chunks = [x.reshape(chunk_num, chunk_size, *x.shape[1:]) if x.ndim != 0
              else jnp.full(chunk_num, x) for x in chunks]

    return remainder, chunks

def _chunk_cat(remainder_array, chunked_array):
    """Reshape and concatenate one remainder and one chunked particle arrays."""
    array = chunked_array.reshape(-1, *chunked_array.shape[2:])

    if remainder_array is not None:
        array = jnp.concatenate((remainder_array, array), axis=0)

    return array


def enmesh(i1, d1, a1, s1, b12, a2, s2, grad):
    r"""From coordinates and displacements on a grid, compute multilinear mesh indices
    and fractions on another grid.

    Parameters
    ----------
    i1 : (num, dim) ArrayLike
        Integer coordinates of points on grid 1.
    d1 : (num, dim) ArrayLike
        Float displacements from the points on grid 1.
    a1 : float
        Cell size of grid 1.
    s1 : dim-tuple of int, or None
        Periodic boundary shape of grid 1. If None, no wrapping.
    b12 : ArrayLike
        Offset of origin of grid 2 to that of grid 1.
    a2 : float or None
        Cell size of grid 2. If None, ``a2`` is the same as ``a1``.
    s2 : dim-tuple of int, or None
        Shape of grid 2. If not None, negative out-of-bounds indices of ``i2`` are set
        to ``s2``, avoiding some of them being treated as in bounds, thus allowing them
        to be dropped by ``add()`` and ``get()`` of ``jax.Array.at``.
    grad : bool
        Whether to return gradients of ``f2``.

    Returns
    -------
    i2 : (num, 2**dim, dim) jax.Array
        Mesh indices on grid 2.
    f2 : (num, 2**dim) jax.Array
        Multilinear fractions on grid 2.
    f2_grad : (num, 2**dim, dim) jax.Array
        Multilinear fraction gradients on grid 2.

    Notes
    -----
    Consider position :math:`\bm{P}` along the j-th axis

    .. math::

        P_j = (i_{1j} + n_{1j} s_{1j}) a_1 + d_{1j}
            = b_{12j} + i_{2j} a_2 + d_{2j}

    where :math:`\bm{n}_1` indexes all periodic images if :math:`\bm{s}_1` is given. The
    goal is to find all pairs of :math:`\bm{i}_2` and :math:`\bm{d}_2` that satisfy
    :math:`0 \leq i_{2j} < s_{2j}` (:math:`0 \leq i_{2j} < \lceil s_{1j} * a_1 / a_2
    \rceil` if periodic), and :math:`-a_2 \leq d_{2j} < a_2`, and then compute
    multilinear fractions from :math:`\bm{d}_2`.

    """
    i1 = jnp.asarray(i1)
    d1 = jnp.asarray(d1)
    a1 = jnp.float64(a1) if a2 is not None else jnp.array(a1, dtype=d1.dtype)
    if s1 is not None:
        s1 = jnp.array(s1, dtype=i1.dtype)
    b12 = jnp.float64(b12)
    if a2 is not None:
        a2 = jnp.float64(a2)
    if s2 is not None:
        s2 = jnp.array(s2, dtype=i1.dtype)

    dim = i1.shape[1]
    neighbors = (jnp.arange(2**dim, dtype=i1.dtype)[:, jnp.newaxis]
                 >> jnp.arange(dim, dtype=i1.dtype)
                ) & 1

    if a2 is not None:
        P = i1 * a1 + d1 - b12
        P = P[:, jnp.newaxis]  # insert neighbor axis
        i2 = P + neighbors * a2  # multilinear

        if s1 is not None:
            L = s1 * a1
            i2 %= L

        i2 //= a2
        d2 = P - i2 * a2

        if s1 is not None:
            d2 -= jnp.rint(d2 / L) * L  # also abs(d2) < a2 is expected

        i2 = i2.astype(i1.dtype)
        d2 = d2.astype(d1.dtype)
        a2 = a2.astype(d1.dtype)

        d2 /= a2
    else:
        i12, d12 = jnp.divmod(b12, a1)
        i1 -= i12.astype(i1.dtype)
        d1 -= d12.astype(d1.dtype)

        # insert neighbor axis
        i1 = i1[:, jnp.newaxis]
        d1 = d1[:, jnp.newaxis]

        # multilinear
        d1 /= a1
        i2 = jnp.floor(d1).astype(i1.dtype)
        i2 += neighbors
        d2 = d1 - i2
        i2 += i1

        if s1 is not None:
            i2 %= s1

    f2 = 1 - jnp.abs(d2)

    if s1 is None and s2 is not None:  # all i2 >= 0 if s1 is not None
        i2 = jnp.where(i2 < 0, s2, i2)

    if grad:
        sign = jnp.sign(-d2)
        f2g = []
        for i in range(dim):
            not_i = tuple(range(i + 1, dim)) + tuple(range(0, i))
            f2g.append(sign[..., i] * f2[..., not_i].prod(axis=-1))
        f2g = jnp.stack(f2g, axis=-1)
        f2 = f2.prod(axis=-1)

        return i2, f2, f2g
    else:
        f2 = f2.prod(axis=-1)

        return i2, f2


def fftfreq(shape, spacing, dtype=jnp.float64, sparse=True):
    """(Angular) wavevectors for FFT.

    Parameters
    ----------
    shape : tuple of int
        Shape of the real field.
    spacing : float or None, optional
        Grid spacing. None is equivalent to spacing of 2π with angular wavevector period
        of 1, or equivalently spacing of 1 with (non-angular) wavevector period of 1.
    dtype : DTypeLike
    sparse : bool, optional
        Whether to return sparse broadcastable or dense wavevector grids.

    Returns
    -------
    kvec : list of jax.Array
        Wavevectors.

    Notes
    -----

    The angular wavevectors differ from the numpy ``fftfreq`` and ``rfftfreq`` by a
    multiplicative factor of 2π.

    """
    period = 1
    if spacing is not None:
        period = 2 * jnp.pi / spacing

    kvec = []
    for axis, s in enumerate(shape[:-1]):
        k = jnp.fft.fftfreq(s) * period
        kvec.append(k.astype(dtype))

    k = jnp.fft.rfftfreq(shape[-1]) * period
    kvec.append(k.astype(dtype))

    kvec = jnp.meshgrid(*kvec, sparse=sparse, indexing='ij')

    return kvec


def rfftnfreq(shape, spacing, dtype=jnp.float64):
    import warnings
    warnings.warn('Deprecated, name changed to ``fftfreq`` for API consistency.',
                  DeprecationWarning)  # TODO
    return fftfreq(shape, spacing, dtype)


def fft(f, shape=None, axes=None, norm=None):
    """FFT between real and Hermitian complex fields, calling ``fftfwd`` or ``fftinv``.

    Parameters
    ----------
    f : ArrayLike
        Input field. See ``a`` in ``numpy.fft.rfftn`` and ``numpy.fft.irfftn``.
    shape : sequence of int, optional
        See ``s`` in ``numpy.fft.rfftn`` and ``numpy.fft.irfftn``.
    axes : sequence of int, optional
        See ``numpy.fft.rfftn`` and ``numpy.fft.irfftn``.
    norm : float or {'backward', 'ortho', 'forward'}, optional
        Grid spacing for normalization if float, otherwise passed to
        ``jax.numpy.fft.rfftn`` or ``jax.numpy.fft.irfftn``.

    Returns
    -------
    f : jax.Array
        Output field. See ``out`` in ``numpy.fft.rfftn`` and ``numpy.fft.irfftn``.

    """
    if jnp.isrealobj(f):
        return fftfwd(f, shape, axes, norm)

    return fftinv(f, shape, axes, norm)


def fftfwd(f, shape=None, axes=None, norm=None):
    r"""Forward FFT from real to Hermitian complex fields, wrapping
    ``jax.numpy.fft.rfftn``.

    Parameters
    ----------
    f : ArrayLike
        Input field. See ``a`` in ``numpy.fft.rfftn``.
    shape : sequence of int, optional
        See ``s`` in ``numpy.fft.rfftn``.
    axes : sequence of int, optional
        See ``numpy.fft.rfftn``.
    norm : float or {'backward', 'ortho', 'forward'}, optional
        Grid spacing for normalization if float, otherwise passed to
        ``jax.numpy.fft.rfftn``.

    Returns
    -------
    f : jax.Array
        Output field. See ``out`` in ``numpy.fft.rfftn``.

    Raises
    ------
    ValueError
        If input field is not real.

    Notes
    -----
    Given the grid spacing, the normalization convention is

    .. math::

        f(\bm{k}) = \int \mathrm{d}\bm{x} f(\bm(x}) e^{-i \bm{k} \cdot \bm{x}}
                    \approx \frac{V}{N} \sum_\bm{x} f(\bm{x}) e^{-i \bm{k} \cdot \bm{x}}

    where :math:`V/N` is the cell volume, :math:`V` is the box volume, :math:`N` is the
    number of grid points/cells to be summed over.

    """
    f = jnp.asarray(f)

    if not jnp.isrealobj(f):
        raise ValueError('input field must be real')

    if norm in {None, 'backward', 'ortho', 'forward'}:
        return jnp.fft.rfftn(f, s=shape, axes=axes, norm=norm)

    d = f.ndim
    if shape is not None:  # len(shape) == len(axes) if both are not None
        d = len(shape)
    if axes is not None:
        d = len(axes)

    return norm**d * jnp.fft.rfftn(f, s=shape, axes=axes, norm='backward')


def fftinv(f, shape=None, axes=None, norm=None):
    r"""Inverse FFT from Hermitian completx to real fields, wrapping
    ``jax.numpy.fft.irfftn``.

    Parameters
    ----------
    f : ArrayLike
        Input field. See ``a`` in ``numpy.fft.irfftn``.
    shape : sequence of int, optional
        See ``s`` in ``numpy.fft.irfftn``.
    axes : sequence of int, optional
        See ``numpy.fft.irfftn``.
    norm : float or {'backward', 'ortho', 'forward'}, optional
        Grid spacing for normalization if float, otherwise passed to
        ``jax.numpy.fft.irfftn``.

    Returns
    -------
    f : jax.Array
        Output field. See ``out`` in ``numpy.fft.irfftn``.

    Raises
    ------
    ValueError
        If input field is not complex.

    Notes
    -----
    Given the grid spacing, the normalization convention is

    .. math::

        f(\bm{x}) = \int \frac{\mathrm{d}\bm{k}}{(2\pi)^d} f(\bm(k}) e^{i \bm{k} \cdot \bm{x}}
                    \approx \frac{1}{V} \sum_\bm{k} f(\bm{k}) e^{i \bm{k} \cdot \bm{x}}

    where :math:`d` is the FFT dimension, :math:`V` is the box volume.

    """
    f = jnp.asarray(f)

    if not jnp.iscomplexobj(f):
        raise ValueError('input field must be Hermitian complex')

    if norm in {None, 'backward', 'ortho', 'forward'}:
        return jnp.fft.irfftn(f, s=shape, axes=axes, norm=norm)

    d = f.ndim
    if shape is not None:  # len(shape) == len(axes) if both are not None
        d = len(shape)
    if axes is not None:
        d = len(axes)

    return norm**-d * jnp.fft.irfftn(f, s=shape, axes=axes, norm='backward')
