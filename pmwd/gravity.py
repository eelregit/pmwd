import jax.numpy as jnp
from jax import custom_vjp

from pmwd.scatter import scatter
from pmwd.gather import gather


def rfftnfreq(shape, spacing, dtype=float):
    """Broadcastable "``sparse``" wavevectors for ``numpy.fft.rfftn``.

    Parameters
    ----------
    shape : tuple of int
        Shape of ``rfftn`` input.
    spacing : float
        Grid spacing in [L].
    dtype : dtype_like

    Returns
    -------
    kvec : list of jax.numpy.ndarray
        Wavevectors.

    """
    freq_period = 2. * jnp.pi / spacing

    kvec = []
    for axis, s in enumerate(shape[:-1]):
        k = jnp.fft.fftfreq(s).astype(dtype) * freq_period
        kvec.append(k)

    k = jnp.fft.rfftfreq(shape[-1]).astype(dtype) * freq_period
    kvec.append(k)

    kvec = jnp.meshgrid(*kvec, indexing='ij', sparse=True)

    return kvec


@custom_vjp
def laplace(kvec, src, cosmo=None):
    """Laplace kernel in Fourier space."""
    k2 = sum(k**2 for k in kvec)

    pot = jnp.where(k2 != 0, - src / k2, 0)

    return pot


def laplace_fwd(kvec, src, cosmo):
    pot = laplace(kvec, src, cosmo)
    return pot, (kvec, cosmo)

def laplace_bwd(res, pot_cot):
    """Custom vjp to avoid NaN when using where, as well as to save memory.

    .. _JAX FAQ:
        https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where

    """
    kvec, cosmo = res
    src_cot = laplace(kvec, pot_cot, cosmo)
    return None, src_cot, None

laplace.defvjp(laplace_fwd, laplace_bwd)


def neg_grad(k, pot, spacing):
    nyquist = jnp.pi / spacing
    eps = nyquist * jnp.finfo(k.dtype).eps
    neg_ik = jnp.where(jnp.abs(jnp.abs(k) - nyquist) <= eps, 0, -1j * k)

    grad = neg_ik * pot

    return grad


def gravity(a, ptcl, cosmo, conf):
    """Particles' gravitational accelerations in [H_0^2], solved on a mesh with FFT."""
    kvec = rfftnfreq(conf.mesh_shape, conf.cell_size, dtype=conf.float_dtype)

    dens = jnp.zeros(conf.mesh_shape, dtype=conf.float_dtype)

    inv_dens_mean = conf.mesh_size / conf.ptcl_num
    dens = scatter(ptcl, dens, inv_dens_mean, conf.cell_size, chunk_size=conf.chunk_size)
    dens -= 1  # overdensity

    dens *= 1.5 * cosmo.Omega_m.astype(conf.float_dtype)

    dens = jnp.fft.rfftn(dens)  # normalization canceled by that of irfftn below

    pot = laplace(kvec, dens, cosmo)

    acc = []
    for k in kvec:
        grad = neg_grad(k, pot, conf.cell_size)

        grad = jnp.fft.irfftn(grad, s=conf.mesh_shape)
        grad = grad.astype(conf.float_dtype)  # no jnp.complex32

        grad = gather(ptcl, grad, 0, conf.cell_size, chunk_size=conf.chunk_size)

        acc.append(grad)
    acc = jnp.stack(acc, axis=-1)

    return acc
