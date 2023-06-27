import jax.numpy as jnp
from jax import custom_vjp

from pmwd.scatter import scatter
from pmwd.gather import gather
from pmwd.pm_util import fftfreq, fftfwd, fftinv
from pmwd.sto.so import sotheta, pot_sharp, grad_sharp


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
    """Gravitational accelerations of particles in [H_0^2], solved on a mesh with FFT."""
    kvec = fftfreq(conf.mesh_shape, conf.cell_size, dtype=conf.float_dtype)

    dens = scatter(ptcl, conf)
    dens -= 1  # overdensity

    dens *= 1.5 * cosmo.Omega_m.astype(conf.float_dtype)

    dens = fftfwd(dens)  # normalization canceled by that of irfftn below

    pot = laplace(kvec, dens, cosmo)

    if conf.so_type is not None:  # spatial optimization
        theta = sotheta(cosmo, conf, a)
        pot = pot_sharp(pot, kvec, theta, cosmo, conf, a)

    acc = []
    for k in kvec:
        grad = neg_grad(k, pot, conf.cell_size)

        if conf.so_type is not None:  # spatial optimization
            grad = grad_sharp(grad, k, theta, cosmo, conf, a)

        grad = fftinv(grad, shape=conf.mesh_shape)
        grad = grad.astype(conf.float_dtype)  # no jnp.complex32

        grad = gather(ptcl, conf, grad)

        acc.append(grad)
    acc = jnp.stack(acc, axis=-1)

    return acc
