import jax
import jax.numpy as jnp

from pmwd.background import distance
from pmwd.pm_util import _chunk_split, _chunk_cat
from pmwd.lens.lenses import potential, deflect
from pmwd.lens.sources import profile
from pmwd.lens.util import _canonicalize_ang_pos, _canonicalize_array


# Question: how to change the strong lensing lens eq when combined with weak lensing?


def displace(x, lens, src, cosmo):
    """Displace ray angular positions through the lenses to the sources.

    Parameters
    ----------
    x : ArrayLike of shape (..., 2)
        Original ray angular positions in :math:`A`.
    lens : Lenses
    src : Sources
    cosmo : Cosmology

    Returns
    -------
    x : jax.Array of the same dtype and shape (..., num_sources, 2)
        Displaced ray angular positions at the source redshifts in :math:`A`.

    Notes
    -----

    .. math::

        TODO fully general case

    """
    # HACK, also distance(..., a_ref=lens.a[0]) below
    #if lens.a.size != 1:
    #    #raise ValueError('multiple lens planes not supported')
    #    raise NotImplementedError  #FIXME fully general case, see NOTES.rst

    x = _canonicalize_ang_pos(x)
    lens, src, cosmo = lens.astype(x.dtype), src.astype(x.dtype), cosmo.astype(x.dtype)

    alpha = deflect(x, lens)  # \hat\alpha
    alpha = alpha[..., jnp.newaxis, :]  # shape = (..., 1, 2)
    a_src = src.a[:, jnp.newaxis]  # shape = (num_sources, 1)
    alpha *= (distance(a_src, cosmo, type='transverse', a_ref=lens.a[0])
              / distance(a_src, cosmo, type='transverse'))

    return x[..., jnp.newaxis, :] - alpha  # shape = (..., num_sources, 2)


def delay(x, x_ref, lens, src, cosmo):
    """Time delays for sources at infinity, in days???

    Parameters
    ----------
    x : ArrayLike of shape (..., 2)
        Ray angular positions in :math:`A`.
    x_ref : ArrayLike of shape (num_sources, 2) or (2,)
        Reference ray angular positions, common or per source, in :math:`A`.
    lens : Lenses
    src : Sources
    cosmo : Cosmology

    """
    #x = _canonicalize_ang_pos(x)
    #x_ref = _canonicalize_ang_pos(x_ref)
    raise NotImplementedError


def ray_trace(x, lens, src, cosmo, chunk_size=2**24):
    r"""Ray trace through the lenses to the sources.

    Parameters
    ----------
    x : ArrayLike of shape (..., 2)
        Ray angular positions in :math:`A`.
    lens : Lenses
    src : Sources
    cosmo : Cosmology
    chunk_size : int, optional
        Chunk size to split rays in batches, to save memory when backpropagating the
        lensing potential gradients.

    Returns
    -------
    I : ArrayLike of x.dtype and shape (...,)
        Lensed image of source surface brightnesses, per :math:`A^2`.

    """
    x = _canonicalize_ang_pos(x)

    shape = x.shape[:-1]
    x = x.reshape(-1, 2)
    remainder, (x,) = _chunk_split(x.shape[0], chunk_size, x)

    carry = lens, src, cosmo
    I0 = None
    if remainder is not None:
        x0 = remainder[0]
        I0 = _ray_trace_chunk_ckpt(carry, x0)[1]
    I = jax.lax.scan(_ray_trace_chunk_ckpt, carry, x)[1]

    I = _chunk_cat(I0, I)
    I = I.reshape(shape)

    return _canonicalize_array(I, dtype=x.dtype)


def _ray_trace_chunk(carry, x):
    lens, src, cosmo = carry
    x = displace(x, lens, src, cosmo)
    return carry, profile(x, src)

#TODO test this mem cost:
_ray_trace_chunk_ckpt = _ray_trace_chunk
#TODO test the CSE flag
#_ray_trace_chunk_ckpt = jax.checkpoint(_ray_trace_chunk, prevent_cse=False)
