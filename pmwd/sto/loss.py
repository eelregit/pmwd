import jax
import jax.numpy as jnp


from pmwd.particles import Particles, ptcl_rpos
from pmwd.pm_util import rfftnfreq
from pmwd.sto.util import ptcl2dens, power_tfcc


def _loss_dens_mse(dens, dens_t):
    return jnp.mean((dens - dens_t)**2)


def _loss_disp_mse(ptcl, ptcl_t):
    return jnp.mean((ptcl.disp - ptcl_t.disp)**2)


def _loss_vel_mse(ptcl, ptcl_t):
    return jnp.mean((ptcl.vel - ptcl_t.vel)**2)


@jax.custom_vjp
def _loss_scale_wmse(kvec, f, g):
    # mse of two fields in Fourier space, uniform weights
    k2 = sum(k**2 for k in kvec)
    d = f - g
    loss = jnp.sum(jnp.where(k2 != 0, jnp.abs(d)**2 / k2**1.5, 0)
                   ) / jnp.array(d.shape).prod()
    return jnp.log(loss), (loss, k2, d)

def _scale_wmse_fwd(kvec, f, g):
    loss, res = _loss_scale_wmse(kvec, f, g)
    return loss, res

def _scale_wmse_bwd(res, loss_cot):
    loss, k2, d = res
    d_shape = d.shape
    abs_valgrad = jax.value_and_grad(jnp.abs)
    d, d_grad = jax.vmap(abs_valgrad)(d.ravel())
    d = d.reshape(d_shape)
    d_grad = d_grad.reshape(d_shape)

    loss_cot /= loss
    f_cot = loss_cot * jnp.where(k2 != 0, 2 * d * d_grad / k2**1.5, 0
                                 ) / jnp.array(d_shape).prod()
    return None, f_cot, None

_loss_scale_wmse.defvjp(_scale_wmse_fwd, _scale_wmse_bwd)


def _loss_tfcc(dens, dens_t, cell_size, wtf=1):
    k, tf, cc = power_tfcc(dens, dens_t, cell_size)
    return wtf * jnp.sum((1 - tf)**2) + jnp.sum((1 - cc)**2)


def _loss_Lanzieri():
    """The loss defined by Eq.(4) in 2207.05509v2 (Lanzieri2022)."""
    pass


def loss_func(ptcl, tgt, conf, mesh_shape=3):

    # get the target ptcl
    pos_t, vel_t = tgt
    disp_t = pos_t - ptcl.pmid * conf.cell_size
    ptcl_t = Particles(conf, ptcl.pmid, disp_t, vel_t)

    # get the density fields for the loss
    (dens, dens_t), (mesh_shape, cell_size) = ptcl2dens(
                                               (ptcl, ptcl_t), conf, mesh_shape)
    dens_k = jnp.fft.rfftn(dens)
    dens_t_k = jnp.fft.rfftn(dens_t)
    kvec_dens = rfftnfreq(mesh_shape, cell_size, dtype=conf.float_dtype)

    # get the disp from particles' grid Lagrangian positions
    disp, disp_t = (ptcl_rpos(p, Particles.gen_grid(p.conf), p.conf)
                    for p in (ptcl, ptcl_t))
    shape_ = (-1,) + conf.ptcl_grid_shape
    disp_k = jnp.fft.rfftn(disp.T.reshape(shape_), axes=range(-3, 0))
    disp_t_k = jnp.fft.rfftn(disp_t.T.reshape(shape_), axes=range(-3, 0))
    kvec_disp = rfftnfreq(conf.ptcl_grid_shape, conf.ptcl_spacing, dtype=conf.float_dtype)

    loss = 0.
    # loss += _loss_scale_wmse(kvec_dens, dens_k, dens_t_k)
    # loss += _loss_scale_wmse(kvec_disp, disp_k, disp_t_k)
    # loss += _loss_dens_mse(dens, dens_t)
    # loss += _loss_disp_mse(ptcl, ptcl_t)
    # loss += _loss_vel_mse(ptcl, ptcl_t)
    loss += _loss_tfcc(dens, dens_t, cell_size)

    return loss
