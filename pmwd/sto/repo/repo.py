@jax.custom_vjp
def _loss_scale_wmse(kvec, f, g, k2pow):
    # mse of two fields in Fourier space, uniform weights
    k2 = sum(k**2 for k in kvec)
    d = f - g
    loss = jnp.sum(jnp.where(k2 != 0, jnp.abs(d)**2 / k2**k2pow, 0)
                   ) / jnp.array(d.shape).prod()
    return jnp.log(loss), (loss, k2, d, k2pow)

def _scale_wmse_fwd(kvec, f, g, k2pow):
    loss, res = _loss_scale_wmse(kvec, f, g, k2pow)
    return loss, res

def _scale_wmse_bwd(res, loss_cot):
    loss, k2, d, k2pow = res
    d_shape = d.shape
    abs_valgrad = jax.value_and_grad(jnp.abs)
    d, d_grad = jax.vmap(abs_valgrad)(d.ravel())
    d = d.reshape(d_shape)
    d_grad = d_grad.reshape(d_shape)

    loss_cot /= loss
    f_cot = loss_cot * jnp.where(k2 != 0, 2 * d * d_grad / k2**k2pow, 0
                                 ) / jnp.array(d_shape).prod()
    return None, f_cot, None, None

_loss_scale_wmse.defvjp(_scale_wmse_fwd, _scale_wmse_bwd)


def _loss_tfcc(dens, dens_t, cell_size, wtf=1):
    k, tf, cc = power_tfcc(dens, dens_t, cell_size)
    return wtf * jnp.sum((1 - tf)**2) + jnp.sum((1 - cc)**2)


def _loss_Lanzieri(disp, disp_t, dens, dens_t, cell_size):
    """The loss defined by Eq.(4) in 2207.05509v2 (Lanzieri2022)."""
    loss = jnp.sum((disp - disp_t)**2)
    k, ps, N = powspec(dens, cell_size)
    k, ps_t, N = powspec(dens_t, cell_size)
    loss += 0.1 * jnp.sum((ps / ps_t - 1)**2)
    return loss


def _loss_abs_power(f, g, spacing=1, cc_pow=1):
    k, tf, cc = power_tfcc(f, g, spacing)
    return (jnp.abs(1 - tf) + (1 - cc)**cc_pow).sum()