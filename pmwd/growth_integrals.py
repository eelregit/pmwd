from functools import partial

import jax
from jax import jit, custom_vjp, ensure_compile_time_eval, grad, vmap
import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax.lax import scan

import sys, os
from flax import linen as nn
from flax.training import checkpoints
from typing import Sequence

from pmwd.cosmology import H_deriv, Omega_m_a


#
def odeint_rk4(fun, y0, t, *args):

    def rk4(carry, t):
        y, t_prev = carry
        h = t - t_prev
        k1 = fun(y, t_prev, *args)
        k2 = fun(y + h * k1 / 2, t_prev + h / 2, *args)
        k3 = fun(y + h * k2 / 2, t_prev + h / 2, *args)
        k4 = fun(y + h * k3, t, *args)
        y = y + 1.0 / 6.0 * h * (k1 + 2 * k2 + 2 * k3 + k4)
        return (y, t), y
    (yf, _), y = scan(rk4, (y0, t[0]), t)
    return y


#
class Simple_MLP(nn.Module):
    """MLP architecture for the growth function emulators
    """
    features:Sequence[int]
    nodes:int

    @nn.compact
    def __call__(self, inputs):
        x=inputs
        for feat in self.features[:-2]:
            x=nn.Dense(feat)(x)
            x=nn.elu(x)
        t=nn.Dense(self.nodes-2)(nn.elu(nn.Dense(self.features[-2])(x)))
        c=nn.Dense(self.nodes+1)(nn.elu(nn.Dense(self.features[-1])(x)))
        t=jnp.concatenate([jnp.zeros((t.shape[0], 4)), jnp.cumsum(jax.nn.softmax(t), axis=1), jnp.ones((t.shape[0], 4))], axis=1)
        c=jnp.concatenate([jnp.zeros((c.shape[0], 1)), c], axis=1)
        return t, c

    
@jit
def _deBoorVectorized(x,t,c):
    p=3
    k=jnp.digitize(x,t)-1
    d=[c[j+k-p] for j in range(0,p+1)]
    for r in range(1,p+1):
        for j in range(p,r-1,-1):
            alpha=(x-t[j+k-p])/(t[j+1+k-r]-t[j+k-p])
            d[j]=(1.0-alpha)*d[j-1]+alpha*d[j]
    return d[p]
deBoor = vmap(_deBoorVectorized,in_axes=(None,0,0))



class Growth_MLP():
    """MLP for growth function and its derivatives
    """
    def __init__(self):
        self.build()

    def build(self):
        layer_sizes = [64,64,64]
        nodes = 8
        model = Simple_MLP(features=layer_sizes,nodes=nodes)
        params = {}
        dirname = os.path.dirname(__file__)
        for order in range(1, 3):
            for deriv in range(3):
                key = "{}{}".format(order, deriv)
                params[key] = checkpoints.restore_checkpoint(ckpt_dir=dirname + "/nets/d%dd%dcheckpoint_0"%(order, deriv),target=None)['params']
        self.model, self.params = model, params
                
        
    @partial(jit, static_argnums=(0,))
    def _growth(self, cosmo, a, params, order):

        reshape = False
        if len(cosmo.shape) == 1: 
            reshape = True
            cosmo = jnp.reshape(cosmo, (1, -1))
        t,c = self.model.apply(params, cosmo)
        g = deBoor(jnp.clip(a,0,0.99999),t,c)
        g = g * a**(order-1)
        if reshape: 
            return g[0]
        else: 
            return g

               
    def __call__(self, cosmo, a, order=1, deriv=0):

        key = "{}{}".format(order, deriv)
        params = self.params[key]
        return self._growth(cosmo, a, params, order)



@jit
def growth_integ(cosmo, conf):
    """Intergrate with jax odeint (Dopri5) and tabulate (LPT) growth functions and derivatives at given scale
    factors.

    Parameters
    ----------
    cosmo : Cosmology
    conf : Configuration

    Returns
    -------
    cosmo : Cosmology
        A new instance containing a growth table, or the input one if it already exists.
        The growth table has the shape ``(num_lpt_order, num_derivatives,
        num_scale_factors)`` and ``conf.cosmo_dtype``.

    Notes
    -----

    TODO: ODE math

    """
    with ensure_compile_time_eval():
        eps = jnp.finfo(conf.cosmo_dtype).eps
        a_ic = 0.5 * jnp.cbrt(eps).item()  # ~ 3e-6 for float64, 2e-3 for float32
        if a_ic >= conf.a_lpt_step:
            a_ic = 0.1 * conf.a_lpt_step

    a = conf.growth_a
    lna = jnp.log(a.at[0].set(a_ic))

    num_order, num_deriv, num_a = 2, 3, len(a)

    # TODO necessary to add lpt_order support?
    # G and lna can either be at a single time, or have leading time axes
    def ode(G, lna, cosmo):
        a = jnp.exp(lna)
        dlnH_dlna = H_deriv(a, cosmo)
        Omega_fac = 1.5 * Omega_m_a(a, cosmo)
        G1, G1p, G2, G2p = jnp.split(G, num_order * (num_deriv-1), axis=-1)
        G1pp = -(3 + dlnH_dlna - Omega_fac) * G1 - (4 + dlnH_dlna) * G1p
        G2pp = Omega_fac * G1**2 - (8 + 2*dlnH_dlna - Omega_fac) * G2 - (6 + dlnH_dlna) * G2p
        return jnp.concatenate((G1p, G1pp, G2p, G2pp), axis=-1)

    G_ic = jnp.array((1, 0, 3/7, 0), dtype=conf.cosmo_dtype)

    G = odeint(ode, G_ic, lna, cosmo, rtol=conf.growth_rtol, atol=conf.growth_atol)

    G_deriv = ode(G, lna[:, jnp.newaxis], cosmo)

    G = G.reshape(num_a, num_order, num_deriv-1)
    G_deriv = G_deriv.reshape(num_a, num_order, num_deriv-1)
    G = jnp.concatenate((G, G_deriv[..., -1:]), axis=2)
    G = jnp.moveaxis(G, 0, 2)

    # D_m /a^m = G
    # D_m'/a^m = m G + G'
    # D_m"/a^m = m^2 G + 2m G' + G"
    m = jnp.array((1, 2), dtype=conf.cosmo_dtype)[:, jnp.newaxis]
    growth = jnp.stack((
        G[:, 0],
        m * G[:, 0] + G[:, 1],
        m**2 * G[:, 0] + 2 * m * G[:, 1] + G[:, 2],
    ), axis=1)

    return cosmo.replace(growth=growth)



def growth_integ_rk4(cosmo, conf, growth_a=None):
    """Intergrate with Runge-Kutta 4 and tabulate (LPT) growth functions and derivatives at given scale
    factors.

    Parameters
    ----------
    cosmo : Cosmology

    Returns
    -------
    cosmo : Cosmology
        A new instance containing a growth table, or the input one if it already exists.
        The growth table has the shape ``(num_lpt_order, num_derivatives,
        num_scale_factors)`` and ``cosmo.conf.cosmo_dtype``.

    Notes
    -----

    TODO: ODE math

    """
    if cosmo.growth is not None:
        return cosmo

    conf = cosmo.conf

    with ensure_compile_time_eval():
        eps = jnp.finfo(conf.cosmo_dtype).eps
        growth_a_ic = 0.5 * jnp.cbrt(eps).item()  # ~ 3e-6 for float64, 2e-3 for float32
        if growth_a_ic >= conf.a_lpt_step:
            growth_a_ic = 0.1 * conf.a_lpt_step

    a = conf.growth_a
    lna = jnp.log(a.at[0].set(growth_a_ic))

    num_order, num_deriv, num_a = 2, 3, len(a)

    # TODO necessary to add lpt_order support?
    # G and lna can either be at a single time, or have leading time axes
    def ode(G, lna, cosmo):
        a = jnp.exp(lna)
        H_fac = H_deriv(a, cosmo)
        Omega_fac = 1.5 * Omega_m_a(a, cosmo)
        G_1, Gp_1, G_2, Gp_2 = jnp.split(G, num_order * (num_deriv-1), axis=-1)
        Gpp_1 = (-3 - H_fac + Omega_fac) * G_1 + (-4 - H_fac) * Gp_1
        Gpp_2 = Omega_fac * G_1**2 + (-8 - 2*H_fac + Omega_fac) * G_2 + (-6 - H_fac) * Gp_2
        return jnp.concatenate((Gp_1, Gpp_1, Gp_2, Gpp_2), axis=-1)

    G_ic = jnp.array((1, 0, 3/7, 0), dtype=conf.cosmo_dtype)

    #ode_jit = jit(ode)
    ode_jit = ode
    G = odeint_rk4(ode_jit, G_ic, lna, cosmo)

    G_deriv = ode(G, lna[:, jnp.newaxis], cosmo)

    G = G.reshape(num_a, num_order, num_deriv-1)
    G_deriv = G_deriv.reshape(num_a, num_order, num_deriv-1)
    G = jnp.concatenate((G, G_deriv[..., -1:]), axis=2)
    G = jnp.moveaxis(G, 0, 2)

    m = jnp.array((1, 2), dtype=conf.cosmo_dtype)[:, jnp.newaxis]
    growth = jnp.stack((
        G[:, 0],
        m * G[:, 0] + G[:, 1],
        m**2 * G[:, 0] + 2 * m * G[:, 1] + G[:, 2],
    ), axis=1)

    return cosmo.replace(growth=growth)



def growth_integ_mlp(cosmo, conf):
    """Use Growth MLP to tabulate (LPT) growth functions and derivatives at given scale
    factors.

    Parameters
    ----------
    cosmo : Cosmology

    Returns
    -------
    cosmo : Cosmology
        A new instance containing a growth table, or the input one if it already exists.
        The growth table has the shape ``(num_lpt_order, num_derivatives,
        num_scale_factors)`` and ``cosmo.conf.cosmo_dtype``.

    Notes
    -----

    TODO: ODE math

    """
    if cosmo.growth is not None:
        return cosmo

    conf = cosmo.conf
    growth_fn = Growth_MLP()
    
    with ensure_compile_time_eval():
        eps = jnp.finfo(conf.cosmo_dtype).eps
        growth_a_ic = 0.5 * jnp.cbrt(eps).item()  # ~ 3e-6 for float64, 2e-3 for float32
        if growth_a_ic >= conf.a_lpt_step:
            growth_a_ic = 0.1 * conf.a_lpt_step

    a = conf.growth_a
    lna = jnp.log(a.at[0].set(growth_a_ic))

    num_order, num_deriv, num_a = 2, 3, len(a)

    growth = []
    a = jnp.array(a)
    for order in [1,2]:
        growth.append(jnp.array([growth_fn(jnp.array([cosmo.Omega_m]), a, order, deriv) for deriv in range(3)]))
    
    growth = jnp.stack(growth, axis=0)

    return cosmo.replace(growth=growth)
