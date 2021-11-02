from functools import partial

import numpy as np
import jax.numpy as jnp
import jax.test_util as jtu
from jax import vjp, float0
from jax import random
from jax.tree_util import tree_map

from .pm import Particles


def gen_pmid(ptcl_grid_shape, dtype='i8'):
    ndim = len(ptcl_grid_shape)
    pmid = jnp.meshgrid(*[jnp.arange(s, dtype=dtype) for s in ptcl_grid_shape],
                        indexing='ij')
    pmid = jnp.stack(pmid, axis=-1).reshape(-1, ndim)
    return pmid

def gen_disp(ptcl_grid_shape, std, dtype='f8'):
    key = random.PRNGKey(0)
    ndim = len(ptcl_grid_shape)
    disp = std * random.normal(key, ptcl_grid_shape + (ndim,),
                                    dtype=dtype)
    disp = disp.reshape(-1, ndim)
    return disp

def gen_val(ptcl_grid_shape, chan_shape, mean, std, dtype='f8'):
    key = random.PRNGKey(0)
    val = mean + std * random.normal(key, ptcl_grid_shape + chan_shape,
                                     dtype=dtype)
    val = val.reshape(-1, *chan_shape)
    return val

def gen_ptcl(ptcl_grid_shape, disp_std, vel_ratio=None, acc_ratio=None,
             chan_shape=None, val_mean=1., val_std=0.,
             int_dtype='i8', real_dtype='f8'):
    pmid = gen_pmid(ptcl_grid_shape, dtype=int_dtype)
    disp = gen_disp(ptcl_grid_shape, disp_std, dtype=real_dtype)

    vel = None
    if vel_ratio is not None:
        vel = vel_ratio * disp

    acc = None
    if acc_ratio is not None:
        acc = acc_ratio * disp

    val = None
    if chan_shape is not None:
        val = gen_val(ptcl_grid_shape, chan_shape, val_mean, val_std,
                      dtype=real_dtype)

    ptcl = Particles(pmid, disp, vel=vel, acc=acc, val=val)

    return ptcl


def gen_mesh(shape, mean=0., std=1.):
    key = random.PRNGKey(0)
    mesh = mean + std * random.normal(key, shape)
    return mesh


def randn_float0_like(x, mean=0., std=1., key=random.PRNGKey(0)):
    if issubclass(x.dtype.type, (jnp.bool_, jnp.integer)):
        # FIXME after https://github.com/google/jax/issues/4433 is addressed
        return np.empty(x.shape, dtype=float0)
    return mean + std * random.normal(key, shape=x.shape, dtype=x.dtype)


def tree_randn_float0_like(tree, mean=None, std=None):
    if mean is None:
        mean = tree_map(lambda x: 0., tree)
    if std is None:
        std = tree_map(lambda x: 1., tree)
    return tree_map(randn_float0_like, tree, mean, std)


def check_custom_vjp(fun, primals, cot_out_mean=None, cot_out_std=None,
                     args=(), kwargs={}, atol=None, rtol=None):
    """Compare custom and automatic vjp's of a decorated function.

    Parameters:
        fun: function decorated with `custom_vjp`
        primals: function inputs whose cotangent vectors are to be compared
        cot_out_mean: mean of randn output cotangents. Default is a pytree of 0
        cot_out_std: std of randn output cotangents. Default is a pytree of 1
        args: positional inputs to be fixed by `partial`
        kwargs: keyword inputs to be fixed by `partial`
        atol: absolute tolerance
        rtol: relative tolerance

    Note:
        Setting leaves of both `cot_out_mean` and `cot_out_std` pytrees to
        zeros can disable the corresponding output cotangents.
    """
    fun_orig = fun.__wrapped__  # original function without custom vjp

    if args or kwargs:
        fun = partial(fun, *args, **kwargs)
        fun_orig = partial(fun_orig, *args, **kwargs)

    primals_out, vjpfun = vjp(fun, *primals)
    primals_out_orig, vjpfun_orig = vjp(fun_orig, *primals)
    jtu.check_close(primals_out, primals_out_orig, atol=atol, rtol=rtol)

    cot_out = tree_randn_float0_like(primals_out, cot_out_mean, cot_out_std)

    cot = vjpfun(cot_out)
    cot_orig = vjpfun_orig(cot_out)
    jtu.check_close(cot, cot_orig, atol=atol, rtol=rtol)
