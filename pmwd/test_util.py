from functools import partial

import numpy as np
import jax.numpy as jnp
import jax.test_util as jtu
from jax import custom_vjp, vjp
from jax import random
from jax.tree_util import tree_map

from pmwd.particles import Particles
from pmwd.util import is_float0_array, float0_like


def gen_ptcl(conf, disp_std=None, vel_std=None, acc_std=None,
             attr_shape=None, attr_mean=1, attr_std=0, seed=0):
    ptcl = Particles.gen_grid(conf)

    key = random.PRNGKey(seed)
    disp_key, vel_key, acc_key, attr_key = random.split(key, num=4)

    vec_shape = (conf.ptcl_num, conf.dim)
    dtype = conf.float_dtype

    disp = ptcl.disp
    if disp_std is not None:
        disp += disp_std * random.normal(disp_key, shape=vec_shape, dtype=dtype)

    vel = None
    if vel_std is not None:
        vel = vel_std * random.normal(vel_key, shape=vec_shape, dtype=dtype)

    acc = None
    if acc_std is not None:
        acc = acc_std * random.normal(acc_key, shape=vec_shape, dtype=dtype)

    attr = None
    if attr_shape is not None:
        attr_shape = (conf.ptcl_num,) + attr_shape
        attr = attr_mean + attr_std * random.normal(attr_key, shape=attr_shape,
                                                    dtype=dtype)

    return ptcl.replace(disp=disp, vel=vel, acc=acc, attr=attr)


def gen_mesh(shape, dtype, mean=0, std=1, seed=0):
    if std == 0:
        return jnp.full(shape, mean, dtype=dtype)

    key = random.PRNGKey(seed)
    mesh = mean + std * random.normal(key, shape=shape, dtype=dtype)
    return mesh


def randn_float0_like(x, mean=0, std=1, seed=0):
    # see primal_dtype_to_tangent_dtype() from jax/core.py
    if not jnp.issubdtype(x.dtype, np.inexact):
        return float0_like(x)

    key = random.PRNGKey(seed)
    y = mean + std * random.normal(key, shape=x.shape, dtype=x.dtype)
    return y


def tree_randn_float0_like(tree, mean=None, std=None):
    if mean is None:
        mean = tree_map(lambda x: 0, tree)
    if std is None:
        std = tree_map(lambda x: 1, tree)
    return tree_map(randn_float0_like, tree, mean, std)


def _safe_sub(x, y):
    if is_float0_array(x) and is_float0_array(y):
        return x
    return x - y


def check_eq(xs, ys, err_msg=''):
    jtu.check_eq(xs, ys, err_msg=err_msg)
    return tree_map(_safe_sub, xs, ys)


def check_close(xs, ys, atol=None, rtol=None, err_msg=''):
    jtu.check_close(xs, ys, atol=atol, rtol=rtol, err_msg=err_msg)
    return tree_map(_safe_sub, xs, ys)


def check_custom_vjp(fun, primals, partial_args=(), partial_kwargs={},
                     cot_out_mean=None, cot_out_std=None,
                     atol=None, rtol=None):
    """Compare custom and automatic vjp's of a decorated function.

    Setting matching leaves of ``cot_out_mean`` and ``cot_out_std`` pytrees to zeros
    disables the corresponding output cotangents.

    Parameters
    ----------
    fun : callable
        Function decorated with ``custom_vjp``.
    primals : iterable
        Function inputs whose cotangent vectors are to be compared.
    partial_args : iterable
        Positional function inputs to be fixed by ``partial``.
    partial_kwargs : mapping
        Keyword function inputs to be fixed by ``partial``.
    cot_out_mean : pytree
        Means of normally distributed output cotangents. Default is a pytree of 0.
    cot_out_std : pytree
        Standard deviations of normally distributed output cotangents. Default is a
        pytree of 1.
    atol : float, optional
        Absolute tolerance.
    rtol : float, optional
        Relative tolerance.

    Returns
    -------
    cot : jax.numpy.ndarray
        Input cotangents by custom vjp.
    cot_orig : jax.numpy.ndarray
        Input cotangents by automatic vjp.
    cot_diff : jax.numpy.ndarray
        Input cotangent differences.

    Raises
    ------
    TypeError
        If ``fun`` has no ``custom_vjp``.

    """
    if not isinstance(fun, custom_vjp):
        raise TypeError(f'{fun.__name__} has no custom_vjp')
    fun_orig = fun.__wrapped__  # original function without custom_vjp

    if partial_args or partial_kwargs:
        fun = partial(fun, *partial_args, **partial_kwargs)
        fun_orig = partial(fun_orig, *partial_args, **partial_kwargs)

    primals_out, vjpfun = vjp(fun, *primals)
    primals_out_orig, vjpfun_orig = vjp(fun_orig, *primals)
    jtu.check_close(primals_out, primals_out_orig, atol=atol, rtol=rtol)

    cot_out = tree_randn_float0_like(primals_out, cot_out_mean, cot_out_std)

    cot = vjpfun(cot_out)
    cot_orig = vjpfun_orig(cot_out)
    cot_diff = check_close(cot, cot_orig, atol=atol, rtol=rtol)

    return cot, cot_orig, cot_diff
