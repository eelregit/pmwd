from functools import partial

import jax.test_util as jtu

from jax import vjp
from jax import random
from jax.tree_util import tree_map


def randn_like(key, x):
    return random.normal(key, shape=x.shape, dtype=x.dtype)


def check_custom_vjp(fun, primals, args=(), kwargs={}, atol=None, rtol=None):
    """Compare the custom vjp of the wrapped function
    to the automatic vjp of the original function

    Parameters:
        fun: function wrapped by `custom_vjp`
        primals: function inputs whose cotangent vectors are to be compared
        args: positional function inputs to be fixed by `partial`
        kwargs: keyword function inputs to be fixed by `partial`
        atol: absolute tolerance
        rtol: relative tolerance
    """
    key = random.PRNGKey(0)

    fun_orig = fun.__wrapped__  # original function without custom vjp
    if args or kwargs:
        fun = partial(fun, *args, **kwargs)
        fun_orig = partial(fun_orig, *args, **kwargs)

    primals_out, vjpfun = vjp(fun, *primals)
    cot_out = tree_map(partial(randn_like, key), primals_out)
    cot = vjpfun(cot_out)

    primals_out_orig, vjpfun_orig = vjp(fun_orig, *primals)
    cot_orig = vjpfun_orig(cot_out)

    jtu.check_close(primals_out, primals_out_orig, atol=atol, rtol=rtol)
    jtu.check_close(cot, cot_orig, atol=atol, rtol=rtol)
