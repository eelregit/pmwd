import jax.numpy as jnp
from jax import vjp, custom_vjp
from scipy.optimize import fsolve


def bspadding(t, c, p):
    """Pad the knots and controls for clamped B-spline, i.e. fixed at 0 and 1 in
       this case.

    Notes
    -----
    https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-curve.html
    """
    c = jnp.pad(c, (1, 1), mode='constant', constant_values=(0., 1.))
    t = jnp.pad(t, (l:=(p+3)//2, p+3-l), mode='edge')
    t = t.at[:p+1].set(0.)
    t = t.at[-(p+1):].set(1.)
    # c = jnp.pad(c, (l:=(p+1)//2, p+1-l), mode='constant', constant_values=(0., 1.))
    # t = jnp.pad(t, (p+1, p+1),mode='constant', constant_values=(0., 1.))
    return t, c


# TODO switch to sum_i control_i * B_i(x), and cache B_i(x)?
def bspline(x, t, c, p):
    """B-spline with De Boor's algorithm.

    Parameters
    ----------
    x: array
        positions to interp
    t: array
        positions of knots
    c: array
        control points
    p: int
        degree
    """
    t, c = bspadding(t, c, p)

    k = jnp.digitize(x, t, right=True) - 1

    d = [c[j + k - p] for j in range(0, p+1)]
    for r in range(1, p+1):
        for j in range(p, r-1, -1):
            alpha = (x - t[j+k-p]) / (t[j+1+k-r] - t[j+k-p])
            d[j] = (1.0 - alpha) * d[j-1] + alpha * d[j]
    return d[p]


def bspline_inv(y, t, c, p):
    """Inverse of the B-spline above."""
    return fsolve(lambda x: bspline(x, t, c, p) - y, y)


def bspline_params(conf):
    """The input parameters for B-spline except the control points."""
    c = conf.a_bsp_controls
    p = conf.a_bsp_degree
    t = jnp.linspace(1/(1+conf.a_bsp_knots_num), 1., conf.a_bsp_knots_num,
                     endpoint=False)

    t_start = jnp.squeeze(bspline_inv(conf.a_to_start, t, c, p))
    t_nbody = jnp.linspace(t_start, 1., 1+conf.a_nbody_num)
    return p, t, t_nbody


@custom_vjp
def a_nbody_to(conf):
    """Replace a_nbody with the TO a_nbody from the B-spline function."""
    c = conf.a_bsp_controls
    p, t, t_nbody = bspline_params(conf)
    a_nbody = bspline(t_nbody, t, c, p)
    conf = conf.replace(a_nbody=a_nbody)
    return conf


def a_nbody_to_fwd(conf):
    conf = a_nbody_to(conf)
    return conf, conf

def a_nbody_to_bwd(conf, conf_cot):
    c = conf.a_bsp_controls
    p, t, t_nbody = bspline_params(conf)
    a_nbody, bspline_vjp = vjp(bspline, t_nbody, t, c, p)
    x_cot, t_cot, c_cot, p_cot = bspline_vjp(conf_cot.a_nbody)
    conf_cot = conf_cot.replace(a_bsp_controls=c_cot)
    return conf_cot

a_nbody_to.defvjp(a_nbody_to_fwd, a_nbody_to_bwd)
