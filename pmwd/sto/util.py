import jax.numpy as jnp
import pickle
from pmwd.sto.mlp import mlp_size

from pmwd.scatter import scatter
from pmwd.spec_util import powspec
from pmwd.particles import Particles


def pv2ptcl(pos, vel, pmid, conf):
    """Get ptcl given (pos, vel) and (pmid, conf)."""
    disp = pos - pmid * conf.cell_size
    return Particles(conf, pmid, disp, vel)


def scatter_dens(ptcls, conf, mesh_shape):
    """A wrapper to scatter particles onto a given mesh shape for dens."""
    # mesh_shape should be int or float
    cell_size = conf.ptcl_spacing / mesh_shape
    mesh_shape = tuple(round(mesh_shape * s) for s in conf.ptcl_grid_shape)
    denss = (scatter(p, conf, mesh=jnp.zeros(mesh_shape, dtype=conf.float_dtype),
                     val=1, cell_size=cell_size) for p in ptcls)
    return denss, cell_size


def power_tfcc(f, g, spacing, cut_nyq=False):
    """A wrapper to get the trans func and corr coef of two fields."""
    # estimate power spectra
    k, ps, N, bins = powspec(f, spacing, cut_nyq=cut_nyq)
    k, ps_t, N, bins = powspec(g, spacing, cut_nyq=cut_nyq)
    k, ps_cross, N, bins = powspec(f, spacing, g=g, cut_nyq=cut_nyq)
    ps_cross = ps_cross.real

    # the transfer function and correlation coefficient
    tf = jnp.sqrt(ps / ps_t)
    cc = ps_cross / jnp.sqrt(ps * ps_t)

    return k, tf, cc


def load_soparams(so_params):
    if isinstance(so_params, str):
        with open(so_params, 'rb') as f:
            so_params = pickle.load(f)['so_params']
    n_input, so_nodes = mlp_size(so_params)
    return so_params, n_input, so_nodes
