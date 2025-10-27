import jax.numpy as jnp
from jax.tree_util import tree_map, tree_flatten
import pickle

from pmwd.scatter import scatter
from pmwd.spec_util import powspec
from pmwd.particles import Particles
from pmwd.sto.so.mlp import mlp_size


def tree_stack(trees):
    """Stack a list of pytrees into a single pytree."""
    return tree_map(lambda *xs: jnp.stack(xs), *trees)


def tree_unstack(tree):
    """Unstack a single pytree into a list of pytrees."""
    leaves, treedef = tree_flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]


def pv2ptcl(pos, vel, pmid, conf):
    """Get ptcl given (pos, vel) and (pmid, conf)."""
    disp = pos - pmid * conf.cell_size
    return Particles(conf, pmid, disp, vel)


def scatter_dens(ptcl_list, conf, mesh_shape, offset=0):
    """A wrapper to scatter a list of particles onto a given mesh shape for a
    list of dens."""
    # mesh_shape should be int or float
    cell_size = conf.ptcl_spacing / mesh_shape
    val = mesh_shape**conf.dim
    mesh_shape = tuple(round(mesh_shape * s) for s in conf.ptcl_grid_shape)
    dens_list = [scatter(p, conf, mesh=jnp.zeros(mesh_shape, dtype=conf.float_dtype),
                         val=val, cell_size=cell_size, offset=offset)
                         for p in ptcl_list]
    return dens_list, cell_size


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
