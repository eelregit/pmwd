import jax.numpy as jnp
from pmwd.scatter import scatter
from pmwd.spec_util import powspec


def ptcl2dens(ptcls, conf, mesh_shape):
    if mesh_shape is None:  # the mesh in pmwd sim
        cell_size = conf.cell_size
        mesh_shape = conf.mesh_shape
    else:  # float or int
        cell_size = conf.ptcl_spacing / mesh_shape
        mesh_shape = tuple(round(mesh_shape * s) for s in conf.ptcl_grid_shape)
    denss = (scatter(p, conf, mesh=jnp.zeros(mesh_shape, dtype=conf.float_dtype),
                     val=1, cell_size=cell_size) for p in ptcls)
    return denss, (mesh_shape, cell_size)


def power_tfcc(dens, dens_t, cell_size):
    # estimate power spectra
    k, ps, N, bins = powspec(dens, cell_size, cut_nyq=False)
    k, ps_t, N, bins = powspec(dens_t, cell_size, cut_nyq=False)
    k, ps_cross, N, bins = powspec(dens, cell_size, g=dens_t, cut_nyq=False)
    ps_cross = ps_cross.real

    # the transfer function and correlation coefficient
    tf = jnp.sqrt(ps / ps_t)
    cc = ps_cross / jnp.sqrt(ps * ps_t)

    return k, tf, cc


def mlp_size(mlp_params):
    """Infer the sizes of input and hidden layers given a list of MLP params."""
    n_input, n_nodes = [], []
    for params in mlp_params:
        dic = params['params']
        n_input.append(dic['layers_0']['kernel'].shape[0])
        n_nodes.append([dic[f'layers_{i}']['kernel'].shape[1] for i in range(len(dic))])

    return n_input, n_nodes
