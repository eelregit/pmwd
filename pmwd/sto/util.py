import jax.numpy as jnp
from pmwd.scatter import scatter


def ptcl2dens(ptcls, conf, mesh_shape):
    if mesh_shape is None:  # 2x the mesh in pmwd sim
        cell_size = conf.cell_size / 2
        mesh_shape = tuple(2 * ms for ms in conf.mesh_shape)
    else:  # float or int
        cell_size = conf.ptcl_spacing / mesh_shape
        mesh_shape = tuple(round(mesh_shape * s) for s in conf.ptcl_grid_shape)
    denss = (scatter(p, conf, mesh=jnp.zeros(mesh_shape, dtype=conf.float_dtype),
                     val=1, cell_size=cell_size) for p in ptcls)
    return denss, (mesh_shape, cell_size)