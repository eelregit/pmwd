import pytest
import jax.numpy as jnp
import jax.test_util as jtu
from jax.tree_util import tree_map

from pmwd.configuration import Configuration
from pmwd.cosmology import SimpleLCDM
from pmwd.boltzmann import boltzmann
from pmwd.nbody import nbody
from pmwd.test_util import gen_ptcl, check_custom_vjp


@pytest.mark.parametrize('mesh_shape', [(4, 9), (7, 8)], ids=['evenodd', 'oddeven'])
class TestNbody:
    def test_nbody_reversibility(self, mesh_shape):
        conf = Configuration(
            cell_size=1.,
            mesh_shape=mesh_shape,
            float_dtype=jnp.dtype(jnp.float64),
            a_start=1/2,
            a_stop=1.,
            a_nbody_maxstep=1/32,
        )
        cosmo = SimpleLCDM(conf)
        cosmo = boltzmann(cosmo, conf)

        ptcl = gen_ptcl(conf, disp_std=7, vel_std=1)
        obsvbl = None

        ptcl1, _ = nbody(ptcl, obsvbl, cosmo, conf)
        ptcl0, _ = nbody(ptcl1, obsvbl, cosmo, conf, reverse=True)

        ptcl0 = ptcl0.replace(acc=None)  # because ptcl.acc is None
        jtu.check_eq(ptcl0, ptcl)

    def test_nbody_custom_vjp(self, mesh_shape):
        conf = Configuration(
            cell_size=1.,
            mesh_shape=mesh_shape,
            float_dtype=jnp.dtype(jnp.float64),
            a_start=1/2,
            a_stop=1.,
            a_nbody_maxstep=1/32,
        )
        cosmo = SimpleLCDM(conf)
        cosmo = boltzmann(cosmo, conf)

        ptcl = gen_ptcl(conf, disp_std=7, vel_std=1)
        obsvbl = None

        # zero acc cot otherwise it also backprops in the automatic vjp
        cot_out_std = tree_map(lambda x: 1, (ptcl, obsvbl))
        cot_out_std = (cot_out_std[0].replace(acc=0), cot_out_std[1])

        primals = ptcl, obsvbl, cosmo, conf
        check_custom_vjp(nbody, primals, cot_out_std=cot_out_std,
                         atol=1e-14, rtol=1e-14)
