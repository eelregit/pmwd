import jax
import jax.numpy as jnp
from jax import vmap, checkpoint

from pmwd.sto.so.mlp import MLP
from pmwd.sto.so.soft import soft_k, soft_kv


def pot_sharp(pot, kvec, theta, cosmo, conf, a):
    """SO of the laplace potential, function of 3D k vector (g function)."""

    if conf.so_type == 'NN' and conf.so_nodes[0] is not None:
        kvec = map(jnp.abs, kvec)  # make even function of kvec

        # sparse to dense kvec, e.g. (128, 128, 65, 3)
        kv = jnp.stack(jnp.broadcast_arrays(*kvec), axis=-1)

        # sort for permutation symmetry of the spatial dimensions
        kv = jnp.sort(kv, axis=-1)

        ft = soft_kv(kv, theta)  # input features
        mlp = MLP(features=conf.so_nodes[0])
        g = mlp.apply(cosmo.so_params[0], ft)[..., 0]

        # use the code below if GPU memory is not sufficient
        # @checkpoint  # checkpoint for saving memory in backward AD
        # def sonn_kvec_slice(kv_):
        #     ft = soft_kv(kv_, theta)  # input features
        #     mlp = MLP(features=conf.so_nodes[0])
        #     g = mlp.apply(cosmo.so_params[0], ft)[..., 0]  # rm the trailing axis of dim one
        #     return g
        # # map for reduced memory usage in the forward run
        # g = jax.lax.map(sonn_kvec_slice, kv)

        pot *= g

    return pot


def grad_sharp(grad, k, theta, cosmo, conf, a):
    """SO of the gradient, function of 1D k component (f function)."""

    if conf.so_type == 'NN' and conf.so_nodes[1] is not None:
        k = jnp.abs(k)  # make even function of k

        ft = soft_k(k, theta)  # input features
        mlp = MLP(features=conf.so_nodes[1])
        f = mlp.apply(cosmo.so_params[1], ft)[..., 0]  # rm the trailing axis of dim one

        grad *= f

    return grad
