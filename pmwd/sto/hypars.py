import numpy as np
import jax.numpy as jnp
import optax

from pmwd.sto.so import soft_len
from pmwd.sto.mlp import init_mlp_params

n_epochs = 150

# data
sobol_ids_global = np.arange(0, 8)
snap_ids = np.arange(0, 121, 3)
shuffle_snaps = False

# optimizer
learning_rate = 1e-2

def get_optimizer(lr):
    return optax.adam(lr)

def lr_scheduler(lr, skd_state, loss, patience=10, factor=0.5, threshold=1e-4, lr_min=1e-4):
    """Reduce the learning rate on plateau, a mini implementation."""
    if skd_state is None:  # initialize
        return lr, (0, loss)

    counter, loss_low = skd_state

    if loss_low - loss < abs(loss_low) * threshold:  # not better
        counter += 1
        if counter == patience:  # reduce LR
            lr = max(lr * factor, lr_min)
            counter = 0
    else:  # is better
        counter = 0
        loss_low = loss

    skd_state = (counter, loss_low)

    return lr, skd_state


# so neural nets
so_type = 2

if so_type == 2:
    n_input = [soft_len(l_fac=3), soft_len()]

if so_type == 3:
    n_input = [soft_len()] * 3

so_nodes = [[n * 2 // 3, n // 3, 1] for n in n_input]
so_params = init_mlp_params(n_input, so_nodes, scheme='last_ws_b1')

# mannually turn off nets by setting the corresponding so_nodes to None
# for i in [0, 2]: so_nodes[i] = None
