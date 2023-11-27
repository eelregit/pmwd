import numpy as np
import jax.numpy as jnp
import optax
import pickle

from pmwd.sto.so import soft_len
from pmwd.sto.mlp import init_mlp_params

n_epochs = 1000

# data
sobol_ids_global = np.arange(0, 64)
snap_ids = np.arange(0, 121, 1)
shuffle_epoch = True  # shuffle the order of sobols across epochs

# loss
log_loss_eps = 1e-2

# optimizer
learning_rate = 3e-5
optimizer = optax.adam(learning_rate)
# optimizer = optax.chain(
#     optax.clip_by_global_norm(3.),
#     optax.adam(learning_rate),
# )
# optimizer = optax.amsgrad(learning_rate, eps=1e-4)
# optimizer = optax.adamw(learning_rate, weight_decay=1e-4)

# so neural nets
so_type = 'NN'
n_input = [soft_len(k_fac=3), soft_len()]

so_nodes = [[3*n, 3*n, 3*n, 1] for n in n_input]

# start a new training
so_params = init_mlp_params(n_input, so_nodes, scheme='last_ws_b1')
opt_state = optimizer.init(so_params)

# load and continue a training
# with open('params/3016932/e1000.pickle', 'rb') as f:
#     dic = pickle.load(f)
#     so_params = dic['so_params']
#     opt_state = dic['opt_state']



def get_optimizer(lr):
    return optax.adam(lr)
    # return optax.adamw(lr, weight_decay=1e-3)


def lr_scheduler(lr, skd_state, loss, patience=15, factor=0.5, threshold=1e-4, lr_min=1e-7):
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

dropout_rate = None  # set to None for no dropout applied

# mannually turn off nets by setting the corresponding so_nodes to None
# for i in [0, 2]: so_nodes[i] = None
