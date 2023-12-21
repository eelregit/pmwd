import numpy as np
import jax.numpy as jnp
import optax
import pickle

from pmwd.sto.so import soft_len
from pmwd.sto.mlp import init_mlp_params

n_epochs = 2

# data
sobol_ids_global = np.arange(0, 2)
snap_ids = np.arange(0, 121, 3)
shuffle_epoch = True  # shuffle the order of sobols across epochs

# loss
loss_pars = {
    'log_eps': 1e-3,
}

# optimizer
learning_rate = 1e-5
optimizer = optax.adamw(learning_rate, b1=0.9, b2=0.9, weight_decay=3e-5)
optimizer = optax.MultiSteps(optimizer, 1)
# optimizer = optax.chain(
#     optax.clip_by_global_norm(3.),
#     optax.adam(learning_rate),
# )
# optimizer = optax.amsgrad(learning_rate, eps=1e-4)
# optimizer = optax.adamw(learning_rate, weight_decay=1e-4)

# so neural nets
so_type = 'NN'
n_input = [soft_len(k_fac=3), soft_len()]

so_nodes = [[3*n] * 5 + [1] for n in n_input]



# start a new training
so_params = init_mlp_params(n_input, so_nodes, scheme='last_ws_b1')
opt_state = optimizer.init(so_params)

# load and continue a training
# with open(f'params/3031768/e2000.pickle', 'rb') as f:
#     dic = pickle.load(f)
#     so_params = dic['so_params']
#     # opt_state = dic['opt_state']
#     opt_state = optimizer.init(so_params)



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
