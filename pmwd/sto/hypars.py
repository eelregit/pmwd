import numpy as np
import optax

from pmwd.sto.so import soft_len
from pmwd.sto.mlp import init_mlp_params

n_epochs = 5000

###  data  ###
sobol_ids_global = np.arange(0, 64)
snap_ids = np.arange(0, 121, 2)
shuffle_epoch = True  # shuffle the order of sobols across epochs

###  loss  ###
loss_pars = {
    'log_eps': 0,
}

###  optimizer  ###
learning_rate = 1e-5
optimizer = optax.adam(learning_rate)
optimizer = optax.MultiSteps(optimizer, 1)

###  model  ###
so_type = 'NN'
soft_i = 'soft_v1'
n_input = [soft_len(soft_i, 'g'), soft_len(soft_i, 'f')]

so_nodes = [[3*n] * 5 + [1] for n in n_input]

# start a new training
so_params = init_mlp_params(n_input, so_nodes, scheme='last_ws')
opt_state = optimizer.init(so_params)

# load and continue a training
# with open(f'params/3031768/e2000.pickle', 'rb') as f:
#     dic = pickle.load(f)
#     so_params = dic['so_params']
#     # opt_state = dic['opt_state']
#     opt_state = optimizer.init(so_params)
