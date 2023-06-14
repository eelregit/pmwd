import numpy as np
import optax

from pmwd.sto.so import soft_len
from pmwd.sto.mlp import init_mlp_params

n_epochs = 150

# data
sobol_ids_global = np.arange(0, 8)
snap_ids = np.arange(0, 121, 3)
shuffle_snaps = False

# optimizer
learning_rate = 1e-4
optimizer = optax.adam(learning_rate)
# optimizer = optax.adamw(learning_rate, weight_decay=1e-4)

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
