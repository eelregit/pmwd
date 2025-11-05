import numpy as np
import optax
import pickle

from pmwd.sto.so.so import soft_len
from pmwd.sto.so.mlp import init_mlp_params

n_epochs = 5000

###  data  ###
data_conf = {
    'data_dir': '/mnt/home/llu/ceph/sto/g4run/gs512',
    'sobol_file': '/mnt/home/llu/ceph/sto/pmwd/sto/g4gen/sobol.txt',
    'sobol_ids_global': np.arange(0, 64),
    'snap_ids': np.arange(0, 121, 2),
    'shuffle_epoch': True,  # shuffle the order of sobols across epochs
}

###  loss  ###
loss_conf = {
    'log_eps': 0,
    'loss_mesh_shape': 3,
    'grid_offset': 0,
    'loss_fields': ['disp', 'dens'],
}

###  optimizer  ###
opt_conf = {
    'learning_rate': 1e-5,
}
opt_conf['optimizer'] = optax.adam(opt_conf['learning_rate'])

###  model  ###
model_conf = {
    'mesh_shape': 1,
    'n_steps': 61,
    'so_type': 'NN',
    'soft_i': 'soft_v1',
}
model_conf['n_input'] = [soft_len(model_conf['soft_i'], 'g'),
                         soft_len(model_conf['soft_i'], 'f')]
model_conf['so_nodes'] = [[3*n] * 5 + [1] for n in model_conf['n_input']]

###  start a new training  ###
so_params = init_mlp_params(model_conf['n_input'], model_conf['so_nodes'],
                            scheme='last_ws')
opt_state = opt_conf['optimizer'].init(so_params)

# ###  load and continue a training  ##
# param_fn = 'params/3031768/e2000.pickle'
# with open(param_fn, 'rb') as f:
#     dic = pickle.load(f)
#     so_params = dic['so_params']
#     # opt_state = dic['opt_state']
#     opt_state = optimizer.init(so_params)
