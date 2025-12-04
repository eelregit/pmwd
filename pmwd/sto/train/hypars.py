import numpy as np
import optax
import pickle

from pmwd.sto.so.soft import soft_len
from pmwd.sto.so.mlp import init_mlp_params

n_epochs = 1000

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
    'log_eps': 0.,
    'loss_mesh_shape': 3,
    'grid_offset': 0.,
    'loss_fields': ['disp', 'dens'],
}

###  optimizer  ###
opt_conf = {
    'learning_rate': 1e-5,
}
opt_conf['optimizer'] = optax.adam(opt_conf['learning_rate'])

###  model  ###
if len(data_conf['snap_ids']) == 121:
    n_steps = 121
    a_stop = 1 + 1/128
if len(data_conf['snap_ids']) == 61:
    n_steps = 61
    a_stop = 1 + 1/64
model_conf = {
    'n_steps': n_steps,
    'a_stop': a_stop,
    'mesh_shape': 1,
    'so_type': 'NN',
}
model_conf['n_input'] = [soft_len('g'), soft_len('f')]
model_conf['so_nodes'] = [[128, 128, 128, 1], [64, 64, 64, 1]]

###  start a new training  ###
so_params = init_mlp_params(model_conf['n_input'], model_conf['so_nodes'],
                            scheme='last_ws')
opt_state = opt_conf['optimizer'].init(so_params)

###  load and continue a training  ###
# job_id, epoch_id = 3031768, 2000
# param_fn = f'params/{job_id}/e{epoch_id}.pickle'
# with open(param_fn, 'rb') as f:
#     dic = pickle.load(f)
#     so_params = dic['so_params']
#     opt_state = dic['opt_state']
#     # opt_state = optimizer.init(so_params)
