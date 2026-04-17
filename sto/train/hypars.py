import os
import numpy as np
import optax
import jax.numpy as jnp

from pmwd.sto.so.soft import soft_len
from pmwd.configuration import Configuration

###  data  ###
data_conf = {
    'data_dir': '/mnt/home/llu/ceph/sto/g4run/gs512',
    'sobol_file': '/mnt/home/llu/ceph/sto/pmwd/sto/g4gen/sobol.txt',
    'sobol_ids_global': np.arange(0, 512),
    'snap_ids': np.arange(0, 121, 4),
    'shuffle': True,  # shuffle the order of sobols across epochs
}

###  model  ###
if len(data_conf['snap_ids']) == 121:  # np.arange(0, 121, 1)
    a_nbody_num = 121
    a_stop = 1 + 1/128
if len(data_conf['snap_ids']) == 61:  # np.arange(0, 121, 2)
    a_nbody_num = 61
    a_stop = 1 + 1/64
if len(data_conf['snap_ids']) == 31:  # np.arange(0, 121, 4)
    a_nbody_num = 31
    a_stop = 1 + 1/32
if len(data_conf['snap_ids']) == 16:  # np.arange(0, 121, 8)
    a_nbody_num = 16
    a_stop = 1 + 1/16

model_conf = Configuration(
    ptcl_spacing = 1.,  # placeholder
    ptcl_grid_shape = (128,) * 3,
    mesh_shape = 1,
    a_start = 1 / 16,
    a_stop = a_stop,
    float_dtype = jnp.float32,
    observe_snapshots = True,
    a_nbody_num = a_nbody_num,
    so_type = 'NN',
    so_nodes = ((128,) * 5 + (1,), (64,) * 5 + (1,)),
)
n_input = (soft_len('g'), soft_len('f'))

###  loss  ###
loss_conf = {
    'log_eps': 0.,
    'loss_fields': ['disp', 'dens'],
}

###  optimizer  ###
opt_conf = {
    'learning_rate': 1e-4,
}
# customize batch size with grad accumulation
batch_size = 64
n_procs = os.getenv('SLURM_NTASKS') # total num devices, i.e. sims per step
if n_procs:
    n_procs = int(n_procs)
else:
    n_procs = 1
grad_accu_steps = batch_size // n_procs

opt_conf['optimizer'] = optax.MultiSteps(
    optax.adamw(opt_conf['learning_rate'], weight_decay=0.01),
    grad_accu_steps,
    use_grad_mean=True,
)
