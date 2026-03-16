import os
import argparse
import pickle
import numpy as np
import jax.numpy as jnp
from jax.tree_util import tree_map
from tqdm import tqdm
from scipy.stats.qmc import Sobol

from pmwd.sto.utils import load_soparams
from pmwd.sto.data.sample import scale_Sobol
from pmwd.sto.data.initial import gen_cc
from pmwd.sto.so.mlp import MLP
import pmwd.sto.so.soft_simple as soft_simple
import pmwd.sto.so.soft_physical as soft_physical


def sample_sonn_data(sidx, so_params_fn, mesh_shape, m=8, n_steps=61,
                     fn=None, sr_features='simple'):
    """Generate the data samples of SO NNs. Use the training sample of
    simulation setups. Use another Sobol to sample a and k."""
    nsims = len(sidx)
    # sample a and k with Sobol, different for each simulation setup
    n = 2**m
    ak = {}
    for x, d in zip(['f', 'g'], [2, 4]):
        ak[x] = np.empty((nsims, n, d))
        for seed in range(nsims):
            sampler = Sobol(d, scramble=True, seed=seed)
            ak[x][seed] = sampler.random(n=n)

    # load the simulation setups
    sims = scale_Sobol(ind=sidx)

    # rescale a and k
    log_kn_min, log_kn_max = jnp.log10(2/128), jnp.log10(1)

    a_s = {}
    for x in ['f', 'g']:
        a_s[x] = 1/16 + (1 + 1/128 - 1/16) * ak[x][:, :, 0]

    norm_k_s = {}
    norm_k_s['f'] = 10**(log_kn_min + (log_kn_max - log_kn_min) * ak['f'][:, :, 1])
    # norm_k_s['g'] will have shape (nsims, n, 3)
    norm_k_s['g'] = 10**(log_kn_min + (log_kn_max - log_kn_min) * ak['g'][:, :, 1:])

    data = {'f_X': [], 'g_X': [], 'g_X_us': []}

    # Decide which feature module to use for SR
    sr_module = soft_physical if sr_features == 'physical' else soft_simple
    
    # feature names for SR
    for x in ['f', 'g']:
        data[f'{x}_names'] = sr_module.soft_names(x)
        data[f'{x}_names_tex'] = sr_module.soft_names_tex(x)

    # Prepare for NN evaluation
    so_params, n_input, so_nodes = load_soparams(so_params_fn)
    nn_X = {'f': [], 'g': []}

    # construct X
    print(f'constructing X using {sr_features} features for SR')
    for i in tqdm(range(nsims)):
        conf, cosmo = gen_cc(sims[i], mesh_shape=mesh_shape, a_nbody_num=n_steps,
                             cal_boltz=True)

        for x in ['f', 'g']:
            k_s = norm_k_s[x][i] * jnp.pi / conf.cell_size
            for a, k in zip(a_s[x][i], k_s):
                # Always calculate simple features for NN labels
                theta_simple = soft_simple.sotheta(cosmo, conf, a)
                if x == 'f':
                    nn_ft = soft_simple.soft_k(k, theta_simple)
                if x == 'g':
                    k_sort = jnp.sort(k)
                    nn_ft = soft_simple.soft_kv(k_sort, theta_simple)
                nn_X[x].append(nn_ft)
                
                # Input for SR (can be physical)
                if sr_features == 'physical':
                    theta_phys = soft_physical.sotheta(cosmo, conf, a)
                    if x == 'f':
                        sr_ft = soft_physical.soft_k(k, theta_phys)
                    if x == 'g':
                        sr_ft = soft_physical.soft_kv(k_sort, theta_phys)
                else:
                    sr_ft = nn_ft

                if x == 'f':
                    data['f_X'].append(sr_ft)
                if x == 'g':
                    data['g_X'].append(sr_ft)
                    # For us (un-sorted) version, we use same feature type
                    if sr_features == 'physical':
                        data['g_X_us'].append(soft_physical.soft_kv(k, theta_phys))
                    else:
                        data['g_X_us'].append(soft_simple.soft_kv(k, theta_simple))

    # finalize X arrays
    for x in ['f', 'g']:
        data[f'{x}_X'] = jnp.array(data[f'{x}_X'])
        nn_X[x] = jnp.array(nn_X[x])
    data['g_X_us'] = np.array(data['g_X_us'])

    # evaluate y
    print('evaluating y (using soft_simple features)')
    for x, nidx in zip(['f', 'g'], [1, 0]):
        nn = MLP(features=so_nodes[nidx])
        data[f'{x}_y'] = nn.apply(so_params[nidx], nn_X[x]).ravel()

    # save the data to file
    if fn is not None:
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        jnp.savez(fn, feature_set=sr_features, **data)
        print(f'nn data saved: {fn}')

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample SO NN data for symbolic regression.')
    parser.add_argument('--jobid', type=int, required=True, help='Slurm job ID')
    parser.add_argument('--epoch', type=int, required=True, help='Epoch number')
    parser.add_argument('--nsims', type=int, default=64, help='Number of simulations to sample')
    parser.add_argument('--exp', type=str, default='standard', help='Experiment name')
    parser.add_argument('--experiments_dir', type=str, default='../experiments', 
                        help='Base directory for experiments')
    parser.add_argument('--out_dir', type=str, default='nn_data', help='Output directory for samples')
    parser.add_argument('--sr_features', type=str, choices=['simple', 'physical'], default='simple',
                        help='Which feature set to use for SR input')
    
    args = parser.parse_args()

    mesh_shape = 1
    fn = os.path.join(args.out_dir, f'j{args.jobid}_e{args.epoch}.{args.sr_features}.npz')
    so_params_fn = os.path.join(args.experiments_dir, args.exp, 'params', 
                                str(args.jobid), f'e{args.epoch:03d}.pickle')

    if not os.path.exists(so_params_fn):
        raise FileNotFoundError(f"Model parameters not found at {so_params_fn}")

    data = sample_sonn_data(np.arange(args.nsims), so_params_fn, mesh_shape, fn=fn, 
                           sr_features=args.sr_features)

