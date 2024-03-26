import pickle
import numpy as np
import jax.numpy as jnp
from jax.tree_util import tree_map
from tqdm import tqdm
from scipy.stats.qmc import Sobol

from pmwd.sto.util import load_soparams
from pmwd.sto.data import scale_Sobol, gen_cc
from pmwd.sto.mlp import MLP
from pmwd.sto.so import sotheta, soft_k, soft_kvec, soft_names, soft_names_tex


def sample_sonn_data(sidx, so_params, nnv, nnv_o, mesh_shape, m=8, n_steps=61,
                     fn=None):
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
    norm_k_s['f'] = 10**(log_kn_min + (log_kn_max - log_kn_min) * ak[x][:, :, 1])
    norm_k_s['g'] = 10**(log_kn_min + (log_kn_max - log_kn_min) * ak[x][:, :, 1:])

    soft_i = 'soft_' + nnv
    data = {'Note': f'NN input features are {nnv}.',
            'f_X_'+nnv: [], 'g_X_'+nnv: [], 'g_X_'+nnv+'_us': []}
    if nnv_o is not None:
        for n in nnv_o:
            data['f_X_'+n] = []
            data['g_X_'+n] = []
            data['g_X_'+n+'_us'] = []

    # feature names
    for x in ['f', 'g']:
        data[f'{x}_{nnv}_names'] = soft_names(soft_i, x)
        data[f'{x}_{nnv}_names_tex'] = soft_names_tex(soft_i, x)

        if nnv_o is not None:
            for n in nnv_o:
                s = 'soft_' + n
                data[f'{x}_{n}_names'] = soft_names(s, x)
                data[f'{x}_{n}_names_tex'] = soft_names_tex(s, x)

    # construct X
    print('constructing X')
    for i in tqdm(range(nsims)):
        conf, cosmo = gen_cc(sims[i], mesh_shape=mesh_shape, a_nbody_num=n_steps,
                             soft_i=soft_i, cal_boltz=True)
        for x in ['f', 'g']:
            k_s = norm_k_s[x][i] * jnp.pi / conf.cell_size
            for a, k in zip(a_s[x][i], k_s):
                theta = sotheta(cosmo, conf, a)
                if x == 'f':
                    data['f_X_'+nnv].append(soft_k(soft_i, k, theta))
                if x == 'g':
                    k_sort = jnp.sort(k)  # sort for permutation symmetry
                    data['g_X_'+nnv].append(soft_kvec(soft_i, k_sort, theta))
                    data['g_X_'+nnv+'_us'].append(soft_kvec(soft_i, k, theta))

                if nnv_o is not None:
                    for n in nnv_o:
                        s = 'soft_' + n
                        theta = sotheta(cosmo, conf, a, soft_i=s)
                        if x == 'f':
                            data['f_X_'+n].append(soft_k(s, k, theta))
                        if x == 'g':
                            data['g_X_'+n].append(soft_kvec(s, k_sort, theta))
                            data['g_X_'+n+'_us'].append(soft_kvec(s, k, theta))

    data['f_X_'+nnv] = jnp.array(data['f_X_'+nnv])
    data['g_X_'+nnv] = jnp.array(data['g_X_'+nnv])
    data['g_X_'+nnv+'_us'] = np.array(data['g_X_'+nnv+'_us'])
    if nnv_o is not None:
        for n in nnv_o:
            data['f_X_'+n] = jnp.array(data['f_X_'+n])
            data['g_X_'+n] = jnp.array(data['g_X_'+n])
            data['g_X_'+n+'_us'] = jnp.array(data['g_X_'+n+'_us'])

    # evaluate y
    print('evaluating y')
    so_params, n_input, so_nodes = load_soparams(so_params)
    for x, nidx in zip(['f', 'g'], [1, 0]):
        nn = MLP(features=so_nodes[nidx])
        data[f'{x}_y'] = nn.apply(so_params[nidx], data[f'{x}_X_{nnv}']).ravel()

    # save the data to file
    if fn is not None:
        jnp.savez(fn, **data)
        print(f'nn data saved: {fn}')

    return data


if __name__ == "__main__":
    jobid = 3177874
    epoch = 3000
    exp = 'so-1'
    mesh_shape = 1
    nnv = 'v2'  # the feature set of NN
    nnv_o = ['v1', 'v3']  # other features to use/add for SR

    fn = f'nn_data/j{jobid}_e{epoch}.npz'
    so_params = f'../experiments/{exp}/params/{jobid}/e{epoch:03d}.pickle'

    data = sample_sonn_data(np.arange(64), so_params, nnv, nnv_o, mesh_shape, fn=fn)
