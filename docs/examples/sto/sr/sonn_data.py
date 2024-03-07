import pickle
import itertools
import numpy as np
import jax.numpy as jnp
from flax.linen import relu
from tqdm import tqdm
from scipy.stats.qmc import Sobol

from pmwd.sto.util import load_soparams
from pmwd.sto.data import gen_sobol, scale_Sobol, gen_cc
from pmwd.sto.mlp import MLP
from pmwd.sto.so import sotheta, soft_k, soft_kvec


def sample_sonn_data(sidx, so_params, soft_i, mesh_shape, m=8, n_steps=61, fn=None):
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
    log_kn_min, log_kn_max = jnp.log10(2/128), jnp.log10(np.sqrt(3))

    a_s = {}
    for x in ['f', 'g']:
        a_s[x] = 1/16 + (1 + 1/128 - 1/16) * ak[x][:, :, 0]

    norm_k_s = {}
    norm_k_s['f'] = 10**(log_kn_min + (log_kn_max - log_kn_min) * ak[x][:, :, 1])
    norm_k_s['g'] = jnp.sort(10**(log_kn_min + (log_kn_max - log_kn_min) * ak[x][:, :, 1:]),
                             axis=-1)  # sort for permutation symmetry

    # construct X
    print('constructing X')
    X = {'f': [], 'g': []}
    for i in range(nsims):
        cal_boltz = True if soft_i == 'soft_a' else False
        conf, cosmo = gen_cc(sims[i], mesh_shape=mesh_shape, a_nbody_num=n_steps,
                             soft_i=soft_i, cal_boltz=cal_boltz)
        for x in ['f', 'g']:
            k_s = norm_k_s[x][i] * jnp.pi / conf.ptcl_spacing
            for a, k in zip(a_s[x][i], k_s):
                theta = sotheta(cosmo, conf, a)
                if x == 'f':
                    X[x].append(soft_k(soft_i, k, theta))
                if x == 'g':
                    X[x].append(soft_kvec(soft_i, k, theta))
    X['f'] = jnp.asarray(X['f'])
    X['g'] = jnp.asarray(X['g'])

    # evaluate y
    print('evaluating y')
    so_params, n_input, so_nodes = load_soparams(so_params)
    y = {}
    for x, nidx in zip(['f', 'g'], [1, 0]):
        nn = MLP(features=so_nodes[nidx], activator=relu, regulator=jnp.exp)
        y[x] = nn.apply(so_params[nidx], X[x]).ravel()

    # save the data to file simply with pickle
    nn_data = {
        'f': {'X': np.asarray(X['f']), 'y': np.asarray(y['f'])},
        'g': {'X': np.asarray(X['g']), 'y': np.asarray(y['g'])}
    }
    if fn is not None:
        with open(fn, 'wb') as f:
            pickle.dump(nn_data, f)
        print(f'nn_data saved: {fn}')

    return X, y


def gen_sonn_data(so_params, soft_i, mesh_shape, n_steps=61, fn=None,
                  vis_mesh_shape=3, m_extra={'f': 3, 'g': 3}):
    """Generate the (input, output) samples of so neural networks, using Sobol."""
    # sample configurations and cosmological params [:, :9]
    # a [:, 9] and k (k1 k2 k3) [:, 10:]
    # using Sobol for f and g neural nets
    sobol = {}
    sobol_s = {}
    a_s = {}
    for x, d in zip(['f', 'g'], [11, 13]):
        sobol[x] = gen_sobol(d=d, m=d+m_extra[x], extra=0)
        sobol_s[x] = scale_Sobol(sobol=sobol[x][:, :9].T)
        a_s[x] = 1/16 + (1 + 1/128 - 1/16) * sobol[x][:, 9]

    # sample k w.r.t. the ptcl grid nyquist
    log_kn_min, log_kn_max = jnp.log10(2/128), jnp.log10(vis_mesh_shape)
    norm_k_s = {}
    norm_k_s['f'] = 10**(log_kn_min + (log_kn_max - log_kn_min) * sobol['f'][:, 10])
    norm_k_s['g'] = jnp.sort(10**(log_kn_min + (log_kn_max - log_kn_min) * sobol['g'][:, 10:]),
                             axis=-1)  # sort for permutation symmetry

    # construct X for the network
    print('constructing X')
    X = {'f': [], 'g': []}
    for x in ['f', 'g']:
        for sob, a, norm_k in tqdm(zip(sobol_s[x], a_s[x], norm_k_s[x]), total=len(sobol_s[x])):
            cal_boltz = True if soft_i == 'soft_a' else False
            conf, cosmo = gen_cc(sob, mesh_shape=mesh_shape, a_nbody_num=n_steps,
                                 soft_i=soft_i, cal_boltz=cal_boltz)
            k = norm_k * jnp.pi / conf.ptcl_spacing
            theta = sotheta(cosmo, conf, a)
            if x == 'f':
                X[x].append(soft_k(soft_i, k, theta))
            if x == 'g':
                X[x].append(soft_kvec(soft_i, k, theta))
        X[x] = jnp.asarray(X[x])

    # evaluate y
    print('evaluating y')
    so_params, n_input, so_nodes = load_soparams(so_params)
    y = {}
    for x, nidx in zip(['f', 'g'], [1, 0]):
        nn = MLP(features=so_nodes[nidx], activator=relu, regulator=jnp.exp)
        y[x] = nn.apply(so_params[nidx], X[x]).ravel()

    # save the data to file simply with pickle
    nn_data = {
        'f': {'X': np.asarray(X['f']), 'y': np.asarray(y['f'])},
        'g': {'X': np.asarray(X['g']), 'y': np.asarray(y['g'])}
    }
    if fn is not None:
        with open(fn, 'wb') as f:
            pickle.dump(nn_data, f)
        print(f'nn_data saved: {fn}')

    return X, y


if __name__ == "__main__":
    jobid = 3177874
    epoch = 3000
    exp = 'so-1'
    soft_i = 'soft_d'
    mesh_shape = 1
    fn = f'nn_data/{jobid}_e{epoch}.pickle'
    so_params = f'../experiments/{exp}/params/{jobid}/e{epoch:03d}.pickle'

    # X, y = gen_sonn_data(so_params, soft_i, mesh_shape, fn=fn)
    X, y = sample_sonn_data(np.arange(64), so_params, soft_i, mesh_shape, fn=fn)
