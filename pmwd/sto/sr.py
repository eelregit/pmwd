"""Some util functions related to the Symbolic Regression."""
import jax
import jax.numpy as jnp
import numpy as np
from pysr import PySRRegressor
import itertools
from tqdm.notebook import tqdm
from sklearn.ensemble import RandomForestRegressor
import pickle

from pmwd.sto.util import load_soparams
from pmwd.sto.data import gen_sobol, scale_Sobol, gen_cc
from pmwd.sto.mlp import MLP
from pmwd.sto.so import sotheta, soft_k, soft_kvec


def gen_sonn_data(so_params, net='f', m=None, fn=None):
    """Generate the (input, output) samples of so neural networks, using Sobol."""
    # sample theta params [:, :9], a [:, 9] and k (k1 k2 k3) [:, 10:]
    # using Sobol for f and g neural nets
    if net == 'f':
        d, nidx = 11, 1
    if net == 'g':
        d, nidx = 13, 0
    if m is None:
        m = d
    sobol = gen_sobol(d=d, m=m, extra=0)
    sobol_s = scale_Sobol(sobol=sobol[:, :9].T)

    a_s = 1/16 + (1 + 1/128 - 1/16) * sobol[:, 9]
    logk_min, logk_max = -3, 2
    if net == 'f':
        k_s = 10**(logk_min + (logk_max - logk_min) * sobol[:, 10])
    if net == 'g':
        k_s = 10**(logk_min + (logk_max - logk_min) * sobol[:, 10:])
        k_s = np.sort(k_s, axis=-1)  # sort for permutation symmetry

    # pmwd only params
    mesh_shape = 1
    n_steps = 100

    # construct X for the network
    print('constructing X')
    X = []
    for sob, a, k in tqdm(zip(sobol_s, a_s, k_s), total=len(sobol_s)):
        conf, cosmo = gen_cc(sob, mesh_shape=mesh_shape, a_nbody_num=n_steps)
        theta = sotheta(cosmo, conf, a)
        if net == 'f':
            X.append(soft_k(k, theta))
        if net == 'g':
            X.append(soft_kvec(k, theta))
    X = jnp.asarray(X)

    # evaluate y
    print('evaluating y')
    so_params, n_input, so_nodes = load_soparams(so_params)
    nn = MLP(features=so_nodes[nidx])
    y = nn.apply(so_params[nidx], X).ravel()

    # save the data to file simply with pickle
    if fn is not None:
        with open(fn, 'wb') as f:
            pickle.dump([X, y], f)
        print(f'[X, y] saved: {fn}')

    return X, y


def gen_sonn_data_old(so_params):
    """Generate the (input, output) samples of so neural networks."""
    # TODO this method would give a dataset that is too large
    # maybe we should sample the input of f and g (i.e. k*scales etc.) directly
    so_params, n_input, so_nodes = load_soparams(so_params)
    nets = [MLP(features=nodes) for nodes in so_nodes]

    # we need to sample k, a, and params (using sobol)
    sobol_s = scale_Sobol(ind=slice(0, 8, 1))
    a_s = jnp.linspace(1/16, 1+1/128, 121)
    k_s = jnp.logspace(-3, 2, 100)
    kv_s = jnp.stack(jnp.meshgrid(k_s, k_s, k_s, indexing='ij'), axis=-1)
    kv_s = jnp.sort(kv_s, axis=-1)  # sort for permutation symmetry

    # pmwd only params
    mesh_shape_s = [1]
    n_steps_s = [100]

    # construct X for the two networks
    print('constructing X')
    X = {'f': [], 'g': []}
    for sobol, mesh_shape, n_steps in tqdm(itertools.product(
                                        sobol_s, mesh_shape_s, n_steps_s)):
        conf, cosmo = gen_cc(sobol, mesh_shape=mesh_shape, a_nbody_num=n_steps)
        for a in a_s:
            theta = sotheta(cosmo, conf, a)
            X['f'].append(soft_k(k_s, theta))
            X['g'].append(soft_kvec(kv_s, theta))

    # evaluate y
    print('evaluating y')
    y = {}
    for i, x in enumerate(('f', 'g')):
        X[x] = jnp.asarray(X[x])
        X[x] = X[x].reshape((-1, X[x].shape[-1]))
        y[x] = nets[i].apply(so_params[i], X[x])

    X_names = None
    return X, y, X_names


def rf_feature_importance(X, y, n_estimators=100, max_depth=3, random_state=None):
    """Get the feature importance and indices using random forest, from high to low."""
    clf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                random_state=random_state)
    clf.fit(X, y)
    idx = np.argsort(clf.feature_importances_)[::-1]  # high to low

    return clf.feature_importances_[idx], idx


def run_pysr(X, y):
    """Run PySR on the data and return the LaTeX expressions."""
    model = PySRRegressor(
        niterations=40,
        binary_operators=['+', '-', '*', '/'],
        unary_operators=['exp', 'log'],
        loss='loss(prediction, target) = (prediction - target)^2'
    )
    model.fit(X, y)
    tex_tab_str = model.latex_table()
    return tex_tab_str
