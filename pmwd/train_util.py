import jax
import jax.numpy as jnp
from jax.lax import pmean
import numpy as np
import optax
from functools import partial
from torch.utils.data import Dataset
import os
from joblib import Parallel, delayed

from pmwd import (
    Configuration,
    Cosmology,
    boltzmann,
    white_noise,
    linear_modes,
    lpt,
    nbody,
    scatter,
    Particles,
)
from pmwd.pm_util import rfftnfreq
from pmwd.io_util import read_gadget_hdf5


def scale_Sobol(fn='sobol.txt', ind=slice(None)):
    """Scale the Sobol sequence samples, refer to the Table in the paper."""
    sobol = np.loadtxt(fn)[ind].T
    # functions mapping uniform random samples in [0, 1] to a desired one
    f_uni = lambda x, a, b : a + x * (b - a)
    f_log_uni = lambda x, a, b : np.exp(f_uni(x, np.log(a), np.log(b)))
    def f_log_trap(x, a, b, c):
        # a, b, c are the locations of the first 3 points of the symmetric trapezoid
        h = 1 / (c - a)
        x1 = (b - a) * h / 2
        x2 = (2*c - b - a) * h / 2
        y = np.zeros_like(x)
        m = (x < x1)
        y[m] = a + np.sqrt(2 * (b - a) * x[m] / h)
        m = (x1 <= x) & (x < x2)
        y[m] = x[m] / h + (a + b) / 2
        m = (x2 <= x)
        y[m] = c + b - a - np.sqrt((1 - x[m]) * 2 * (b - a) / h)
        return np.exp(y)

    # 0: box size, log-trapezoidal
    sobol[0] = f_log_trap(sobol[0], np.log(128)+np.log(0.2),
                          np.log(512)+np.log(0.2), np.log(128)+np.log(5))
    # 1: snapshot offset, uniform
    sobol[1] = f_uni(sobol[1], 0, 1/128)
    # 2: A_s_1e9, log-uniform
    sobol[2] = f_log_uni(sobol[2], 1, 4)
    # 3: n_s, log-uniform
    sobol[3] = f_log_uni(sobol[3], 0.75, 1.25)
    # 4: Omega_m, log-uniform
    sobol[4] = f_log_uni(sobol[4], 1/5, 1/2)
    # 5: Omega_b / Omega_m, log-uniform
    sobol[5] = f_log_uni(sobol[5], 1/8, 1/4)
    sobol[5] *= sobol[4]  # get Omega_b
    # 6: Omega_k / (1 - Omega_k), uniform
    sobol[6] = f_uni(sobol[6], -1/3, 1/3)
    sobol[6] = sobol[6] / (1 + sobol[6])  # get Omega_k
    # 7: h, log-uniform
    sobol[7] = f_log_uni(sobol[7], 0.5, 1)
    # 8: softening ratio, log-uniform
    sobol[8] = f_log_uni(sobol[8], 1/50, 1/20)
    sobol[8] *= sobol[0] / 128  # * ptcl_spacing = softening length

    return sobol.T


def gen_cc(sobol, mesh_shape=1, a_out=1, a_nbody_num=63, so_nodes=None,
           a_start=1/16, a_stop=1+1/128):
    """Setup conf and cosmo given a sobol."""
    conf = Configuration(
        ptcl_spacing = sobol[0] / 128,
        ptcl_grid_shape = (128,) * 3,
        a_start = a_start,
        a_stop = a_stop,
        float_dtype = jnp.float64,
        mesh_shape = mesh_shape,
        a_out = a_out,
        a_nbody_num = a_nbody_num,
        so_nodes = so_nodes,
    )

    cosmo = Cosmology(
        conf = conf,
        A_s_1e9 = sobol[2],
        n_s = sobol[3],
        Omega_m = sobol[4],
        Omega_b = sobol[5],
        Omega_k_ = sobol[6],
        h = sobol[7],
    )

    return conf, cosmo


def gen_ic(seed, conf, cosmo):
    """Generate the initial condition with lpt for nbody."""
    modes = white_noise(seed, conf)

    cosmo = boltzmann(cosmo, conf)
    modes = linear_modes(modes, cosmo, conf)
    ptcl, obsvbl = lpt(modes, cosmo, conf)

    return ptcl, cosmo


def read_g4data(sobol_ids, sims_dir, snaps_per_sim, fn_sobol):
    data = {}
    def load_sobol(i, sidx):
        data[i] = {}
        sobol = scale_Sobol(fn_sobol, sidx)
        for j in range(snaps_per_sim):
            snap_file = os.path.join(sims_dir, f'{sidx:03}',
                                     'output', f'snapshot_{j:03}')
            pos, vel, a = read_gadget_hdf5(snap_file)
            data[i][j] = (pos, vel, a, sidx, sobol)
    Parallel(n_jobs=min(16, len(sobol_ids)), prefer='threads', require='sharedmem')(
        delayed(load_sobol)(i, sidx) for i, sidx in enumerate(sobol_ids))
    return data


class G4snapDataset(Dataset):

    def __init__(self, sims_dir, sobol_ids=None, sobols_edge=None,
                 snaps_per_sim=121, fn_sobol='sobol.txt'):
        self.sims_dir = sims_dir
        if sobol_ids is None:
            sobol_ids = np.arange(*sobols_edge)
        self.n_sims = len(sobol_ids)
        self.snaps_per_sim = snaps_per_sim
        self.n_snaps = self.n_sims * self.snaps_per_sim

        self.data = read_g4data(sobol_ids, sims_dir, snaps_per_sim, fn_sobol)

    def __len__(self):
        return self.n_snaps

    def __getitem__(self, idx):
        i_sobol = idx // self.snaps_per_sim
        i_snap = idx % self.snaps_per_sim

        return self.data[i_sobol][i_snap]


def _loss_dens_mse(dens, dens_t):
    return jnp.sum((dens - dens_t)**2)


def _loss_disp_mse(ptcl, ptcl_t):
    return jnp.sum((ptcl.disp - ptcl_t.disp)**2)


def _loss_vel_mse(ptcl, ptcl_t):
    return jnp.sum((ptcl.vel - ptcl_t.vel)**2)


@jax.custom_vjp
def _loss_scale_wmse(kvec, dens_k, dens_t_k):
    k2 = sum(k**2 for k in kvec)
    diff = dens_k - dens_t_k
    loss = jnp.sum(jnp.where(k2 != 0, jnp.abs(diff)**2 / k2**1.5, 0))
    return loss, (k2, diff)

def _loss_scale_wmse_fwd(kvec, dens_k, dens_t_k):
    loss, res = _loss_scale_wmse(kvec, dens_k, dens_t_k)
    return loss, res

def _loss_scale_wmse_bwd(res, loss_cot):
    k2, diff = res
    abs_valgrad = jax.value_and_grad(jnp.abs)
    abs_diff, dabs_diff = jax.vmap(abs_valgrad)(diff.ravel())
    abs_diff = abs_diff.reshape(k2.shape)
    dabs_diff = dabs_diff.reshape(k2.shape)
    dens_k_cot = loss_cot * jnp.where(k2 != 0, 2 * abs_diff * dabs_diff / k2**1.5, 0)
    return None, dens_k_cot, None

_loss_scale_wmse.defvjp(_loss_scale_wmse_fwd, _loss_scale_wmse_bwd)


def loss_func(ptcl, tgt, conf, mesh_shape=None):
    loss = 0

    # get the target ptcl
    pos_t, vel_t = tgt
    disp_t = pos_t - ptcl.pmid * conf.cell_size
    ptcl_t = Particles(conf, ptcl.pmid, disp_t, vel_t)

    # get the density fields for the loss
    if mesh_shape is None:  # default to 2x the mesh in pmwd sim
        cell_size = conf.cell_size / 2
        mesh_shape = tuple(2 * ms for ms in conf.mesh_shape)
    else:  # float or int
        cell_size = conf.ptcl_spacing / mesh_shape
        mesh_shape = tuple(round(mesh_shape * s) for s in conf.ptcl_grid_shape)
    dens, dens_t = (scatter(p, conf, mesh=jnp.zeros(mesh_shape, dtype=conf.float_dtype),
                            val=1, cell_size=cell_size) for p in (ptcl, ptcl_t))
    dens_k = jnp.fft.rfftn(dens)
    dens_t_k = jnp.fft.rfftn(dens_t)
    kvec = rfftnfreq(mesh_shape, cell_size, dtype=conf.float_dtype)

    loss += _loss_scale_wmse(kvec, dens_k, dens_t_k)
    # loss += _loss_dens_mse(dens, dens_t)
    # loss += _loss_disp_mse(ptcl, ptcl_t)

    return loss


def obj(tgt, ptcl_ic, so_params, cosmo, conf):
    cosmo = cosmo.replace(so_params=so_params)
    ptcl, obsvbl = nbody(ptcl_ic, None, cosmo, conf)
    loss = loss_func(obsvbl[0], tgt, conf)
    return loss


@partial(jax.pmap, axis_name='global', in_axes=(0, None), out_axes=None)
def _global_mean(loss, grad):
    loss = pmean(loss, axis_name='global')
    grad = pmean(grad, axis_name='global')
    return loss, grad


def _init_pmwd(pmwd_params):
    a_out, sidx, sobol, mesh_shape, n_steps, so_nodes = pmwd_params

    # generate ic, cosmo, conf
    conf, cosmo = gen_cc(sobol, mesh_shape=(mesh_shape,)*3, a_out=a_out,
                         a_nbody_num=n_steps, so_nodes=so_nodes)
    ptcl_ic, cosmo = gen_ic(sidx, conf, cosmo)

    return ptcl_ic, cosmo, conf


def train_step(tgt, so_params, pmwd_params, learning_rate, opt_state):
    ptcl_ic, cosmo, conf = _init_pmwd(pmwd_params)

    # loss and grad
    obj_valgrad = jax.value_and_grad(obj, argnums=2)
    loss, grad = obj_valgrad(tgt, ptcl_ic, so_params, cosmo, conf)

    # average over global devices
    loss = jnp.expand_dims(loss, axis=0)  # for pmap
    loss, grad = _global_mean(loss, grad)

    # optimize
    optimizer = optax.adam(learning_rate=learning_rate)
    updates, opt_state = optimizer.update(grad, opt_state, so_params)
    so_params = optax.apply_updates(so_params, updates)

    return so_params, loss, opt_state


def visins(fn_root, tgt, so_params, pmwd_params, mesh_shape=None):
    """Util for visually inspection during/after training."""
    # run pmwd with given params
    ptcl_ic, cosmo, conf = _init_pmwd(pmwd_params)
    cosmo = cosmo.replace(so_params=so_params)
    _, obsvbl = nbody(ptcl_ic, None, cosmo, conf)
    ptcl = obsvbl[0]

    # get the target ptcl
    pos_t, vel_t = tgt
    disp_t = pos_t - ptcl.pmid * conf.cell_size
    ptcl_t = Particles(conf, ptcl.pmid, disp_t, vel_t)

    # get the density fields
    if mesh_shape is None:  # default to 2x the mesh in pmwd sim
        cell_size = conf.cell_size / 2
        mesh_shape = tuple(2 * ms for ms in conf.mesh_shape)
    else:  # float or int
        cell_size = conf.ptcl_spacing / mesh_shape
        mesh_shape = tuple(round(mesh_shape * s) for s in conf.ptcl_grid_shape)
    dens, dens_t = (scatter(p, conf, mesh=jnp.zeros(mesh_shape, dtype=conf.float_dtype),
                            val=1, cell_size=cell_size) for p in (ptcl, ptcl_t))

    ### power spectra
    k, ps, N = powspec(dens, cell_size)
    ps = ps.real
    k, ps_t, N = powspec(dens_t, cell_size)
    ps_t = ps_t.real
    k, ps_cross, N = powspec(dens, cell_size, g=dens_t)
    ps_cross = ps_cross.real

    # check the correlation coefficient and ratio of auto power
    psr = ps / ps_t
    cc = ps_cross / jnp.sqrt(ps * ps_t)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3), tight_layout=True)
    ax.plot(k, psr, label=r'$P(k)/P_t(k)$')
    ax.plot(k, cc, label=r'corr. coef.')
    ax.set_xscale('log')
    ax.set_xlabel(r'$k$')
    ax.set_xlim(k[0], 1)
    ax.legend()
    fig.savefig(f'{fn_root}_ps.pdf')

    ### TODO 2d density plot

