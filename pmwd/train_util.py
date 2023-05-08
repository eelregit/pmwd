import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial
from torch.utils.data import Dataset
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from pmwd.configuration import Configuration
from pmwd.cosmology import Cosmology
from pmwd.boltzmann import boltzmann
from pmwd.modes import white_noise, linear_modes
from pmwd.lpt import lpt
from pmwd.nbody import nbody
from pmwd.scatter import scatter
from pmwd.particles import Particles, ptcl_rpos
from pmwd.pm_util import rfftnfreq
from pmwd.io_util import read_gadget_hdf5
from pmwd.spec_util import powspec


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
        softening_length = sobol[8],
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


def read_g4data(sims_dir, sobol_ids, snap_ids, fn_sobol):
    data = {}
    def load_sobol(sobo):
        data[sobo] = {}
        sobol = scale_Sobol(fn_sobol, sobo)
        for snap in snap_ids:
            snap_file = os.path.join(sims_dir, f'{sobo:03}',
                                     'output', f'snapshot_{snap:03}')
            pos, vel, a = read_gadget_hdf5(snap_file)
            data[sobo][snap] = (pos, vel, a, sobo, sobol)
    Parallel(n_jobs=min(8, len(sobol_ids)), prefer='threads', require='sharedmem')(
        delayed(load_sobol)(sobo) for sobo in sobol_ids)
    return data


class G4snapDataset(Dataset):

    def __init__(self, sims_dir, sobol_ids=None, sobols_edge=None,
                 snap_ids=None, snaps_per_sim=121, fn_sobol='sobol.txt'):
        self.sims_dir = sims_dir

        if sobol_ids is None:
            sobol_ids = np.arange(*sobols_edge)
        self.sobol_ids = sobol_ids

        if snap_ids is not None:
            self.snap_ids = snap_ids
            self.snaps_per_sim = len(snap_ids)
        else:
            self.snap_ids = np.arange(snaps_per_sim)
            self.snaps_per_sim = snaps_per_sim

        self.n_sims = len(sobol_ids)
        self.n_snaps = self.n_sims * self.snaps_per_sim

        self.data = read_g4data(sims_dir, self.sobol_ids, self.snap_ids, fn_sobol)

    def __len__(self):
        return self.n_snaps

    def __getitem__(self, idx):
        sobol_id = self.sobol_ids[idx // self.snaps_per_sim]
        snap_id = self.snap_ids[idx % self.snaps_per_sim]

        return self.data[sobol_id][snap_id]


def _loss_dens_mse(dens, dens_t):
    return jnp.mean((dens - dens_t)**2)

def _loss_disp_mse(ptcl, ptcl_t):
    return jnp.mean((ptcl.disp - ptcl_t.disp)**2)

def _loss_vel_mse(ptcl, ptcl_t):
    return jnp.mean((ptcl.vel - ptcl_t.vel)**2)


@jax.custom_vjp
def _scale_wmse(kvec, f, g):
    # mse of two fields in Fourier space, uniform weights
    k2 = sum(k**2 for k in kvec)
    d = f - g
    loss = jnp.sum(jnp.where(k2 != 0, jnp.abs(d)**2 / k2**1.5, 0)
                   ) / jnp.array(d.shape).prod()
    return jnp.log(loss), (loss, k2, d)

def _scale_wmse_fwd(kvec, f, g):
    loss, res = _scale_wmse(kvec, f, g)
    return loss, res

def _scale_wmse_bwd(res, loss_cot):
    loss, k2, d = res
    d_shape = d.shape
    abs_valgrad = jax.value_and_grad(jnp.abs)
    d, d_grad = jax.vmap(abs_valgrad)(d.ravel())
    d = d.reshape(d_shape)
    d_grad = d_grad.reshape(d_shape)

    loss_cot /= loss
    f_cot = loss_cot * jnp.where(k2 != 0, 2 * d * d_grad / k2**1.5, 0
                                 ) / jnp.array(d_shape).prod()
    return None, f_cot, None

_scale_wmse.defvjp(_scale_wmse_fwd, _scale_wmse_bwd)


def loss_func(ptcl, tgt, conf, mesh_shape=4):

    # get the target ptcl
    pos_t, vel_t = tgt
    disp_t = pos_t - ptcl.pmid * conf.cell_size
    ptcl_t = Particles(conf, ptcl.pmid, disp_t, vel_t)

    # get the density fields for the loss
    if mesh_shape is None:  # 2x the mesh in pmwd sim
        cell_size = conf.cell_size / 2
        mesh_shape = tuple(2 * ms for ms in conf.mesh_shape)
    else:  # float or int
        cell_size = conf.ptcl_spacing / mesh_shape
        mesh_shape = tuple(round(mesh_shape * s) for s in conf.ptcl_grid_shape)
    dens, dens_t = (scatter(p, conf, mesh=jnp.zeros(mesh_shape, dtype=conf.float_dtype),
                            val=1, cell_size=cell_size) for p in (ptcl, ptcl_t))
    dens_k = jnp.fft.rfftn(dens)
    dens_t_k = jnp.fft.rfftn(dens_t)
    kvec_dens = rfftnfreq(mesh_shape, cell_size, dtype=conf.float_dtype)

    # get the disp from particles' grid Lagrangian positions
    disp, disp_t = (ptcl_rpos(p, Particles.gen_grid(p.conf), p.conf)
                    for p in (ptcl, ptcl_t))
    disp_k = jnp.fft.rfftn(disp, axes=range(-3, 0))
    disp_t_k = jnp.fft.rfftn(disp_t, axes=range(-3, 0))
    kvec_disp = rfftnfreq(conf.ptcl_grid_shape, conf.ptcl_spacing, dtype=conf.float_dtype)

    loss = 0
    loss += _scale_wmse(kvec_dens, dens_k, dens_t_k)
    loss += _scale_wmse(kvec_disp, disp_k, disp_t_k)
    # loss += _loss_dens_mse(dens, dens_t)
    # loss += _loss_disp_mse(ptcl, ptcl_t)

    return loss


def obj(tgt, ptcl_ic, so_params, cosmo, conf):
    cosmo = cosmo.replace(so_params=so_params)
    _, obsvbl = nbody(ptcl_ic, None, cosmo, conf)
    loss = loss_func(obsvbl[0], tgt, conf)
    return loss


@partial(jax.pmap, axis_name='global', in_axes=(0, None), out_axes=None)
def _global_mean(loss, grad):
    loss = jax.lax.pmean(loss, axis_name='global')
    grad = jax.lax.pmean(grad, axis_name='global')
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


def visins(tgt, so_params, pmwd_params, fn_root=None, mesh_shape=4):
    """Util for visual inspection during/after training."""
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
    if mesh_shape is None:  # 2x the mesh in pmwd sim
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

    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.6), tight_layout=True)
    ax.plot(k, psr, label=r'$P(k)/P_t(k)$')
    ax.plot(k, cc, label=r'corr. coef.')
    ax.set_xscale('log')
    ax.set_xlabel(r'$k$')
    ax.set_xlim(k[0], k[-1])
    ax.set_ylim(0.9, 1.1)
    ax.legend()

    if fn_root is not None:
        fig.savefig(f'{fn_root}_ps.pdf')

    return fig
