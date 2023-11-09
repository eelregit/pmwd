from functools import partial

from PIL import Image, ImageFont, ImageDraw
import jax
import jax.numpy as jnp
from jax.example_libraries.optimizers import adam
import matplotlib.pyplot as plt

from pmwd import (
    Configuration,
    SimpleLCDM,
    boltzmann,
    white_noise, linear_modes,
    lpt,
    nbody,
    scatter,
)
from pmwd.vis_util import simshow



def model(modes, cosmo, conf):
    modes = linear_modes(modes, cosmo, conf)
    ptcl, obsvbl = lpt(modes, cosmo, conf)
    ptcl, obsvbl = nbody(ptcl, obsvbl, cosmo, conf)
    dens = scatter_2d(ptcl, conf)
    return dens


def scatter_2d(ptcl, conf):
    dens = jnp.zeros(tuple(2*s for s in conf.mesh_shape), dtype=conf.float_dtype)
    dens = scatter(ptcl, conf, mesh=dens, val=1, cell_size=conf.cell_size / 2)
    return dens.sum(axis=2)


pmwdshow = partial(simshow, figsize=(2.7, 1.6), cmap='cividis', colorbar=False,
                   interpolation='none')
modeshow = partial(simshow, figsize=(3.2, 1.6), colorbar=True, interpolation='none')


def obj(tgt, modes, cosmo, conf):
    dens = model(modes, cosmo, conf)
    return (dens - tgt).var() / tgt.var()

obj_valgrad = jax.value_and_grad(obj, argnums=1)


def optim(tgt, modes, cosmo, conf, iters=100, lr=0.1):
    init, update, get_params = adam(lr)
    state = init(modes)

    def step(step, state, tgt, cosmo, conf):
        modes = get_params(state)
        value, grads = obj_valgrad(tgt, modes, cosmo, conf)
        state = update(step, grads, state)
        return value, state

    tgt = jnp.asarray(tgt)
    for i in range(iters):
        value, state = step(i, state, tgt, cosmo, conf)

    modes = get_params(state)
    return value, modes


text = 'pmwd'
font = ImageFont.truetype('../../nova/NovaRoundSlim-BookOblique.ttf', 32)

ptcl_spacing = 10.
ptcl_grid_shape = (16, 27, 16)
mesh_shape = (32, 54, 32)
im_shape = (64, 108)
xy = (52, 28)

im = Image.new('L', im_shape[::-1], 255)
draw = ImageDraw.Draw(im)
draw.text(xy, text, font=font, anchor='mm')

# normalize the image to make the target
im_tgt = 1 - jnp.asarray(im) / 255
im_tgt *= jnp.prod(jnp.array(ptcl_grid_shape)) / im_tgt.sum()


conf = Configuration(ptcl_spacing, ptcl_grid_shape, mesh_shape)
cosmo = SimpleLCDM(conf)
seed = 0
modes = white_noise(seed, conf, real=True)
cosmo = boltzmann(cosmo, conf)


plt.style.use('adjoint.mplstyle')
plt.rcParams['savefig.pad_inches'] = 0

ptcl, obsvbl = lpt(linear_modes(modes, cosmo, conf), cosmo, conf)
fig, _ = pmwdshow(scatter_2d(ptcl, conf))
fig.savefig('optim_ini.pdf')
plt.close(fig)

ptcl, obsvbl = nbody(ptcl, obsvbl, cosmo, conf)
fig, _ = pmwdshow(scatter_2d(ptcl, conf))
fig.savefig('optim_0.pdf')
plt.close(fig)

fig, _ = modeshow(modes.mean(axis=2), cmap='RdBu_r', vmin=-1.2, vmax=1.2)
fig.savefig('optim_modes_mean_0.pdf')
plt.close(fig)

fig, _ = modeshow(modes.std(axis=2), cmap='viridis', vmin=0.2, vmax=1.8)
fig.savefig('optim_modes_std_0.pdf')
plt.close(fig)

ptcl, obsvbl = nbody(ptcl, obsvbl, cosmo, conf, reverse=True)
fig, _ = pmwdshow(scatter_2d(ptcl, conf))
fig.savefig('optim_rev.pdf')
plt.close(fig)


for iters in [10, 100, 1000]:
    loss, modes_optim = optim(im_tgt, modes, cosmo, conf, iters=iters)
    print(f'{iters} iters: {loss}')

    fig, _ = pmwdshow(model(modes_optim, cosmo, conf))
    fig.savefig(f'optim_{iters}.pdf')
    plt.close(fig)

    fig, _ = modeshow(modes_optim.mean(axis=2), cmap='RdBu_r', vmin=-1.2, vmax=1.2)
    fig.savefig(f'optim_modes_mean_{iters}.pdf')
    plt.close(fig)

    fig, _ = modeshow(modes_optim.std(axis=2), cmap='viridis', vmin=0.2, vmax=1.8)
    fig.savefig(f'optim_modes_std_{iters}.pdf')
    plt.close(fig)
