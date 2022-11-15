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
    return ptcl


def scatter_2d(ptcl, conf):
    dens = jnp.zeros(tuple(2*s for s in conf.mesh_shape), dtype=conf.float_dtype)
    dens = scatter(ptcl, conf, mesh=dens, val=1, cell_size=conf.cell_size / 2)
    return dens.sum(axis=2)


pmwdshow = partial(simshow, figsize=(27/2, 8), cmap='cividis', colorbar=False)


def obj(tgt, modes, cosmo, conf):
    ptcl = model(modes, cosmo, conf)
    dens = scatter_2d(ptcl, conf)
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
font = ImageFont.truetype('../nova/NovaRoundSlim-BookOblique.ttf', 32)

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


ptcl, obsvbl = lpt(linear_modes(modes, cosmo, conf), cosmo, conf)
fig, _ = pmwdshow(scatter_2d(ptcl, conf))
fig.savefig('optim_ini.pdf')
plt.close(fig)

ptcl, obsvbl = nbody(ptcl, obsvbl, cosmo, conf)
fig, _ = pmwdshow(scatter_2d(ptcl, conf))
fig.savefig('optim_0.pdf')
plt.close(fig)

ptcl, obsvbl = nbody(ptcl, obsvbl, cosmo, conf, reverse=True)
fig, _ = pmwdshow(scatter_2d(ptcl, conf))
fig.savefig('optim_rev.pdf')
plt.close(fig)

loss, modes_optim = optim(im_tgt, modes, cosmo, conf, iters=10)
fig, _ = pmwdshow(scatter_2d(model(modes_optim, cosmo, conf), conf))
fig.savefig('optim_10.pdf')
plt.close(fig)

loss, modes_optim = optim(im_tgt, modes, cosmo, conf, iters=100)
fig, _ = pmwdshow(scatter_2d(model(modes_optim, cosmo, conf), conf))
fig.savefig('optim_100.pdf')
plt.close(fig)

loss, modes_optim = optim(im_tgt, modes, cosmo, conf, iters=1000)
fig, _ = pmwdshow(scatter_2d(model(modes_optim, cosmo, conf), conf))
fig.savefig('optim_1000.pdf')
plt.close(fig)
