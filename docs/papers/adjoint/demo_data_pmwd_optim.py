import numpy as np
import jax
import jax.numpy as jnp
from tqdm.notebook import tqdm
import optax
from PIL import Image, ImageFont, ImageDraw
import h5py

from pmwd import (
    Configuration,
    SimpleLCDM,
    boltzmann,
    white_noise, linear_modes,
    lpt,
    nbody,
    scatter,
)


ptcl_spacing = 10.
ptcl_grid_shape = (16, 27, 16)
mesh_shape = (32, 54, 32)

text = 'pmwd'
font = ImageFont.truetype('../../nova/NovaRoundSlim-BookOblique.ttf', 32)
im_shape = (64, 108)
im = Image.new('L', im_shape[::-1], 255)
draw = ImageDraw.Draw(im)
draw.text((52, 28), text, font=font, anchor='mm')
im_tgt = 1 - jnp.asarray(im) / 255  # normalize the image to make the target
im_tgt *= jnp.prod(jnp.array(ptcl_grid_shape)) / im_tgt.sum()


def model(modes, cosmo, conf):
    modes = linear_modes(modes, cosmo, conf)
    ptcl, obsvbl = lpt(modes, cosmo, conf)
    ptcl, obsvbl = nbody(ptcl, obsvbl, cosmo, conf)
    return ptcl


def scatter_2d(ptcl, conf):
    dens = []
    # hack onto a 2x finer mesh
    for idx in [(0, 1), (2, 1)]:
        conf_ = conf.replace(
            ptcl_grid_shape=[ptcl_grid_shape[i] for i in idx],
            mesh_shape=im_shape
        )
        ptcl_ = ptcl[:, idx]  # project to first 2 dim
        ptcl_ = ptcl_.replace(
            conf=conf_,
            pmid=ptcl_.pmid*2
        )
        dens.append(scatter(ptcl_, conf_, val=1))
    return dens


def obj(tgt, modes, cosmo, conf):
    ptcl = model(modes, cosmo, conf)
    dens = scatter_2d(ptcl, conf)
    loss = 0.
    for den in dens:
        loss += (den - tgt).var() / tgt.var()
    return loss, ptcl


obj_valgrad = jax.value_and_grad(obj, argnums=1, has_aux=True)


def save_pos(f, i, ptcl, conf):
    pos = np.asarray(ptcl.pos())
    _ = f.create_dataset(f'{i}', data=pos)


def optim(tgt, modes, cosmo, conf, optimizer, f=None, iters=100):
    state = optimizer.init(modes)

    def step(i, modes, state, tgt, cosmo, conf):
        (loss, ptcl), grads = obj_valgrad(tgt, modes, cosmo, conf)
        updates, state = optimizer.update(grads, state, modes)
        modes = optax.apply_updates(modes, updates)
        return modes, state, loss, ptcl

    tgt = jnp.asarray(tgt)
    for i in tqdm(range(iters)):
        modes, state, loss, ptcl = step(i, modes, state, tgt, cosmo, conf)
        if f is not None:
            save_pos(f, i, ptcl, conf)

    return loss, modes


conf = Configuration(ptcl_spacing, ptcl_grid_shape, mesh_shape)
cosmo = SimpleLCDM(conf)
cosmo = boltzmann(cosmo, conf)
seed = 0
modes = white_noise(seed, conf)


iters = 3000

lr, b1 = 0.02, 0.9
optimizer = optax.adam(learning_rate=lr, b1=b1)
optm_str = f'lr_{lr:g}_b1_{b1:g}'

f = h5py.File(f'data/pmwd_optim_{iters}_{optm_str}.h5', 'w')
loss, modes_optim = optim(im_tgt, modes, cosmo, conf, optimizer, f, iters=iters)
f.close()
