import numpy as np
import jax.numpy as jnp
import h5py
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family':'serif','serif':['Palatino']})

import imageio.v2 as imageio
from gaepsi2.painter import paint

import nvix.camera as nvc
import nvix.utils as nvu
import nvix.shutter as nvs
from pmwd.vis_util import CosmicWebNorm

f = h5py.File('data/time_evo.h5', 'r')
fo = h5py.File('data/pmwd_optim_3000_lr_0.02_b1_0.9.h5', 'r')

# setup the camera
# model box
box = np.array([[0, 160],
                [0, 270],
                [0, 160]])
box_size = box[:, 1] - box[:, 0]

# the model matrix to shift the box to the origin
M_model = np.eye(4)
M_model[:3, 3] = -box.mean(axis=1)

# the camera is on z-axis and facing the origin
eye = np.array([0, 0, box_size[-1]*8])
target = np.array([0, 0, 0])
D = np.abs(eye[-1])

# up direction, fov and aspect
up = np.array([-1, 0, 0])
fov_up = np.arctan2(box_size[0]/2, D) * 2 * 1.8
aspect = box_size[1] / box_size[0]

# distance of camera to near and far planes
near = D - box_size[-1]
far = D + box_size[-1] * 2

cam = nvc.Camera(eye, target, up, fov_up, aspect, near, far)

# window size, i.e. image resolution
res = round(box_size[1] * 3)
window = (res, round(res / aspect))

# smoothing length
sml = 5.

# get the norm for plotting
def get_norm(X):
    X = nvs.shutter(X.T, cam, M_model=M_model, window=window)
    X = paint(np.array(X).T, np.full(X.shape[1], sml), np.ones(X.shape[1]), window)[0]
    norm = CosmicWebNorm(X)
    return norm

webnorm = get_norm(f[f'fw_{len(f)//2-1}'][:])

def plot_snap(X, cam, sml, a=None, it=None, norm='linear', vmin=0, vmax=None, fn=None, cmap='cividis',
              text=None):
    fig, ax = plt.subplots(1, 1, figsize=(window[0]/100, window[1]/100), dpi=100)
    plt.subplots_adjust(0, 0, 1, 1)
    ax.axis('off')

    # draw the box
    xs, ys, zs = np.meshgrid(*box)
    vs = np.vstack([_.ravel() for _ in (xs, ys, zs)])
    vs = nvs.shutter(vs, cam, M_model=M_model, window=window)
    nvu.draw_box(vs, ax)

    X = nvs.shutter(X.T, cam, M_model=M_model, window=window)
    X = paint(np.array(X).T, np.full(X.shape[1], sml), np.ones(X.shape[1]), window)[0]

    if vmax is None:
        ax.imshow(X.T, cmap=cmap, origin='lower', norm=webnorm,
                  interpolation='lanczos', interpolation_stage='rgba')
    else:
        ax.imshow(X.T, cmap=cmap, origin='lower', norm=norm, vmin=0, vmax=vmax,
                  interpolation='lanczos', interpolation_stage='rgba')

    if a is not None:
        ax.text(0.7, 0.08, f'Scale factor a = {a:.3f}', c='w', transform=ax.transAxes, ha='left', va='top', fontsize=16)
    if it is not None:
        ax.text(0.6, 0.08, f'Optimization iteration: {it+1:d}', c='w', transform=ax.transAxes, ha='left', va='top', fontsize=16)
    if text == 'forward':
        ax.text(0.5, 0.95,
                'Forward time evolution', fontsize=16,
                c='w', transform=ax.transAxes, ha='center', va='center')
    if text == 'reverse':
        ax.text(0.5, 0.95,
                'Reverse time evolution', fontsize=16,
                c='w', transform=ax.transAxes, ha='center', va='center')
    if text == 'optim':
        ax.text(0.5, 0.95,
                'Optimizing initial conditions', fontsize=16,
                c='w', transform=ax.transAxes, ha='center', va='center')

    if fn is not None:
        fig.savefig(fn)
    else:
        plt.show()
    plt.close()


#### Forward and reverse

fpt = 1  # frames per snapshot
snaps = len(f) // 2
frames = snaps * fpt
fps = 30

# generate the plots
a1 = 180
angles_f = np.linspace(0, a1, frames)
angles_r = np.linspace(a1, a1*2, frames)
for i, angle in enumerate(tqdm(angles_f)):
    cam_ = nvc.rotate(cam, 'y', angle)
    dset = f[f'fw_{i//fpt}']
    plot_snap(dset[:], cam_, sml, dset.attrs['a'], fn=f'tmp_time/fw_{i}.png', text='forward')
for i, angle in enumerate(tqdm(angles_r)):
    cam_ = nvc.rotate(cam, 'y', angle)
    dset = f[f're_{i//fpt}']
    plot_snap(dset[:], cam_, sml, dset.attrs['a'], fn=f'tmp_time/re_{i}.png', text='reverse')

# make the video
writer = imageio.get_writer('videos/time_evolution.mp4', fps=fps, format='FFMPEG', codec='libx265', output_params=['-crf', '30'])
for i in range(frames):  # forward
    writer.append_data(imageio.imread(f'tmp_time/fw_{i}.png'))
for _ in range(fps):  # pause for 1 sec
    writer.append_data(imageio.imread(f'tmp_time/fw_{frames-1}.png'))
for i in range(frames):  # reverse
    writer.append_data(imageio.imread(f'tmp_time/re_{i}.png'))
for _ in range(fps):  # pause for 1 sec
    writer.append_data(imageio.imread(f'tmp_time/re_{frames-1}.png'))
writer.close()


#### Optim

frames = 900
cycles = 1
parts = 3
fpp = frames // parts
fps = 30

its = []

for a, b in [(0, 10), (10, 70)]:
    fpi = fpp // (b - a)
    for i in range(a, b):
        its += [i] * fpi
# the rest
its_ = np.rint(np.linspace(b, 999, fpp)).astype(int)
its = np.concatenate((its, its_))

t = np.arange(frames) / frames
da = np.abs(np.sin(t * 4*np.pi * cycles))
angles = da.cumsum()
angles = angles / angles[-1] * cycles * 360


for i in tqdm(range(frames)):
    X = fo[f'{its[i]}'][:]
    cam_ = nvc.rotate(cam, 'y', angles[i])
    plot_snap(X, cam_, sml, it=its[i], fn=f'tmp_optim/op_{i}.png', text='optim')

# make the video
writer = imageio.get_writer('videos/pmwd_optim.mp4', fps=fps, format='FFMPEG', codec='libx265', output_params=['-crf', '30'])
for i in tqdm(range(frames)):
    writer.append_data(imageio.imread(f'tmp_optim/op_{i}.png'))
writer.close()


#### make the combined video
writer = imageio.get_writer('videos/pmwd_demo.mp4', fps=fps)

frames = frames = snaps * fpt
for i in range(frames):  # forward
    writer.append_data(imageio.imread(f'tmp_time/fw_{i}.png'))
for _ in range(fps):  # pause
    writer.append_data(imageio.imread(f'tmp_time/fw_{frames-1}.png'))
for i in range(frames):  # reverse
    writer.append_data(imageio.imread(f'tmp_time/re_{i}.png'))
for i in range(frames):  # fast forwrad
    if i % 2 == 0:
        writer.append_data(imageio.imread(f'tmp_time/ff_{i}.png'))
for _ in range(fps):  # pause
    writer.append_data(imageio.imread(f'tmp_time/ff_{frames-1}.png'))

frames = 1000
for i in tqdm(range(frames)):
    writer.append_data(imageio.imread(f'tmp_optim/op_{i}.png'))

writer.close()
