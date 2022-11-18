import os

import numpy as np
import matplotlib.pyplot as plt


mesh_shapes = [1, 2]
ptcl_grid_sizes = [2**n for n in range(5, 10)]


number = 1
repeat = 8
exec_times = np.empty((2 + len(mesh_shapes), len(ptcl_grid_sizes)))
fname = 'perf.txt'

if not os.path.exists(fname):
    import timeit

    from pmwd import (
        Configuration,
        SimpleLCDM,
        boltzmann,
        white_noise, linear_modes,
        lpt,
        nbody,
    )


    for i, mesh_shape in enumerate(mesh_shapes):
        for j, ptcl_grid_size in enumerate(ptcl_grid_sizes):
            ptcl_spacing = 1.
            ptcl_grid_shape = (ptcl_grid_size,) * 3
            conf = Configuration(ptcl_spacing, ptcl_grid_shape, mesh_shape=mesh_shape)
            print(f'ptcl_grid_size={ptcl_grid_size}, mesh_shape={mesh_shape}:')

            cosmo = SimpleLCDM(conf)

            seed = 0
            modes = white_noise(seed, conf)


            boltzmann(cosmo, conf)  # do not overwrite cosmo
            t = timeit.repeat('boltzmann(cosmo, conf).h.block_until_ready()',
                              repeat=repeat, number=number, globals=globals())
            t = min(t) / number
            exec_times[0, j] = t
            print(f'boltz {t}')
            cosmo = boltzmann(cosmo, conf)  # overwrite cosmo


            ptcl, obsvbl = lpt(linear_modes(modes, cosmo, conf), cosmo, conf)  # do not overwrite modes
            t = timeit.repeat('lpt(linear_modes(modes, cosmo, conf), cosmo, conf)[0].vel.block_until_ready()',
                              repeat=repeat, number=number, globals=globals())
            t = min(t) / number
            print(f'  lpt {t}')
            exec_times[1, j] = t


            nbody(ptcl, obsvbl, cosmo, conf)  # do not overwrite ptcl and obsvbl
            t = timeit.repeat('nbody(ptcl, obsvbl, cosmo, conf)[0].vel.block_until_ready()',
                              repeat=repeat, number=number, globals=globals())
            t = min(t) / number / (1 + conf.a_nbody_num)
            print(f'nbody {t}')
            exec_times[2+i, j] = t

    np.savetxt(fname, exec_times)

exec_times = np.loadtxt(fname)


plt.style.use('font.mplstyle')

fig, ax = plt.subplots(figsize=(4, 3))

ax.loglog(ptcl_grid_sizes, exec_times[3], c='k', ls='-', label='$N$-body per step, $2^3$x mesh')
ax.loglog(ptcl_grid_sizes, exec_times[2], c='k', ls='--', label='$N$-body per step, $1$x mesh')
ax.loglog(ptcl_grid_sizes, exec_times[1], c='C0', ls='-.', label='2LPT')
ax.loglog(ptcl_grid_sizes, exec_times[0], c='gray', ls=':', label='growth', zorder=1)
ax.set_xticks(ptcl_grid_sizes, labels=[f'${s}^3$' for s in ptcl_grid_sizes])
ax.tick_params(axis='x', which='minor', bottom=False, top=False)
ax.set_xlim(min(ptcl_grid_sizes), max(ptcl_grid_sizes))
ax.set_xlabel('ptcl grid size')
ax.set_ylabel('wall time [sec]')
ax.legend()

fig.savefig('perf.pdf')
plt.close(fig)
