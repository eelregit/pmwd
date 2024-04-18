"""Generate the csv file of scaled Sobol parameters for convenience."""
import numpy as np
from pmwd.sto.sample import scale_Sobol

sobol = scale_Sobol()

header = ','.join(['index', 'box size [Mpc]', 'offset Delta_a',
                   'A_s_1e9', 'n_s', 'Omega_m', 'Omega_b',
                   'Omega_k', 'h', 'softening length'])
fmt = ['%d', '%.0f', '%.4f',
       '%.2f', '%.2f', '%.2f', '%.3f',
       '%.2f', '%.2f', '%.3f']

np.savetxt('scaled_sobol_params.csv',
           np.hstack((np.arange(sobol.shape[0]).reshape(-1, 1), sobol)),
           delimiter=',', header=header, fmt=fmt, comments='')
