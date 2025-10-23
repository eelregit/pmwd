import numpy as np


def gen_sobol(filename=None, d=9, m=9, extra=9, seed=55868, seed_max=65536):
    from scipy.stats.qmc import Sobol, discrepancy

    nicer_seed = seed
    if seed is None:
        disc_min = np.inf
        for s in range(seed_max):
            sampler = Sobol(d, scramble=True, seed=s)  # d is the dimensionality
            sample = sampler.random_base2(m)  # m is the log2 of the number of samples
            disc = discrepancy(sample, method='MD')
            if disc < disc_min:
                nicer_seed = s
                disc_min = disc
        print(f'0 <= seed = {nicer_seed} < {seed_max}, minimizes mixture discrepancy = '
                f'{disc_min}')
        # nicer_seed = 55868, mixture discrepancy = 0.016109347957680598

    sampler = Sobol(d, scramble=True, seed=nicer_seed)
    sample = sampler.random(n=2**m + extra)  # extra is the additional testing samples
    if filename is None:
        return sample
    else:
        np.savetxt(filename, sample)


def scale_Sobol(sobol=None, fn='sobol.txt', ind=slice(None)):
    """Scale the Sobol sequence samples, refer to the Table in the paper."""
    if sobol is None:
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
    # the trapezoid shape is given by the product of the mesh shape and cell size
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
