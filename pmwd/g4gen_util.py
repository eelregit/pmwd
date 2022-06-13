import numpy as np
from collections import OrderedDict
from scipy.stats.qmc import Sobol
import os

from pmwd.configuration import Configuration
from pmwd.cosmology import Cosmology
from pmwd.boltzmann import boltzmann
from pmwd.lpt import white_noise, lpt
from pmwd.io_util import writeGadgetHDF5


class ParamGenerator:
    """To sample parameters and generate files needed for running Gadget4.
    """

    def __init__(self, m=9, ptcl_grid_shape=128, num_snapshot=64):
        """
        Sample parameters using a Sobol sequence for quasi Monte Carlo.
        Note that the Sobol sequence is not scrambled, i.e. a definite sequence is always used.

        Parameters
        ----------
        m : int
            2**m sample points are generated.
        ptcl_grid_shape : int
            1D ptcl grid shape for the cube.
        """
        self.num_sims = 2**m

        ### raw setup ranges to sample
        range_dic = OrderedDict([
            # PM
            ('0: log mesh-to-ptcl shape ratio', [np.log(1), np.log(4)]),
            ('1: log cell size', [-np.log(5), np.log(5)]),
            ('2: log number time step', [np.log10(10), np.log10(1000)]),
            # Gadget4
            ('3: log softening length to ptcl spacing ratio', [-np.log(50), -np.log(20)]),
            # both PM & Gadget4
            ('4: snapshot (positive) offset in delta ln(a)', [0, np.log(2)/16]),
            ('5: ln(A_s_1e9)', [np.log(1), np.log(4)]),
            ('6: ln(n_s)', [np.log(0.75), np.log(1.25)]),
            ('7: ln(Omega_m)', [-np.log(5), -np.log(2)]),
            ('8: log Omega_b-to-Omega_m ratio', [-np.log(16), -np.log(2)]),
            ('9: ln(h)', [-np.log(2), np.log(1)]),
            ('10: ln(1 - Omega_k) (related to SU)', [-np.log(2), np.log(2)]),
        ])
        ranges = np.array(list(range_dic.values()))
        rands = Sobol(d=len(range_dic), scramble=False).random_base2(m=m)
        sample = rands * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]

        ### convert to parameters in pmwd and Gadget
        self.pars = OrderedDict()

        # cosmological parameters
        self.pars['A_s_1e9'] = np.exp(sample[:, 5])
        self.pars['n_s'] = np.exp(sample[:, 6])
        self.pars['Omega_m'] = np.exp(sample[:, 7])
        self.pars['Omega_b'] = np.exp(sample[:, 8]) * self.pars['Omega_m']
        self.pars['h'] = np.exp(sample[:, 9])
        self.pars['Omega_k'] = 1 - np.exp(sample[:, 10])

        # particle and mesh sizes
        self.pars['ptcl grid shape'] = ptcl_grid_shape
        self.pars['mesh shape'] = np.rint(np.exp(sample[:, 0]) *
                                          self.pars['ptcl grid shape']).astype(int)
        self.pars['cell size'] = np.exp(sample[:, 1])

        # time integral
        self.pars['number time step'] = np.power(sample[:, 2], 10)
        self.pars['a_start'] = np.exp(-np.log(16) + sample[:, 4])
        self.pars['a_stop'] = np.exp(np.log(1) + sample[:, 4])
        self.pars['a_nbody_maxstep'] = (self.pars['a_stop'] - self.pars['a_start']
                                        ) / self.pars['number time step']

        # Gadget
        self.pars['softening length ratio'] = np.exp(sample[:, 3])

        # snapshot output
        self.pars['number of snapshots'] = num_snapshot
        self.pars['times of snapshots'] = np.logspace(np.log10(self.pars['a_start']),
                                                      np.log10(self.pars['a_stop']),
                                                      num=self.pars['number of snapshots'],
                                                      axis=-1)

    def print_params(self, i):
        """Print the parameters in a Sobol sample.

        Parameters
        ----------
        i : int
            Index of the sample.
        """
        for k, v in self.pars.items():
            if k == 'times of snapshots':
                print(f'{k:>25} : {v[i]}')
            elif isinstance(v, np.ndarray) and len(v) == self.num_sims:
                print(f'{k:>25} : {v[i]:12g}   in   [{v.min():12g}, {v.max():12g}]')
            else:
                print(f'{k:>25} : {v:12g}')

    def init_conf(self, i):
        """Initialize a pmwd Configuration instance.

        Parameters
        ----------
        i : int
            Index of the sample.
        """
        conf = Configuration(
            cell_size=self.pars['cell size'][i],
            mesh_shape=(self.pars['mesh shape'][i],)*3,
            ptcl_grid_shape=(self.pars['ptcl grid shape'],)*3,
            a_start=self.pars['a_start'][i],
            a_stop=self.pars['a_stop'][i],
            a_nbody_maxstep=self.pars['a_nbody_maxstep'][i],
        )
        return conf

    def init_cosmo(self, i, conf=None):
        """Initialize a pmwd Cosmology instance.

        Parameters
        ----------
        i : int
            Index of the sample.
        """
        if not conf:
            conf = self.init_conf(i)

        cosmo = Cosmology(
            conf=conf,
            A_s_1e9=self.pars['A_s_1e9'][i],
            n_s=self.pars['n_s'][i],
            Omega_m=self.pars['Omega_m'][i],
            Omega_b=self.pars['Omega_b'][i],
            Omega_k_=self.pars['Omega_k'][i],
            h=self.pars['h'][i],
        )
        return cosmo

    def gen_IC(self, file_dir, i, conf=None, cosmo=None, seed=16807):
        """Generate and write the initial condition.

        Parameters
        ----------
        file_dir : str
            The directory to output the files and run Gadget.
        i : int
            Index of the sample.
        conf : Configuration
            Configuration parameters in pmwd.
        cosmo : Cosmology
            Cosmological parameters in pmwd.
        seed : int
            Seed passed to pmwd.lpt.white_noise for pseudo-RNG.
        """
        if not conf:
            conf = self.init_conf(i)
        if not cosmo:
            cosmo = self.init_cosmo(i, conf=conf)

        modes = white_noise(seed, conf)
        cosmo = boltzmann(cosmo, conf)
        ptcl, obsvbl = lpt(modes, cosmo, conf)

        writeGadgetHDF5(os.path.join(file_dir, 'ic'), 1, ptcl, cosmo, conf, conf.a_start)

    def gen_GadgetFiles(self,
                        file_dir,
                        i,
                        conf=None,
                        cosmo=None,
                        temp_config='./templates/Config.sh',
                        temp_param='./templates/param.txt',
                        temp_job='./templates/job.sh'):
        """
        Generate (Config.sh, param.txt, job.sh, outtimes.txt) files
        for Gadget4 dark matter only simulations.

        Parameters
        ----------
        file_dir : str
            The directory to output the files and run Gadget.
        i : int
            Index of the sample.
        conf : Configuration
            Configuration parameters in pmwd.
        cosmo : Cosmology
            Cosmological parameters in pmwd.
        temp_config : str
            The template for generating the Config.sh file.
        temp_param : str
            The template for generating the param.txt file.
        temp_job : str
            The template for generating the job.sh file.
        """
        if not conf:
            conf = self.init_conf(i)
        if not cosmo:
            cosmo = self.init_cosmo(i, conf=conf)

        soften_length = self.pars['softening length ratio'][i] * conf.ptcl_spacing

        with open(temp_config, 'r') as f, \
             open(os.path.join(file_dir, 'Config.sh'), 'w') as fo:
            config = f.read()
            fo.write(config)

        with open(temp_param, 'r') as f, \
             open(os.path.join(file_dir, 'param.txt'), 'w') as fo:
            param = f.read()
            param = param.format(box_size=conf.box_size[0],
                                 Omega_m=cosmo.Omega_m,
                                 Omega_de=cosmo.Omega_de,
                                 Omega_b=cosmo.Omega_b,
                                 h=cosmo.h,
                                 soften_cmv=soften_length,
                                 soften_phy=soften_length*1.5,) # to make it irrelevant ?
            fo.write(param)

        with open(temp_job, 'r') as f, \
             open(os.path.join(file_dir, 'job.sh'), 'w') as fo:
            job = f.read()
            job = job.format(job_index=i)
            fo.write(job)

        # times (scale factors) of output snapshots
        np.savetxt(os.path.join(file_dir, 'outtimes.txt'),
                   self.pars['times of snapshots'][i], fmt='%.15f')

    def deploy(self,
               base_dir,
               seed=16807,
               i_start=0,
               i_end=None,
               temp_config='./templates/Config.sh',
               temp_param='./templates/param.txt',
               temp_job='./templates/job.sh',
               submit_job=True):
        """A setup pipeline of the 2**m Gadget4 simulations.

        Parameters
        ----------
        base_dir : str
            The directory where the simulations are run.
            Each simulation will be in a sub-directory created here.
        seed : int
            Seed passed to pmwd.lpt.white_noise for pseudo-RNG.
        i_start : int
            The start index of the sample in the Sobol sequence (included).
        i_end : int
            The end index of the sample in the Sobol sequence (included).
        temp_config : str
            The template for generating the Config.sh file.
        temp_param : str
            The template for generating the param.txt file.
        temp_job : str
            The template for generating the job.sh file.
        """
        if not i_end:
            i_end = self.num_sims - 1
        elif i_end > self.num_sims - 1:
            raise ValueError('i_end should be smaller than 2**m')

        for i in range(i_start, i_end+1):
            # create the sub-directories for each simulation
            file_dir = os.path.join(base_dir, f'sim_{i:03}')
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)

            # conf, cosmo and IC
            conf = self.init_conf(i)
            cosmo = self.init_cosmo(i, conf=conf)
            self.gen_IC(file_dir, i, conf=conf, cosmo=cosmo, seed=seed)

            # generate the Gadget4 files
            self.gen_GadgetFiles(file_dir, i, conf=conf, cosmo=cosmo,
                                 temp_config=temp_config,
                                 temp_param=temp_param,
                                 temp_job=temp_job)

            # submit the job (compile + run)
            if submit_job:
                os.system(f'cd {file_dir}; sbatch job.sh')
