"""A script to setup directories and files etc for Gadget-4 simulations."""
import os
import sys
import jax

import numpy as np

from pmwd.train_util import gen_ic
from pmwd.io_util import write_gadget_hdf5


def gen_g4files(sim_dir, i, fn_sobol='sobol.txt',
                tpl_config='Config.sh', tpl_param='param.txt', tpl_job='job.sh'):
    """
    Generate input files with Sobol i parameters for Gadget4 dark matter only
    simulations.

    Parameters
    ----------
    sim_dir : str
        The directory to output the files and run Gadget.
    tpl_config : str
        The template for generating the Config.sh file.
    tpl_param : str
        The template for generating the param.txt file.
    tpl_job : str
        The template for generating the job.sh file.
    """
    # generate initial condition on GPU, to be consistent with training
    with jax.default_device(jax.devices('gpu')[0]):
        ptcl, cosmo, conf, sobol = gen_ic(i, fn_sobol, re_sobol=True)
    write_gadget_hdf5(os.path.join(sim_dir, 'ic'), conf.a_start, ptcl, cosmo, conf)

    with (open(tpl_config, 'r') as f,
          open(os.path.join(sim_dir, 'Config.sh'), 'w') as fo):
        config = f.read()
        fo.write(config)

    with (open(tpl_param, 'r') as f,
          open(os.path.join(sim_dir, 'param.txt'), 'w') as fo):
        param = f.read()
        param = param.format(
            a_start=conf.a_start,
            a_stop=1 + sobol[1],
            box_size=conf.box_size[0],
            M_g=conf.M * 1e3,
            L_cm=conf.L * 1e2,
            V_cm_s=conf.V * 1e2,
            H_0=conf.H_0,
            G=conf.G,
            Omega_m=cosmo.Omega_m,
            Omega_de=cosmo.Omega_de,
            Omega_b=cosmo.Omega_b,
            h=cosmo.h,
            soften_cmv=sobol[8],
            soften_phy=sobol[8] * 1.5,  # making this irrelevant
        )
        fo.write(param)

    # times (scale factors) of output snapshots
    outtimes = np.linspace(1/16, 1, 121) + sobol[1]
    np.savetxt(os.path.join(sim_dir, 'outtimes.txt'), outtimes, fmt='%.16f')

    # slurm job script
    with (open(tpl_job, 'r') as f,
          open(os.path.join(sim_dir, 'job.sh'), 'w') as fo):
        job = f.read()
        job = job.format(job_index=i)
        fo.write(job)

    print(f'> files in {sim_dir} generated')


if __name__ == "__main__":
    i_start, i_stop = int(sys.argv[1]), int(sys.argv[2])
    base_dir = 'g4sims'

    for i in range(i_start, i_stop):  # not including i_stop
        # create the sub-directories for each simulation
        sim_dir = os.path.join(base_dir, f'{i:03}')
        os.makedirs(sim_dir, exist_ok=True)

        # generate ic and necessary files
        gen_g4files(sim_dir, i)

        # submit the job (compile + run)
        os.system(f'cd {sim_dir} && sbatch job.sh')
