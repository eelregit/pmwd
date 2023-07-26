#!/usr/bin/bash

#SBATCH --job-name=g4-{job_index}
#SBATCH --output=%x-%j.out
#SBATCH --partition=ccm
#SBATCH --constraint=rome
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=7-00:00:00


module --force purge
module load modules/2.0-20220630
module load slurm
module load openblas
module load gcc/11.2.0 openmpi/4.0.7
module load gsl/2.7
module load fftw/3.3.10
module load hdf5/1.12.1


cd $HOME/gadget4
export SYSTYPE=Generic-gcc
make -j 64 DIR=$SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR


hostname; pwd; date

time -p mpirun ./Gadget4 param.txt

date
