#!/usr/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --constraint=a100-80gb,ib
#SBATCH --exclusive
#SBATCH --mem=0

module purge

source ~/mambaforge/bin/activate
srun python test_jax_dist.py
