#!/usr/bin/bash

#SBATCH --job-name=trainso
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --constraint=h100
#SBATCH --mem-per-gpu=125G
#SBATCH --time=0-12

module purge

source ~/mambaforge/bin/activate
srun --cpu-bind=cores  python run_train.py


# Notes:
# - We are using one gpu for one proc, and the proc should use the same gpu
#   throughout the whole computation (i.e. gpu should be binded to the proc).
# - If we set --gpus-per-task=1, then each proc can only see one gpu (i.e.
#   CUDA_VISIBLE_DEVICES is simply 0). And --gpu-bind might be necessary to
#   specify the binding between proc and gpu. But Slurm could have bugs here.
# - If we set --gpus-per-node=<ntasks-per-node>, then each proc can see all gpus
#   on its node. Then we can use CUDA_DEVICE_ORDER and CUDA_VISIBLE_DEVICES to
#   bind the gpu to proc. This is the approach that we currently take.
