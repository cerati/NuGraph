#!/bin/bash
#SBATCH -J exatrkx_train
#SBATCH -t 1440
#SBATCH -p wc_gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -A fwk
#SBATCH -q regular
#SBATCH --cpus-per-task=12
#SBATCH --signal=SIGUSR1@90

srun python scripts/train.py $@
