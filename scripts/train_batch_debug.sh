#!/bin/bash
#SBATCH -A m4599_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

export NUGRAPH_DATA=/global/u1/c/cerati/NuGraph
export NUGRAPH_DIR=/global/u1/c/cerati/NuGraph
export NUGRAPH_LOG=/global/u1/c/cerati/NuGraph/logs
srun python scripts/train.py $@
