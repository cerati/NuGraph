#!/bin/bash
#SBATCH -J nugraph_train
#SBATCH -t 12:00:00
#SBATCH -p general
#SBATCH --gres=gpu:1
#SBATCH -q normal
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=64G

source /etc/profile.d/conda.sh
conda activate /net/projects2/fermi2526/conda/nugraph-25-10
which python
srun python scripts/train.py $@
