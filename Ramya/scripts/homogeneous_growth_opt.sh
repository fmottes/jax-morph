#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -n 1                      # Number of cores (-n)
#SBATCH -N 1                      # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-20:00                # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --partition=seas_gpu,seas_compute
#SBATCH --mem-per-gpu=32G
#SBATCH --constraint="v100"
#SBATCH -o ../output/slurm.%N.%j.out # STDOUT
#SBATCH -e ../output/slurm.%N.%j.err # STDERR
set -e
set -u

# load modules
module load python

# activate environment
mamba activate jax_py310

# run code
cd /n/home10/rdeshpande/morphogenesis/jax-morph/Ramya/scripts/
python homogeneous_growth_opt.py $1 $2 $3 $4 $5
