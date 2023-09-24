#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -n 1                      # Number of cores (-n)
#SBATCH -N 1                      # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-10:00                # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --partition=seas_gpu,seas_compute
#SBATCH --mem-per-gpu=32G
#SBATCH --constraint="a100"
#SBATCH -o ./output/slurm.%N.%j.out # STDOUT
#SBATCH -e ./output/slurm.%N.%j.err # STDERR
set -e
set -u
python optimization_script.py $1 $2 $3 $4 $5
