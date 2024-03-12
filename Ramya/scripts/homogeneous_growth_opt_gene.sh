#!/bin/bash
#SBATCH -J homoggrowth_opt_gene     # Job name
#SBATCH -c 4                # Number of cores (-c)
#SBATCH --mem=32000         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -t 0-40:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu         # Partition to submit to
#SBATCH --gres=gpu:1        # Request 1 GPU "generic resource"
#SBATCH --constraint=a100   # Request a100 GPU
#SBATCH -o out_%j.out       # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e err_%j.err       # File to which STDERR will be written, %j inserts jobid

# load modules
module load python

# activate environment
mamba activate jax_py310_new

# run code
cd /n/home10/rdeshpande/morphogenesis/jax-morph/Ramya/scripts
python homogeneous_growth_opt_gene.py $1 $2 $3 $4
