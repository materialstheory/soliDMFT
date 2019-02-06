#!/bin/bash
#SBATCH --job-name="lno-test"
#SBATCH --time=1:00:00
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=36
#SBATCH --account=eth3
#SBATCH --ntasks-per-core=1
#SBATCH --constraint=mc
#SBATCH --partition=normal
#SBATCH --output=out.%j
#SBATCH --error=err.%j

#======START=====

srun shifter run --mpi materialstheory/triqs python run_dmft.py

#=====END====
