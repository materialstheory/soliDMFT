#!/bin/bash
#SBATCH --job-name="svo-test"
#SBATCH --time=1:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=36
#SBATCH --account=eth3
#SBATCH --ntasks-per-core=1
#SBATCH --constraint=mc
#SBATCH --partition=normal
#SBATCH --output=out.%j
#SBATCH --error=err.%j

#======START=====

srun shifter run --mpi materialstheory/triqs python /apps/ethz/eth3/dmatl-theory-git/uni-dmft/run_dmft.py

# or with self build container:
# srun shifter run --mpi load/library/triqs_vasp_csc python /apps/ethz/eth3/dmatl-theory-git/uni-dmft/run_dmft.py
#=====END====
