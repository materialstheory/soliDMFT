#!/bin/bash
#SBATCH --job-name="lno-scf"
#SBATCH --time=2:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=36
#SBATCH --account=eth3
#SBATCH --ntasks-per-core=1
#SBATCH --constraint=mc
#SBATCH --partition=normal
#SBATCH --output=out.%j
#SBATCH --error=err.%j

#======START=====

srun shifter run --mpi load/library/triqs_vasp_csc vasp_std

#=====END====
