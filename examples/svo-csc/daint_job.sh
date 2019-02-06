#!/bin/bash
#SBATCH --job-name="svo-csc-test"
#SBATCH --time=0:30:00
#SBATCH --nodes=4
#SBATCH --account=eth3
#SBATCH --constraint=mc
#SBATCH --partition=debug
#SBATCH --output=out.%j
#SBATCH --error=err.%j
#SBATCH --gres=craynetwork:4

#======START=====

SCRIPTDIR=/apps/ethz/eth3/dmatl-theory-git/uni-dmft/run_dmft.py
IMAGE=load/library/triqs_vasp_csc
NTRIQS=4
NVASP=1

bash /apps/ethz/eth3/dmatl-theory-git/uni-dmft/vasp_dmft_daint.sh -n $NTRIQS -v $NVASP -i $IMAGE $SCRIPTDIR


#=====END====
