# Running jobs on CSCS daint

First one has to load the desired docker images with shifter
on daint: https://user.cscs.ch/tools/containers/ . For a public available image
this can be done via:
```
shifter pull materialstheory/triqs
```
if you wish to use your pre-build docker image, for example if one wants to
include VASP one needs to first save the docker image locally (where one has
built it):
```
docker save --output=triqs_vasp_csc.tar triqs_vasp_csc
```
and then upload it to daint and then load it via:
```
shifter load /apps/ethz/eth3/dmatl-theory-git/uni-dmft/Docker/triqs_vasp_csc_image.tar triqs_vasp_csc
```
than one can run it has shown in the example job scripts in this directory.

### one shot job on daint

one shot is quite straight forward. Just get the newest version of these
scripts, go to a working directory and then create job file that looks like
this:
```
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
```
thats it. This line automatically runs the docker image and executes the
`run_dmft.py` script.

### CSC calculations on daint

This requires a bit more scripting to start both VASP and triqs at the same
time. There is a special `vasp_dmft_daint.sh` bash script version in the top dir
of this git repo.

Now the options `-n` controls the number of nodes where triqs runs and `-v` the
number of nodes where VASP runs. Just ask for the highest amount of nodes that
you want to use. (usually the number of triqs nodes)

A slurm job script should look like this:
```
#!/bin/bash
#SBATCH --job-name="svo-csc-test"
#SBATCH --time=4:00:00
#SBATCH --nodes=4
#SBATCH --account=eth3
#SBATCH --constraint=mc
#SBATCH --partition=normal
#SBATCH --output=out.%j
#SBATCH --error=err.%j
#SBATCH --gres=craynetwork:4

# path to run_dmft.py script
SCRIPTDIR=/apps/ethz/eth3/dmatl-theory-git/uni-dmft/run_dmft.py
# Shifter image that is utilized
IMAGE=load/library/triqs_vasp_csc
# number of nodes for triqs
NTRIQS=${SLURM_JOB_NUM_NODES}
# number of nodes for vasp
NVASP=1

bash /apps/ethz/eth3/dmatl-theory-git/uni-dmft/vasp_dmft_daint.sh -n $NTRIQS -v $NVASP -i $IMAGE $SCRIPTDIR
```
Please specify only these `#SBATCH options` to not confuse slurm with the setup.
__Always make sure that `$NTRIQS+$NVASP` is equal the total number of nodes!__
In general I found 1 node for Vasp is in most case enough. Using more than one
node results in a lot of MPI communication, which in turn slows down the
calculation significantly. For a 80 atom unit cell 2 nodes are useful, but for a
20 atom unit cell not at all!
