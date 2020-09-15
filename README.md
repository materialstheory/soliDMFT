# soliDMFT
This program allows to perform DFT+DMFT ''one-shot'' and CSC
calculations from h5 archives or VASP input files for multiband systems using
the TRIQS package, in combination with the CThyb solver and SumkDFT from
DFT-tools. Runs with triqs 3.x.x

For all calculations the start script is 'run_dmft.py'.

Written by A. Hampel, M. Merkel, S. Beck, and J. S. Casares from
Materials Theory at ETH Zurich.


## Source code files and their use

- __run_dmft.py:__ main file that runs the calculation and start a CSC flow by
  invoking  `csc_flow_control` or a one shot calculation directly by invoking
  `dmft_cycle` on a given h5 archive
- __read_config.py:__ contains the functions to read the dmft config file. Take a
  look in `read_config_doc.md` for a detailed list of parameters
- __dmft_cycle.py:__ contains the `dmft_cycle` function that run a predefined
  number of DMFT iterations
- __csc_flow.py:__ contains the `csc_flow_control` function to steer the CSC
  calculation and call then ones per DFT+DMFT cycle the `dmft_cycle` function
- __observables.py:__ contains all functions necessary to calculate and write the
  observables during the run, which are stored in a general dictionary: `observables`
- __toolset.py:__ contains several small helper functions


## Introduction to docker

You can get docker via 'sudo apt install docker.io' and then test your docker
installation by running 'docker run hello-world'.

Docker runs containers, which are instances of images. To first construct an image,
see section __CSC calculations locally__ or __Running on CSCS daint__. This image can
then be called with the 'docker run' command, with helpful flags specified in
section __Getting started__.

## Getting started

To start take a look in the `example` directory. There one finds several
examples to run. Best start with the svo-one-shot example. The
`dmft_config.ini` file contains the configuration for the DMFT run, which is
explained in the read\_config method in the main script. The `svo.h5` is the DMFT
input data, which is obtained from projection on localized Wannier functions
(see folder `svo-one-shot/converter`).

Furthermore, there is a `read_config_doc.md` file containing the docstrings from
the main script in a readable format. If one wishes to do CSC calculations the
docker container must contain also a installed VASP version >5.4.4 that
understands the ICHARG=5 flag.

To run the one shot examples one can use the triqs docker images on,
https://hub.docker.com/r/materialstheory/triqs/ , the official ones on
https://hub.docker.com/r/flatironinstitute/triqs/ , or our own Docker image
under `docker` , where VASP is already pre-installed.

then one can run docker from any directory as:  
```
docker run --rm -it -u $(id -u) -v ~/git/d-matl-theory-git/uni-dmft:/work materialstheory/triqs bash
```
go to the example directory inside the running container and the run it via:  
```
mpirun -n 4 python work/run_dmft.py
```
or run it directly via:  
```
docker run --rm -it -u $(id -u) -v ~/git/d-matl-theory-git/uni-dmft:/work materialstheory/triqs bash -c 'cd /work/tests/svo-one-shot/ && python /work/run_dmft.py'
```
the more elaborate version of the Docker container found in this repo in the
folder `Docker` can be best started as:
```
docker run --rm -it --shm-size=4g -e USER_ID=`id -u` -e GROUP_ID=`id -g` -p 8378:8378 -v $PWD:/work -v /mnt/eth-home/git/d-matl-theory-git/uni-dmft:/uni-dmft triqs_vasp_csc bash
```
where the `-e` flags will translate your current user and group id into the
container and make sure writing permissions are correct for the mounted volumes.
Note also the option --shm-size, which increases shared memory size. This is hard
coded in Docker to 64m and is often not sufficient and will produce SIBUS 7 errors
when starting programs with mpirun! (see also [https://github.com/moby/moby/issues/2606](https://github.com/moby/moby/issues/2606)).
The '-v' flags mounts a host directory as the docker directory given after the colon.
This way docker can permanently save data; otherwise, it will restart with clean directories each time.
All the flags are explained in 'docker run --help'.
Moreover, you can start by executing
```
jupyter.sh
```
a jupyter-lab server from the current dir.

### CSC calculations locally

Here one needs a special docker image with vasp included. This can be done by
building the Dockerfile in `/Docker/`:
```
docker build -t triqs_vasp_csc -f Dockerfile_MPICH ./
```
Then start this docker image as done above and go to the directory with all
necessary input files (start with `svo-csc` example). You need a pre-converged
CHGCAR and preferably a WAVECAR, a set of INCAR, POSCAR, KPOINTS and POTCAR
files, the PLO cfg file `plo.cfg` and the usual DMFT input file
`dmft_config.ini`, which specifies the number of ranks for the DFT code and the DFT code executable in the `[dft]` section.

The whole machinery is started by calling `run_dmft.py` as normal. Importantly the flag `csc = True` has to be set in the general section in the config file. Then:
```
mpirun -n 12 /work/run_dmft.py
```
The programm will then run the `csc_flow_control` routine, which starts VASP accordingly by spawning a new child process. After VASP is finished it will run the converter, run the dmft_cycle, and then VASP again until the given
limit of DMFT iterations is reached. This should also work on most HPC systems (tested on slurm with OpenMPI), as the the child mpirun call is performed without the slurm environment variables. This tricks slrum into starting more ranks than it has available. Note, that maybe a slight adaption of the environment variables is needed to make sure VASP is found on the second node. The variables are stored `args` in the function `start_vasp_from_master_node` of the module `csc_flow.py`

One remark regarding the number of iterations per DFT cycle. Since VASP uses a
block Davidson scheme for minimizing the energy functional not all eigenvalues
of the Hamiltonian are updated simultaneously therefore one has to make several
iterations before the changes from DMFT in the charge density are completely
considered. The default value are __6__ DFT iterations, which is very
conservative, and can be changed by changing the config parameter `n_iter` in the `[dft]` section. In general one should use `IALGO=90` in VASP, which performs an exact diagonalization rather than a partial diagonalization scheme, but this is very slow for larger systems.


## Running on CSCS daint

in some directories one can also find example job files to run everything on
daint. Note, that one has to first load the desired docker images with sarus
on daint: https://user.cscs.ch/tools/containers/sarus/ . For a public available image
this can be done via:
```
sarus pull materialstheory/triqs
```
if you wish to use your pre-build docker image, for example if one wants to
include VASP one needs to first save the docker image locally (where one has
built it):
```
docker save --output=triqs-2.1-vasp.tar triqs-2.1-vasp
```
and then upload it to daint and then load it via:
```
sarus load /apps/ethz/eth3/daint/docker-images/triqs-2.1-vasp.tar triqs-2.1-vasp
```
than one can run it has shown in the example files.

I will upload from time to time the most recent docker images into the directory
`/apps/ethz/eth3/daint/docker-images` on daint.

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

srun sarus run --mpi --mount=type=bind,source=$SCRATCH,destination=$SCRATCH --mount=type=bind,source=/apps,destination=/apps load/library/triqs-2.1-vasp bash -c "cd $PWD ; python /apps/ethz/eth3/dmatl-theory-git/uni-dmft/run_dmft.py"
```
thats it. This line automatically runs the docker image and executes the
`run_dmft.py` script. Unfortunately the new sarus container enginge does not mounts automatically user directories. Therefore, one needs to specify with `--mount` to mount the scratch and apps folder manually. Then, one executes in the container bash to first go into the current dir and then executes python and the dmft script.

### CSC calculations on daint

CSC calculations need the parameter `csc = True` and the mandatory parameters from the group `dft`.
Then, uni-dmft automatically starts VASP on as many cores as specified.
Note that VASP runs on cores that are already used by uni-dmft.
This minimizes the time that cores are idle while not harming the performance because these two processes are never active at the same time.

A slurm job script should look like this:
```
#!/bin/bash
#SBATCH --job-name="svo-csc-test"
#SBATCH --time=4:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=36
#SBATCH --account=eth3
#SBATCH --ntasks-per-core=1
#SBATCH --constraint=mc
#SBATCH --partition=normal
#SBATCH --output=out.%j
#SBATCH --error=err.%j

# path to run_dmft.py script
SCRIPTDIR=/apps/ethz/eth3/dmatl-theory-git/uni-dmft/run_dmft.py
# Sarus image that is utilized
IMAGE=load/library/triqs-vasp

srun --mpi=pmi2 sarus run --mount=type=bind,source=/apps,destination=/apps --mount=type=bind,source=$SCRATCH,destination=$SCRATCH $IMAGE bash -c "cd $PWD ; python $SCRIPTDIR"
```
Note that here the mpi option is given to the `srun` command and not the sarus command, as for one-shot calculations.
This is important for the python to be able to start VASP.

In general I found 1 node for Vasp is in most cases enough, which means that we set `n_cores` in the dmft\_config.ini to 36 here.
Using more than one node results in a lot of MPI communication, which in turn slows down the calculation significantly.
For a 80 atom unit cell 2 nodes are useful, but for a 20 atom unit cell not at all!

## LOCPROJ bug for individual projections:

Example use of LOCPROJ for t2g manifold of SrVO3 (the order of the orbitals seems to be mixed up... this example leads to x^2 -y^2, z^2, yz... )
In the current version there is some mix up in the mapping between selected orbitals in the INCAR and actual selected in the LOCPROJ. This is 
what the software does (left side is INCAR, right side is resulting in the LOCPROJ)

* xy -> x2-y2
* yz -> z2
* xz -> yz
* x2-y2 -> xz
* z2 -> xy

```
LOCPROJ = 2 : dxz : Pr 1
LOCPROJ = 2 : dx2-y2 : Pr 1
LOCPROJ = 2 : dz2 : Pr 1
```
However, if the complete d manifold is chosen, the usual VASP order (xy, yz, z2, xz, x2-y2) is obtained in the LOCPROJ. This is done as shown below
```
LOCPROJ = 2 : d : Pr 1
```

## convergence of projectors with Vasp

for a good convergence of the projectors it is important to convergence the wavefunctions to high accuracy. Otherwise this often leads to off-diagonal elements in the the local Green's function. To check convergence pay attention to the rms and rms(c) values in the Vasp output. The former specifies the convergence of the KS wavefunction and the latter is difference of the input and out charge density. Note, this does not necessarily coincide with good convergence of the total energy in DFT! Here an example of two calculations for the same system, both converged down to `EDIFF= 1E-10` and Vasp stopped. First run:

```
       N       E                     dE             d eps       ncg     rms          rms(c)
...
DAV:  25    -0.394708006287E+02   -0.65893E-09   -0.11730E-10 134994   0.197E-06  0.992E-05
...
```
second run with different smearing:
```
...
DAV:  31    -0.394760088659E+02    0.39472E-09    0.35516E-13 132366   0.110E-10  0.245E-10
...
```
The total energy is lower as well. But more importantly the second calculation produces well converged projectors preserving symmetries way better, with less off-diagonal elements in Gloc, making it way easier for the solver. Always pay attention to rms.


## orbital order in the W90 converter

Some interaction Hamiltonians are sensitive to the order of orbitals (i.e. density-density or Slater Hamiltonian), others are invariant under rotations in orbital space (i.e. the Kanamori Hamiltonian).
For the former class and W90-based DMFT calculations, we need to be careful because the order of W90 (z^2, xz, yz, x^2-y^2, xy) is different from the order expected by TRIQS (xy, yz, z^2, xz, x^2-y^2).
Therefore, we need to specify the order of orbitals in the projections block (example for Pbnm or P21/n cell, full d shell):
```
begin projections
# site 0
f=0.5,0.0,0.0:dxy
f=0.5,0.0,0.0:dyz
f=0.5,0.0,0.0:dz2
f=0.5,0.0,0.0:dxz
f=0.5,0.0,0.0:dx2-y2
# site 1
f=0.5,0.0,0.5:dxy
f=0.5,0.0,0.5:dyz
f=0.5,0.0,0.5:dz2
f=0.5,0.0,0.5:dxz
f=0.5,0.0,0.5:dx2-y2
# site 2
f=0.0,0.5,0.0:dxy
f=0.0,0.5,0.0:dyz
f=0.0,0.5,0.0:dz2
f=0.0,0.5,0.0:dxz
f=0.0,0.5,0.0:dx2-y2
# site 3
f=0.0,0.5,0.5:dxy
f=0.0,0.5,0.5:dyz
f=0.0,0.5,0.5:dz2
f=0.0,0.5,0.5:dxz
f=0.0,0.5,0.5:dx2-y2
end projections
```
Warning: simply using `Fe:dxy,dyz,dz2,dxz,dx2-y2` does not work, VASP/W90 brings the d orbitals back to W90 standard order.

The 45-degree rotation for the sqrt2 x sqrt2 x 2 cell can be ignored because the interaction Hamiltonian is invariant under swapping x^2-y^2 and xy.


## remarks on the Vasp version

One can use the official Vasp 5.4.4 patch 1 version with a few modifications:

- there is an bug in `fileio.F` around line 1710 where the code tries print out something like "reading the density matrix from Gamma", but this should be done only by the master node. Adding a `IF (IO%IU0>=0) THEN ... ENDIF` around it fixes this
- in the current version of the dft_tools interface the file `LOCPROJ` should contain the fermi energy in the header. Therefore  one should replace the following line in `locproj.F`:
```
WRITE(99,'(4I6,"  # of spin, # of k-points, # of bands, # of proj" )') NS,NK,NB,NF
```
by
```
WRITE(99,'(4I6,F12.7,"  # of spin, # of k-points, # of bands, # of proj, Efermi" )') W%WDES%NCDIJ,NK,NB,NF,EFERMI
```
and add the variable `EFERMI` accordingly in the function call.
- Vasp gets sometimes stuck and does not write the `OSZICAR` file correctly due to a stuck buffer. Adding a flush to the buffer to have a correctly written `OSZICAR` to extract the DFT energy helps, by adding in `electron.F` around line 580 after
```
CALL STOP_TIMING("G",IO%IU6,"DOS")
```
two lines:
```
flush(17)
print *, ' '
```
- this one is __essential__ vor the current version of the DMFT code. Vasp spends a very long time in the function `LPRJ_LDApU` and this function is not needed! Is it used for some basic checks and a manual LDA+U implementation. Removing the call to this function in `electron.F` in line 644 speeds up the calculation by up to 30%! If this is not done, Vasp will create a GAMMA file each iteration which needs to be removed manually to not overwrite the DMFT GAMMA file!
- make sure that mixing in VASP stays turned on. Don't set IMIX or the DFT steps won't converge!
