# soliDMFT
This program allows to perform DFT+DMFT ''one-shot'' and charge self-consistent
calculations from h5 archives or VASP input files for multiband systems using
the TRIQS package, in combination with the CThyb solver and SumkDFT from
DFT-tools. Runs with both triqs 1.4.2 and triqs 2.x.x (https://github.com/TRIQS/)

For one-shot calculations one starts directly by running `run_dmft.py`, whereas
for CSC calculations one uses the `vasp_dmft.sh` bash scripts to start both VASP
and triqs at once.

Copyright (C) 2019, A. Hampel, S. Beck, M. Merkel, and C. Ederer from Materials Theory Group
at ETH Zurich.

This application is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version (see <http://www.gnu.org/licenses/>). You should have received a
copy of the License (file: LICENSE) with this repository.

It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

Please also note that we do not provide user support for this code.

## source code files

- __run_dmft.py:__ main file that runs the calculation and start a CSC flow by
  invoking  `csc_flow_control` or a one shot calculation directly by invoking
  `dmft_cycle` on a given h5 archive
- __read_config.py:__ contains the functions to read the dmft config file. Take a
  look in `read_config.py` for a detailed list of parameters
- __dmft_cycle.py:__ contains the `dmft_cycle` function that run a predefined
  number of DMFT iterations
- __csc_flow.py:__ contains the `csc_flow_control` function to steer the CSC
  calculation and call then ones per DFT+DMFT cycle the `dmft_cycle` function
- __observables.py:__ contains all functions necessary to calculate and write the
  observables during the run, which are stored in a general dictionary: `observables`
- __toolset.py:__ contains several small helper functions


## getting started

For one-shot and CSC calculations one starts directly by running `run_dmft.py`. In case of CSC calculations one has to set `cscs = True` in the config file (see below).

In the `example` directory one finds several
examples to run. Best start with the svo-one-shot example. The
`dmft_config.ini` file contains the configuration for the DMFT run, which is
explained in the read_config method in the main script. The `svo.h5` is the DMFT
input data, which is obtained from projection on localized Wannier functions
(see folder `svo-one-shot/converter`).

If one wishes to do CSC calculations the
docker container must contain also a installed VASP version >5.4.4 that
understands the ICHARG=5 flag.

To run the one shot examples one can use the triqs docker images on,
https://hub.docker.com/r/materialstheory/triqs/ or the official ones on
https://hub.docker.com/r/flatironinstitute/triqs/.

then one can run docker as:  
```
docker run --rm -it -u $(id -u) -v ~/git/uni-dmft:/work materialstheory/triqs bash
```
go to the example directory inside the running container and the run it via:  
```
mpirun -n 4 python work/run_dmft.py
```
or run it directly via:  
```
docker run --rm -it -u $(id -u) -v ~/git/uni-dmft:/work materialstheory/triqs bash -c 'cd /work/tests/svo-one-shot/ && python /work/run_dmft.py'
```
the more elaborate version of the Docker container found in this repo in the
folder `Docker` can be best started as:
```
docker run --rm -it --shm-size=4g -e USER_ID=`id -u` -e GROUP_ID=`id -g` -p 8378:8378 -v $PWD:/work -v ~/git/uni-dmft:/uni-dmft triqs_vasp_csc bash
```
where the `-e` flags will translate your current user and group id into the
container and make sure writing permissions are correct for the mounted volumes.
Moreover, you can start by executing
```
jupyter.sh
```
a jupyter-lab server from the current dir.

### CSC calculations locally

Here one needs a special docker image with vasp included. This can be done by
building the Dockerfile in `/Docker/`:
```
docker build -t triqs_vasp_csc ./
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
limit of DMFT iterations is reached. This should also work on most HPC systems (tested on slurm with OpenMPI), as the the child mpirun call is performed without the slurm environment variables. This tricks slrum into starting more ranks than it has available.

One remark regarding the number of iterations per DFT cycle. Since VASP uses a
block Davidson scheme for minimizing the energy functional not all eigenvalues
of the Hamiltonian are updated simultaneously therefore one has to make several
iterations before the changes from DMFT in the charge density are completely
considered. The default value are __6__ DFT iterations, which is very
conservative, and can be changed by changing the config parameter `n_iter` in the `[dft]` section. In general one should use `IALGO=90` in VASP, which performs an exact diagonalization rather than a partial diagonalization scheme, but this is very slow for larger systems.

## remarks on the VASP version

I now use the official Vasp 5.4.4 patch 1 version with a few modifications:

- there is an bug in `fileio.F` around line 1710 where the code tries print out
  something like "reading the density matrix from Gamma", but this should be
  done only by the master node. So I added a `IF (IO%IU0>=0) THEN ... ENDIF`
  around it
- in the current version of the dft_tools interface the file `LOCPROJ` should
  contain the fermi energy in the header. Therefore I replaced the following
  line in `locproj.F`:
```
WRITE(99,'(4I6,"  # of spin, # of k-points, # of bands, # of proj" )') NS,NK,NB,NF
```
by
```
WRITE(99,'(4I6,F12.7,"  # of spin, # of k-points, # of bands, # of proj, Efermi" )') W%WDES%NCDIJ,NK,NB,NF,EFERMI
```
and added the variable `EFERMI` accordingly in the function call.
- Vasp gets sometimes stuck and does not write the `OSZICAR` file correctly due
  to a stuck buffer. I added a flush to the buffer to have a correctly written
  `OSZICAR` to extract the DFT energy. I added in `electron.F` around line 580
  after
```
CALL STOP_TIMING("G",IO%IU6,"DOS")
```
two lines:
```
flush(17)
print *, ' '
```
which did the job.
- this one is __essential__ vor the current version of the DMFT code. I tried to
  fasten everything and I figured that Vasp spends a very long time in the
  function `LPRJ_LDApU`. I checked and this function is not needed! Is it used
  for some basic checks and a manual LDA+U implementation. Removing the call to
  this function in `electron.F` in line 644 speeds up the calculation by up to
  30%! However, originally I added a copy of the file `GAMMA` to `GAMMA.old` to
  keep the original GAMMA file from DMFT. This is not needed now anymore and I
  removed the `shutil` copy call in our python code.
- make sure that mixing in VASP is turned of IMIX=0
