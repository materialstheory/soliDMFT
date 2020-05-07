# Docker files

The normal Dockerfile builds a complete container starting with Ubuntu 18.04 with openmpi, IntelMKL, VASP, wannier90 2.1, triqs 2.x.x, maxent, and SoliDMFT included.

The Dockerfile in mpich_wannier90 builds the former Docker container build on MPICH instead of OpenMPI for shifter on daint.

The Dockerfile is build in the usual way via:
```
docker build -t triqs-2.2 -f Dockerfile_OpenMPI ./
```

Note that one needs a working vasp version as archive (csc_vasp.tar.gz) in this
directory to make the CSC calculation work.

Recent error: in case you run into an error related to the MKL libraries (error message involves https://apt.repos.intel.com/mkl all InRelease), a fix is to pull the latest image from the docker hub.
A current working command is then
```
docker pull docker.io/materialstheory/base-bionic-sci-python
docker build --no-cache -t triqs-2.2 -f Dockerfile_MPICH ./
```