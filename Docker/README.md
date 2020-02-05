# Docker files

The normal Dockerfile builds a complete container starting with Ubuntu 18.04 with openmpi, IntelMKL, VASP, wannier90 2.1, triqs 2.x.x, maxent, and SoliDMFT included.

The Dockerfile in mpich_wannier90 builds the former Docker container build on MPICH instead of OpenMPI for shifter on daint.

The Dockerfile is build in the usual way via:
```
docker build -t triqs-2.2 -f Dockerfile_OpenMPI ./
```

Note that one needs a working vasp version as archive (csc_vasp.tar.gz) in this
directory to make the CSC calculation work.
