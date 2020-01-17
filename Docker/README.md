# Dockerfile for triqs 2.1

Dockerfile and supplements for creating a docker container that works with MPICH
(also in a cluster environment). The Dockerfile is build in the usual way via:
```
docker build -t triqs-2.2 ./
```
Note that one needs a working vasp version as archive (csc_vasp.tar.gz) in this
directory to make the CSC calculation work.
