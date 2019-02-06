# Examples

Here one finds multiple examples for running the DFT+DMFT code. For the CSC
examples a working, adapted, VASP version 5.4.4 and suitable POTCAR files are
needed. There are three example systems: SrVO3, LuNiO3 and SrRuO3. For the first
two there are one-shot and CSC examples, and for SrRuO3 there is only a one-shot
example, but for a ferro-magnetic calculation. For the LuNiO3 CSC examples
please use the following PAW-PBE POTCARs: Lu_3, Ni_pv, O. For the SrVO3 CSC
example please use the PAW-PBE POTCARs: Sr_sv, V, O.

In the example for svo-one-shot is also a folder 'converter' which contains the
work flow for creating the h5 archive.  

To run any of the one-shot examples execute:
```
mpirun -n 4 python work/run_dmft.py
```
within one of the directories. For the CSC calculations follow the instructions
in the main readme file of this repository.
