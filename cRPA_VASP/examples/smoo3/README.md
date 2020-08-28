# cRPA for SrMoO3

## how to run
just sbatch crpa_rusty.sh on rusty and the whole machinery is run automatically. 
Afterwards use python eval_U.py to extract effective slater parameters from 
UIJKL file. 

## setup
cRPA with Vasp is done in 3 steps:
1. INCAR.DFT is used to pre-converge the setup
2. INCAR.EXACT is used to run an exact diagonalization and compute the dielectric funcion
3. INCAR.CRPA is the actual cRPA calculation using the converged WAVECAR and WAVEDER to 
compute the screened Coulomb interaction

