#!/bin/bash

# this is the bash script that spawns both vasp and triqs with the according
# options. The dmft_script variable at call time should be set to the run_dmft.py
# script from this repo. This is the modified version for daint, see extras below

show_help()
{
echo "
Usage: vasp_dmft [-n <number of nodes triqs>] [-v <number of nodes vasp>] [-i shifter image] -c dmft_config_file [<dmft_script.py>]

       If the number of cores is not specified it is set to 1 by default.

       <dmft_script.py> must call the run_dmft.py script from this Repo which
       will take care of the csc_flow and dmft cycles. If the script name is
       omitted the default name 'run_dmft.py' is used.
"
}

while getopts "n:v:i:c:" opt; do
  case $opt in
     n) NTRIQS=$OPTARG;;
     v) NVASP=$OPTARG;;
     c) DMFT_CFG=$OPTARG;;
     i) SHIFTER_IMG=$OPTARG;;
     :)
       echo "  Error: Option -$OPTARG requires an argument" >&2
       show_help
       exit 1
       ;;
      \?)
         echo "  Error: Invalid option -$OPTARG" >&2
  esac
done

if [ -z "$NTRIQS" ]; then
  echo "  Number of nodes for triqs not specified with option -n" >&2
  show_help
  exit 1
fi

if [ -z "$NVASP" ]; then
  echo "  Number of nodes for vasp not specified with option -v" >&2
  show_help
  exit 1
fi

if [ -z "$SHIFTER_IMG" ]; then
  echo "  Name of shifter image not specified with -i" >&2
  show_help
  exit 1
fi

if [ -z "$DMFT_CFG" ]; then
  echo "  dmft config file name set to dmft_config.ini"
  DMFT_CFG=dmft_config.ini
fi

shift $((OPTIND-1))

if [ -z "$1" ]; then
  DMFT_SCRIPT=run_dmft.py
else
  DMFT_SCRIPT=$1
fi


echo "  Number of nodes for triqs: $NTRIQS"
echo "  Number of nodes for vasp: $NVASP"
echo "  Shifter Image: $SHIFTER_IMG"
echo "  Script name: $DMFT_SCRIPT"
echo "  config file name: $DMFT_CFG"

TASKSPERN=$(($SLURM_CPUS_ON_NODE/2))

# remove vasp.lock to make sure vasp starts
rm -f vasp.lock
rm -f STOPCAR

# on daint we need to spawn two containers for vasp and triqs on seperate nodes
stdbuf -o 0 srun -u --mem=40G --gres=craynetwork:1 --cpu_bind=none --exclusive -N $NVASP --ntasks-per-core=1 --ntasks-per-node=$TASKSPERN --hint=nomultithread shifter run --mpi $SHIFTER_IMG vasp_std &

srun -u --mem=18G --gres=craynetwork:1 --cpu_bind=none -N $NTRIQS --exclusive --ntasks-per-core=1 --ntasks-per-node=$TASKSPERN --hint=nomultithread shifter run --mpi $SHIFTER_IMG python $DMFT_SCRIPT $DMFT_CFG || kill %1
