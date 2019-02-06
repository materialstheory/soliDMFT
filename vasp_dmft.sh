#!/bin/bash

# this is the bash script that spawns both vasp and triqs with the according
# options. The dmft_script variable at call time should be set to the run_dmft.py
# script from this repo

show_help()
{
echo "
Usage: vasp_dmft [-n <number of cores for triqs>] [-v <number of cores for vasp>] -c dmft_config_file [<dmft_script.py>]

       <dmft_script.py> must call the run_dmft.py script from this Repo which
       will take care of the csc_flow and dmft cycles. If the script name is
       omitted the default name 'run_dmft.py' is used.
"
}

while getopts ":n:v:c" opt; do
  case $opt in
    n) NTRIQS=$OPTARG;;
    v) NVASP=$OPTARG;;
    c) DMFT_CFG=$OPTARG;;
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

echo "  Number of cores for triqs: $NTRIQS"
echo "  Number of cores for vasp: $NVASP"
echo "  Script name: $DMFT_SCRIPT"
echo "  config file name: $DMFT_CFG"

# remove vasp.lock to make sure vasp starts
rm -f vasp.lock
rm -f STOPCAR

# start VASP
stdbuf -o 0 mpirun -np $NVASP vasp_std &

# start triqs
mpirun -np $NTRIQS python $DMFT_SCRIPT $DMFT_CFG || kill %1
