#!/bin/bash -e
#SBATCH -A csc333_crusher
#SBATCH -t 1:59:59
#SBATCH -N 1
#SBATCH -J celer-regression
#SBATCH -o crusher-%J.out
#SBATCH -e crusher-%J.err

if [ -z "$SLURM_JOB_ID" ]; then
  set -ex
  sbatch $0 
  sleep 5
  squeue -u $USER
  exit $?
fi

source /ccs/home/s3j/.local/src/celeritas-crusher/scripts/env/crusher.sh

echo "Running on $HOSTNAME at $(date)"
python3 run-problems.py crusher
echo "Completed at $(date)"
exit 0
