#!/bin/bash -e
#SBATCH -A csc333_crusher
#SBATCH -t 0:59:59
#SBATCH -N 1
#SBATCH -J celer-regression
#SBATCH -o crusher-%J.out
#SBATCH -e crusher-%J.err

if [ -z "$SLURM_JOB_ID" ]; then
  set -x
  sbatch $0 && squeue -u $USER
  exit $?
fi

source /ccs/home/s3j/.local/src/celeritas/scripts/env/crusher.sh

echo "Running on $HOSTNAME at $(date)"
python3 run-problems.py crusher
echo "Completed at $(date)"
exit 0
