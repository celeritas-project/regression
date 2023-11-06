#!/bin/bash -e
#SBATCH -A csc404
#SBATCH -t 1:59:59
#SBATCH -N 1
#SBATCH -J celer-regression
#SBATCH -o frontier-%J.out
#SBATCH -e frontier-%J.err

if [ -z "$SLURM_JOB_ID" ]; then
  set -x
  sbatch $0 && squeue -u $USER
  exit $?
fi

source /ccs/home/s3j/.local/src/celeritas-frontier/scripts/env/frontier.sh

echo "Running on $HOSTNAME at $(date)"
python3 run-problems.py frontier
echo "Completed at $(date)"
exit 0
