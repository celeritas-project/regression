#!/bin/bash -e
#SBATCH -A hep143
#SBATCH -t 0:59:59
#SBATCH -N 1
#SBATCH -J celer-regression
#SBATCH -o frontier-%J.out
#SBATCH -e frontier-%J.err
#SBATCH -q debug

if [ -z "$SLURM_JOB_ID" ]; then
  set -x
  sbatch $0 && squeue -u $USER
  exit $?
fi

source /ccs/home/s3j/Code/celeritas-frontier/scripts/env/frontier.sh

echo "Running on $HOSTNAME at $(date)"
python3 run-problems.py frontier
echo "Completed at $(date)"
exit 0
