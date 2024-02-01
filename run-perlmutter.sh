#!/bin/bash -e
#SBATCH -A m2616
#SBATCH -t 01:59:59
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -J celer-regression
#SBATCH -o perlmutter-%J.out
#SBATCH -e perlmutter-%J.err

if [ -z "$SLURM_JOB_ID" ]; then
  set -x
  sbatch $0 && squeue -u $USER
  exit $?
fi

source $CFS/atlas/esseivaj/devel/celeritas/scripts/env/perlmutter.sh

echo "Running on $HOSTNAME at $(date)"
python3 run-problems.py perlmutter "$@"
echo "Completed at $(date)"
exit 0
