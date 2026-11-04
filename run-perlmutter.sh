#!/bin/bash -e
#SBATCH -A m2845
#SBATCH -t 03:59:59
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -J celer-regression
#SBATCH -o perlmutter-%J.out
#SBATCH -e perlmutter-%J.err

if [ -z "$SLURM_JOB_ID" ]; then
  set -x
  sbatch $0 "$@" && squeue -u $USER
  exit $?
fi

source /pscratch/sd/e/esseivaj/dev/celeritas/scripts/env/perlmutter.sh
# The spack view's lua paths hide the 'lfs' module from srun's cli_filter plugin
unset LUA_PATH LUA_CPATH

echo "Running on $HOSTNAME at $(date)"
python3 run-problems.py perlmutter "$@"
echo "Completed at $(date)"
exit 0
