#!/bin/bash -e
#SBATCH -A m2616
#SBATCH -t 04:00:00
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -C gpu
#SBATCH -J celer-regression
#SBATCH -o perlmutter-%J.out
#SBATCH -e perlmutter-%J.err

if [ -z "$SLURM_JOB_ID" ]; then
  exec sbatch $0 "$@"
fi

source $CFS/atlas/esseivaj/devel/celeritas/scripts/env/perlmutter.sh

echo "Running on $HOSTNAME at $(date)"
dcgmi profile --pause
python3 run-problems.py perlmutter $1
dcgmi profile --resume
echo "Completed at $(date)"

exit 0
