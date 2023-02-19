#!/bin/bash -e
#BSUB -P csc404
#BSUB -W 1:59
#BSUB -nnodes 1
#BSUB -J celer-regression
#BSUB -o summit-%J.out
#BSUB -e summit-%J.err

if [ -z "$LSB_JOBID" ]; then
  set -ex
  bsub $0
  sleep 5
  bjobs -u $USER
  exit $?
fi

source $PROJWORK/csc404/celeritas/summit-env.sh

echo "Running on $HOSTNAME at $(date)"
python3 run-problems.py summit
echo "Completed at $(date)"
exit 0
