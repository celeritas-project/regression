#!/bin/bash -e
#BSUB -P csc404
#BSUB -W 0:30
#BSUB -nnodes 1
#BSUB -J celer-regression
#BSUB -o summit-%J.out
#BSUB -e summit-%J.err

if [ -z "$LSB_JOBID" ]; then
  exec bsub $0
fi

source $PROJWORK/csc404/celeritas/summit-env.sh

echo "Running on $HOSTNAME at $(date)"
python3 run-problems.py
echo "Completed at $(date)"
exit 0
