#!/bin/bash -e

module load geant4-data/10.7.3-kzsk python

echo "Running on $HOSTNAME at $(date)"
python3 run-problems.py wildstyle
echo "Completed at $(date)"
exit 0