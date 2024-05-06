#!/bin/sh -e

PROJID=hep143
_worldwork=${WORLDWORK}/${PROJID}

module load PrgEnv-amd/8.5.0 cpe/23.12 amd/5.7.1 craype-x86-trento \
  libfabric/1.15.2.0 miniforge3/23.11.0

set -x
export MODULEPATH=${_worldwork}/share/lmod/linux-sles15-x86_64/Core:${MODULEPATH}
export G4FORCE_RUN_MANAGER_TYPE=MT
export G4FORCENUMBEROFTHREADS=2
export CELER_LOG=debug
export CELER_LOG_LOCAL=debug

module load geant4-data/11.0

exec ./celer-g4 failing.inp.json
