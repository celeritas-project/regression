#!/bin/bash -ex

source $HOME/.local/src/celeritas-summit/scripts/env/summit.sh
cd /gpfs/alpine/proj-shared/csc404/celeritas/build-ndebug \
  && find . -name '*.cu.o' -exec rm {} + \
  && cmake -UCeleritas_GIT_DESCRIBE . \
  && ninja celer-sim celer-g4
cd /gpfs/alpine/proj-shared/csc404/celeritas/build-ndebug-novg \
  && cmake -UCeleritas_GIT_DESCRIBE . \
  && ninja celer-sim celer-g4
