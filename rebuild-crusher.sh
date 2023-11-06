#!/bin/bash -ex

source $HOME/.local/src/celeritas-crusher/scripts/env/crusher.sh
cd $HOME/.local/src/celeritas-crusher/build-ndebug \
  && cmake -UCeleritas_GIT_DESCRIBE . \
  && ninja celer-sim
