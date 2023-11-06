#!/bin/bash -ex

source $HOME/.local/src/celeritas-frontier/scripts/env/frontier.sh
cd $HOME/.local/src/celeritas-frontier/build-ndebug \
  && cmake -UCeleritas_GIT_DESCRIBE . \
  && ninja celer-sim
