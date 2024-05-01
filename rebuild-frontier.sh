#!/bin/bash -ex

source $HOME/Code/celeritas-frontier/scripts/env/frontier.sh
cd $HOME/Code/celeritas-frontier/build-ndebug \
  && cmake -UCeleritas_GIT_DESCRIBE . \
  && ninja celer-sim celer-g4
