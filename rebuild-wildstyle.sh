#!/bin/bash -ex

cd $HOME/Code/celeritas/build-reldeb \
  && cmake -UCeleritas_GIT_DESCRIBE . \
  && ninja celer-sim celer-g4
cd $HOME/Code/celeritas/build-reldeb-vecgeom \
  && cmake -UCeleritas_GIT_DESCRIBE . \
  && ninja celer-sim celer-g4
