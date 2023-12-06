#!/bin/bash -ex

source $CFS/atlas/esseivaj/devel/celeritas/scripts/env/perlmutter.sh
cd $CFS/atlas/esseivaj/devel/celeritas/build-ndebug \
  && find . -name '*.cu.o' -exec rm {} + \
  && cmake -UCeleritas_GIT_DESCRIBE . \
  && ninja celer-sim celer-g4
cd $CFS/atlas/esseivaj/devel/celeritas/build-ndebug-novg \
  && find . -name '*.cu.o' -exec rm {} + \
  && cmake -UCeleritas_GIT_DESCRIBE . \
  && ninja celer-sim celer-g4
