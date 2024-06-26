#!/bin/sh -e

PROJID=hep143
_worldwork=${WORLDWORK}/${PROJID}
_ccsproj=/ccs/proj/${PROJID}

module load PrgEnv-amd/8.3.3 amd/5.6.0 craype-x86-trento \
  libfabric/1.15.2.0 cray-pmi/6.1.8
# Disable warning "Using generic mem* routines instead of tuned routines"
export RFE_811452_DISABLE=1

# Avoid linking multiple different libsci (one with openmp, one without)
module unload cray-libsci
# Avoid libraries interfering with I/O
module unload darshan-runtime

# Set up compilers
test -n "${CRAYPE_DIR}"
export CXX=${CRAYPE_DIR}/bin/CC
export CC=${CRAYPE_DIR}/bin/cc

# Do NOT load the accelerator target, because it adds
# -fopenmp-targets=amdgcn-amd-amdhsa 
# -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a
# which implicitly defines __CUDA_ARCH__
# module load craype-accel-amd-gfx90a

# Set up celeritas
export SPACK_ROOT=${_ccsproj}/spack
export PATH=${_worldwork}/opt-view/bin:${_ccsproj}/opt-view/bin:${_ccsproj}/conda-frontier/bin:$PATH
export CMAKE_PREFIX_PATH=${_worldwork}/opt-view:${CMAKE_PREFIX_PATH}
export MODULEPATH=${_worldwork}/share/lmod/linux-sles15-x86_64/Core:${MODULEPATH}

# Set up Geant4 data 
module load geant4-data/11.0
test -n "${G4ENSDFSTATEDATA}"
test -e "${G4ENSDFSTATEDATA}"

# Make llvm available
test -n "${ROCM_PATH}"
export PATH=$PATH:${ROCM_PATH}/llvm/bin
