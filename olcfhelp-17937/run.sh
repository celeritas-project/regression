#!/bin/sh -e

if [ -z "$1" ]; then
  echo "usage: $0 {560|571}" >&2
  exit 1
fi

. "./env-$1.sh"
export OMP_NUM_THREADS=1
srun --gpus-per-task=1 ./build-$1/bin/celer-sim inp.json | jq . > results-$1.out.json
