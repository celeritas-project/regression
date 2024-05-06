#!/bin/sh -e

if [ -z "$1" ]; then
  echo "usage: $0 {560|571}" >&2
  exit 1
fi

. "./env-$1.sh"
cd celeritas
ln -s scripts/cmake-presets/frontier.json CMakeUserPresets.json || true
cmake --preset=ndebug -B ../build-$1 -S .
cd ../build-$1
cmake --build . --target celer-sim
