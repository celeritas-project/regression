#!/bin/bash -ex

exec salloc \
-A m2616 \
-t 02:00:00 \
-N 1 \
-q interactive \
-C gpu \
-J celer-debugging
