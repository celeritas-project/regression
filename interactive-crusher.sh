#!/bin/bash -ex

exec salloc \
-A csc333_crusher \
-t 0:59:59 \
-N 1 \
-J celer-debug
