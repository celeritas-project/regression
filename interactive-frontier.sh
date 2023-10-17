#!/bin/bash -ex

exec salloc \
-A csc404 \
-t 0:09:59 \
-N 1 \
-q debug \
-J celer-debugging
