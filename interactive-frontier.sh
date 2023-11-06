#!/bin/bash -ex

exec salloc \
-A csc404 \
-t 0:59:00 \
-N 1 \
-q debug \
-J celer-debugging
