#!/bin/bash -ex

exec bsub \
-P csc404 \
-W 0:59 \
-nnodes 1 \
-q debug \
-J celer-debugging \
-Is /bin/bash
