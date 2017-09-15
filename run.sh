#!/bin/sh
#th tc.lua

t=$1
lscpu
. /home/zhaoxiao/test/inteltorch/install/bin/torch-activate
export KMP_AFFINITY=compact,1,0,granularity=fine

if [ $t == 'bdw' ]; then
  export OMP_NUM_THREADS=44
  th benchmark.lua
fi
if [ $t == 'knl' ]; then
  export OMP_NUM_THREADS=68
  th benchmark.lua
fi
if [ $t == 'knm' ]; then
  export OMP_NUM_THREADS=72
  th benchmark.lua
fi
if [ $t == 'skx' ]; then
  export OMP_NUM_THREADS=56
  th benchmark.lua
fi

