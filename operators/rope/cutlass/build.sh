#!/bin/bash
cd "$(dirname "$0")"
source "../../../scripts/find_cutlass.sh"
echo "Using CUTLASS: ${CUTLASS_ROOT}"
CUDA_ARCH=${CUDA_ARCH:-"sm_80"}
nvcc -O3 --compiler-options -fPIC -shared -arch=${CUDA_ARCH} -std=c++17 \
    -I${CUTLASS_INCLUDE} \
    -o rope_cutlass.so kernel.cu
echo "Build done: rope_cutlass.so (arch=${CUDA_ARCH})"
