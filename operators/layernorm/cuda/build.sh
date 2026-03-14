#!/bin/bash
cd "$(dirname "$0")"
CUDA_ARCH=${CUDA_ARCH:-"sm_80"}
nvcc -O3 --compiler-options -fPIC -shared -arch=${CUDA_ARCH} -o layernorm.so kernel.cu
echo "Build done: layernorm.so"
