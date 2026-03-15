#!/bin/bash
# 编译 vector_add CuTe (CUTLASS) kernel

cd "$(dirname "$0")"

# 自动检测 CUTLASS 路径
source "../../../scripts/find_cutlass.sh"
echo "Using CUTLASS: ${CUTLASS_ROOT}"

CUDA_ARCH=${CUDA_ARCH:-"sm_80"}

nvcc \
    -O3 \
    --compiler-options -fPIC \
    -shared \
    -arch=${CUDA_ARCH} \
    -std=c++17 \
    -I${CUTLASS_INCLUDE} \
    -o vector_add_cutlass.so \
    kernel.cu

echo "Build done: vector_add_cutlass.so (arch=${CUDA_ARCH})"
