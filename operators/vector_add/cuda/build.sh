#!/bin/bash
# 编译 vector_add CUDA kernel 为 shared library

cd "$(dirname "$0")"

CUDA_ARCH=${CUDA_ARCH:-"sm_80"}  # 默认 Ampere (A100/RTX30xx)

nvcc \
    -O3 \
    --compiler-options -fPIC \
    -shared \
    -arch=${CUDA_ARCH} \
    -o vector_add.so \
    kernel.cu

echo "Build done: vector_add.so"
