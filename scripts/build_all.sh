#!/bin/bash
# 一键编译所有 CUDA kernels

set -e

CUDA_ARCH=${CUDA_ARCH:-"sm_80"}
ROOT=$(cd "$(dirname "$0")/.." && pwd)

echo "Building all CUDA kernels for arch=${CUDA_ARCH}..."
echo ""

for op in vector_add transpose softmax layernorm matmul attention rms_norm rope group_gemm; do
    build_sh="$ROOT/operators/$op/cuda/build.sh"
    if [ -f "$build_sh" ]; then
        echo "Building $op..."
        CUDA_ARCH=$CUDA_ARCH bash "$build_sh"
    fi
done

echo ""
echo "All builds complete!"
