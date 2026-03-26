#!/bin/bash
# 编译 CUTLASS 3.x 高层 API Group GEMM（sm90a / Hopper）
cd "$(dirname "$0")"
source "../../../scripts/find_cutlass.sh"
echo "Using CUTLASS: ${CUTLASS_ROOT}"

CUDA_ARCH=${CUDA_ARCH:-"sm_80"}

# sm90 高层 API 需要 sm_90a（带 'a' 后缀）才能用 wgmma 指令
if [ "${CUDA_ARCH}" = "sm_90" ]; then
    CUDA_ARCH="sm_90a"
fi

nvcc -O3 --compiler-options -fPIC -shared \
    -arch=${CUDA_ARCH} \
    -std=c++17 \
    --expt-relaxed-constexpr \
    -lineinfo \
    -I${CUTLASS_INCLUDE} \
    -o group_gemm_cutlass_hl.so kernel_grouped.cu

echo "Build done: group_gemm_cutlass_hl.so (arch=${CUDA_ARCH})"
