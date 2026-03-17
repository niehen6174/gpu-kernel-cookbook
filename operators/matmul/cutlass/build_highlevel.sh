#!/bin/bash
# 编译 CUTLASS 高层 API matmul（sm90 / Hopper）
cd "$(dirname "$0")"
source "../../../scripts/find_cutlass.sh"
echo "Using CUTLASS: ${CUTLASS_ROOT}"

CUDA_ARCH=${CUDA_ARCH:-"sm_80"}

# 注意：CUTLASS 3.x/4.x 高层 API 需要额外编译选项：
#   -arch=sm_90a   — wgmma 指令必须用 sm_90a（带 'a' 后缀），sm_90 不够
#   --expt-relaxed-constexpr  — 允许 constexpr 在 device 代码中使用
#   -lineinfo      — 保留行号（方便 Nsight profiling）
#
# 如果 CUDA_ARCH=sm_90，自动升级为 sm_90a
if [ "${CUDA_ARCH}" = "sm_90" ]; then
    CUDA_ARCH="sm_90a"
fi

nvcc -O3 --compiler-options -fPIC -shared \
    -arch=${CUDA_ARCH} \
    -std=c++17 \
    --expt-relaxed-constexpr \
    -lineinfo \
    -I${CUTLASS_INCLUDE} \
    -o matmul_cutlass_hl.so kernel_highlevel.cu

echo "Build done: matmul_cutlass_hl.so (arch=${CUDA_ARCH})"
