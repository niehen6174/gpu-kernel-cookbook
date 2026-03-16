/*
 * Transpose — CuTe C++ 实现
 *
 * 学习目标：演示 CuTe 的 2D Layout 和 local_tile 在转置场景的用法：
 *   - make_layout(M, N) 构造 2D Tensor
 *   - 通过 Layout 交换 stride 实现逻辑转置
 *   - Shared memory Tensor 的创建与 bank conflict padding
 *   - cute::copy 在 smem ↔ gmem 之间的协同拷贝
 *
 * =========================================================================
 * V1: 直接用 CuTe 2D Tensor 索引
 *   - 把输入/输出包装成 (rows, cols) Layout
 *   - thread(tx, ty) 直接通过 (row, col) 坐标读写
 *   - 写方向 non-coalesced（与 CUDA v1 等价，用于教学对比）
 *
 * V2: CuTe Shared Memory Tiling（带 bank conflict padding）
 *   - 用 make_layout(Shape<TILE,TILE+1>{}) 创建带 padding 的 smem Layout
 *   - Phase1: coalesced 读全局内存 → smem
 *   - Phase2: 从 smem（转置坐标）coalesced 写全局内存
 * =========================================================================
 */

#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cuda_runtime.h>

using namespace cute;

constexpr int TILE = 32;

// -------------------------------------------------------------------------
// V1: 2D CuTe Tensor 直接索引（non-coalesced 写，教学用）
// -------------------------------------------------------------------------
__global__ void transpose_cute_v1(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int rows, int cols)
{
    // 用 2D Layout 包装指针
    // make_layout(rows, cols) → Shape=(rows,cols), Stride=(cols,1)（行主序）
    auto A = make_tensor(make_gmem_ptr(input),  make_layout(make_shape(rows, cols),
                                                             make_stride(cols, 1)));
    // output 是转置矩阵 (cols, rows)，行主序
    auto B = make_tensor(make_gmem_ptr(output), make_layout(make_shape(cols, rows),
                                                             make_stride(rows, 1)));

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        // A(row, col) → B(col, row)
        // 读 A：同 warp 的 threadIdx.x 连续 → col 连续 → coalesced ✓
        // 写 B：B(col, row) 按 col 分配给同 warp → stride=rows → non-coalesced ✗
        B(col, row) = A(row, col);
    }
}

// -------------------------------------------------------------------------
// V2: CuTe Shared Memory Tiling（coalesced 读写 + bank conflict 消除）
// -------------------------------------------------------------------------
__global__ void transpose_cute_v2(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int rows, int cols)
{
    // 全局内存 Tensor（行主序）
    auto A = make_tensor(make_gmem_ptr(input),  make_layout(make_shape(rows, cols),
                                                             make_stride(cols, 1)));
    auto B = make_tensor(make_gmem_ptr(output), make_layout(make_shape(cols, rows),
                                                             make_stride(rows, 1)));

    // Shared memory：TILE×(TILE+1) — +1 列 padding 消除 bank conflict
    // make_layout(Shape<TILE, TILE+1>{}) 是编译期常量 layout
    __shared__ float smem_raw[TILE * (TILE + 1)];
    auto smem = make_tensor(make_smem_ptr(smem_raw),
                            make_layout(make_shape(Int<TILE>{}, Int<TILE + 1>{}),
                                        make_stride(Int<TILE + 1>{}, Int<1>{})));
    // smem(i, j) = smem_raw[i*(TILE+1) + j]

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    // Phase 1: coalesced 读全局内存 → smem
    // tile 起始位置：全局行 by*TILE + ty，全局列 bx*TILE + tx
    int grow = by * TILE + ty;
    int gcol = bx * TILE + tx;

    if (grow < rows && gcol < cols) {
        // A(grow, gcol)：同 warp tx=0..31 → 连续列 → coalesced ✓
        smem(ty, tx) = A(grow, gcol);
    }
    __syncthreads();

    // Phase 2: 从 smem 转置读 → coalesced 写全局内存
    // 输出 block 位置：行 bx*TILE + ty，列 by*TILE + tx（行列互换）
    int orow = bx * TILE + ty;
    int ocol = by * TILE + tx;

    if (orow < cols && ocol < rows) {
        // smem(tx, ty)：同 warp tx=0..31 → stride=TILE+1 → 不同 bank ✓
        // B(orow, ocol)：同 warp tx=0..31 → 连续列 → coalesced ✓
        B(orow, ocol) = smem(tx, ty);
    }
}

// -------------------------------------------------------------------------
// Host 函数
// -------------------------------------------------------------------------
extern "C" {

void transpose_cutlass_v1(const float* input, float* output, int rows, int cols) {
    dim3 block(TILE, TILE);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
    transpose_cute_v1<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}

void transpose_cutlass_v2(const float* input, float* output, int rows, int cols) {
    dim3 block(TILE, TILE);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
    transpose_cute_v2<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}

}  // extern "C"
