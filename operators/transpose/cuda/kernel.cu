/*
 * Matrix Transpose CUDA Kernel
 *
 * 问题：给定 M×N 矩阵 A（行主序），计算 N×M 矩阵 B = A^T
 *
 * =========================================================================
 * V1: Naive（内存不连续，写入不 coalesced）
 * =========================================================================
 * 每个 thread 处理一个元素：B[j][i] = A[i][j]
 *
 * 问题分析：
 *   - 读 A[i][j]：连续 thread 读连续列 → coalesced ✓
 *   - 写 B[j][i]：连续 thread 写不连续行 → scattered write ✗
 *   - 对于写操作来说，每个 warp（32 threads）写到内存的不同行，
 *     导致 32 个 cache line 的无效写，带宽利用率极低
 *
 * =========================================================================
 * V2: Shared Memory Tiling（消除 bank conflict + coalesced 访问）
 * =========================================================================
 * 核心思想：
 *   1. 每个 block 处理一个 TILE×TILE 的子矩阵
 *   2. 先从全局内存 coalesced 读取 tile 到 shared memory
 *   3. 在 shared memory 中完成 transpose（局部操作）
 *   4. 再从 shared memory coalesced 写回全局内存
 *
 * Shared Memory Bank Conflict 问题：
 *   - GPU shared memory 分为 32 个 bank（每 bank 4 字节）
 *   - 如果一个 warp 的 32 个 thread 同时访问同一 bank → bank conflict
 *   - 在 32×32 tile 的 transpose 中，列读取 s[0][0], s[1][0], ..., s[31][0]
 *     都落在 bank 0（每行 32 个 float，间隔 128 字节，32 bank × 4 = 128）→ 严重冲突！
 *   - 解决：在 shared memory 数组加 1 列 padding：s[TILE][TILE+1]
 *     使得行跨度变为 (TILE+1)*4 字节，不再对齐到同一 bank
 *
 * 内存访问模式（32×32 tile，每个 thread 处理 NUM_PER_THREAD 行）：
 *   load:  thread(tx,ty) 读 A[by*T + ty + k*stride][bx*T + tx]  → coalesced
 *   store: thread(tx,ty) 写 B[bx*T + ty + k*stride][by*T + tx]  → coalesced
 *          (通过 shared memory 中转完成行列互换)
 *
 * =========================================================================
 * 性能对比（参考 A100 80GB）：
 *   Naive    :  ~120 GB/s  (peak BW ~2000 GB/s, 利用率 ~6%)
 *   Shared   :  ~800 GB/s  (利用率 ~40%)
 *   cuBLAS   : ~1600 GB/s  (利用率 ~80%)
 * =========================================================================
 */

#include <cuda_runtime.h>

#define TILE 32

// -------------------------------------------------------------------------
// V1: Naive - 直接转置，写不 coalesced
// -------------------------------------------------------------------------
__global__ void transpose_v1_naive(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // 列方向
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // 行方向

    if (x < cols && y < rows) {
        // 读: input[y*cols + x]  (连续行方向 → coalesced)
        // 写: output[x*rows + y] (连续列方向 → NOT coalesced!)
        output[x * rows + y] = input[y * cols + x];
    }
}

// -------------------------------------------------------------------------
// V2: Shared Memory Tiling + Bank Conflict 避免 (padding +1)
//
// block 大小: (TILE, TILE/NUM_PER_THREAD) = (32, 8)
// 每个 thread 处理 NUM_PER_THREAD 行 (4 行)
// -------------------------------------------------------------------------
template <int BLOCK_SZ, int NUM_PER_THREAD>
__global__ void transpose_v2_shared(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int rows, int cols) {
    // BLOCK_SZ+1: padding 消除 bank conflict
    __shared__ float sdata[BLOCK_SZ][BLOCK_SZ + 1];

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    // x: 全局列索引（对应 input 的列维度）
    int x = bx * BLOCK_SZ + tx;
    // y: 全局行索引（当前 block 的起始行）
    int y = by * BLOCK_SZ + ty;

    // 每个 thread 处理 NUM_PER_THREAD 行
    // ROW_STRIDE = BLOCK_SZ / NUM_PER_THREAD = 32/4 = 8
    constexpr int ROW_STRIDE = BLOCK_SZ / NUM_PER_THREAD;

    // === Step 1: 从全局内存 coalesced 读到 shared memory ===
    // 同一 warp 中的 32 个 thread，tx = 0..31，x 连续 → coalesced 读
    if (x < cols) {
        #pragma unroll
        for (int y_off = 0; y_off < BLOCK_SZ; y_off += ROW_STRIDE) {
            if (y + y_off < rows) {
                sdata[ty + y_off][tx] = input[(y + y_off) * cols + x];
            }
        }
    }
    __syncthreads();  // 等待所有 thread 完成 shared memory 写入

    // === Step 2: 从 shared memory 读取转置后的数据，写回全局内存 ===
    // 注意：这里 x 和 y 的含义互换了
    x = by * BLOCK_SZ + tx;  // 现在 x 对应 output 的列（= input 的行 block）
    y = bx * BLOCK_SZ + ty;  // 现在 y 对应 output 的行（= input 的列 block）

    if (x < rows) {
        #pragma unroll
        for (int y_off = 0; y_off < BLOCK_SZ; y_off += ROW_STRIDE) {
            if (y + y_off < cols) {
                // sdata[tx][ty+y_off]: 读 shared memory 的列 → 实现转置
                // 写到 output[(y+y_off)*rows + x]: 连续写 → coalesced
                output[(y + y_off) * rows + x] = sdata[tx][ty + y_off];
            }
        }
    }
}

// -------------------------------------------------------------------------
// Host 函数
// -------------------------------------------------------------------------
extern "C" {

void transpose_cuda_v1(const float* input, float* output, int rows, int cols) {
    dim3 block(TILE, TILE);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
    transpose_v1_naive<<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}

void transpose_cuda_v2(const float* input, float* output, int rows, int cols) {
    constexpr int BLOCK_SZ = 32;
    constexpr int NUM_PER_THREAD = 4;
    dim3 block(BLOCK_SZ, BLOCK_SZ / NUM_PER_THREAD);  // (32, 8)
    dim3 grid((cols + BLOCK_SZ - 1) / BLOCK_SZ, (rows + BLOCK_SZ - 1) / BLOCK_SZ);
    transpose_v2_shared<BLOCK_SZ, NUM_PER_THREAD><<<grid, block>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}

} // extern "C"
