/*
 * Vector Add CUDA Kernel
 *
 * 原理：
 *   C[i] = A[i] + B[i]
 *
 * 并行策略：
 *   每个 CUDA thread 负责计算一个元素。
 *   grid = ceil(N / BLOCK_SIZE)，block = BLOCK_SIZE。
 *
 * 关键概念：
 *   - threadIdx.x: block 内的线程索引
 *   - blockIdx.x:  block 在 grid 中的索引
 *   - blockDim.x:  每个 block 的线程数
 *   - 全局线程 ID = blockIdx.x * blockDim.x + threadIdx.x
 *   - 边界检查：若 N 不是 BLOCK_SIZE 的整数倍，最后一个 block 中部分 thread 不做计算
 *
 * 版本1（naive）: 每个 thread 处理 1 个元素
 * 版本2（向量化）: 每个 thread 通过 float4 处理 4 个元素，提升内存吞吐
 */

#include <cuda_runtime.h>
#include <stdio.h>

// -------------------------------------------------------------------------
// Kernel v1: 每个 thread 处理 1 个 float
// -------------------------------------------------------------------------
__global__ void vector_add_v1(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// -------------------------------------------------------------------------
// Kernel v2: 向量化读写，每个 thread 处理 4 个 float（float4）
//
// float4 是 CUDA 内置的 128-bit 向量类型，一次 LD/ST 指令加载 4 个 float。
// 这样做的好处：
//   1. 减少指令发射次数（4x fewer memory instructions）
//   2. 更好地利用内存带宽（128-bit 事务对齐时是最优的）
//   3. 减少地址计算开销
//
// 注意：要求 N 是 4 的倍数（实际中可用 padding 保证）
// -------------------------------------------------------------------------
__global__ void vector_add_v2_vectorized(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int N) {
    // 把 float* 转为 float4* 进行向量化访问
    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    float4*       C4 = reinterpret_cast<float4*>(C);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = N / 4;

    if (i < n4) {
        float4 a = A4[i];
        float4 b = B4[i];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        C4[i] = c;
    }
    // 处理尾部（N 不是 4 的倍数时）
    int tail_start = n4 * 4;
    if (i == 0) {
        for (int j = tail_start; j < N; j++) {
            C[j] = A[j] + B[j];
        }
    }
}

// -------------------------------------------------------------------------
// Host 调用函数
// -------------------------------------------------------------------------
extern "C" {

void vector_add_cuda_v1(const float* A, const float* B, float* C, int N) {
    const int BLOCK_SIZE = 256;
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vector_add_v1<<<grid, BLOCK_SIZE>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

void vector_add_cuda_v2(const float* A, const float* B, float* C, int N) {
    const int BLOCK_SIZE = 256;
    int n4 = (N + 3) / 4;
    int grid = (n4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vector_add_v2_vectorized<<<grid, BLOCK_SIZE>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

} // extern "C"
