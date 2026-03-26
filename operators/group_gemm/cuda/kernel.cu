/*
 * Group GEMM CUDA Kernel
 *
 * 计算 G 组独立的矩阵乘法：C[g] = A[g] × B[g]
 *
 * =========================================================================
 * 背景：Group GEMM vs Batched GEMM
 *   - Batched GEMM：所有 batch 的 (M, K, N) 相同，cuBLAS 直接支持
 *   - Group GEMM：每组的 (M_g, K_g, N_g) 可以不同，常见于 MoE（混合专家模型）
 *
 * 在 Transformer MoE 层中，不同 expert 会分到不同数量的 token，
 * 因此需要 group GEMM 来并行处理大小不一的 GEMM 问题。
 *
 * =========================================================================
 * 内存布局（flat / jagged）
 *
 *   A_flat: (sum_M, K)  所有 group 的 A 矩阵按行拼接
 *   B_flat: (G*K, N)    所有 group 的 B 矩阵按行拼接（等 K 时简单）
 *   C_flat: (sum_M, N)  所有 group 的 C 矩阵按行拼接
 *
 *   a_offsets[g]: group g 在 A_flat 的起始行
 *   b_offsets[g]: group g 在 B_flat 的起始行
 *   c_offsets[g]: group g 在 C_flat 的起始行
 *   m_sizes[g]:   group g 的 M_g
 *
 * =========================================================================
 * V1: Fixed-size group GEMM（所有 group 的 M, K, N 相同）
 *
 *   grid = (ceil(N/TILE), ceil(M/TILE), G)
 *   blockIdx.z = group_id → 直接偏移到该 group 的数据
 *   教学重点：如何用 3D grid 处理 batch 维度
 *
 * V2: Variable-size group GEMM（M_g 可不同，K/N 相同）
 *
 *   grid = (ceil(N/TILE) * MAX_TILES_M, G)
 *   通过 a_offsets / m_sizes 查表 → 支持变大小
 *   教学重点：prefix-sum offset 数组的使用
 *
 * V3: Fixed-size，Shared Memory Tiling（V1 + tiling 优化）
 *
 *   与 V1 相同的 grid，但加入 shared memory tiling
 *   减少 HBM 访问，提升 compute/memory 比
 * =========================================================================
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define TILE 32

// -------------------------------------------------------------------------
// V1: Fixed-size group GEMM (naive, no shared memory)
// grid = (ceil(N/TILE), ceil(M/TILE), G)
// -------------------------------------------------------------------------
__global__ void group_gemm_v1_fixed(
    const float* __restrict__ A,    // (G, M, K)
    const float* __restrict__ B,    // (G, K, N)
    float*       __restrict__ C,    // (G, M, N)
    int G, int M, int K, int N)
{
    int col      = blockIdx.x * TILE + threadIdx.x;
    int row      = blockIdx.y * TILE + threadIdx.y;
    int group_id = blockIdx.z;

    if (row >= M || col >= N) return;

    // 定位当前 group 的 A/B/C 起始地址
    const float* a = A + (long long)group_id * M * K;
    const float* b = B + (long long)group_id * K * N;
    float*       c = C + (long long)group_id * M * N;

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += a[row * K + k] * b[k * N + col];
    }
    c[row * N + col] = acc;
}


// -------------------------------------------------------------------------
// V2: Variable-size group GEMM (jagged A, fixed K, N)
// grid = (MAX_TILES_M * ceil(N/TILE), G)
// a_offsets[g]: group g 的 A 矩阵在 A_flat 中的起始行
// m_sizes[g]:   group g 的 M_g
// -------------------------------------------------------------------------
__global__ void group_gemm_v2_var(
    const float* __restrict__ A_flat,   // (sum_M, K)
    const float* __restrict__ B_flat,   // (G, K, N)  each group K×N
    float*       __restrict__ C_flat,   // (sum_M, N)
    const int*   __restrict__ a_offsets,  // [G]
    const int*   __restrict__ m_sizes,    // [G]
    int G, int K, int N)
{
    int tile_id  = blockIdx.x;  // flat tile index within group
    int group_id = blockIdx.y;

    int num_tiles_n = (N + TILE - 1) / TILE;
    int tile_m = tile_id / num_tiles_n;
    int tile_n = tile_id % num_tiles_n;

    int M_g = m_sizes[group_id];
    int row  = tile_m * TILE + threadIdx.y;
    int col  = tile_n * TILE + threadIdx.x;

    if (row >= M_g || col >= N) return;

    int a_row0 = a_offsets[group_id];
    int c_row0 = a_offsets[group_id];   // same prefix sum for C

    const float* a = A_flat + (long long)(a_row0 + row) * K;
    const float* b = B_flat + (long long)group_id * K * N;
    float*       c = C_flat + (long long)(c_row0 + row) * N;

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += a[k] * b[k * N + col];
    }
    c[col] = acc;
}


// -------------------------------------------------------------------------
// V3: Fixed-size group GEMM + shared memory tiling
// grid = (ceil(N/TILE), ceil(M/TILE), G)
// -------------------------------------------------------------------------
__global__ void group_gemm_v3_tiled(
    const float* __restrict__ A,    // (G, M, K)
    const float* __restrict__ B,    // (G, K, N)
    float*       __restrict__ C,    // (G, M, N)
    int G, int M, int K, int N)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row      = blockIdx.y * TILE + ty;
    int col      = blockIdx.x * TILE + tx;
    int group_id = blockIdx.z;

    const float* a = A + (long long)group_id * M * K;
    const float* b = B + (long long)group_id * K * N;
    float*       c = C + (long long)group_id * M * N;

    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int ak_col = t * TILE + tx;
        int bk_row = t * TILE + ty;

        sA[ty][tx] = (row < M && ak_col < K) ? a[row * K + ak_col] : 0.0f;
        sB[ty][tx] = (bk_row < K && col < N) ? b[bk_row * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        c[row * N + col] = acc;
    }
}


// -------------------------------------------------------------------------
// Host functions (extern "C" for ctypes)
// -------------------------------------------------------------------------
extern "C" {

void group_gemm_cuda_v1(
    const float* A, const float* B, float* C,
    int G, int M, int K, int N)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, G);
    group_gemm_v1_fixed<<<grid, block>>>(A, B, C, G, M, K, N);
    cudaDeviceSynchronize();
}

void group_gemm_cuda_v2(
    const float* A_flat, const float* B_flat, float* C_flat,
    const int* a_offsets, const int* m_sizes,
    int G, int total_M, int K, int N)
{
    // 预估每组最大的 M_g 以决定 tile 数
    // 为简单起见用 total_M 作上界
    int max_tiles_m = (total_M + TILE - 1) / TILE;
    int num_tiles_n = (N + TILE - 1) / TILE;
    dim3 block(TILE, TILE);
    dim3 grid(max_tiles_m * num_tiles_n, G);
    group_gemm_v2_var<<<grid, block>>>(
        A_flat, B_flat, C_flat,
        a_offsets, m_sizes,
        G, K, N);
    cudaDeviceSynchronize();
}

void group_gemm_cuda_v3(
    const float* A, const float* B, float* C,
    int G, int M, int K, int N)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, G);
    group_gemm_v3_tiled<<<grid, block>>>(A, B, C, G, M, K, N);
    cudaDeviceSynchronize();
}

}  // extern "C"
