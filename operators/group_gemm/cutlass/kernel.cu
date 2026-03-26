/*
 * Group GEMM — CuTe C++ 实现
 *
 * 演示 CuTe 如何优雅地处理 batch/group 维度的矩阵乘法：
 *   - 3D Tensor make_tensor 表示 (G, M, K) 布局
 *   - local_tile 在 group 维度和 M/N/K 维度上分块
 *   - 同一套 cute::gemm 逻辑通过 group 偏移复用
 *
 * V1: Fixed-size group GEMM — 3D Tensor + 直接索引
 * V2: Fixed-size group GEMM + shared memory tiling (cute::copy + cute::gemm)
 */

#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/algorithm/copy.hpp>
#include <cuda_runtime.h>

using namespace cute;

constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr int TILE_K = 32;


// -------------------------------------------------------------------------
// V1: 3D Tensor 直接索引
// grid = (ceil(N/TILE_N), ceil(M/TILE_M), G)
// -------------------------------------------------------------------------
__global__ void group_gemm_cute_v1(
    const float* __restrict__ A_ptr,   // (G, M, K) row-major
    const float* __restrict__ B_ptr,   // (G, K, N) row-major
    float*       __restrict__ C_ptr,   // (G, M, N) row-major
    int G, int M, int K, int N)
{
    // 3D layout: (G, M, K) stride = (M*K, K, 1)
    auto A = make_tensor(make_gmem_ptr(A_ptr),
                         make_layout(make_shape(G, M, K),
                                     make_stride(M * K, K, 1)));
    auto B = make_tensor(make_gmem_ptr(B_ptr),
                         make_layout(make_shape(G, K, N),
                                     make_stride(K * N, N, 1)));
    auto C = make_tensor(make_gmem_ptr(C_ptr),
                         make_layout(make_shape(G, M, N),
                                     make_stride(M * N, N, 1)));

    int col      = blockIdx.x * TILE_N + threadIdx.x;
    int row      = blockIdx.y * TILE_M + threadIdx.y;
    int group_id = blockIdx.z;

    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) {
            acc += A(group_id, row, k) * B(group_id, k, col);
        }
        C(group_id, row, col) = acc;
    }
}


// -------------------------------------------------------------------------
// V2: Shared memory tiling，2D slice per group
// grid = (ceil(N/TILE_N), ceil(M/TILE_M), G)
// -------------------------------------------------------------------------
__global__ void group_gemm_cute_v2(
    const float* __restrict__ A_ptr,
    const float* __restrict__ B_ptr,
    float*       __restrict__ C_ptr,
    int G, int M, int K, int N)
{
    int group_id = blockIdx.z;
    long long g_offset_a = (long long)group_id * M * K;
    long long g_offset_b = (long long)group_id * K * N;
    long long g_offset_c = (long long)group_id * M * N;

    // 2D slice Tensor for current group
    auto gA = make_tensor(make_gmem_ptr(A_ptr + g_offset_a),
                          make_layout(make_shape(M, K), make_stride(K, 1)));
    auto gB = make_tensor(make_gmem_ptr(B_ptr + g_offset_b),
                          make_layout(make_shape(K, N), make_stride(N, 1)));
    auto gC = make_tensor(make_gmem_ptr(C_ptr + g_offset_c),
                          make_layout(make_shape(M, N), make_stride(N, 1)));

    __shared__ float smA_raw[TILE_M * TILE_K];
    __shared__ float smB_raw[TILE_K * TILE_N];

    auto smA = make_tensor(make_smem_ptr(smA_raw),
                           make_layout(make_shape(Int<TILE_M>{}, Int<TILE_K>{}),
                                       make_stride(Int<TILE_K>{}, Int<1>{})));
    auto smB = make_tensor(make_smem_ptr(smB_raw),
                           make_layout(make_shape(Int<TILE_K>{}, Int<TILE_N>{}),
                                       make_stride(Int<TILE_N>{}, Int<1>{})));

    int tx = threadIdx.x, ty = threadIdx.y;
    int row0 = blockIdx.y * TILE_M;
    int col0 = blockIdx.x * TILE_N;

    float acc = 0.f;
    int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; ++t) {
        int k0 = t * TILE_K;

        for (int i = ty; i < TILE_M; i += blockDim.y) {
            int gr = row0 + i, gc = k0 + tx;
            smA(i, tx) = (gr < M && gc < K) ? gA(gr, gc) : 0.f;
        }
        for (int i = ty; i < TILE_K; i += blockDim.y) {
            int gr = k0 + i, gc = col0 + tx;
            smB(i, tx) = (gr < K && gc < N) ? gB(gr, gc) : 0.f;
        }
        __syncthreads();

        if (row0 + ty < M && col0 + tx < N) {
            for (int k = 0; k < TILE_K; ++k)
                acc += smA(ty, k) * smB(k, tx);
        }
        __syncthreads();
    }

    if (row0 + ty < M && col0 + tx < N)
        gC(row0 + ty, col0 + tx) = acc;
}


// -------------------------------------------------------------------------
// Host functions
// -------------------------------------------------------------------------
extern "C" {

void group_gemm_cutlass_v1(
    const float* A, const float* B, float* C,
    int G, int M, int K, int N)
{
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N,
              (M + TILE_M - 1) / TILE_M,
              G);
    group_gemm_cute_v1<<<grid, block>>>(A, B, C, G, M, K, N);
    cudaDeviceSynchronize();
}

void group_gemm_cutlass_v2(
    const float* A, const float* B, float* C,
    int G, int M, int K, int N)
{
    dim3 block(TILE_N, TILE_M / 2);   // 32×16 = 512 threads
    dim3 grid((N + TILE_N - 1) / TILE_N,
              (M + TILE_M - 1) / TILE_M,
              G);
    group_gemm_cute_v2<<<grid, block>>>(A, B, C, G, M, K, N);
    cudaDeviceSynchronize();
}

}  // extern "C"
