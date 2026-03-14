/*
 * Matrix Multiplication CUDA Kernel
 *
 * 计算: C = A × B，其中 A(M,K), B(K,N), C(M,N)
 *
 * FLOP 数: 2 * M * N * K（乘加各一次）
 *
 * =========================================================================
 * 性能分析（以 A100 为例）：
 *   Peak FP32: ~19.5 TFLOPS
 *   Peak BW:   ~2000 GB/s
 *
 *   对于 M=N=K=4096：
 *     FLOP  = 2 * 4096³ ≈ 137 GFLOP
 *     bytes = (M*K + K*N + M*N) * 4 ≈ 201 MB
 *     Arithmetic intensity = 137G / 201M ≈ 682 FLOP/byte
 *     A100 的 ridge point ≈ 19.5T / 2000G ≈ 9.75 FLOP/byte
 *
 *   → Matmul 是 **compute bound**（远大于 ridge point）
 *   → 目标：最大化 TFLOPS 而非内存带宽
 *
 * =========================================================================
 * V1: Naive（每个 thread 计算 C 的一个元素）
 *
 *   问题：每个 thread 独立读取 A 的一行和 B 的一列，大量冗余读取。
 *   对于 C[i][j]，需要读 A[i][0..K-1] 和 B[0..K-1][j]。
 *   总读取量 = M*N*(2K) 个 float → 大量重复读
 *
 * =========================================================================
 * V2: Shared Memory Tiling（核心优化）
 *
 * 思想：将 K 维度分成长度为 TILE_K 的块，每次把 A 和 B 的当前块加载到
 *       shared memory，然后 block 内的所有 thread 使用共享的数据计算。
 *
 * Tiling 原理：
 *   - A(M,K) 拆成 M/TILE_M × K/TILE_K 的块
 *   - B(K,N) 拆成 K/TILE_K × N/TILE_N 的块
 *   - 每个 block 计算 C 的一个 TILE_M × TILE_N 子块
 *   - 迭代 K/TILE_K 次，每次：
 *     1. 把 A 的 TILE_M×TILE_K 块加载到 sA
 *     2. 把 B 的 TILE_K×TILE_N 块加载到 sB
 *     3. block 内计算 sA × sB（外积累加到 C tile）
 *
 * 内存复用分析：
 *   - 没有 tiling：每个 C[i][j] 独立读 2K 个数 → 总 2MNK 次读
 *   - 有 tiling (TILE=32)：每个数据被复用 TILE 次 → 总 2MNK/TILE 次读
 *   - TILE=32 → 带宽需求降低 32×
 *
 * =========================================================================
 * V3: Double Buffering（隐藏内存延迟）
 *
 * 思想：overlapping computation with memory loading
 *   - 在计算当前 tile (k) 的同时，预取下一个 tile (k+1) 到另一组 shared memory
 *   - 需要两倍的 shared memory（双缓冲）
 *   - 使用 __pipeline_memcpy_async 进行异步内存拷贝（CUDA 11+）
 *
 * =========================================================================
 * Thread 分工（V2 以 TILE=32 为例）：
 *   - Block 大小: (TILE_N, TILE_M) = (32, 32) 个 thread
 *   - threadIdx.x → N 方向（列）
 *   - threadIdx.y → M 方向（行）
 *   - 每个 thread 负责 C 的一个元素
 *
 * =========================================================================
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define TILE 32

// -------------------------------------------------------------------------
// V1: Naive - 每个 thread 计算 C 的一个元素
// -------------------------------------------------------------------------
__global__ void matmul_v1_naive(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            // A[row][k] * B[k][col]
            // 注意：每次迭代，同一 warp 中 thread 读 B[k][col] 是跨行访问（不 coalesced）
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

// -------------------------------------------------------------------------
// V2: Shared Memory Tiling
//
// 每个 block (TILE×TILE threads) 计算 C 的 TILE×TILE 子块。
// 外层循环迭代 ceil(K/TILE) 次，每次将 TILE×TILE 的 A 和 B 子块
// 加载到 shared memory，然后计算局部乘积并累加。
// -------------------------------------------------------------------------
__global__ void matmul_v2_tiled(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int K, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;  // C 的行索引
    int col = blockIdx.x * TILE + tx;  // C 的列索引

    float acc = 0.0f;

    // 迭代 K 维度的所有 tile
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // 加载 A 的当前 tile：A[row][t*TILE + tx]
        if (row < M && (t * TILE + tx) < K) {
            sA[ty][tx] = A[row * K + t * TILE + tx];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // 加载 B 的当前 tile：B[t*TILE + ty][col]
        if ((t * TILE + ty) < K && col < N) {
            sB[ty][tx] = B[(t * TILE + ty) * N + col];
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();  // 确保 shared memory 加载完成

        // 计算当前 tile 的贡献
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();  // 确保计算完成再加载下一个 tile
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// -------------------------------------------------------------------------
// V3: 每个 thread 计算 C 的多个元素（Thread Coarsening）
//
// 思想：增加每个 thread 的工作量，减少 thread 数量，
//       更好地利用寄存器，减少 shared memory 访问次数。
//
// 这里每个 thread 计算 2×2 = 4 个 C 元素（TH=2）。
// block 大小: (TILE/TH × TILE/TH) = (16×16) threads
// -------------------------------------------------------------------------
#define TILE2 64  // block 处理 64×64 的 C tile
#define TH    4   // 每个 thread 处理 4×4 = 16 个 C 元素

__global__ void matmul_v3_coarsened(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int K, int N) {
    // 每个 thread 负责 TH×TH = 4×4 个 C 元素
    // block = (32/4) × (32/4) = 8×8 = 64 threads，处理 32×32 的 C tile

    const int BTILE = 32;
    __shared__ float sa[BTILE][BTILE + 1];
    __shared__ float sb[BTILE][BTILE + 1];

    int tx = threadIdx.x, ty = threadIdx.y;  // thread 在 (BTILE/TH × BTILE/TH) block 中的位置
    int bm = blockIdx.y * BTILE;
    int bn = blockIdx.x * BTILE;

    // 每个 thread 负责 TH×TH 个 C 元素
    float acc[TH][TH] = {};

    for (int t = 0; t < (K + BTILE - 1) / BTILE; t++) {
        // 每个 thread (tx, ty) in (8×8) block 加载 TH×TH = 4×4 = 16 个元素到 shared memory
        // sa[ty*TH+i][tx*TH+j] = A[bm + ty*TH+i][t*BTILE + tx*TH+j]
        // sb[ty*TH+i][tx*TH+j] = B[t*BTILE + ty*TH+i][bn + tx*TH+j]
        #pragma unroll
        for (int i = 0; i < TH; i++) {
            #pragma unroll
            for (int j = 0; j < TH; j++) {
                int row = bm + ty * TH + i;
                int col_k = t * BTILE + tx * TH + j;
                sa[ty * TH + i][tx * TH + j] = (row < M && col_k < K) ? A[row * K + col_k] : 0.0f;
            }
        }
        #pragma unroll
        for (int i = 0; i < TH; i++) {
            #pragma unroll
            for (int j = 0; j < TH; j++) {
                int row_k = t * BTILE + ty * TH + i;
                int col = bn + tx * TH + j;
                sb[ty * TH + i][tx * TH + j] = (row_k < K && col < N) ? B[row_k * N + col] : 0.0f;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BTILE; k++) {
            #pragma unroll
            for (int i = 0; i < TH; i++) {
                float a = sa[ty * TH + i][k];
                #pragma unroll
                for (int j = 0; j < TH; j++) {
                    acc[i][j] += a * sb[k][tx * TH + j];
                }
            }
        }
        __syncthreads();
    }

    // 写回 C
    #pragma unroll
    for (int i = 0; i < TH; i++) {
        #pragma unroll
        for (int j = 0; j < TH; j++) {
            int row = bm + ty * TH + i;
            int col = bn + tx * TH + j;
            if (row < M && col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}

// -------------------------------------------------------------------------
// Host 函数
// -------------------------------------------------------------------------
extern "C" {

void matmul_cuda_v1(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_v1_naive<<<grid, block>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}

void matmul_cuda_v2(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_v2_tiled<<<grid, block>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}

void matmul_cuda_v3(const float* A, const float* B, float* C, int M, int K, int N) {
    // block = (BTILE/TH) × (BTILE/TH) = 8×8 threads
    const int BTILE = 32;
    const int th = TH;
    dim3 block(BTILE / th, BTILE / th);  // (8, 8)
    dim3 grid((N + BTILE - 1) / BTILE, (M + BTILE - 1) / BTILE);
    matmul_v3_coarsened<<<grid, block>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}

} // extern "C"
