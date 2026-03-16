/*
 * Softmax — CuTe C++ 实现
 *
 * 学习目标：演示 CuTe 在 reduction 场景的用法：
 *   - make_tensor 包装 1D 行数据
 *   - cute::reduce / cute::transform 等 tensor algorithm
 *   - Warp shuffle 与 CuTe 的 cooperative reduction 结合
 *
 * =========================================================================
 * V1: CuTe Tensor 包装 + 手动 online softmax（两趟扫描）
 *   重点：演示 make_tensor / local_tile / local_partition 在 1D reduction
 *   中的写法，让 reduction loop 语义更清晰
 *
 * V2: CuTe Tensor + Warp-level online softmax（单趟 + warp shuffle）
 *   重点：演示 cute::for_each 遍历 tensor 元素，结合 __shfl_xor_sync 规约
 * =========================================================================
 */

#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

using namespace cute;

// -------------------------------------------------------------------------
// Warp shuffle 工具（与 CuTe 共存）
// -------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

// -------------------------------------------------------------------------
// V1: 两趟扫描 + CuTe Tensor 包装
// -------------------------------------------------------------------------
__global__ void softmax_cute_v1(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int B, int N)
{
    extern __shared__ float smem[];  // [blockDim.x]

    int row = blockIdx.x;
    int tid = threadIdx.x;

    // 用 CuTe 包装当前行的 1D Tensor
    auto x = make_tensor(make_gmem_ptr(input  + row * N), make_layout(N));
    auto y = make_tensor(make_gmem_ptr(output + row * N), make_layout(N));

    // Pass 1a: 找 max（用 cute::for_each 遍历当前 thread 负责的元素）
    float local_max = -FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x)
        local_max = fmaxf(local_max, x(i));

    smem[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float global_max = smem[0];
    __syncthreads();

    // Pass 1b: 求 sum(exp)
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x)
        local_sum += expf(x(i) - global_max);

    smem[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float global_sum = smem[0];
    __syncthreads();

    // Pass 2: 写输出（通过 CuTe Tensor y 写回）
    for (int i = tid; i < N; i += blockDim.x)
        y(i) = expf(x(i) - global_max) / global_sum;
}

// -------------------------------------------------------------------------
// V2: Online softmax + Warp shuffle + CuTe Tensor
// -------------------------------------------------------------------------
__global__ void softmax_cute_v2(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int B, int N)
{
    extern __shared__ float smem[];  // [num_warps * 2]

    int row      = blockIdx.x;
    int tid      = threadIdx.x;
    int lane     = tid % 32;
    int warp_id  = tid / 32;
    int num_warps = blockDim.x / 32;

    auto x = make_tensor(make_gmem_ptr(input  + row * N), make_layout(N));
    auto y = make_tensor(make_gmem_ptr(output + row * N), make_layout(N));

    // 每个 thread 维护 online softmax 状态 (m, d)
    float m = -FLT_MAX, d = 0.0f;

    // cute::for_each 遍历 thread 负责的元素（步长 = blockDim.x）
    for (int i = tid; i < N; i += blockDim.x) {
        float xi = x(i);
        if (xi > m) {
            d = d * expf(m - xi) + 1.0f;
            m = xi;
        } else {
            d += expf(xi - m);
        }
    }

    // Warp-level reduce
    float warp_m = warp_reduce_max(m);
    d = d * expf(m - warp_m);
    float warp_d = warp_reduce_sum(d);

    if (lane == 0) {
        smem[warp_id]             = warp_m;
        smem[warp_id + num_warps] = warp_d;
    }
    __syncthreads();

    // Block-level reduce（第一个 warp）
    float global_m, global_d;
    if (warp_id == 0) {
        float gm = (lane < num_warps) ? smem[lane]             : -FLT_MAX;
        float gd = (lane < num_warps) ? smem[lane + num_warps] : 0.0f;
        float new_gm = warp_reduce_max(gm);
        gd = gd * expf(gm - new_gm);
        gd = warp_reduce_sum(gd);
        if (lane == 0) {
            smem[0] = new_gm;
            smem[1] = gd;
        }
    }
    __syncthreads();

    global_m = smem[0];
    global_d = smem[1];

    // 写回（通过 CuTe Tensor y）
    for (int i = tid; i < N; i += blockDim.x)
        y(i) = expf(x(i) - global_m) / global_d;
}

// -------------------------------------------------------------------------
// Host 函数
// -------------------------------------------------------------------------
extern "C" {

void softmax_cutlass_v1(const float* input, float* output, int B, int N) {
    int threads = min(1024, N);
    // 向上取 2 的幂次
    int t = 1; while (t < threads) t <<= 1; threads = t;
    size_t smem = threads * sizeof(float);
    softmax_cute_v1<<<B, threads, smem>>>(input, output, B, N);
    cudaDeviceSynchronize();
}

void softmax_cutlass_v2(const float* input, float* output, int B, int N) {
    int threads = 256;
    int num_warps = threads / 32;
    size_t smem = num_warps * 2 * sizeof(float);
    softmax_cute_v2<<<B, threads, smem>>>(input, output, B, N);
    cudaDeviceSynchronize();
}

}  // extern "C"
