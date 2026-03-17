/*
 * RMSNorm — CuTe C++ 实现
 *
 * 演示 CuTe Tensor 包装 + warp reduce 在 RMSNorm 场景的用法。
 *
 * V1: CuTe Tensor 包装 2D (B,N) 行，warp-reduce sum-of-squares
 *   两趟：
 *     Pass 1: 累积 sum(x²) → rms_inv
 *     Pass 2: y = x * rms_inv * w
 *
 * V2: CuTe + float4 向量化加载
 *   通过 make_tensor 以 stride=(N/4, 4, 1) 等效实现向量化，
 *   实际仍以 float4 方式提升带宽利用率
 */

#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cuda_runtime.h>
#include <math.h>

using namespace cute;

// -------------------------------------------------------------------------
// Warp reduce sum
// -------------------------------------------------------------------------
__device__ __forceinline__
float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

#define MAX_WARPS 32

// -------------------------------------------------------------------------
// V1: Two-pass + CuTe Tensor 包装
// -------------------------------------------------------------------------
__global__ void rms_norm_cute_v1(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float*       __restrict__ y,
    int N, float eps)
{
    extern __shared__ float smem[];

    int tid      = threadIdx.x;
    int warp_id  = tid / 32;
    int lane     = tid % 32;
    int num_warps = blockDim.x / 32;
    int row      = blockIdx.x;

    // CuTe Tensor 包装当前行
    auto xrow = make_tensor(make_gmem_ptr(x + row * N), make_layout(N));
    auto yrow = make_tensor(make_gmem_ptr(y + row * N), make_layout(N));
    auto wvec = make_tensor(make_gmem_ptr(w),            make_layout(N));

    // Pass 1: 累积 sum(x²) via CuTe 索引
    float local_ss = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float xi = xrow(i);
        local_ss += xi * xi;
    }

    local_ss = warp_reduce_sum(local_ss);
    if (lane == 0) smem[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < num_warps) ? smem[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();

    float rms_inv = rsqrtf(smem[0] / N + eps);

    // Pass 2: y = x * rms_inv * w
    for (int i = tid; i < N; i += blockDim.x) {
        yrow(i) = xrow(i) * rms_inv * wvec(i);
    }
}

// -------------------------------------------------------------------------
// V2: CuTe + float4 向量化
// -------------------------------------------------------------------------
__global__ void rms_norm_cute_v2(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float*       __restrict__ y,
    int N, float eps)
{
    extern __shared__ float smem[];

    int tid      = threadIdx.x;
    int warp_id  = tid / 32;
    int lane     = tid % 32;
    int num_warps = blockDim.x / 32;
    int row      = blockIdx.x;
    int N4       = N / 4;

    // CuTe Tensor 以 float4 stride 包装（每 4 个元素一组）
    // layout: (N4, 4), stride: (4, 1) — 等效 float4 视图
    auto xrow4 = make_tensor(make_gmem_ptr(reinterpret_cast<const float4*>(x + row * N)),
                             make_layout(N4));
    auto wvec4 = make_tensor(make_gmem_ptr(reinterpret_cast<const float4*>(w)),
                             make_layout(N4));
    auto yrow4 = make_tensor(make_gmem_ptr(reinterpret_cast<float4*>(y + row * N)),
                             make_layout(N4));

    // Pass 1: sum(x²) via float4
    float local_ss = 0.0f;
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 xi = xrow4(i);
        local_ss += xi.x*xi.x + xi.y*xi.y + xi.z*xi.z + xi.w*xi.w;
    }

    local_ss = warp_reduce_sum(local_ss);
    if (lane == 0) smem[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < num_warps) ? smem[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();

    float rms_inv = rsqrtf(smem[0] / N + eps);

    // Pass 2: float4 写出
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 xi = xrow4(i);
        float4 wi = wvec4(i);
        float4 out;
        out.x = xi.x * rms_inv * wi.x;
        out.y = xi.y * rms_inv * wi.y;
        out.z = xi.z * rms_inv * wi.z;
        out.w = xi.w * rms_inv * wi.w;
        yrow4(i) = out;
    }
    // 处理尾部
    const float* xbase = x + row * N;
    float*       ybase = y + row * N;
    for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
        ybase[i] = xbase[i] * rms_inv * w[i];
    }
}

// -------------------------------------------------------------------------
// Host 函数
// -------------------------------------------------------------------------
extern "C" {

void rms_norm_cutlass_v1(float* x, float* w, float* y, int B, int N, float eps) {
    int threads = 256;
    size_t smem = MAX_WARPS * sizeof(float);
    rms_norm_cute_v1<<<B, threads, smem>>>(x, w, y, N, eps);
    cudaDeviceSynchronize();
}

void rms_norm_cutlass_v2(float* x, float* w, float* y, int B, int N, float eps) {
    int threads = 256;
    size_t smem = MAX_WARPS * sizeof(float);
    rms_norm_cute_v2<<<B, threads, smem>>>(x, w, y, N, eps);
    cudaDeviceSynchronize();
}

} // extern "C"
