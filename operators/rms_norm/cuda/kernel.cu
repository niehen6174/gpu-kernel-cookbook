/*
 * RMSNorm CUDA Kernel
 *
 * 数学定义：
 *   rms(x) = sqrt( (1/N) * Σ x_i² + ε )
 *   y_i    = x_i / rms(x) * w_i
 *
 * 与 LayerNorm 的区别：不减均值，只做 RMS 归一化。
 *
 * =========================================================================
 * V1: Two-pass warp-reduce
 *   Pass 1: 并行累积 sum(x²)，warp reduce → block reduce → rms
 *   Pass 2: 逐元素应用 y[i] = x[i] / rms * w[i]
 *
 * V2: 向量化（float4）加载 + warp shuffle reduce
 *   每线程处理 4 个元素，提高带宽利用率
 *
 * V3: fused_add_rmsnorm
 *   先 residual = x + residual（inplace），再 RMSNorm
 * =========================================================================
 */

#include <cuda_runtime.h>
#include <math.h>

// -------------------------------------------------------------------------
// 工具：Warp reduce sum
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
// V1: Two-pass + warp/block reduce
// -------------------------------------------------------------------------
__global__ void rms_norm_v1(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float*       __restrict__ y,
    int N, float eps)
{
    extern __shared__ float smem[];  // MAX_WARPS floats

    int tid      = threadIdx.x;
    int warp_id  = tid / 32;
    int lane     = tid % 32;
    int num_warps = blockDim.x / 32;
    int row      = blockIdx.x;

    const float* xrow = x + row * N;
    float*       yrow = y + row * N;

    // Pass 1: 每线程累积 sum(x²)
    float local_ss = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float xi = xrow[i];
        local_ss += xi * xi;
    }

    // Warp reduce
    local_ss = warp_reduce_sum(local_ss);

    // 写到 shared memory
    if (lane == 0) smem[warp_id] = local_ss;
    __syncthreads();

    // Block reduce（第一个 warp）
    if (warp_id == 0) {
        float val = (lane < num_warps) ? smem[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();

    float rms_inv = rsqrtf(smem[0] / N + eps);

    // Pass 2: 归一化 + 乘 weight
    for (int i = tid; i < N; i += blockDim.x) {
        yrow[i] = xrow[i] * rms_inv * w[i];
    }
}

// -------------------------------------------------------------------------
// V2: Vectorized float4 + warp reduce
// -------------------------------------------------------------------------
__global__ void rms_norm_v2(
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

    const float4* x4 = reinterpret_cast<const float4*>(x + row * N);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    float4*       y4 = reinterpret_cast<float4*>(y + row * N);
    int N4 = N / 4;

    // Pass 1: float4 累积 sum(x²)
    float local_ss = 0.0f;
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 xi = x4[i];
        local_ss += xi.x * xi.x + xi.y * xi.y + xi.z * xi.z + xi.w * xi.w;
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
        float4 xi = x4[i];
        float4 wi = w4[i];
        float4 out;
        out.x = xi.x * rms_inv * wi.x;
        out.y = xi.y * rms_inv * wi.y;
        out.z = xi.z * rms_inv * wi.z;
        out.w = xi.w * rms_inv * wi.w;
        y4[i] = out;
    }
    // 处理尾部（N 不是 4 的倍数）
    int base = N4 * 4;
    for (int i = base + tid; i < N; i += blockDim.x) {
        y[row * N + i] = x[row * N + i] * rms_inv * w[i];
    }
}

// -------------------------------------------------------------------------
// V3: Fused residual add + RMSNorm (inplace on residual)
// -------------------------------------------------------------------------
__global__ void fused_add_rms_norm_v3(
    float*       __restrict__ x,        // input x
    float*       __restrict__ residual, // inplace updated: residual = x + residual
    const float* __restrict__ w,
    float*       __restrict__ y,        // normed output
    int N, float eps)
{
    extern __shared__ float smem[];

    int tid      = threadIdx.x;
    int warp_id  = tid / 32;
    int lane     = tid % 32;
    int num_warps = blockDim.x / 32;
    int row      = blockIdx.x;

    float* xrow = x + row * N;
    float* rrow = residual + row * N;
    float* yrow = y + row * N;

    // Pass 1: residual = x + residual, sum((x+r)²)
    float local_ss = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float xi = xrow[i] + rrow[i];
        rrow[i] = xi;      // update residual inplace
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

    // Pass 2: y = (x+r) / rms * w  (read from updated residual)
    for (int i = tid; i < N; i += blockDim.x) {
        yrow[i] = rrow[i] * rms_inv * w[i];
    }
}

// -------------------------------------------------------------------------
// Host 函数
// -------------------------------------------------------------------------
extern "C" {

void rms_norm_cuda_v1(float* x, float* w, float* y, int B, int N, float eps) {
    int threads = min(1024, (N + 31) / 32 * 32);
    threads = min(threads, 1024);
    size_t smem = MAX_WARPS * sizeof(float);
    rms_norm_v1<<<B, threads, smem>>>(x, w, y, N, eps);
    cudaDeviceSynchronize();
}

void rms_norm_cuda_v2(float* x, float* w, float* y, int B, int N, float eps) {
    // 要求 N 是 4 的倍数（常见隐藏维度都满足）
    int threads = 256;
    size_t smem = MAX_WARPS * sizeof(float);
    rms_norm_v2<<<B, threads, smem>>>(x, w, y, N, eps);
    cudaDeviceSynchronize();
}

void fused_add_rms_norm_cuda(float* x, float* residual, float* w, float* y,
                              int B, int N, float eps) {
    int threads = 256;
    size_t smem = MAX_WARPS * sizeof(float);
    fused_add_rms_norm_v3<<<B, threads, smem>>>(x, residual, w, y, N, eps);
    cudaDeviceSynchronize();
}

} // extern "C"
