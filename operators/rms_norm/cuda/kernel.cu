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
// V3: Single-pass register caching + __ldg weight
//
// 核心优化：消除 x 的第二次 DRAM 读
//
// V2 的问题（ncu 分析）：
//   - Pass 1 读 x（64 MB），Pass 2 再读一次 x（L2 Hit ~60%，仍有 ~25 MB DRAM 流量）
//   - 总 DRAM 流量约 153 MB，而理论最优为 128 MB（读 x 一次 + 写 y 一次）
//   - WarpStateStats：71.98% stall 在 L1TEX，根源是多余的全局内存访问
//
// 优化策略：
//   1. 将 x 缓存到寄存器（Pass 1 load 时顺便存入 reg_x[]）
//   2. Pass 2 直接从寄存器读取 x，不再访问 DRAM
//   3. weight w 用 __ldg() 走 read-only texture cache，提高 L1 命中率
//
// 代价：寄存器用量从 26 → ~40，Occupancy 从 ~91% → ~75%
//       但 DRAM 流量减少约 16%，预期抵消 Occupancy 下降
//
// 适用范围：ELEMS_PER_THREAD = N / (4 * blockDim.x)
//   N=4096, threads=256 → ELEMS_PER_THREAD=4（每线程 4 个 float4 = 16 floats）
// -------------------------------------------------------------------------
template <int ELEMS_PER_THREAD>
__global__ void rms_norm_v3_impl(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float*       __restrict__ y,
    int N, float eps)
{
    extern __shared__ float smem[];

    int tid       = threadIdx.x;
    int warp_id   = tid / 32;
    int lane      = tid % 32;
    int num_warps = blockDim.x / 32;
    int row       = blockIdx.x;
    int N4        = N / 4;

    const float4* x4 = reinterpret_cast<const float4*>(x + row * N);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    float4*       y4 = reinterpret_cast<float4*>(y + row * N);

    // Pass 1: 加载 x 到寄存器，同时计算 sum(x²)
    float4 reg_x[ELEMS_PER_THREAD];
    float local_ss = 0.0f;

    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int i = tid + e * blockDim.x;
        if (i < N4) {
            reg_x[e] = __ldg(&x4[i]);
            local_ss += reg_x[e].x * reg_x[e].x
                      + reg_x[e].y * reg_x[e].y
                      + reg_x[e].z * reg_x[e].z
                      + reg_x[e].w * reg_x[e].w;
        }
    }

    // Warp + block reduce
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

    // Pass 2: 从寄存器读 x（无 DRAM 访问），weight 用 __ldg 走 texture cache
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int i = tid + e * blockDim.x;
        if (i < N4) {
            float4 wi = __ldg(&w4[i]);
            float4 out;
            out.x = reg_x[e].x * rms_inv * wi.x;
            out.y = reg_x[e].y * rms_inv * wi.y;
            out.z = reg_x[e].z * rms_inv * wi.z;
            out.w = reg_x[e].w * rms_inv * wi.w;
            y4[i] = out;
        }
    }
    // scalar tail for N not divisible by 4
    for (int i = N4*4 + tid; i < N; i += blockDim.x)
        y[row*N+i] = __ldg(&x[row*N+i]) * rms_inv * __ldg(&w[i]);
}

// -------------------------------------------------------------------------
// fused_add_rms_norm_v3: float4 + register cache (optimized)
//
// 相比 v4（scalar two-pass）的改进：
//   1. float4 向量化：128-bit LDG/STG，减少 75% 指令
//   2. 寄存器缓存更新后的 residual (r = x + r_old)
//      Pass 1 load x + r_old → 计算 r = x + r_old → 写回 DRAM，缓存 r 到寄存器
//      Pass 2 从寄存器读 r，完全零 DRAM re-read
//   3. __ldg 读 w，走 read-only cache
//
// DRAM 访问：x(1R) + residual(1R+1W) + w(1R) + y(1W) = 5 passes
// 对比 v4：  x(1R) + residual(1R+1W) + residual(1R) + w(1R) + y(1W) = 6 passes
// -------------------------------------------------------------------------
template <int ELEMS_PER_THREAD>
__global__ void __launch_bounds__(256)
fused_add_rms_norm_v3_impl(
    const float* __restrict__ x,
    float*       __restrict__ residual,  // inplace: residual = x + residual
    const float* __restrict__ w,
    float*       __restrict__ y,
    int N, float eps)
{
    extern __shared__ float smem[];

    int tid       = threadIdx.x;
    int warp_id   = tid / 32;
    int lane      = tid % 32;
    int num_warps = blockDim.x / 32;
    int row       = blockIdx.x;
    int N4        = N / 4;

    const float4* x4 = reinterpret_cast<const float4*>(x        + row * N);
    float4*       r4 = reinterpret_cast<float4*>      (residual + row * N);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    float4*       y4 = reinterpret_cast<float4*>      (y        + row * N);

    float4 rR[ELEMS_PER_THREAD];  // register cache: updated residual
    float4 rW[ELEMS_PER_THREAD];

    float local_ss = 0.0f;
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int i = tid + e * blockDim.x;
        if (i < N4) {
            float4 xi = __ldg(&x4[i]);
            float4 ri = __ldg(&r4[i]);
            // fused add: r = x + residual
            rR[e].x = xi.x + ri.x;
            rR[e].y = xi.y + ri.y;
            rR[e].z = xi.z + ri.z;
            rR[e].w = xi.w + ri.w;
            // write updated residual back (single DRAM write)
            r4[i] = rR[e];
            // load weight with __ldg
            rW[e] = __ldg(&w4[i]);
            // accumulate sum(r²)
            local_ss += rR[e].x*rR[e].x + rR[e].y*rR[e].y
                      + rR[e].z*rR[e].z + rR[e].w*rR[e].w;
        }
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

    // Pass 2: normalize from registers (zero DRAM re-read)
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int i = tid + e * blockDim.x;
        if (i < N4) {
            float4 out;
            out.x = rR[e].x * rms_inv * rW[e].x;
            out.y = rR[e].y * rms_inv * rW[e].y;
            out.z = rR[e].z * rms_inv * rW[e].z;
            out.w = rR[e].w * rms_inv * rW[e].w;
            y4[i] = out;
        }
    }
    // scalar tail
    for (int i = N4*4 + tid; i < N; i += blockDim.x) {
        float r = __ldg(&x[row*N+i]) + __ldg(&residual[row*N+i]);
        residual[row*N+i] = r;
        y[row*N+i] = r * rms_inv * __ldg(&w[i]);
    }
}

// -------------------------------------------------------------------------
// V4: Fused residual add + RMSNorm (inplace on residual), scalar baseline
// -------------------------------------------------------------------------
__global__ void fused_add_rms_norm_v4(
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

void rms_norm_cuda_v3(float* x, float* w, float* y, int B, int N, float eps) {
    // 单次读取 x，缓存到寄存器，消除 Pass 2 的 DRAM 重读
    // N=4096, threads=256 → ELEMS_PER_THREAD = 4096/(4*256) = 4
    int threads = 256;
    int elems = N / (4 * threads);
    size_t smem = MAX_WARPS * sizeof(float);
    if      (elems <= 1) rms_norm_v3_impl<1><<<B, threads, smem>>>(x, w, y, N, eps);
    else if (elems == 2) rms_norm_v3_impl<2><<<B, threads, smem>>>(x, w, y, N, eps);
    else if (elems == 4) rms_norm_v3_impl<4><<<B, threads, smem>>>(x, w, y, N, eps);
    else if (elems == 8) rms_norm_v3_impl<8><<<B, threads, smem>>>(x, w, y, N, eps);
    else                 rms_norm_v3_impl<1><<<B, threads, smem>>>(x, w, y, N, eps);
    cudaDeviceSynchronize();
}

void fused_add_rms_norm_cuda(float* x, float* residual, float* w, float* y,
                              int B, int N, float eps) {
    int threads = 256;
    size_t smem = MAX_WARPS * sizeof(float);
    fused_add_rms_norm_v4<<<B, threads, smem>>>(x, residual, w, y, N, eps);
    cudaDeviceSynchronize();
}

void fused_add_rms_norm_cuda_v3(const float* x, float* residual, const float* w, float* y,
                                 int B, int N, float eps) {
    int threads = 256;
    int elems = N / (4 * threads);
    size_t smem = MAX_WARPS * sizeof(float);
    if      (elems <= 1) fused_add_rms_norm_v3_impl<1><<<B, threads, smem>>>(x, residual, w, y, N, eps);
    else if (elems == 2) fused_add_rms_norm_v3_impl<2><<<B, threads, smem>>>(x, residual, w, y, N, eps);
    else if (elems == 4) fused_add_rms_norm_v3_impl<4><<<B, threads, smem>>>(x, residual, w, y, N, eps);
    else if (elems == 8) fused_add_rms_norm_v3_impl<8><<<B, threads, smem>>>(x, residual, w, y, N, eps);
    else                 fused_add_rms_norm_v3_impl<1><<<B, threads, smem>>>(x, residual, w, y, N, eps);
    cudaDeviceSynchronize();
}

} // extern "C"
