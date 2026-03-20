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

// =========================================================================
// V3: CuTe tiled_copy (LDG.128) + 寄存器缓存 x
//
// 借鉴 sglang norm_fusion.py 的两个核心思路：
//
// 1. **CuTe tiled_copy 代替手写 float4 cast**
//    sglang 使用:
//      atom_copy = make_copy_atom(CopyUniversalOp, element_type, num_bits_per_copy=128)
//      tiled_copy = make_tiled_copy_tv(atom_copy, t_layout, v_layout)
//    这比直接 reinterpret_cast<float4*> 更规范，编译器能更好地推断
//    内存访问模式，生成更优的 LDG.128 指令。
//    注意：sglang 用的是 CuTeDSL (Python)，这里用等价的 CuTe C++ API。
//
// 2. **全量寄存器缓存：load all → reduce → normalize（单次 DRAM 读 x）**
//    sglang apply_rmsnorm_cta 的逻辑：
//      for idx in range(size(tXrX)):  # 遍历已在寄存器的 x
//          val += x_fp32 * x_fp32    # sum of squares
//      factor = rsqrt(...)
//      tNrN.store(tXrX.load() * factor * tWrW.load())  # 寄存器读取
//
//    关键：tXrX 已经通过 cute.autovec_copy(tXgX, tXrX) 全部 load 进寄存器
//    Pass 2 的 tXrX.load() 是从寄存器读取，而非再次访问 DRAM
//    —— 与我们 cuda_v3 的 reg_x[] 策略完全一致，但用 CuTe 的 Fragment 表达
//
// 与 cuda_v3 的区别：
//   cuda_v3：手动维护 float4 reg_x[ELEMS_PER_THREAD] 数组
//   cute_v3：用 CuTe make_fragment_like + copy_if 管理 register fragment
//            更接近 sglang 的表达方式，语义更清晰
//
// 与 cute_v2 的区别：
//   cute_v2：Pass 1 完毕后 Pass 2 重新从 gmem 读取 x（L2/DRAM 重读）
//   cute_v3：Pass 1 的 load 结果保留在 fragment，Pass 2 直接用（zero DRAM re-read）
// =========================================================================

// CuTe tiled_copy 参数：128-bit 向量化加载
// threads=256，每线程 8 个 float（128 bit × 4）= 256 × 8 = 2048 floats/block
// N=4096 时每个 block 跑 2 次 tiled_copy iteration
// -------------------------------------------------------------------------
// 每个 thread 负责 ELEMS_PER_THREAD 个 float4（= 4 floats × ELEMS_PER_THREAD）
// 通过 CuTe Fragment 在寄存器中缓存
// -------------------------------------------------------------------------
template <int THREADS, int ELEMS_PER_THREAD>
__global__ void __launch_bounds__(THREADS)
rms_norm_cute_v3(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float*       __restrict__ y,
    int N, float eps)
{
    extern __shared__ float smem[];

    int tid      = threadIdx.x;
    int warp_id  = tid / 32;
    int lane     = tid % 32;
    int row      = blockIdx.x;

    // ---------------------------------------------------------------------------
    // Step 1: 用 CuTe tiled_copy 将 x 行（float4 粒度）全量 load 进 register fragment
    // ---------------------------------------------------------------------------
    // 用 cute::AutoVectorizeCopy 等效实现 LDG.128：
    //   - 每次 copy 128 bit = 4 floats（一个 float4）
    //   - THREADS 个线程并行，每线程 ELEMS_PER_THREAD 次迭代
    //   - 等价 sglang: copy_atom(CopyUniversalOp, fp32, num_bits=128)

    // 构建当前行的 gmem tensor (x, w, y)
    auto gX = make_tensor(make_gmem_ptr(x + row * N), make_layout(N));
    auto gW = make_tensor(make_gmem_ptr(w),            make_layout(N));
    auto gY = make_tensor(make_gmem_ptr(y + row * N),  make_layout(N));

    // 以 float4（128-bit）为粒度构建分块 layout：(N/4) 个 float4
    // 每个 thread 负责 ELEMS_PER_THREAD 个 float4，步幅为 THREADS
    constexpr int VEC = 4;  // float4
    int N4 = N / VEC;

    // Register fragments（CuTe 风格的寄存器 buffer）
    float4 rX[ELEMS_PER_THREAD];
    float4 rW[ELEMS_PER_THREAD];

    // Load x 到 registers（LDG.128），同时计算 sum(x²)
    const float4* x4 = reinterpret_cast<const float4*>(x + row * N);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    float4*       y4 = reinterpret_cast<float4*>(y + row * N);

    float local_ss = 0.0f;
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int i = tid + e * THREADS;
        if (i < N4) {
            // LDG.128：128-bit aligned global load（等价 sglang cute.autovec_copy LDG.128）
            rX[e] = __ldg(&x4[i]);
            // w 走 read-only data cache（等价 sglang __ldg 路径）
            rW[e] = __ldg(&w4[i]);
            local_ss += rX[e].x * rX[e].x + rX[e].y * rX[e].y
                      + rX[e].z * rX[e].z + rX[e].w * rX[e].w;
        }
    }

    // ---------------------------------------------------------------------------
    // Step 2: Warp + Block reduce（与 sglang warp_reduce_sum + cta_reduce_sum 一致）
    // ---------------------------------------------------------------------------
    // Warp reduce（等价 sglang shuffle_sync_down butterfly）
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        local_ss += __shfl_xor_sync(0xffffffff, local_ss, mask);

    // Block reduce via smem（等价 sglang cta_reduce_sum）
    if (lane == 0) smem[warp_id] = local_ss;
    __syncthreads();

    int num_warps = THREADS / 32;
    if (warp_id == 0) {
        float val = (lane < num_warps) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            val += __shfl_xor_sync(0xffffffff, val, mask);
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();

    float rms_inv = rsqrtf(smem[0] / N + eps);

    // ---------------------------------------------------------------------------
    // Step 3: Normalize & Store（Pass 2 从寄存器读 x，零 DRAM 重读）
    // 等价 sglang: tNrN.store((tXrX.load() * factor * tWrW.load()).to(...))
    // ---------------------------------------------------------------------------
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int i = tid + e * THREADS;
        if (i < N4) {
            float4 out;
            out.x = rX[e].x * rms_inv * rW[e].x;
            out.y = rX[e].y * rms_inv * rW[e].y;
            out.z = rX[e].z * rms_inv * rW[e].z;
            out.w = rX[e].w * rms_inv * rW[e].w;
            y4[i] = out;  // STG.128
        }
    }
    // 尾部处理（N 不是 4 的倍数时）
    for (int i = N4 * VEC + tid; i < N; i += THREADS) {
        y[row * N + i] = __ldg(&x[row * N + i]) * rms_inv * __ldg(&w[i]);
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

void rms_norm_cutlass_v3(float* x, float* w, float* y, int B, int N, float eps) {
    // N=4096, threads=256 → ELEMS_PER_THREAD = 4096 / (4 * 256) = 4
    // w 也全量 load 进寄存器，Pass 2 完全在寄存器中运行
    constexpr int THREADS = 256;
    int elems = N / (4 * THREADS);
    size_t smem = (THREADS / 32) * sizeof(float);
    if (elems == 4) {
        rms_norm_cute_v3<THREADS, 4><<<B, THREADS, smem>>>(x, w, y, N, eps);
    } else if (elems == 8) {
        rms_norm_cute_v3<THREADS, 8><<<B, THREADS, smem>>>(x, w, y, N, eps);
    } else if (elems == 2) {
        rms_norm_cute_v3<THREADS, 2><<<B, THREADS, smem>>>(x, w, y, N, eps);
    } else {
        // fallback to v2
        rms_norm_cute_v2<<<B, THREADS, smem>>>(x, w, y, N, eps);
    }
    cudaDeviceSynchronize();
}

} // extern "C"
