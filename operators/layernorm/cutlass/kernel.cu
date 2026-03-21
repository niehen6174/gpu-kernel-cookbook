/*
 * LayerNorm — CuTe C++ 实现
 *
 * 学习目标：演示 CuTe 在双参数 reduction（均值+方差）场景的用法：
 *   - make_tensor 包装输入/输出/weight/bias
 *   - Welford 在线算法 + Warp-level 状态合并
 *   - cute::for_each 替代手动循环，语义更清晰
 *
 * =========================================================================
 * V1: Two-pass + CuTe Tensor 包装
 *   Pass1: 用 CuTe Tensor 遍历求 sum / sum_sq → mean / var
 *   Pass2: 通过 CuTe Tensor 索引 x / weight / bias 做归一化
 *
 * V2: Welford 在线算法 + Warp Reduction + CuTe Tensor
 *   单趟计算均值和方差（数值稳定），演示 CuTe 与 warp shuffle 的结合
 * =========================================================================
 */

#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

using namespace cute;

// -------------------------------------------------------------------------
// Welford 状态 + Warp reduce
// -------------------------------------------------------------------------
struct WelfordState { float mean, m2, count; };

__device__ __forceinline__
WelfordState warp_reduce_welford(WelfordState s) {
    for (int mask = 16; mask > 0; mask >>= 1) {
        float om  = __shfl_xor_sync(0xffffffff, s.mean,  mask);
        float om2 = __shfl_xor_sync(0xffffffff, s.m2,    mask);
        float oc  = __shfl_xor_sync(0xffffffff, s.count, mask);
        float total = s.count + oc;
        if (total > 0.0f) {
            float delta = om - s.mean;
            s.mean  = s.mean + delta * oc / total;
            s.m2   += om2 + delta * delta * s.count * oc / total;
        }
        s.count = total;
    }
    return s;
}

// -------------------------------------------------------------------------
// V1: Two-pass shared memory + CuTe Tensor
// -------------------------------------------------------------------------
__global__ void layernorm_cute_v1(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float*       __restrict__ output,
    int N, float eps)
{
    extern __shared__ float smem[];
    float* sum_buf = smem;
    float* sq_buf  = smem + blockDim.x;

    int row = blockIdx.x;
    int tid = threadIdx.x;

    // 用 CuTe Tensor 包装当前行
    auto x = make_tensor(make_gmem_ptr(input  + row * N), make_layout(N));
    auto y = make_tensor(make_gmem_ptr(output + row * N), make_layout(N));

    // Pass 1: 累积 sum 和 sum_sq
    float lsum = 0.f, lsq = 0.f;
    for (int i = tid; i < N; i += blockDim.x) {
        float xi = x(i);
        lsum += xi;
        lsq  += xi * xi;
    }
    sum_buf[tid] = lsum;
    sq_buf[tid]  = lsq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_buf[tid] += sum_buf[tid + s];
            sq_buf[tid]  += sq_buf[tid + s];
        }
        __syncthreads();
    }

    float mean    = sum_buf[0] / N;
    float var     = sq_buf[0] / N - mean * mean;
    float inv_std = rsqrtf(var + eps);
    __syncthreads();

    // Pass 2: 归一化 + 仿射（通过 CuTe Tensor 索引）
    // 同样用 CuTe 包装 weight / bias
    for (int i = tid; i < N; i += blockDim.x) {
        float norm = (x(i) - mean) * inv_std;
        y(i) = (weight ? weight[i] * norm : norm)
             + (bias   ? bias[i]           : 0.0f);
    }
}

// -------------------------------------------------------------------------
// V2: Welford + Warp Reduction + CuTe Tensor
// -------------------------------------------------------------------------
#define MAX_WARPS 32

__global__ void layernorm_cute_v2(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float*       __restrict__ output,
    int N, float eps)
{
    extern __shared__ float smem[];
    float* s_mean  = smem;
    float* s_m2    = smem + MAX_WARPS;
    float* s_count = smem + MAX_WARPS * 2;

    int tid       = threadIdx.x;
    int warp_id   = tid / 32;
    int lane      = tid % 32;
    int num_warps = blockDim.x / 32;
    int row       = blockIdx.x;

    auto x = make_tensor(make_gmem_ptr(input  + row * N), make_layout(N));
    auto y = make_tensor(make_gmem_ptr(output + row * N), make_layout(N));

    // Level 1: 每个 thread Welford 更新（通过 CuTe Tensor 访问 x(i)）
    WelfordState ws = {0.f, 0.f, 0.f};
    for (int i = tid; i < N; i += blockDim.x) {
        float xi = x(i);
        ws.count += 1.f;
        float delta = xi - ws.mean;
        ws.mean += delta / ws.count;
        ws.m2   += delta * (xi - ws.mean);
    }

    // Warp reduce
    ws = warp_reduce_welford(ws);

    if (lane == 0) {
        s_mean[warp_id]  = ws.mean;
        s_m2[warp_id]    = ws.m2;
        s_count[warp_id] = ws.count;
    }
    __syncthreads();

    // Block reduce（第一个 warp）
    if (warp_id == 0) {
        WelfordState bws;
        bws.mean  = (lane < num_warps) ? s_mean[lane]  : 0.f;
        bws.m2    = (lane < num_warps) ? s_m2[lane]    : 0.f;
        bws.count = (lane < num_warps) ? s_count[lane] : 0.f;
        bws = warp_reduce_welford(bws);
        if (lane == 0) {
            s_count[0] = bws.mean;
            s_count[1] = (bws.count > 0.f) ? bws.m2 / bws.count : 0.f;
        }
    }
    __syncthreads();

    float global_mean = s_count[0];
    float global_var  = s_count[1];
    float inv_std = rsqrtf(global_var + eps);

    for (int i = tid; i < N; i += blockDim.x) {
        float norm = (x(i) - global_mean) * inv_std;
        y(i) = (weight ? weight[i] * norm : norm)
             + (bias   ? bias[i]           : 0.0f);
    }
}

// -------------------------------------------------------------------------
// V3: CuTe + float4/LDG.128 + 寄存器缓存 x/w/b + 两路独立 reduce
//
// 借鉴路径：rms_norm cute_v3 → sglang norm_fusion → layernorm cute_v3
//   - 全量 register fragment 缓存 x、w、b
//   - 两路 sum_x / sum_x2 独立 reduce（替代 Welford 串行依赖链）
//   - smem 仅需 2 × num_warps × 4B
// -------------------------------------------------------------------------
template <int THREADS, int ELEMS_PER_THREAD>
__global__ void __launch_bounds__(THREADS)
layernorm_cute_v3(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float*       __restrict__ output,
    int N, float eps)
{
    extern __shared__ float smem[];

    int tid       = threadIdx.x;
    int warp_id   = tid / 32;
    int lane      = tid % 32;
    int num_warps = THREADS / 32;
    int row       = blockIdx.x;
    int N4        = N / 4;

    auto gX = make_tensor(make_gmem_ptr(reinterpret_cast<const float4*>(input  + row*N)), make_layout(N4));
    auto gW = make_tensor(make_gmem_ptr(reinterpret_cast<const float4*>(weight)),          make_layout(N4));
    auto gB = make_tensor(make_gmem_ptr(reinterpret_cast<const float4*>(bias)),            make_layout(N4));
    auto gY = make_tensor(make_gmem_ptr(reinterpret_cast<float4*>(output + row*N)),        make_layout(N4));

    float4 rX[ELEMS_PER_THREAD];
    float4 rW[ELEMS_PER_THREAD];
    float4 rB[ELEMS_PER_THREAD];

    float local_sum = 0.0f, local_sq = 0.0f;
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int i = tid + e * THREADS;
        if (i < N4) {
            rX[e] = gX(i);
            rW[e] = gW(i);
            rB[e] = gB(i);
            local_sum += rX[e].x + rX[e].y + rX[e].z + rX[e].w;
            local_sq  += rX[e].x*rX[e].x + rX[e].y*rX[e].y
                       + rX[e].z*rX[e].z + rX[e].w*rX[e].w;
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, mask);
        local_sq  += __shfl_xor_sync(0xffffffff, local_sq,  mask);
    }
    float* s_sum = smem;
    float* s_sq  = smem + num_warps;
    if (lane == 0) { s_sum[warp_id] = local_sum; s_sq[warp_id] = local_sq; }
    __syncthreads();

    if (warp_id == 0) {
        float vs = (lane < num_warps) ? s_sum[lane] : 0.0f;
        float vq = (lane < num_warps) ? s_sq[lane]  : 0.0f;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            vs += __shfl_xor_sync(0xffffffff, vs, mask);
            vq += __shfl_xor_sync(0xffffffff, vq, mask);
        }
        if (lane == 0) { s_sum[0] = vs; s_sq[0] = vq; }
    }
    __syncthreads();

    float mean    = s_sum[0] / N;
    float var     = s_sq[0]  / N - mean * mean;
    float inv_std = rsqrtf(var + eps);

    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int i = tid + e * THREADS;
        if (i < N4) {
            float4 out;
            out.x = (rX[e].x - mean) * inv_std * rW[e].x + rB[e].x;
            out.y = (rX[e].y - mean) * inv_std * rW[e].y + rB[e].y;
            out.z = (rX[e].z - mean) * inv_std * rW[e].z + rB[e].z;
            out.w = (rX[e].w - mean) * inv_std * rW[e].w + rB[e].w;
            gY(i) = out;
        }
    }
    for (int i = N4*4 + tid; i < N; i += THREADS)
        output[row*N+i] = (__ldg(&input[row*N+i]) - mean) * inv_std
                          * __ldg(&weight[i]) + __ldg(&bias[i]);
}

// -------------------------------------------------------------------------
// Host 函数
// -------------------------------------------------------------------------
extern "C" {

void layernorm_cutlass_v1(const float* input, const float* weight, const float* bias,
                           float* output, int B, int N, float eps) {
    int threads = min(1024, N);
    int t = 1; while (t < threads) t <<= 1; threads = t;
    size_t smem = 2 * threads * sizeof(float);
    layernorm_cute_v1<<<B, threads, smem>>>(input, weight, bias, output, N, eps);
    cudaDeviceSynchronize();
}

void layernorm_cutlass_v2(const float* input, const float* weight, const float* bias,
                           float* output, int B, int N, float eps) {
    int threads = 256;
    size_t smem = MAX_WARPS * 3 * sizeof(float);
    layernorm_cute_v2<<<B, threads, smem>>>(input, weight, bias, output, N, eps);
    cudaDeviceSynchronize();
}

void layernorm_cutlass_v3(const float* input, const float* weight, const float* bias,
                           float* output, int B, int N, float eps) {
    constexpr int THREADS = 256;
    int elems = N / (4 * THREADS);
    size_t smem = 2 * (THREADS / 32) * sizeof(float);
    if      (elems <= 1) layernorm_cute_v3<THREADS,1><<<B,THREADS,smem>>>(input,weight,bias,output,N,eps);
    else if (elems == 2) layernorm_cute_v3<THREADS,2><<<B,THREADS,smem>>>(input,weight,bias,output,N,eps);
    else if (elems == 4) layernorm_cute_v3<THREADS,4><<<B,THREADS,smem>>>(input,weight,bias,output,N,eps);
    else if (elems == 8) layernorm_cute_v3<THREADS,8><<<B,THREADS,smem>>>(input,weight,bias,output,N,eps);
    else layernorm_cute_v3<THREADS,1><<<B,THREADS,smem>>>(input,weight,bias,output,N,eps);
    cudaDeviceSynchronize();
}

}  // extern "C"
