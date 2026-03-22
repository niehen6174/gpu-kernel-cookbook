/*
 * LayerNorm CUDA Kernel
 *
 * 数学定义：
 *   给定输入 x ∈ R^N，可学习参数 γ (weight), β (bias)：
 *
 *   μ = mean(x) = (1/N) * Σ x_i
 *   σ² = var(x) = (1/N) * Σ (x_i - μ)²
 *   y_i = γ * (x_i - μ) / sqrt(σ² + ε) + β
 *
 *   ε 是防止除零的小常数（通常 1e-5）
 *
 * 应用场景：
 *   - Transformer 中每个 token 的 hidden state 归一化
 *   - 输入形状：(B, T, C)，对最后一维 C 做 LayerNorm
 *   - B=batch, T=seq_len, C=hidden_dim
 *
 * =========================================================================
 * 实现挑战：
 *   - 需要两趟扫描：第一趟求均值和方差，第二趟归一化
 *   - 可以用 Welford 在线算法，一趟完成均值+方差计算
 *
 * =========================================================================
 * V1: Two-pass shared memory reduction
 *   Pass 1: sum_x, sum_x2 → mean, var
 *   Pass 2: 归一化 + 应用 γ, β
 *
 * V2: Welford 在线算法 + Warp Reduction
 *   Welford 算法：数值稳定地在一趟内计算 (mean, variance)
 *   公式：
 *     n++
 *     delta = x - mean
 *     mean += delta / n
 *     delta2 = x - mean
 *     M2 += delta * delta2    // M2 = Σ(x_i - mean)²
 *   最终：var = M2 / n
 *
 *   优点：
 *     - 比 E[x²] - E[x]² 数值稳定（避免大数相减）
 *     - 一趟扫描（减少内存访问）
 *
 * Warp Reduction 合并 Welford 状态：
 *   合并两个 Welford 状态 (n_a, mean_a, M2_a) 和 (n_b, mean_b, M2_b)：
 *     n_c = n_a + n_b
 *     delta = mean_b - mean_a
 *     mean_c = mean_a + delta * n_b / n_c
 *     M2_c = M2_a + M2_b + delta² * n_a * n_b / n_c
 * =========================================================================
 */

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// -------------------------------------------------------------------------
// V1: Two-pass + shared memory block reduction
// -------------------------------------------------------------------------
__global__ void layernorm_v1(const float* __restrict__ input,
                              const float* __restrict__ weight,  // γ
                              const float* __restrict__ bias,    // β
                              float* __restrict__ output,
                              int N, float eps) {
    extern __shared__ float smem[];
    float* sum_buf = smem;
    float* sq_buf  = smem + blockDim.x;

    int tid = threadIdx.x;
    int row = blockIdx.x;

    const float* x = input + row * N;
    float* y = output + row * N;

    // Pass 1a: 累计 sum(x)
    float local_sum = 0.0f, local_sq = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float xi = x[i];
        local_sum += xi;
        local_sq  += xi * xi;
    }
    sum_buf[tid] = local_sum;
    sq_buf[tid]  = local_sq;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_buf[tid] += sum_buf[tid + s];
            sq_buf[tid]  += sq_buf[tid + s];
        }
        __syncthreads();
    }

    float mean = sum_buf[0] / N;
    float var  = sq_buf[0] / N - mean * mean;
    float inv_std = rsqrtf(var + eps);  // 1 / sqrt(var + eps)
    __syncthreads();

    // Pass 2: 归一化
    for (int i = tid; i < N; i += blockDim.x) {
        float xi = x[i];
        float norm = (xi - mean) * inv_std;
        y[i] = (weight ? weight[i] * norm : norm) +
               (bias   ? bias[i]           : 0.0f);
    }
}

// -------------------------------------------------------------------------
// Warp-level Welford reduction 工具
// -------------------------------------------------------------------------
struct WelfordState {
    float mean;
    float m2;
    float count;
};

__device__ __forceinline__
WelfordState warp_reduce_welford(WelfordState state) {
    // Butterfly reduce
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        float other_mean  = __shfl_xor_sync(0xffffffff, state.mean,  mask);
        float other_m2    = __shfl_xor_sync(0xffffffff, state.m2,    mask);
        float other_count = __shfl_xor_sync(0xffffffff, state.count, mask);

        // 合并两个 Welford 状态
        float total = state.count + other_count;
        if (total > 0.0f) {
            float delta = other_mean - state.mean;
            state.mean  = state.mean + delta * other_count / total;
            state.m2   += other_m2 + delta * delta * state.count * other_count / total;
        }
        state.count = total;
    }
    return state;
}

// -------------------------------------------------------------------------
// V2: Welford + Warp + Block 两级规约
// -------------------------------------------------------------------------
#define MAX_WARPS 32

__global__ void layernorm_v2_welford(const float* __restrict__ input,
                                      const float* __restrict__ weight,
                                      const float* __restrict__ bias,
                                      float* __restrict__ output,
                                      int N, float eps) {
    extern __shared__ float smem[];
    // shared memory: [MAX_WARPS mean, MAX_WARPS m2, MAX_WARPS count]
    float* s_mean  = smem;
    float* s_m2    = smem + MAX_WARPS;
    float* s_count = smem + MAX_WARPS * 2;

    int tid      = threadIdx.x;
    int warp_id  = tid / 32;
    int lane     = tid % 32;
    int num_warps = blockDim.x / 32;
    int row      = blockIdx.x;

    const float* x = input + row * N;
    float* y = output + row * N;

    // === Level 1: 每个 thread 做 Welford 更新 ===
    WelfordState state = {0.0f, 0.0f, 0.0f};
    for (int i = tid; i < N; i += blockDim.x) {
        float xi = x[i];
        state.count += 1.0f;
        float delta = xi - state.mean;
        state.mean += delta / state.count;
        float delta2 = xi - state.mean;
        state.m2 += delta * delta2;
    }

    // === Level 1: Warp 内规约 ===
    state = warp_reduce_welford(state);

    // Warp lane 0 写入 shared memory
    if (lane == 0) {
        s_mean[warp_id]  = state.mean;
        s_m2[warp_id]    = state.m2;
        s_count[warp_id] = state.count;
    }
    __syncthreads();

    // === Level 2: 第一个 warp 做 block-level 规约 ===
    float global_mean, global_var;
    if (warp_id == 0) {
        WelfordState ws;
        ws.mean  = (lane < num_warps) ? s_mean[lane]  : 0.0f;
        ws.m2    = (lane < num_warps) ? s_m2[lane]    : 0.0f;
        ws.count = (lane < num_warps) ? s_count[lane] : 0.0f;

        ws = warp_reduce_welford(ws);

        // 注意：将结果写到 s_count[0] 和 s_count[1]（与 s_mean/s_m2 不重叠的区域）
        if (lane == 0) {
            s_count[0] = ws.mean;
            s_count[1] = (ws.count > 0.0f) ? ws.m2 / ws.count : 0.0f;  // variance
        }
    }
    __syncthreads();

    global_mean = s_count[0];
    global_var  = s_count[1];
    float inv_std = rsqrtf(global_var + eps);

    // 归一化 + 仿射变换
    for (int i = tid; i < N; i += blockDim.x) {
        float norm = (x[i] - global_mean) * inv_std;
        y[i] = (weight ? weight[i] * norm : norm) +
               (bias   ? bias[i]           : 0.0f);
    }
}

// -------------------------------------------------------------------------
// V3: float4 向量化 + 寄存器缓存 x/w/b + 两路独立 reduce
//
// 核心问题（ncu 分析，B=4096，N=1024）：
//   v1 DRAM 21%，v2 DRAM 26% —— 均 latency-bound（两者 Memory/Compute < 60%）
//   v1: scalar load + shared memory block reduce + x 读两次
//   v2: Welford 每步有串行依赖（delta = x - mean_prev），无法向量化
//
// 优化策略：
//   1. float4 向量化：128-bit load，单次指令处理 4 floats，降低指令发射压力
//   2. 寄存器缓存 x/w/b：单次 DRAM 读，Pass2 完全从寄存器读取
//   3. 两路独立 reduce（sum_x + sum_x2）代替 Welford：
//      - Welford: delta 依赖上一步 mean，存在循环依赖，编译器无法向量化
//      - 两路: sum_x += x[i], sum_x2 += x[i]² 互相独立，可完全 unroll
//      - 注意：两路方式在极端值下数值稳定性略逊（但 float32 实际场景无问题）
//   4. __ldg 走 read-only cache 路径，减少 L1 污染
// -------------------------------------------------------------------------
template <int THREADS, int ELEMS_PER_THREAD>
__global__ void __launch_bounds__(THREADS)
layernorm_v3(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float*       __restrict__ output,
    int N, float eps)
{
    extern __shared__ float smem[];  // [num_warps: sum_x] [num_warps: sum_x2]

    int tid       = threadIdx.x;
    int warp_id   = tid / 32;
    int lane      = tid % 32;
    int num_warps = THREADS / 32;
    int row       = blockIdx.x;
    int N4        = N / 4;

    const float4* x4 = reinterpret_cast<const float4*>(input  + row * N);
    const float4* w4 = reinterpret_cast<const float4*>(weight);
    const float4* b4 = reinterpret_cast<const float4*>(bias);
    float4*       y4 = reinterpret_cast<float4*>(output + row * N);

    // Step 1: 全量 load x/w/b 进寄存器，同时计算 sum_x 和 sum_x2
    float4 rX[ELEMS_PER_THREAD];
    float4 rW[ELEMS_PER_THREAD];
    float4 rB[ELEMS_PER_THREAD];

    float local_sum = 0.0f, local_sq = 0.0f;
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int i = tid + e * THREADS;
        if (i < N4) {
            rX[e] = __ldg(&x4[i]);
            rW[e] = __ldg(&w4[i]);
            rB[e] = __ldg(&b4[i]);
            local_sum += rX[e].x + rX[e].y + rX[e].z + rX[e].w;
            local_sq  += rX[e].x*rX[e].x + rX[e].y*rX[e].y
                       + rX[e].z*rX[e].z + rX[e].w*rX[e].w;
        }
    }

    // Step 2: Warp reduce（两路同步进行）
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, mask);
        local_sq  += __shfl_xor_sync(0xffffffff, local_sq,  mask);
    }

    float* s_sum = smem;
    float* s_sq  = smem + num_warps;
    if (lane == 0) {
        s_sum[warp_id] = local_sum;
        s_sq[warp_id]  = local_sq;
    }
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

    // Step 3: Normalize（完全从寄存器读 x/w/b，零 DRAM/L2 re-read）
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int i = tid + e * THREADS;
        if (i < N4) {
            float4 out;
            out.x = (rX[e].x - mean) * inv_std * rW[e].x + rB[e].x;
            out.y = (rX[e].y - mean) * inv_std * rW[e].y + rB[e].y;
            out.z = (rX[e].z - mean) * inv_std * rW[e].z + rB[e].z;
            out.w = (rX[e].w - mean) * inv_std * rW[e].w + rB[e].w;
            y4[i] = out;
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

void layernorm_cuda_v1(const float* input, const float* weight, const float* bias,
                        float* output, int B, int N, float eps) {
    int threads = min(1024, N);
    int t = 1; while (t < threads) t <<= 1; threads = t;
    size_t smem = 2 * threads * sizeof(float);
    layernorm_v1<<<B, threads, smem>>>(input, weight, bias, output, N, eps);
    cudaDeviceSynchronize();
}

void layernorm_cuda_v2(const float* input, const float* weight, const float* bias,
                        float* output, int B, int N, float eps) {
    int threads = 256;
    size_t smem = MAX_WARPS * 3 * sizeof(float);
    layernorm_v2_welford<<<B, threads, smem>>>(input, weight, bias, output, N, eps);
    cudaDeviceSynchronize();
}

void layernorm_cuda_v3(const float* input, const float* weight, const float* bias,
                        float* output, int B, int N, float eps) {
    // N=1024, threads=256 → ELEMS=1024/(4×256)=1
    // N=2048  → ELEMS=2; N=4096 → ELEMS=4
    constexpr int THREADS = 256;
    int elems = N / (4 * THREADS);
    size_t smem = 2 * (THREADS / 32) * sizeof(float);
    if      (elems <= 1) layernorm_v3<THREADS,1><<<B,THREADS,smem>>>(input,weight,bias,output,N,eps);
    else if (elems == 2) layernorm_v3<THREADS,2><<<B,THREADS,smem>>>(input,weight,bias,output,N,eps);
    else if (elems == 4) layernorm_v3<THREADS,4><<<B,THREADS,smem>>>(input,weight,bias,output,N,eps);
    else if (elems == 8) layernorm_v3<THREADS,8><<<B,THREADS,smem>>>(input,weight,bias,output,N,eps);
    else layernorm_v3<THREADS,1><<<B,THREADS,smem>>>(input,weight,bias,output,N,eps);
    cudaDeviceSynchronize();
}

} // extern "C"

// -------------------------------------------------------------------------
// fused_add_layernorm_v1: residual add + two-pass layernorm
//   - residual = x + residual  (inplace)
//   - y = LayerNorm(residual, weight, bias)
// -------------------------------------------------------------------------
__global__ void fused_add_layernorm_v1_kernel(
    const float* __restrict__ x,
    float*       __restrict__ residual,   // inplace: residual = x + residual
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float*       __restrict__ output,
    int N, float eps)
{
    extern __shared__ float smem[];
    float* s_sum = smem;
    float* s_sq  = smem + blockDim.x;

    int tid = threadIdx.x;
    int row = blockIdx.x;

    const float* xrow = x        + row * N;
    float*       rrow = residual + row * N;
    float*       yrow = output   + row * N;

    // Pass 1: residual += x, accumulate sum and sum_sq
    float local_sum = 0.0f, local_sq = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float ri = xrow[i] + rrow[i];
        rrow[i]    = ri;
        local_sum += ri;
        local_sq  += ri * ri;
    }
    s_sum[tid] = local_sum;
    s_sq[tid]  = local_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sq[tid]  += s_sq[tid  + s];
        }
        __syncthreads();
    }

    float mean    = s_sum[0] / N;
    float var     = s_sq[0]  / N - mean * mean;
    float inv_std = rsqrtf(var + eps);
    __syncthreads();

    // Pass 2: normalize from updated residual
    for (int i = tid; i < N; i += blockDim.x) {
        float norm = (rrow[i] - mean) * inv_std;
        yrow[i] = (weight ? weight[i] * norm : norm)
                + (bias   ? bias[i]           : 0.0f);
    }
}

// -------------------------------------------------------------------------
// fused_add_layernorm_v3: float4 + register cache + dual-path reduce
//   Key difference from layernorm_v3:
//     - First load x + residual, compute fused sum inplace
//     - Cache updated residual in rX registers (not original x)
//     - Pass2: normalize from rX registers (zero extra DRAM)
//
//   DRAM reads : x (1×) + residual (1×) + weight (1×) + bias (1×) = 4 reads
//   DRAM writes: residual (1×) + output (1×)                       = 2 writes
//   Total: 6 × B×N×4 bytes  vs  standalone layernorm_v3: 4 × B×N×4 bytes
//   (extra cost is x_read + residual_read+write, but saves a separate add kernel)
// -------------------------------------------------------------------------
template <int THREADS, int ELEMS_PER_THREAD>
__global__ void __launch_bounds__(THREADS)
fused_add_layernorm_v3_kernel(
    const float* __restrict__ x,
    float*       __restrict__ residual,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float*       __restrict__ output,
    int N, float eps)
{
    extern __shared__ float smem[];   // [num_warps: s_sum] [num_warps: s_sq]

    int tid       = threadIdx.x;
    int warp_id   = tid / 32;
    int lane      = tid % 32;
    int num_warps = THREADS / 32;
    int row       = blockIdx.x;
    int N4        = N / 4;

    const float4* x4 = reinterpret_cast<const float4*>(x        + row * N);
    float4*       r4 = reinterpret_cast<float4*>      (residual + row * N);
    const float4* w4 = reinterpret_cast<const float4*>(weight);
    const float4* b4 = reinterpret_cast<const float4*>(bias);
    float4*       y4 = reinterpret_cast<float4*>      (output   + row * N);

    // Step 1: load x and residual, compute fused residual, cache in registers
    float4 rR[ELEMS_PER_THREAD];   // register cache for updated residual
    float4 rW[ELEMS_PER_THREAD];
    float4 rB[ELEMS_PER_THREAD];

    float local_sum = 0.0f, local_sq = 0.0f;
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int i = tid + e * THREADS;
        if (i < N4) {
            float4 xi = __ldg(&x4[i]);
            float4 ri = __ldg(&r4[i]);
            // fused add: updated_residual = x + residual
            rR[e].x = xi.x + ri.x;
            rR[e].y = xi.y + ri.y;
            rR[e].z = xi.z + ri.z;
            rR[e].w = xi.w + ri.w;
            // write updated residual back to DRAM
            r4[i] = rR[e];
            // load weight and bias
            rW[e] = __ldg(&w4[i]);
            rB[e] = __ldg(&b4[i]);
            // accumulate stats
            local_sum += rR[e].x + rR[e].y + rR[e].z + rR[e].w;
            local_sq  += rR[e].x*rR[e].x + rR[e].y*rR[e].y
                       + rR[e].z*rR[e].z + rR[e].w*rR[e].w;
        }
    }

    // Step 2: two-path warp + block reduce
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, mask);
        local_sq  += __shfl_xor_sync(0xffffffff, local_sq,  mask);
    }

    float* s_sum = smem;
    float* s_sq  = smem + num_warps;
    if (lane == 0) {
        s_sum[warp_id] = local_sum;
        s_sq[warp_id]  = local_sq;
    }
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

    // Step 3: normalize from registers (zero extra DRAM reads)
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int i = tid + e * THREADS;
        if (i < N4) {
            float4 out;
            out.x = (rR[e].x - mean) * inv_std * rW[e].x + rB[e].x;
            out.y = (rR[e].y - mean) * inv_std * rW[e].y + rB[e].y;
            out.z = (rR[e].z - mean) * inv_std * rW[e].z + rB[e].z;
            out.w = (rR[e].w - mean) * inv_std * rW[e].w + rB[e].w;
            y4[i] = out;
        }
    }
    // scalar tail for N not divisible by 4
    for (int i = N4*4 + tid; i < N; i += THREADS) {
        float xi = __ldg(&x[row*N+i]);
        float ri = __ldg(&residual[row*N+i]);
        float fused = xi + ri;
        residual[row*N+i] = fused;
        float norm = (fused - mean) * inv_std;
        output[row*N+i] = __ldg(&weight[i]) * norm + __ldg(&bias[i]);
    }
}

// Host wrappers
extern "C" {

void fused_add_layernorm_cuda_v1(
    const float* x, float* residual,
    const float* weight, const float* bias,
    float* output, int B, int N, float eps)
{
    int threads = min(1024, N);
    int t = 1; while (t < threads) t <<= 1; threads = t;
    size_t smem = 2 * threads * sizeof(float);
    fused_add_layernorm_v1_kernel<<<B, threads, smem>>>(
        x, residual, weight, bias, output, N, eps);
    cudaDeviceSynchronize();
}

void fused_add_layernorm_cuda_v3(
    const float* x, float* residual,
    const float* weight, const float* bias,
    float* output, int B, int N, float eps)
{
    constexpr int THREADS = 256;
    int elems = N / (4 * THREADS);
    size_t smem = 2 * (THREADS / 32) * sizeof(float);
    if      (elems <= 1) fused_add_layernorm_v3_kernel<THREADS,1><<<B,THREADS,smem>>>(x,residual,weight,bias,output,N,eps);
    else if (elems == 2) fused_add_layernorm_v3_kernel<THREADS,2><<<B,THREADS,smem>>>(x,residual,weight,bias,output,N,eps);
    else if (elems == 4) fused_add_layernorm_v3_kernel<THREADS,4><<<B,THREADS,smem>>>(x,residual,weight,bias,output,N,eps);
    else if (elems == 8) fused_add_layernorm_v3_kernel<THREADS,8><<<B,THREADS,smem>>>(x,residual,weight,bias,output,N,eps);
    else                 fused_add_layernorm_v3_kernel<THREADS,1><<<B,THREADS,smem>>>(x,residual,weight,bias,output,N,eps);
    cudaDeviceSynchronize();
}

} // extern "C"
