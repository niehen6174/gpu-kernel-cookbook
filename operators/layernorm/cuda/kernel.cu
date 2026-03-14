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

} // extern "C"
