/*
 * Softmax CUDA Kernel
 *
 * 数学定义：
 *   softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 *
 * 减去 max(x) 是数值稳定技巧（避免 exp 溢出）。
 *
 * =========================================================================
 * 问题场景：
 *   输入: (B, N) 矩阵，对最后一维（长度 N）做 softmax
 *   输出: 同形状
 *
 * =========================================================================
 * V1: 每个 block 处理一行，block 内做 shared memory reduction
 *
 *  Pass 1: 找行最大值 max(x)
 *    - 每个 thread 先处理自己分配到的元素，得到局部最大值
 *    - Block 内 tree reduction：从 s/2 步长收缩到 1
 *
 *  Pass 2: 计算 sum(exp(x_i - max))
 *    - 同样的 reduction 模式
 *
 *  Pass 3: 计算归一化结果
 *    - output[i] = exp(x_i - max) / sum
 *
 * 三趟全局内存访问 → memory bound
 *
 * =========================================================================
 * V2: Online Softmax（Warp-level Reduction）
 *
 * 思想来自论文: "Online normalizer calculation for softmax" (Milakov & Gimelshein, 2018)
 *
 * 核心技巧：在一趟扫描中同时维护 (max, sum_of_exp) 状态。
 * 当新的 max' > max 时，对已有的 sum 进行校正：
 *   sum' = sum * exp(max - max') + exp(x_new - max')
 *
 * 这样只需 2 趟全局内存访问（一趟读 + 一趟写）。
 *
 * Warp Reduction 使用 __shfl_xor_sync 指令：
 *   - __shfl_xor_sync(mask, val, offset): 在 warp 内交换数据，无需 shared memory
 *   - 比 shared memory reduction 更快（寄存器级通信）
 *   - 每次 __shfl_xor_sync 延迟 ~4 cycle vs shared memory ~20+ cycle
 *
 * =========================================================================
 * V3: Warp + Block 两级 Online Softmax
 *
 * 当 N 很大时，一个 warp（32 threads）不够用。
 * 采用两级结构：
 *   Level 1: 每个 warp 内做 online reduction → (warp_max, warp_sum)
 *   Level 2: warp 间通过 shared memory 汇总 → (block_max, block_sum)
 *   Final:   每个 thread 计算自己负责的 output
 *
 * =========================================================================
 */

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// -------------------------------------------------------------------------
// V1: 简单的 shared memory block reduction softmax
// -------------------------------------------------------------------------
__global__ void softmax_v1_shared(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int N) {
    extern __shared__ float smem[];
    float* max_buf = smem;
    float* sum_buf = smem + blockDim.x;

    int tid = threadIdx.x;

    // Pass 1: 找局部最大值
    float local_max = -FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, input[blockIdx.x * N + i]);
    }
    max_buf[tid] = local_max;
    __syncthreads();

    // Tree reduction：找全局最大值
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            max_buf[tid] = fmaxf(max_buf[tid], max_buf[tid + s]);
        }
        __syncthreads();
    }
    float global_max = max_buf[0];
    __syncthreads();

    // Pass 2: 计算 exp(x - max) 的和
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += expf(input[blockIdx.x * N + i] - global_max);
    }
    sum_buf[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_buf[tid] += sum_buf[tid + s];
        }
        __syncthreads();
    }
    float global_sum = sum_buf[0];
    __syncthreads();

    // Pass 3: 计算输出
    for (int i = tid; i < N; i += blockDim.x) {
        output[blockIdx.x * N + i] = expf(input[blockIdx.x * N + i] - global_max) / global_sum;
    }
}

// -------------------------------------------------------------------------
// Warp-level reduction 工具函数
// 使用 __shfl_xor_sync（butterfly reduction）在 warp 内规约
// -------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_max(float val) {
    // 蝶形规约：每次交换距离为 mask 的 lane
    // 16 → 8 → 4 → 2 → 1
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// -------------------------------------------------------------------------
// V2: Online Softmax，每行一个 warp（适合 N <= 1024）
// -------------------------------------------------------------------------
__global__ void softmax_v2_online_warp(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        int N) {
    // 每个 warp 处理一行
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    if (row >= gridDim.x * (blockDim.x / 32)) return;

    const float* row_in = input + row * N;
    float* row_out = output + row * N;

    // Pass 1: Online 求 (max, sum)
    float m = -FLT_MAX;  // 当前最大值
    float d = 0.0f;      // 当前归一化分母

    for (int i = lane; i < N; i += 32) {
        float x = row_in[i];
        if (x > m) {
            d = d * expf(m - x) + 1.0f;  // 校正旧的 sum
            m = x;
        } else {
            d += expf(x - m);
        }
    }

    // Warp-level reduction 合并各 lane 的 (m, d)
    // 需要特殊处理：不能直接 reduce sum，因为 m 不同
    // 方法：先 reduce max，再重新计算 sum
    float global_max = warp_reduce_max(m);
    // 每个 lane 将自己的局部 sum 校正到 global_max
    float local_d = d * expf(m - global_max);
    float global_d = warp_reduce_sum(local_d);

    // Pass 2: 写输出
    for (int i = lane; i < N; i += 32) {
        row_out[i] = expf(row_in[i] - global_max) / global_d;
    }
}

// -------------------------------------------------------------------------
// V3: Warp + Block 两级规约（适合大 N）
// -------------------------------------------------------------------------
#define MAX_WARPS 32

__global__ void softmax_v3_two_level(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int N) {
    extern __shared__ float smem[];  // 大小: MAX_WARPS * 2

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane    = tid % 32;
    int num_warps = blockDim.x / 32;

    const float* row_in = input + row * N;
    float* row_out = output + row * N;

    // === Level 1: 每个 warp 内的 online reduction ===
    float m = -FLT_MAX, d = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float x = row_in[i];
        if (x > m) { d = d * expf(m - x) + 1.0f; m = x; }
        else { d += expf(x - m); }
    }
    // Warp reduce
    float wm = warp_reduce_max(m);
    d = d * expf(m - wm);         // 校正到 warp 内最大值
    float wd = warp_reduce_sum(d);

    // 每个 warp 的代表（lane 0）将结果写入 shared memory
    if (lane == 0) {
        smem[warp_id]             = wm;  // warp max
        smem[warp_id + MAX_WARPS] = wd;  // warp sum
    }
    __syncthreads();

    // === Level 2: 第一个 warp 做 block-level reduction ===
    float global_max, global_sum;
    if (warp_id == 0) {
        float wm2 = (lane < num_warps) ? smem[lane] : -FLT_MAX;
        float wd2 = (lane < num_warps) ? smem[lane + MAX_WARPS] : 0.0f;

        // 需要再做一次 online reduce（对 warp 间的 (max, sum) 对）
        // 简单做法：先 reduce max，再校正 sum
        float gm = warp_reduce_max(wm2);
        float gd = warp_reduce_sum(wd2 * expf(wm2 - gm));

        if (lane == 0) {
            smem[0] = gm;
            smem[1] = gd;
        }
    }
    __syncthreads();

    global_max = smem[0];
    global_sum = smem[1];

    // === 写输出 ===
    for (int i = tid; i < N; i += blockDim.x) {
        row_out[i] = expf(row_in[i] - global_max) / global_sum;
    }
}

// -------------------------------------------------------------------------
// Host 函数
// -------------------------------------------------------------------------
extern "C" {

void softmax_cuda_v1(const float* input, float* output, int B, int N) {
    // B 行，每行长度 N
    int threads = min(1024, N);
    // round up to power of 2
    int t = 1; while (t < threads) t <<= 1;
    threads = t;
    size_t smem = 2 * threads * sizeof(float);
    softmax_v1_shared<<<B, threads, smem>>>(input, output, N);
    cudaDeviceSynchronize();
}

void softmax_cuda_v2(const float* input, float* output, int B, int N) {
    // 每行一个 warp，block 含多个 warp
    int warps_per_block = 4;
    int threads = warps_per_block * 32;
    int grid = (B + warps_per_block - 1) / warps_per_block;
    softmax_v2_online_warp<<<grid, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}

void softmax_cuda_v3(const float* input, float* output, int B, int N) {
    int threads = 256;
    size_t smem = MAX_WARPS * 2 * sizeof(float);
    softmax_v3_two_level<<<B, threads, smem>>>(input, output, N);
    cudaDeviceSynchronize();
}

} // extern "C"
