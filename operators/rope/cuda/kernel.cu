/*
 * RoPE (Rotary Position Embedding) CUDA Kernel
 *
 * 数学定义（非交错 / GPT-NeoX rotate_half 风格）：
 *   将 head_dim 分为前半 x1 = x[..., :d/2] 和后半 x2 = x[..., d/2:]
 *   x'[..., :d/2] = x1 * cos - x2 * sin
 *   x'[..., d/2:] = x2 * cos + x1 * sin
 *
 * 输入布局：
 *   Q, K: (seq_len, num_heads, head_dim) contiguous float32
 *   cos_cache, sin_cache: (max_seq_len, head_dim/2)
 *   positions: (seq_len,) int32 position indices into cache
 *
 * =========================================================================
 * V1: 非交错风格，inplace on Q/K
 *   blockIdx.x = token * num_heads + head
 *   threadIdx.x = 0..head_dim/2-1
 *
 * V2: Vectorized float2 loads，每线程处理 2 个 half-dim 元素
 * =========================================================================
 */

#include <cuda_runtime.h>

// -------------------------------------------------------------------------
// V1: Non-interleaved RoPE
// -------------------------------------------------------------------------
__global__ void rope_v1(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    const int*   __restrict__ positions,
    int seq_len, int num_heads, int head_dim)
{
    int pid       = blockIdx.x;
    int token_idx = pid / num_heads;
    int head_idx  = pid % num_heads;
    int half_dim  = head_dim / 2;
    int tx        = threadIdx.x;  // 0..half_dim-1

    if (tx >= half_dim) return;

    int pos = positions[token_idx];

    float c = cos_cache[pos * half_dim + tx];
    float s = sin_cache[pos * half_dim + tx];

    // Q
    int q_base = (token_idx * num_heads + head_idx) * head_dim;
    float q1 = q[q_base + tx];
    float q2 = q[q_base + tx + half_dim];
    q[q_base + tx]           = q1 * c - q2 * s;
    q[q_base + tx + half_dim] = q2 * c + q1 * s;

    // K
    int k_base = (token_idx * num_heads + head_idx) * head_dim;
    float k1 = k[k_base + tx];
    float k2 = k[k_base + tx + half_dim];
    k[k_base + tx]           = k1 * c - k2 * s;
    k[k_base + tx + half_dim] = k2 * c + k1 * s;
}

// -------------------------------------------------------------------------
// V2: Vectorized float2 loads (2 elements per thread)
// -------------------------------------------------------------------------
__global__ void rope_v2(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    const int*   __restrict__ positions,
    int seq_len, int num_heads, int head_dim)
{
    int pid       = blockIdx.x;
    int token_idx = pid / num_heads;
    int head_idx  = pid % num_heads;
    int half_dim  = head_dim / 2;
    // Each thread handles 2 pairs via float2
    int tx = threadIdx.x;  // 0..half_dim/2-1
    int idx = tx * 2;      // actual half-dim index

    if (idx >= half_dim) return;

    int pos = positions[token_idx];

    // Load 2 cos/sin values
    float2 cos2 = *reinterpret_cast<const float2*>(cos_cache + pos * half_dim + idx);
    float2 sin2 = *reinterpret_cast<const float2*>(sin_cache + pos * half_dim + idx);

    int q_base = (token_idx * num_heads + head_idx) * head_dim;

    // Load q1[idx:idx+2] and q2[idx:idx+2]
    float2 q1 = *reinterpret_cast<float2*>(q + q_base + idx);
    float2 q2 = *reinterpret_cast<float2*>(q + q_base + idx + half_dim);

    float2 q1_out, q2_out;
    q1_out.x = q1.x * cos2.x - q2.x * sin2.x;
    q1_out.y = q1.y * cos2.y - q2.y * sin2.y;
    q2_out.x = q2.x * cos2.x + q1.x * sin2.x;
    q2_out.y = q2.y * cos2.y + q1.y * sin2.y;

    *reinterpret_cast<float2*>(q + q_base + idx)           = q1_out;
    *reinterpret_cast<float2*>(q + q_base + idx + half_dim) = q2_out;

    int k_base = (token_idx * num_heads + head_idx) * head_dim;
    float2 k1 = *reinterpret_cast<float2*>(k + k_base + idx);
    float2 k2 = *reinterpret_cast<float2*>(k + k_base + idx + half_dim);

    float2 k1_out, k2_out;
    k1_out.x = k1.x * cos2.x - k2.x * sin2.x;
    k1_out.y = k1.y * cos2.y - k2.y * sin2.y;
    k2_out.x = k2.x * cos2.x + k1.x * sin2.x;
    k2_out.y = k2.y * cos2.y + k1.y * sin2.y;

    *reinterpret_cast<float2*>(k + k_base + idx)           = k1_out;
    *reinterpret_cast<float2*>(k + k_base + idx + half_dim) = k2_out;
}

// -------------------------------------------------------------------------
// V3: All heads per block, shared cos/sin cache
//
// 核心优化：
//   V1/V2 每 block = 1 (token, head) 对，grid = seq_len * num_heads (131072)
//   每个 block 独立读相同 token 的 cos/sin 行（32 个 head 重复读）
//
//   V3 每 block = 1 token，grid = seq_len (4096)
//   1. cos/sin 整行读入 shared memory，32 个 head 共享 → 32× L1 复用
//   2. threads = num_heads * half_dim（如 32×32=1024）= 32 个完整 warp
//      每 warp 恰好处理一个 head，fully coalesced
//   3. grid 减小 32×，kernel launch 调度开销降低
// -------------------------------------------------------------------------
__global__ void rope_v3(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    const int*   __restrict__ positions,
    int seq_len, int num_heads, int head_dim)
{
    extern __shared__ float smem[];   // 2 * half_dim floats

    int token_idx = blockIdx.x;
    int half_dim  = head_dim / 2;
    int tid       = threadIdx.x;

    float* s_cos = smem;
    float* s_sin = smem + half_dim;

    // 协作加载该 token 的 cos/sin 行到 shared memory
    int pos = positions[token_idx];
    const float* cos_row = cos_cache + pos * half_dim;
    const float* sin_row = sin_cache + pos * half_dim;

    for (int i = tid; i < half_dim; i += blockDim.x) {
        s_cos[i] = cos_row[i];
        s_sin[i] = sin_row[i];
    }
    __syncthreads();

    // 每线程处理一个 (head, lane) 元素
    // 线程排布：tid = head_idx * half_dim + lane
    // → warp i 处理 head i（完整 warp，coalesced 访存）
    int total = num_heads * half_dim;
    for (int t = tid; t < total; t += blockDim.x) {
        int head_idx = t / half_dim;
        int lane     = t % half_dim;

        float c = s_cos[lane];
        float sv = s_sin[lane];

        int base = (token_idx * num_heads + head_idx) * head_dim;

        // Q
        float q1 = q[base + lane];
        float q2 = q[base + lane + half_dim];
        q[base + lane]            = q1 * c - q2 * sv;
        q[base + lane + half_dim] = q2 * c + q1 * sv;

        // K
        float k1 = k[base + lane];
        float k2 = k[base + lane + half_dim];
        k[base + lane]            = k1 * c - k2 * sv;
        k[base + lane + half_dim] = k2 * c + k1 * sv;
    }
}

// -------------------------------------------------------------------------
// Host 函数
// -------------------------------------------------------------------------
extern "C" {

void rope_cuda_v1(float* q, float* k,
                  float* cos_cache, float* sin_cache,
                  int* positions,
                  int seq_len, int num_heads, int head_dim) {
    int half_dim = head_dim / 2;
    int total_blocks = seq_len * num_heads;
    int threads = half_dim;  // one thread per half-dim element
    rope_v1<<<total_blocks, threads>>>(
        q, k, cos_cache, sin_cache, positions,
        seq_len, num_heads, head_dim);
    cudaDeviceSynchronize();
}

void rope_cuda_v2(float* q, float* k,
                  float* cos_cache, float* sin_cache,
                  int* positions,
                  int seq_len, int num_heads, int head_dim) {
    int half_dim = head_dim / 2;
    int total_blocks = seq_len * num_heads;
    int threads = half_dim / 2;  // each thread handles 2 pairs
    rope_v2<<<total_blocks, threads>>>(
        q, k, cos_cache, sin_cache, positions,
        seq_len, num_heads, head_dim);
    cudaDeviceSynchronize();
}

void rope_cuda_v3(float* q, float* k,
                  float* cos_cache, float* sin_cache,
                  int* positions,
                  int seq_len, int num_heads, int head_dim) {
    int half_dim = head_dim / 2;
    // threads = num_heads * half_dim，通常 = 1024（32 heads × 32 half_dim）
    int threads  = min(1024, num_heads * half_dim);
    int grid     = seq_len;
    size_t smem  = 2 * half_dim * sizeof(float);
    rope_v3<<<grid, threads, smem>>>(
        q, k, cos_cache, sin_cache, positions,
        seq_len, num_heads, head_dim);
    cudaDeviceSynchronize();
}

} // extern "C"
