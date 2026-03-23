/*
 * RoPE — CuTe C++ 实现
 *
 * 演示 CuTe 3D Tensor 包装 (seq_len, num_heads, head_dim) 布局，
 * 以及 local_tile 按 head 切片的用法。
 *
 * V1: CuTe 3D Tensor + local_tile，非交错风格（rotate_half）
 *   per-block: (token, head) 对
 *   通过 local_tile 切出每个 head 的 head_dim 长度视图
 *
 * V2: CuTe + float2 向量化 inplace 旋转
 */

#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cuda_runtime.h>

using namespace cute;

// -------------------------------------------------------------------------
// V1: CuTe 3D Tensor + local_tile
// -------------------------------------------------------------------------
__global__ void rope_cute_v1(
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
    int tx        = threadIdx.x;

    if (tx >= half_dim) return;

    int pos = positions[token_idx];

    // CuTe Tensor 包装当前 head 的 head_dim 个元素
    // Q 的布局：(seq_len, num_heads, head_dim)，stride = (num_heads*head_dim, head_dim, 1)
    int q_offset = (token_idx * num_heads + head_idx) * head_dim;
    auto qhead = make_tensor(make_gmem_ptr(q + q_offset), make_layout(head_dim));
    auto khead = make_tensor(make_gmem_ptr(k + q_offset), make_layout(head_dim));

    // cos/sin 布局：(max_seq_len, half_dim)
    auto cos_row = make_tensor(make_gmem_ptr(cos_cache + pos * half_dim), make_layout(half_dim));
    auto sin_row = make_tensor(make_gmem_ptr(sin_cache + pos * half_dim), make_layout(half_dim));

    float c = cos_row(tx);
    float s = sin_row(tx);

    float q1 = qhead(tx);
    float q2 = qhead(tx + half_dim);
    qhead(tx)           = q1 * c - q2 * s;
    qhead(tx + half_dim) = q2 * c + q1 * s;

    float k1 = khead(tx);
    float k2 = khead(tx + half_dim);
    khead(tx)           = k1 * c - k2 * s;
    khead(tx + half_dim) = k2 * c + k1 * s;
}

// -------------------------------------------------------------------------
// V2: CuTe + float2 向量化
// -------------------------------------------------------------------------
__global__ void rope_cute_v2(
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
    int tx        = threadIdx.x;  // 0..half_dim/2-1
    int idx       = tx * 2;       // actual index in half_dim

    if (idx >= half_dim) return;

    int pos = positions[token_idx];
    int q_offset = (token_idx * num_heads + head_idx) * head_dim;

    // CuTe Tensor 包装 float2 视图 — half_dim/2 个 float2
    int half2 = half_dim / 2;
    auto cos2_row = make_tensor(make_gmem_ptr(reinterpret_cast<const float2*>(cos_cache + pos * half_dim)),
                                make_layout(half2));
    auto sin2_row = make_tensor(make_gmem_ptr(reinterpret_cast<const float2*>(sin_cache + pos * half_dim)),
                                make_layout(half2));

    auto q1v = make_tensor(make_gmem_ptr(reinterpret_cast<float2*>(q + q_offset)),
                           make_layout(half2));
    auto q2v = make_tensor(make_gmem_ptr(reinterpret_cast<float2*>(q + q_offset + half_dim)),
                           make_layout(half2));
    auto k1v = make_tensor(make_gmem_ptr(reinterpret_cast<float2*>(k + q_offset)),
                           make_layout(half2));
    auto k2v = make_tensor(make_gmem_ptr(reinterpret_cast<float2*>(k + q_offset + half_dim)),
                           make_layout(half2));

    float2 c2 = cos2_row(tx);
    float2 s2 = sin2_row(tx);

    float2 q1 = q1v(tx), q2 = q2v(tx);
    float2 q1o, q2o;
    q1o.x = q1.x * c2.x - q2.x * s2.x;  q1o.y = q1.y * c2.y - q2.y * s2.y;
    q2o.x = q2.x * c2.x + q1.x * s2.x;  q2o.y = q2.y * c2.y + q1.y * s2.y;
    q1v(tx) = q1o; q2v(tx) = q2o;

    float2 k1 = k1v(tx), k2 = k2v(tx);
    float2 k1o, k2o;
    k1o.x = k1.x * c2.x - k2.x * s2.x;  k1o.y = k1.y * c2.y - k2.y * s2.y;
    k2o.x = k2.x * c2.x + k1.x * s2.x;  k2o.y = k2.y * c2.y + k1.y * s2.y;
    k1v(tx) = k1o; k2v(tx) = k2o;
}

// -------------------------------------------------------------------------
// V3: CuTe + shared cos/sin + all-heads-per-block
//
// 相比 V1/V2（每 block = 1 (token,head) 对）的改进：
//   1. 1 block per token，grid = seq_len（减少 num_heads 倍 block 数）
//   2. cos/sin 整行载入 shared memory，num_heads 个 head 共享复用
//   3. threads = num_heads * half_dim（通常 1024），完整 warp 处理每个 head
//   4. 用 CuTe Tensor 包装 smem cos/sin 以及 Q/K 当前 head 视图
// -------------------------------------------------------------------------
__global__ void rope_cute_v3(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    const int*   __restrict__ positions,
    int seq_len, int num_heads, int head_dim)
{
    extern __shared__ float smem[];

    int token_idx = blockIdx.x;
    int half_dim  = head_dim / 2;
    int tid       = threadIdx.x;

    // CuTe Tensor 包装 shared memory cos/sin
    float* s_cos_ptr = smem;
    float* s_sin_ptr = smem + half_dim;

    auto s_cos = make_tensor(make_smem_ptr(s_cos_ptr), make_layout(half_dim));
    auto s_sin = make_tensor(make_smem_ptr(s_sin_ptr), make_layout(half_dim));

    // 协作加载该 token 的 cos/sin 行
    int pos = positions[token_idx];
    auto g_cos = make_tensor(make_gmem_ptr(cos_cache + pos * half_dim), make_layout(half_dim));
    auto g_sin = make_tensor(make_gmem_ptr(sin_cache + pos * half_dim), make_layout(half_dim));

    for (int i = tid; i < half_dim; i += blockDim.x) {
        s_cos(i) = g_cos(i);
        s_sin(i) = g_sin(i);
    }
    __syncthreads();

    // 每线程处理 1 个 (head, lane) 元素
    int total = num_heads * half_dim;
    for (int t = tid; t < total; t += blockDim.x) {
        int head_idx = t / half_dim;
        int lane     = t % half_dim;

        float c  = s_cos(lane);
        float sv = s_sin(lane);

        // CuTe Tensor 包装当前 head 的 head_dim 元素
        int offset = (token_idx * num_heads + head_idx) * head_dim;
        auto qhead = make_tensor(make_gmem_ptr(q + offset), make_layout(head_dim));
        auto khead = make_tensor(make_gmem_ptr(k + offset), make_layout(head_dim));

        float q1 = qhead(lane), q2 = qhead(lane + half_dim);
        qhead(lane)            = q1 * c - q2 * sv;
        qhead(lane + half_dim) = q2 * c + q1 * sv;

        float k1 = khead(lane), k2 = khead(lane + half_dim);
        khead(lane)            = k1 * c - k2 * sv;
        khead(lane + half_dim) = k2 * c + k1 * sv;
    }
}

// -------------------------------------------------------------------------
// Host 函数
// -------------------------------------------------------------------------
extern "C" {

void rope_cutlass_v1(float* q, float* k,
                     float* cos_cache, float* sin_cache,
                     int* positions,
                     int seq_len, int num_heads, int head_dim) {
    int half_dim = head_dim / 2;
    int total_blocks = seq_len * num_heads;
    int threads = half_dim;
    rope_cute_v1<<<total_blocks, threads>>>(
        q, k, cos_cache, sin_cache, positions,
        seq_len, num_heads, head_dim);
    cudaDeviceSynchronize();
}

void rope_cutlass_v2(float* q, float* k,
                     float* cos_cache, float* sin_cache,
                     int* positions,
                     int seq_len, int num_heads, int head_dim) {
    int half_dim = head_dim / 2;
    int total_blocks = seq_len * num_heads;
    int threads = half_dim / 2;
    rope_cute_v2<<<total_blocks, threads>>>(
        q, k, cos_cache, sin_cache, positions,
        seq_len, num_heads, head_dim);
    cudaDeviceSynchronize();
}

void rope_cutlass_v3(float* q, float* k,
                     float* cos_cache, float* sin_cache,
                     int* positions,
                     int seq_len, int num_heads, int head_dim) {
    int half_dim = head_dim / 2;
    int threads  = min(1024, num_heads * half_dim);
    int grid     = seq_len;
    size_t smem  = 2 * half_dim * sizeof(float);
    rope_cute_v3<<<grid, threads, smem>>>(
        q, k, cos_cache, sin_cache, positions,
        seq_len, num_heads, head_dim);
    cudaDeviceSynchronize();
}

} // extern "C"
