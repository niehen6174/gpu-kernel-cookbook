/*
 * Flash Attention — CuTe C++ 实现
 *
 * 学习目标：演示 CuTe 在 Flash Attention 这类复杂 IO-aware 算法中的应用：
 *   - 4D Tensor（B,H,N,d）的 CuTe Layout 表示
 *   - 用 local_tile 描述 Q/K/V 的分块策略
 *   - 用 make_smem_ptr + make_tensor 创建 shared memory Tensor
 *   - 在 SRAM 内完成 QK^T 计算（避免 N×N 矩阵物化）
 *   - Online softmax 状态 (m, l, O) 的递推更新
 *   - Causal mask 的 tile 级跳过优化
 *
 * =========================================================================
 * CuTe 与 Triton 的对比：
 *
 *   Triton：Python DSL，用 tl.load/store + tl.dot，自动管理 tile 化
 *   CuTe：  C++ 库，用 make_tensor/local_tile 手动管理 Layout，
 *           更接近硬件，可以精确控制内存访问模式
 *
 *   二者的核心算法（online softmax 递推）完全相同：
 *     m_new = max(m_old, tile_max)
 *     alpha = exp(m_old - m_new)        // 校正旧累积
 *     l_new = l_old * alpha + tile_sum  // 更新归一化分母
 *     O_new = O_old * alpha + P * V     // 更新输出
 *
 * =========================================================================
 * V1: 单头 Flash Attention（最清晰，展示 CuTe 核心用法）
 *   - 每个 CUDA block 处理 (1 batch, 1 head, Br 行 Q)
 *   - 用 CuTe make_tensor 表达 Q/K/V 的 2D Layout (N,d)
 *   - 用 local_tile 将 Q 切成 [Br,d] 的 tile，K/V 切成 [Bc,d] 的 tile
 *   - Shared memory Tensor 通过 make_smem_ptr 创建
 *
 * V2: 多头 Flash Attention + Causal Mask
 *   - 用 4D Layout (B,H,N,d) 统一表示 Q/K/V
 *   - 支持 causal mask（跳过上三角 KV tile）
 *   - 每个 CUDA block 处理 (B*H 中的某一个, Br 行 Q)
 * =========================================================================
 */

#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

using namespace cute;

// -------------------------------------------------------------------------
// V1: 单头 Flash Attention，展示 CuTe Tensor + local_tile
//
// 输入：Q, K, V: (N, d) 单头，已偏移到对应 (batch, head)
// -------------------------------------------------------------------------
template <int Br, int Bc, int HeadDim>
__global__ void flash_attn_cute_v1(
    const float* __restrict__ Q_ptr,
    const float* __restrict__ K_ptr,
    const float* __restrict__ V_ptr,
    float* __restrict__       O_ptr,
    int N,
    float scale)
{
    // 1. 用 CuTe 包装 2D (N, d) Tensor
    //    Layout: Shape=(N, HeadDim), Stride=(HeadDim, 1) — 行主序
    auto Q = make_tensor(make_gmem_ptr(Q_ptr),
                         make_layout(make_shape(N, Int<HeadDim>{}),
                                     make_stride(Int<HeadDim>{}, Int<1>{})));
    auto K = make_tensor(make_gmem_ptr(K_ptr),
                         make_layout(make_shape(N, Int<HeadDim>{}),
                                     make_stride(Int<HeadDim>{}, Int<1>{})));
    auto V = make_tensor(make_gmem_ptr(V_ptr),
                         make_layout(make_shape(N, Int<HeadDim>{}),
                                     make_stride(Int<HeadDim>{}, Int<1>{})));
    auto O = make_tensor(make_gmem_ptr(O_ptr),
                         make_layout(make_shape(N, Int<HeadDim>{}),
                                     make_stride(Int<HeadDim>{}, Int<1>{})));

    // 2. 当前 block 处理 Q 的第 blockIdx.x 个 Br-tile
    //    local_tile(Q, tile_shape, tile_coord) → 第 blockIdx.x 个 [Br, HeadDim] 子 Tensor
    int q_tile_idx = blockIdx.x;
    auto Q_tile = local_tile(Q, make_shape(Int<Br>{}, Int<HeadDim>{}),
                             make_coord(q_tile_idx, 0));
    // Q_tile: Shape=(Br, HeadDim), Stride=(HeadDim, 1)，指向 Q[q_tile_idx*Br : ...]

    // 3. Shared memory Tensor
    __shared__ float smQ_raw[Br * HeadDim];
    __shared__ float smK_raw[Bc * HeadDim];
    __shared__ float smV_raw[Bc * HeadDim];
    __shared__ float smS_raw[Br * Bc];

    auto smQ = make_tensor(make_smem_ptr(smQ_raw),
                           make_layout(make_shape(Int<Br>{}, Int<HeadDim>{}),
                                       make_stride(Int<HeadDim>{}, Int<1>{})));
    auto smK = make_tensor(make_smem_ptr(smK_raw),
                           make_layout(make_shape(Int<Bc>{}, Int<HeadDim>{}),
                                       make_stride(Int<HeadDim>{}, Int<1>{})));
    auto smV = make_tensor(make_smem_ptr(smV_raw),
                           make_layout(make_shape(Int<Bc>{}, Int<HeadDim>{}),
                                       make_stride(Int<HeadDim>{}, Int<1>{})));
    auto smS = make_tensor(make_smem_ptr(smS_raw),
                           make_layout(make_shape(Int<Br>{}, Int<Bc>{}),
                                       make_stride(Int<Bc>{}, Int<1>{})));

    int tx = threadIdx.x;

    // 4. 协同将 Q tile 加载到 smem
    int q_row_start = q_tile_idx * Br;
    for (int idx = tx; idx < Br * HeadDim; idx += blockDim.x) {
        int r = idx / HeadDim, c = idx % HeadDim;
        int global_r = q_row_start + r;
        smQ(r, c) = (global_r < N) ? Q_tile(r, c) : 0.0f;
    }

    // 5. 初始化 online softmax 状态（每个 Q 行各一个）
    float m[Br], l[Br], o_acc[Br][HeadDim];
    if (tx == 0) {
        for (int i = 0; i < Br; i++) { m[i] = -FLT_MAX; l[i] = 0.0f; }
    }
    for (int idx = tx; idx < Br * HeadDim; idx += blockDim.x) {
        int r = idx / HeadDim, c = idx % HeadDim;
        o_acc[r][c] = 0.0f;
    }
    __syncthreads();

    // 6. 遍历 KV tiles
    int num_kv_tiles = (N + Bc - 1) / Bc;
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * Bc;

        // 用 local_tile 取出 K/V 的当前 tile
        auto K_tile = local_tile(K, make_shape(Int<Bc>{}, Int<HeadDim>{}),
                                 make_coord(kv_tile, 0));
        auto V_tile = local_tile(V, make_shape(Int<Bc>{}, Int<HeadDim>{}),
                                 make_coord(kv_tile, 0));

        // 协同加载 K/V tile 到 smem
        for (int idx = tx; idx < Bc * HeadDim; idx += blockDim.x) {
            int r = idx / HeadDim, c = idx % HeadDim;
            int global_r = kv_start + r;
            smK(r, c) = (global_r < N) ? K_tile(r, c) : 0.0f;
            smV(r, c) = (global_r < N) ? V_tile(r, c) : 0.0f;
        }
        __syncthreads();

        // 计算 S = smQ * smK^T * scale: [Br, Bc]
        // thread tx 负责 S 的某些列（tx < Bc）
        for (int i = 0; i < Br; i++) {
            for (int j = tx; j < Bc; j += blockDim.x) {
                float qk = 0.0f;
                for (int d = 0; d < HeadDim; d++)
                    qk += smQ(i, d) * smK(j, d);
                smS(i, j) = qk * scale;
            }
        }
        __syncthreads();

        // Online softmax 更新（thread 0 串行更新每行的 m/l/O）
        // 注意：这里为清晰起见用串行；生产代码应并行
        if (tx == 0) {
            for (int i = 0; i < Br; i++) {
                int global_q = q_row_start + i;
                if (global_q >= N) continue;
                int valid_kv = min(Bc, N - kv_start);

                // 找当前 tile 的 row max
                float m_tile = -FLT_MAX;
                for (int j = 0; j < valid_kv; j++)
                    m_tile = fmaxf(m_tile, smS(i, j));

                float m_new = fmaxf(m[i], m_tile);
                float alpha  = expf(m[i] - m_new);  // 校正旧累积

                // 更新 l
                float l_new = l[i] * alpha;
                for (int j = 0; j < valid_kv; j++)
                    l_new += expf(smS(i, j) - m_new);

                // 更新 O：O_new = O_old * alpha + P * V
                for (int d = 0; d < HeadDim; d++) {
                    float o_new = o_acc[i][d] * alpha;
                    for (int j = 0; j < valid_kv; j++)
                        o_new += expf(smS(i, j) - m_new) * smV(j, d);
                    o_acc[i][d] = o_new;
                }

                m[i] = m_new;
                l[i] = l_new;
            }
        }
        __syncthreads();
    }

    // 7. 最终归一化 + 写回（通过 CuTe Tensor O）
    auto O_tile = local_tile(O, make_shape(Int<Br>{}, Int<HeadDim>{}),
                             make_coord(q_tile_idx, 0));
    if (tx == 0) {
        for (int i = 0; i < Br; i++) {
            int global_q = q_row_start + i;
            if (global_q >= N) continue;
            for (int d = 0; d < HeadDim; d++)
                O_tile(i, d) = (l[i] > 0.0f) ? o_acc[i][d] / l[i] : 0.0f;
        }
    }
}


// -------------------------------------------------------------------------
// V2: 多头 Flash Attention，warp-per-row，支持 causal mask
//
// 每个 block 处理 (某 batch, 某 head, 一个 Q tile)
// blockIdx.x = q_tile_idx, blockIdx.y = b*H + h
// blockDim = (32, Br)：每行 Q 分配一个完整 warp（32 threads），
//   每个 thread 负责 d 维度的 2 个元素（d=tx 和 d=tx+32，适配 HeadDim=64）
//
// 关键点：HeadDim=64 需要 64 个 d 分量，但 warp 只有 32 lanes。
//   解决方法：每个 thread 持有 2 个 d 元素（tx, tx+32），
//   内积 reduce 仅在 warp 内（mask=16..1），所有 shuffle 合法。
// -------------------------------------------------------------------------
template <int Br, int Bc, int HeadDim>
__global__ void flash_attn_cute_v2(
    const float* __restrict__ Q_ptr,  // (B, H, N, d)
    const float* __restrict__ K_ptr,
    const float* __restrict__ V_ptr,
    float* __restrict__       O_ptr,
    int B, int H, int N,
    float scale,
    bool causal)
{
    int q_tile_idx = blockIdx.x;
    int bh          = blockIdx.y;
    int b           = bh / H, h = bh % H;
    int head_stride  = N * HeadDim;
    int batch_stride = H * head_stride;

    // 当前 (b, h) 的基址
    const float* Q_bh = Q_ptr + b * batch_stride + h * head_stride;
    const float* K_bh = K_ptr + b * batch_stride + h * head_stride;
    const float* V_bh = V_ptr + b * batch_stride + h * head_stride;
    float*       O_bh = O_ptr + b * batch_stride + h * head_stride;

    // CuTe 包装 2D Tensor (N, HeadDim)
    auto Q = make_tensor(make_gmem_ptr(Q_bh),
                         make_layout(make_shape(N, Int<HeadDim>{}),
                                     make_stride(Int<HeadDim>{}, Int<1>{})));
    auto K = make_tensor(make_gmem_ptr(K_bh),
                         make_layout(make_shape(N, Int<HeadDim>{}),
                                     make_stride(Int<HeadDim>{}, Int<1>{})));
    auto V = make_tensor(make_gmem_ptr(V_bh),
                         make_layout(make_shape(N, Int<HeadDim>{}),
                                     make_stride(Int<HeadDim>{}, Int<1>{})));
    auto O = make_tensor(make_gmem_ptr(O_bh),
                         make_layout(make_shape(N, Int<HeadDim>{}),
                                     make_stride(Int<HeadDim>{}, Int<1>{})));

    // blockDim.x = 32（一个 warp），blockDim.y = Br
    // tx = 0..31，每个 thread 负责 d = {tx, tx+32}
    int ty = threadIdx.y;
    int tx = threadIdx.x;  // 0..31

    int q_row = q_tile_idx * Br + ty;

    // 预载 Q 的 2 个 d 分量到寄存器
    float q0 = (q_row < N) ? Q(q_row, tx)      : 0.0f;
    float q1 = (q_row < N) ? Q(q_row, tx + 32) : 0.0f;

    float o0 = 0.0f, o1 = 0.0f;
    float m_val = -FLT_MAX, l_val = 0.0f;

    // Shared memory：K tile [Bc, HeadDim]，V tile [Bc, HeadDim]
    __shared__ float smK_raw[Bc * HeadDim];
    __shared__ float smV_raw[Bc * HeadDim];

    auto smK = make_tensor(make_smem_ptr(smK_raw),
                           make_layout(make_shape(Int<Bc>{}, Int<HeadDim>{}),
                                       make_stride(Int<HeadDim>{}, Int<1>{})));
    auto smV = make_tensor(make_smem_ptr(smV_raw),
                           make_layout(make_shape(Int<Bc>{}, Int<HeadDim>{}),
                                       make_stride(Int<HeadDim>{}, Int<1>{})));

    // Causal 优化：只遍历到当前 Q tile 能 attend 的 KV 位置
    int kv_end = causal ? min(N, (q_tile_idx + 1) * Br) : N;
    int num_kv_tiles = (kv_end + Bc - 1) / Bc;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * Bc;

        // 协同加载 K/V tile：每个 thread 加载 2 个 d 分量
        auto K_tile = local_tile(K, make_shape(Int<Bc>{}, Int<HeadDim>{}),
                                 make_coord(kv_tile, 0));
        auto V_tile = local_tile(V, make_shape(Int<Bc>{}, Int<HeadDim>{}),
                                 make_coord(kv_tile, 0));

        for (int r = ty; r < Bc; r += Br) {
            int global_r = kv_start + r;
            smK(r, tx)      = (global_r < N) ? K_tile(r, tx)      : 0.0f;
            smK(r, tx + 32) = (global_r < N) ? K_tile(r, tx + 32) : 0.0f;
            smV(r, tx)      = (global_r < N) ? V_tile(r, tx)      : 0.0f;
            smV(r, tx + 32) = (global_r < N) ? V_tile(r, tx + 32) : 0.0f;
        }
        __syncthreads();

        if (q_row < N) {
            int valid_kv = min(Bc, N - kv_start);
            float m_tile = -FLT_MAX;

            float scores[Bc];
            for (int j = 0; j < valid_kv; j++) {
                // 每个 thread 持有 2 个 d 分量的乘积
                float contrib = q0 * smK(j, tx) + q1 * smK(j, tx + 32);
                // Warp butterfly reduce（mask=16..1，全在 warp 内合法）
                // 结果：所有 lane 都得到完整内积和
                for (int mask = 16; mask > 0; mask >>= 1)
                    contrib += __shfl_xor_sync(0xffffffff, contrib, mask);
                scores[j] = contrib * scale;

                // Causal mask
                if (causal && (kv_start + j) > q_row)
                    scores[j] = -FLT_MAX;

                m_tile = fmaxf(m_tile, scores[j]);
            }

            float m_new = fmaxf(m_val, m_tile);
            float alpha  = expf(m_val - m_new);
            float l_new  = l_val * alpha;

            // 更新 O（每个 thread 更新 d = {tx, tx+32} 两列）
            o0 *= alpha;
            o1 *= alpha;
            for (int j = 0; j < valid_kv; j++) {
                float p = expf(scores[j] - m_new);
                l_new += p;
                o0 += p * smV(j, tx);
                o1 += p * smV(j, tx + 32);
            }

            m_val = m_new;
            l_val = l_new;
        }
        __syncthreads();
    }

    // 写回（每个 thread 写 2 个 d 分量）
    if (q_row < N && l_val > 0.0f) {
        O(q_row, tx)      = o0 / l_val;
        O(q_row, tx + 32) = o1 / l_val;
    }
}


// -------------------------------------------------------------------------
// Host 函数
// -------------------------------------------------------------------------
extern "C" {

// V1：单头串行（教学用，清晰展示 CuTe local_tile）
void flash_attention_cutlass_v1(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int N, int d)
{
    float scale = 1.0f / sqrtf((float)d);
    const int Br = 32, Bc = 32;
    int grid = (N + Br - 1) / Br;

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int offset = (b * H + h) * N * d;
            // d=64 特化（通过模板参数）
            flash_attn_cute_v1<Br, Bc, 64><<<grid, 64>>>(
                Q + offset, K + offset, V + offset, O + offset, N, scale);
        }
    }
    cudaDeviceSynchronize();
}

// V2：多头并行 + causal mask 支持
void flash_attention_cutlass_v2(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int N, int d, int causal)
{
    float scale = 1.0f / sqrtf((float)d);
    const int Br = 16, Bc = 16;
    // block: (32 threads/warp, Br rows) — 每行 Q 分配一个 warp
    // tx=0..31，每 thread 负责 d={tx, tx+32}（HeadDim=64）
    dim3 block(32, Br);
    dim3 grid((N + Br - 1) / Br, B * H);

    flash_attn_cute_v2<Br, Bc, 64><<<grid, block>>>(
        Q, K, V, O, B, H, N, scale, (bool)causal);
    cudaDeviceSynchronize();
}

}  // extern "C"
