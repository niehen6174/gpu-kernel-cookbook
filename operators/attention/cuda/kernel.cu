/*
 * Flash Attention CUDA Kernel
 *
 * 论文：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
 *       Dao et al., NeurIPS 2022  (https://arxiv.org/abs/2205.14135)
 *
 * =========================================================================
 * 标准 Attention 的问题（朴素实现）：
 *
 *   Attention(Q, K, V) = softmax(Q K^T / √d) V
 *
 *   朴素实现步骤：
 *     1. S = Q K^T        → (B, H, N, N) 矩阵，O(N²) 显存！
 *     2. P = softmax(S)   → 需要整行 S 才能算 softmax
 *     3. O = P V
 *
 *   对于 N=8192, H=32, B=4, d=128：
 *     S 矩阵大小 = 4 × 32 × 8192 × 8192 × 4 bytes ≈ 32 GB！
 *     这超过了大多数 GPU 的显存。
 *
 * =========================================================================
 * Flash Attention 核心思想：IO-Aware 算法
 *
 * 关键洞察：Attention 是 memory-bound 操作，真正的瓶颈不是 FLOPs，
 * 而是 HBM（高带宽显存）的读写次数。
 *
 * Flash Attention 通过分块（tiling）将整个 Attention 计算在 SRAM（shared memory）
 * 中完成，避免了 S = QK^T 的物化（materialization）：
 *
 *   - 将 Q 分成 Tr 个 block，每个 block Q_i ∈ R^{Br × d}
 *   - 将 K, V 分成 Tc 个 block，每个 block K_j, V_j ∈ R^{Bc × d}
 *   - 对每个 (Q_i, K_j, V_j) 组合：
 *     计算局部 attention，并用 online softmax 维护 (m, l) 状态
 *     累积更新输出 O_i
 *
 * IO 复杂度对比：
 *   标准 Attention:  O(N² d) HBM 读写
 *   Flash Attention: O(N d²/M) HBM 读写（M = SRAM 大小）
 *
 * =========================================================================
 * Online Softmax 在 Flash Attention 中的应用：
 *
 * 由于 Q 和 K,V 分块处理，无法一次看到完整的一行 S，
 * 需要用 online softmax 逐块更新。
 *
 * 维护状态：
 *   m_i: 当前看到的最大值（用于数值稳定）
 *   l_i: 当前 exp 的累计和（归一化分母）
 *   O_i: 当前的输出累计值
 *
 * 更新规则（处理新的 K_j 块时）：
 *   S_ij = Q_i K_j^T / sqrt(d)                    # 局部 attention scores
 *   m_ij = max(m_{i-1}, rowmax(S_ij))              # 更新全局最大值
 *   P̃_ij = exp(S_ij - m_ij)                        # 局部 softmax 分子
 *   l_ij = exp(m_{i-1} - m_ij) * l_{i-1} + rowsum(P̃_ij)  # 校正分母
 *   O_i  = diag(exp(m_{i-1} - m_ij)) * O_{i-1} + P̃_ij * V_j  # 更新输出
 *
 * 最终归一化：
 *   O_i = O_i / l_i
 *
 * =========================================================================
 * 下面实现一个单头（single-head）的 Flash Attention kernel，
 * 以清晰展示算法逻辑。
 *
 * 输入：
 *   Q, K, V: (N, d) float32 张量（单头）
 *   N: 序列长度
 *   d: head dimension
 *   Br: Q block size（行方向）
 *   Bc: K/V block size（列方向）
 *
 * SRAM 要求：
 *   sQ:  Br × d
 *   sK:  Bc × d
 *   sV:  Bc × d
 *   sS:  Br × Bc
 *   sO:  Br × d
 *   sm, sl: Br（max 和 sum 向量）
 *
 * =========================================================================
 */

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// -------------------------------------------------------------------------
// Flash Attention V1: 基础实现（单头，无 causal masking）
//
// 参数说明：
//   每个 block 处理 Q 的一个 Br 行的 tile。
//   block 内的 thread 并行处理 d 维度的加载和计算。
// -------------------------------------------------------------------------
template <int Br, int Bc, int d>
__global__ void flash_attention_v1(
    const float* __restrict__ Q,   // (N, d)
    const float* __restrict__ K,   // (N, d)
    const float* __restrict__ V,   // (N, d)
    float* __restrict__ O,         // (N, d) 输出
    int N,
    float scale                    // 1 / sqrt(d)
) {
    // 每个 block 处理 Q 的 Br 行
    int q_tile_idx = blockIdx.x;   // 第几个 Q tile
    int q_row_start = q_tile_idx * Br;

    // Shared memory 分配
    __shared__ float sQ[Br][d];    // Q tile
    __shared__ float sK[Bc][d];    // K tile
    __shared__ float sV[Bc][d];    // V tile
    __shared__ float sS[Br][Bc];   // attention scores tile
    __shared__ float sO[Br][d];    // 输出 accumulator
    __shared__ float sm[Br];       // running max
    __shared__ float sl[Br];       // running sum

    int tx = threadIdx.x;  // 列方向（d 维度）

    // 初始化 m, l, O
    if (tx < Br) {
        sm[tx] = -FLT_MAX;
        sl[tx] = 0.0f;
    }
    for (int j = tx; j < Br * d; j += blockDim.x) {
        int r = j / d, c = j % d;
        sO[r][c] = 0.0f;
    }

    // 加载 Q tile 到 shared memory
    for (int j = tx; j < Br * d; j += blockDim.x) {
        int r = j / d, c = j % d;
        int global_row = q_row_start + r;
        sQ[r][c] = (global_row < N) ? Q[global_row * d + c] : 0.0f;
    }
    __syncthreads();

    // 遍历所有 K, V tile
    int num_kv_tiles = (N + Bc - 1) / Bc;
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * Bc;

        // 加载 K tile 和 V tile
        for (int j = tx; j < Bc * d; j += blockDim.x) {
            int r = j / d, c = j % d;
            int global_row = kv_start + r;
            sK[r][c] = (global_row < N) ? K[global_row * d + c] : 0.0f;
            sV[r][c] = (global_row < N) ? V[global_row * d + c] : 0.0f;
        }
        __syncthreads();

        // 计算 S = Q K^T * scale: [Br, Bc]
        // thread tx 计算 S 的第 tx 列（对于所有 Br 行）
        // 注意：这里每个 thread 负责计算 S 中的部分元素
        for (int i = 0; i < Br; i++) {
            for (int j = tx; j < Bc; j += blockDim.x) {
                float qk = 0.0f;
                for (int k = 0; k < d; k++) {
                    qk += sQ[i][k] * sK[j][k];
                }
                sS[i][j] = qk * scale;
            }
        }
        __syncthreads();

        // 对每个 Q 行（行 i），更新 online softmax 状态
        if (tx < Br) {
            int q_row = q_row_start + tx;
            if (q_row >= N) continue;

            // 找当前 K tile 中对应这行的最大值
            float m_new = sm[tx];
            for (int j = 0; j < min(Bc, N - kv_start); j++) {
                m_new = fmaxf(m_new, sS[tx][j]);
            }

            // 计算校正因子和新的 sum
            float exp_scale = expf(sm[tx] - m_new);  // 旧 max → 新 max 的校正
            float l_new = sl[tx] * exp_scale;
            for (int j = 0; j < min(Bc, N - kv_start); j++) {
                l_new += expf(sS[tx][j] - m_new);
            }

            // 更新输出 O = diag(exp_scale) * O_old + P_new * V
            for (int k = 0; k < d; k++) {
                float o_new = sO[tx][k] * exp_scale;
                for (int j = 0; j < min(Bc, N - kv_start); j++) {
                    o_new += expf(sS[tx][j] - m_new) * sV[j][k];
                }
                sO[tx][k] = o_new;
            }

            // 更新 m, l
            sm[tx] = m_new;
            sl[tx] = l_new;
        }
        __syncthreads();
    }

    // 最终归一化：O = O / l
    for (int j = tx; j < Br * d; j += blockDim.x) {
        int r = j / d, c = j % d;
        int global_row = q_row_start + r;
        if (global_row < N && sl[r] > 0.0f) {
            O[global_row * d + c] = sO[r][c] / sl[r];
        }
    }
}


// -------------------------------------------------------------------------
// Flash Attention V2 风格：更高效的 thread 分配
// （每个 block 的 thread 负责不同的 q 行）
// -------------------------------------------------------------------------
template <int Br, int Bc, int HEAD_DIM>
__global__ void flash_attention_v2(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N,
    float scale
) {
    int q_tile_idx = blockIdx.x;
    int q_row = q_tile_idx * Br + threadIdx.y;  // 每个 thread 行方向负责一个 q 向量

    if (q_row >= N) return;

    __shared__ float sK[Bc][HEAD_DIM];
    __shared__ float sV[Bc][HEAD_DIM];

    // 每个 thread 持有自己的 q 向量（寄存器）
    float q_reg[HEAD_DIM];
    for (int k = 0; k < HEAD_DIM; k++) {
        q_reg[k] = Q[q_row * HEAD_DIM + k];
    }

    float m = -FLT_MAX;
    float l = 0.0f;
    float o_reg[HEAD_DIM] = {};

    int tx = threadIdx.x;

    // 遍历 KV tiles
    for (int kv_tile = 0; kv_tile < (N + Bc - 1) / Bc; kv_tile++) {
        int kv_start = kv_tile * Bc;

        // threadIdx.y × threadIdx.x 共同加载 K, V
        for (int r = threadIdx.y; r < Bc; r += blockDim.y) {
            int kv_row = kv_start + r;
            for (int c = tx; c < HEAD_DIM; c += blockDim.x) {
                sK[r][c] = (kv_row < N) ? K[kv_row * HEAD_DIM + c] : 0.0f;
                sV[r][c] = (kv_row < N) ? V[kv_row * HEAD_DIM + c] : 0.0f;
            }
        }
        __syncthreads();

        // 计算 q 与当前 K tile 的 attention scores
        float m_tile = -FLT_MAX;
        float scores[Bc];  // 注意：Bc 需要是编译期常量
        for (int j = 0; j < min(Bc, N - kv_start); j++) {
            float qk = 0.0f;
            for (int k = 0; k < HEAD_DIM; k++) {
                qk += q_reg[k] * sK[j][k];
            }
            scores[j] = qk * scale;
            m_tile = fmaxf(m_tile, scores[j]);
        }

        // Online softmax 更新
        float m_new = fmaxf(m, m_tile);
        float l_new = l * expf(m - m_new);
        for (int j = 0; j < min(Bc, N - kv_start); j++) {
            float p = expf(scores[j] - m_new);
            l_new += p;
            for (int k = 0; k < HEAD_DIM; k++) {
                o_reg[k] = o_reg[k] * expf(m - m_new) / 1.0f;  // 后面再除
                o_reg[k] += p * sV[j][k];
            }
        }
        // 注意：校正因子需要统一应用
        float correction = expf(m - m_new);
        for (int k = 0; k < HEAD_DIM; k++) {
            o_reg[k] = o_reg[k] * correction;
            for (int j = 0; j < min(Bc, N - kv_start); j++) {
                o_reg[k] += expf(scores[j] - m_new) * sV[j][k];
            }
        }

        m = m_new;
        l = l_new;
        __syncthreads();
    }

    // 写回
    for (int k = 0; k < HEAD_DIM; k++) {
        O[q_row * HEAD_DIM + k] = o_reg[k] / l;
    }
}


// -------------------------------------------------------------------------
// Host 接口（多头版本的简化实现：串行跑多个头）
// -------------------------------------------------------------------------
extern "C" {

void flash_attention_cuda(
    const float* Q,  // (B, H, N, d)
    const float* K,
    const float* V,
    float* O,
    int B, int H, int N, int d
) {
    float scale = 1.0f / sqrtf((float)d);

    // 简化：d=64, Br=Bc=32
    const int Br = 32, Bc = 32;
    int grid = (N + Br - 1) / Br;

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int offset = (b * H + h) * N * d;
            // 动态 d 版本需要模板特化，这里写死 d=64
            // 实际项目中应使用 if-else 或 switch 选择不同的模板实例
            flash_attention_v1<Br, Bc, 64><<<grid, 32>>>(
                Q + offset, K + offset, V + offset, O + offset, N, scale
            );
        }
    }
    cudaDeviceSynchronize();
}

} // extern "C"
