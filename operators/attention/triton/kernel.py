"""
Flash Attention - Triton 实现

参考 Triton 官方的 flash attention tutorial。
这是一个非常经典的 Triton 示例，展示了 Triton 如何处理复杂的
IO-aware 算法（online softmax + tiling）。

核心技术：
1. 分块：Q 按行分块，K/V 按列分块
2. Online softmax：边扫 K 边维护 (m, l, O) 状态
3. tl.dot：调用 tensor core 做矩阵乘
4. Causal mask（可选）：只 attend to 之前的 token

支持：
- 单精度/半精度
- Causal mask
- 批处理（B, H, N, d）
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    softmax_scale,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    B, H, N, D,
    BLOCK_M: tl.constexpr,  # Q tile 的行数
    BLOCK_N: tl.constexpr,  # K/V tile 的行数
    BLOCK_D: tl.constexpr,  # head dim（必须是 2 的幂）
    CAUSAL: tl.constexpr,   # 是否使用 causal mask
):
    """
    每个 program 处理一个 (batch, head, q_tile) 组合。

    program_id(0): q_tile 索引（在 N 方向）
    program_id(1): batch * head 的索引

    输入/输出都是 (B, H, N, D) 形状，用 strides 访问。
    """
    # 确定当前 program 处理哪个 (batch, head, q_tile)
    start_m = tl.program_id(0)   # q tile 在序列方向的索引
    off_bh  = tl.program_id(1)   # batch * head 的平铺索引
    off_b   = off_bh // H
    off_h   = off_bh % H

    # 计算当前 (b, h) 的基地址偏移
    Q_base = Q_ptr + off_b * stride_qb + off_h * stride_qh
    K_base = K_ptr + off_b * stride_kb + off_h * stride_kh
    V_base = V_ptr + off_b * stride_vb + off_h * stride_vh
    O_base = O_ptr + off_b * stride_ob + off_h * stride_oh

    # 当前 Q tile 的行偏移
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_d = tl.arange(0, BLOCK_D)                       # [BLOCK_D]

    # 加载 Q tile: [BLOCK_M, BLOCK_D]
    Q_ptrs = Q_base + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < N) & (offs_d[None, :] < D)
    q = tl.load(Q_ptrs, mask=q_mask, other=0.0)

    # 初始化 online softmax 状态
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)  # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                # running sum
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)       # output accumulator

    # 确定 KV 的结束位置（causal: 只 attend 到位置 <= q 的 k）
    if CAUSAL:
        kv_end = min(N, (start_m + 1) * BLOCK_M)
    else:
        kv_end = N

    # 遍历所有 KV tile
    for start_n in range(0, kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)  # [BLOCK_N]

        # 加载 K tile: [BLOCK_N, BLOCK_D]
        K_ptrs = K_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k_mask = (offs_n[:, None] < N) & (offs_d[None, :] < D)
        k = tl.load(K_ptrs, mask=k_mask, other=0.0)

        # 加载 V tile: [BLOCK_N, BLOCK_D]
        V_ptrs = V_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(V_ptrs, mask=k_mask, other=0.0)

        # 计算 attention scores: S = Q K^T * scale
        # [BLOCK_M, BLOCK_D] @ [BLOCK_D, BLOCK_N] → [BLOCK_M, BLOCK_N]
        s = tl.dot(q, tl.trans(k)) * softmax_scale

        # Causal mask：将 k > q 的位置设为 -inf
        if CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        # 越界 mask
        s = tl.where(offs_n[None, :] < N, s, float("-inf"))

        # Online softmax 更新
        m_ij = tl.max(s, axis=1)          # [BLOCK_M]，当前 tile 的行最大值
        m_new = tl.maximum(m_i, m_ij)     # 更新全局最大值

        # 校正因子：旧的 sum 需要乘以 exp(m_old - m_new)
        alpha = tl.exp(m_i - m_new)       # [BLOCK_M]，校正旧 acc

        # 计算当前 tile 的 exp：P = exp(S - m_new)
        p = tl.exp(s - m_new[:, None])    # [BLOCK_M, BLOCK_N]

        # 更新 l（归一化分母）
        l_new = l_i * alpha + tl.sum(p, axis=1)

        # 更新输出 acc：
        # acc_new = diag(alpha) * acc_old + P @ V
        acc = acc * alpha[:, None] + tl.dot(p, v)

        # 更新 m, l
        m_i = m_new
        l_i = l_new

    # 最终归一化：O = acc / l
    acc = acc / l_i[:, None]

    # 写回 O
    O_ptrs = O_base + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    o_mask = (offs_m[:, None] < N) & (offs_d[None, :] < D)
    tl.store(O_ptrs, acc, mask=o_mask)


def flash_attention_triton(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
    sm_scale: float = None,
) -> torch.Tensor:
    """
    Flash Attention forward pass.

    Args:
        Q, K, V: (B, H, N, D) float32/float16 CUDA tensor
        causal: 是否使用因果注意力（下三角 mask）
        sm_scale: softmax 缩放因子（默认 1/sqrt(D)）

    Returns:
        O: (B, H, N, D) 注意力输出
    """
    assert Q.is_cuda
    B, H, N, D = Q.shape
    assert D in [16, 32, 64, 128], f"Head dim D={D} must be power of 2 and <= 128"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    O = torch.empty_like(Q)

    # Tile 大小：可以根据 SRAM 大小和 D 调整
    BLOCK_M = 64  # Q tile 行数
    BLOCK_N = 64  # KV tile 行数
    BLOCK_D = triton.next_power_of_2(D)

    # grid: (num_q_tiles, B*H)
    grid = (triton.cdiv(N, BLOCK_M), B * H)

    flash_attn_fwd_kernel[grid](
        Q, K, V, O,
        sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        B, H, N, D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        CAUSAL=causal,
        num_warps=4,
        num_stages=2,
    )
    return O
