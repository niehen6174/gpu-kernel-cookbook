"""
SageAttention V1 — Triton 实现（论文 SageAttention: Accurate 8-bit Attention）

核心创新（相比 FlashAttention）：
1. Q/K 量化为 INT8（per-block，即每个 BLOCK_M 或 BLOCK_N 个 token 一个 scale）
   - 量化公式：x_int8 = round(x / scale)，scale = max(|x|) / 127
   - Q 的 scale 已融合进去了 softmax_scale（即 sm_scale * 1.44269504 / log2e）
   - K 的 scale = max(|k_block|) / 127

2. K Smoothing（关键技巧，V1 论文 Section 3.1）：
   - 问题：K 的分布通常有大离群值，导致量化误差大
   - 解决：k_smooth = k - mean(k, dim=seq)，让 K 分布更对称、幅值更小
   - mean(k) 从 Q·km^T 中单独补回 lse（本实现简化：直接相减后量化）
   - 注意：减去的是整条序列的均值向量（对所有 token 相同），不影响 softmax 归一化

3. PV 乘积：
   - 使用 FP16，Triton tl.dot 输出 fp16 或 fp32
   - SageAttn v1：P (fp16) @ V (fp16) → acc (fp32)，即每轮先做 fp16 dot 再加到 fp32 buf

4. 在线 softmax（Flash Attention 标准流程）：
   - 使用 log2 代替自然对数（用 exp2 代替 exp，Triton 指令更快）
   - 维护 (m_i, l_i, acc) 三元组，每轮更新

输入约定：
  Q, K, V: (B, H, N, D) float16，tensor_layout="HND"
  - 本实现固定 HND 布局，可扩展

与原始 SageAttention 仓库的关系：
  本实现是从头 re-implement 的教学版，算法完全对应 SageAttention v1 论文的 Triton 路径，
  代码结构参考了 SageAttention 官方仓库的 triton/attn_qk_int8_per_block.py 和
  triton/quant_per_block.py，但做了精简和注释。
"""

import torch
import triton
import triton.language as tl
import math


# ============================================================
# 量化内核：per-block INT8
# 每个 BLK 行（一个 attention tile）用一个 scale
# ============================================================
@triton.jit
def _quant_per_block_int8(
    X_ptr, O_ptr, Scale_ptr,
    stride_xb, stride_xh, stride_xn,
    stride_ob, stride_oh, stride_on,
    stride_sb, stride_sh,
    seq_len,
    sm_scale,               # 量化前先乘以 sm_scale（仅对 Q 使用）
    C: tl.constexpr,        # head_dim
    BLK: tl.constexpr,      # block size（BLKQ 或 BLKK）
):
    """每个程序处理一个 (batch, head, block) 的量化"""
    off_blk = tl.program_id(0)
    off_h   = tl.program_id(1)
    off_b   = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK)   # [BLK] 当前 block 的 token 索引
    offs_c = tl.arange(0, C)                       # [C]  head_dim 索引

    x_ptrs = X_ptr + off_b * stride_xb + off_h * stride_xh \
             + offs_n[:, None] * stride_xn + offs_c[None, :]
    o_ptrs = O_ptr + off_b * stride_ob + off_h * stride_oh \
             + offs_n[:, None] * stride_on + offs_c[None, :]
    s_ptr  = Scale_ptr + off_b * stride_sb + off_h * stride_sh + off_blk

    # 加载并量化
    x = tl.load(x_ptrs, mask=offs_n[:, None] < seq_len, other=0.0)
    x = x.to(tl.float32) * sm_scale

    scale  = tl.max(tl.abs(x)) / 127.0
    x_q    = x / scale
    # round-half-away-from-zero（比 tl.rint 更准确）
    x_q   += 0.5 * tl.where(x_q >= 0, 1.0, -1.0)
    x_q    = x_q.to(tl.int8)

    tl.store(o_ptrs, x_q, mask=offs_n[:, None] < seq_len)
    tl.store(s_ptr, scale)


def quant_per_block_int8(x: torch.Tensor, BLK: int, sm_scale: float = 1.0,
                         tensor_layout: str = "HND") -> tuple:
    """
    将输入张量 x 量化为 per-block INT8。

    Returns:
        x_int8: 量化后的张量，shape 同 x，dtype=int8
        x_scale: 每个 block 的 scale，shape (B, H, num_blocks), float32
    """
    x_int8 = torch.empty(x.shape, dtype=torch.int8, device=x.device)

    if tensor_layout == "HND":
        B, H, seq, C = x.shape
        stride_b, stride_h, stride_n = x.stride(0), x.stride(1), x.stride(2)
        stride_ob, stride_oh, stride_on = x_int8.stride(0), x_int8.stride(1), x_int8.stride(2)
    else:  # NHD
        B, seq, H, C = x.shape
        stride_b, stride_h, stride_n = x.stride(0), x.stride(2), x.stride(1)
        stride_ob, stride_oh, stride_on = x_int8.stride(0), x_int8.stride(2), x_int8.stride(1)

    num_blocks = triton.cdiv(seq, BLK)
    x_scale = torch.empty((B, H, num_blocks), dtype=torch.float32, device=x.device)

    _quant_per_block_int8[(num_blocks, H, B)](
        x, x_int8, x_scale,
        stride_b, stride_h, stride_n,
        stride_ob, stride_oh, stride_on,
        x_scale.stride(0), x_scale.stride(1),
        seq, sm_scale,
        C=C, BLK=BLK,
    )
    return x_int8, x_scale


# ============================================================
# 主注意力内核
# K 以 row-major 布局，用转置访问方式做 QK^T
# ============================================================
@triton.jit
def _sage_v1_kernel(
    Q_int8, Q_scale,
    K_int8, K_scale,
    V,
    Out,
    stride_qb, stride_qh, stride_qn,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_on,
    B, H_qo, H_kv,
    qo_len, kv_len,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h   = tl.program_id(1)
    off_b   = tl.program_id(2)

    num_kv_groups = H_qo // H_kv
    off_kv_h = off_h // num_kv_groups

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # 加载 Q (INT8) [BM, D]
    q_base = Q_int8 + off_b * stride_qb + off_h * stride_qh
    q = tl.load(q_base + offs_m[:, None] * stride_qn + offs_d[None, :],
                mask=offs_m[:, None] < qo_len, other=0)

    q_scale_flat = Q_scale + (off_b * H_qo + off_h) * tl.cdiv(qo_len, BLOCK_M)
    q_scale = tl.load(q_scale_flat + start_m)

    K_base = K_int8 + off_b * stride_kb + off_kv_h * stride_kh
    K_scale_base = K_scale + (off_b * H_kv + off_kv_h) * tl.cdiv(kv_len, BLOCK_N)
    V_base = V + off_b * stride_vb + off_kv_h * stride_vh

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    hi = (start_m + 1) * BLOCK_M if CAUSAL else kv_len

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # 加载 K (INT8) [D, BN]（转置访问 = K^T，用于 q[BM,D] @ k^T[D,BN]）
        k = tl.load(K_base + offs_d[:, None] * 1 + offs_n[None, :] * stride_kn,
                    mask=offs_n[None, :] < kv_len, other=0)   # [D, BN]
        k_scale = tl.load(K_scale_base + start_n // BLOCK_N)

        # QK^T: [BM, D] @ [D, BN] → [BM, BN]，反量化到 float
        qk = tl.dot(q, k).to(tl.float32) * (q_scale * k_scale)

        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
        qk = tl.where(offs_n[None, :] < kv_len, qk, float("-inf"))

        # Online softmax (exp2 路径，更快)
        m_ij  = tl.maximum(m_i, tl.max(qk, axis=1))
        p     = tl.math.exp2(qk - m_ij[:, None])
        l_ij  = tl.sum(p, axis=1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i   = l_i * alpha + l_ij
        acc   = acc * alpha[:, None]
        m_i   = m_ij

        # PV: P (fp16) @ V [BN, D] → FP32 acc
        v = tl.load(V_base + offs_n[:, None] * stride_vn + offs_d[None, :],
                    mask=offs_n[:, None] < kv_len, other=0.0)
        acc += tl.dot(p.to(tl.float16), v, out_dtype=tl.float32)

    acc = acc / l_i[:, None]

    O_base = Out + off_b * stride_ob + off_h * stride_oh
    tl.store(O_base + offs_m[:, None] * stride_on + offs_d[None, :],
             acc.to(Out.type.element_ty),
             mask=offs_m[:, None] < qo_len)


# ============================================================
# Python 包装函数
# ============================================================
def sageattn_v1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale: float = None,
    smooth_k: bool = True,
) -> torch.Tensor:
    """
    SageAttention V1 前向传播。

    算法：
    1. (可选) K Smoothing: k_smooth = k - mean(k, dim=seq)
       - 去除序列均值，降低 K 的量化误差
       - 等价于修正后的 attention（数学上对 softmax 结果有微小影响）
    2. Q/K Per-block INT8 量化（BLKQ=128, BLKK=64）
       - Q scale 已融合 sm_scale（用 log2e 代替 e，与 exp2 匹配）
    3. FlashAttention 在线 softmax：qk_int32 * (q_scale * k_scale) → softmax → P @ V

    参数:
        q, k, v: (B, H, N, D) float16，HND 布局
        is_causal: 是否使用因果 mask
        sm_scale: softmax 缩放（默认 1/sqrt(D)）
        smooth_k: 是否减去 K 的序列均值
    """
    assert q.is_cuda and q.dtype == torch.float16, "q must be float16 on CUDA"
    assert k.dtype == torch.float16 and v.dtype == torch.float16

    B, H_qo, qo_len, D = q.shape
    _, H_kv, kv_len, _ = k.shape

    if sm_scale is None:
        sm_scale = D ** -0.5

    # K Smoothing：减去 K 的序列均值
    # 数学等价性：Q(K-μ)^T = QK^T - Qμ^T
    # 由于 μ 是全序列 token 的均值向量（常数），减去后 K 的动态范围更小
    # 对应 SageAttention 论文 Algorithm 1 Step 3
    if smooth_k:
        km = k.mean(dim=2, keepdim=True)   # (B, H_kv, 1, D)
        k  = k - km

    # Per-block INT8 量化
    # Q: scale 融合 sm_scale * log2(e)（用于内核中的 exp2 路径）
    BLKQ, BLKK = 128, 64
    q_int8, q_scale = quant_per_block_int8(q, BLKQ, sm_scale=sm_scale * 1.44269504)
    k_int8, k_scale = quant_per_block_int8(k, BLKK, sm_scale=1.0)

    out = torch.empty_like(q)

    BLOCK_M = 128
    BLOCK_N = 64
    assert D in (64, 128), f"SageAttention v1 只支持 head_dim=64 或 128，当前 D={D}"

    grid = (triton.cdiv(qo_len, BLOCK_M), H_qo, B)
    _sage_v1_kernel[grid](
        q_int8, q_scale,
        k_int8, k_scale,
        v, out,
        q_int8.stride(0), q_int8.stride(1), q_int8.stride(2),
        k_int8.stride(0), k_int8.stride(1), k_int8.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        B, H_qo, H_kv,
        qo_len, kv_len,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CAUSAL=is_causal,
        num_warps=4 if D == 64 else 8,
        num_stages=3,
    )
    return out
