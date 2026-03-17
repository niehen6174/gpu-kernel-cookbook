"""
RoPE (Rotary Position Embedding) - Triton 实现

V1: rope_triton — 标准非交错风格（GPT-NeoX / rotate_half）
    每个 program 处理一个 (token, head) 对。
    线程加载 cos/sin（按 position），对 Q/K 应用旋转。

V2: rope_interleaved_triton — 交错风格（Llama-style）
    按相邻配对旋转：(x[2i], x[2i+1])
"""

import triton
import triton.language as tl
import torch


# -------------------------------------------------------------------------
# V1: Non-interleaved (rotate_half style)
# -------------------------------------------------------------------------
@triton.jit
def rope_kernel(
    Q_ptr, K_ptr,
    COS_ptr, SIN_ptr,
    POS_ptr,
    seq_len, num_heads, head_dim,
    q_stride_s, q_stride_h,   # Q strides: seq, head
    k_stride_s, k_stride_h,
    cos_stride,                # cos stride: seq (cos is (max_seq, half_dim))
    HALF_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # >= HALF_DIM
):
    """
    每个 program 处理一个 (token, head) 对。
    program_id(0) = token * num_heads + head
    """
    pid = tl.program_id(0)
    token_idx = pid // num_heads
    head_idx  = pid % num_heads

    # 查找该 token 的 position
    pos = tl.load(POS_ptr + token_idx)

    # cols: 0..HALF_DIM-1
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < HALF_DIM

    # 加载 cos/sin (HALF_DIM 个值)
    cos = tl.load(COS_ptr + pos * cos_stride + cols, mask=mask, other=0.0)
    sin = tl.load(SIN_ptr + pos * cos_stride + cols, mask=mask, other=0.0)

    # ---- 处理 Q ----
    q_base = Q_ptr + token_idx * q_stride_s + head_idx * q_stride_h
    # x1 = q[..., :HALF_DIM], x2 = q[..., HALF_DIM:]
    q1 = tl.load(q_base + cols,           mask=mask, other=0.0)
    q2 = tl.load(q_base + cols + HALF_DIM, mask=mask, other=0.0)
    q1_out = q1 * cos - q2 * sin
    q2_out = q2 * cos + q1 * sin
    tl.store(q_base + cols,           q1_out, mask=mask)
    tl.store(q_base + cols + HALF_DIM, q2_out, mask=mask)

    # ---- 处理 K ----
    k_base = K_ptr + token_idx * k_stride_s + head_idx * k_stride_h
    k1 = tl.load(k_base + cols,           mask=mask, other=0.0)
    k2 = tl.load(k_base + cols + HALF_DIM, mask=mask, other=0.0)
    k1_out = k1 * cos - k2 * sin
    k2_out = k2 * cos + k1 * sin
    tl.store(k_base + cols,           k1_out, mask=mask)
    tl.store(k_base + cols + HALF_DIM, k2_out, mask=mask)


def rope_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple:
    """
    Apply RoPE (non-interleaved) to Q and K inplace.
    q, k: (seq_len, num_heads, head_dim) float32 CUDA, contiguous
    cos_cache, sin_cache: (max_seq_len, head_dim//2)
    positions: (seq_len,) int32/int64
    Returns (q, k) with rotation applied inplace.
    """
    assert q.is_cuda
    q = q.clone()
    k = k.clone()
    seq_len, num_heads, head_dim = q.shape
    half_dim = head_dim // 2

    BLOCK_SIZE = triton.next_power_of_2(half_dim)
    total_programs = seq_len * num_heads

    rope_kernel[(total_programs,)](
        q, k,
        cos_cache, sin_cache,
        positions,
        seq_len, num_heads, head_dim,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        cos_cache.stride(0),
        HALF_DIM=half_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return q, k


# -------------------------------------------------------------------------
# V2: Interleaved (Llama-style: process pairs (x[2i], x[2i+1]))
# -------------------------------------------------------------------------
@triton.jit
def rope_interleaved_kernel(
    Q_ptr, K_ptr,
    COS_ptr, SIN_ptr,
    POS_ptr,
    seq_len, num_heads, head_dim,
    q_stride_s, q_stride_h,
    k_stride_s, k_stride_h,
    cos_stride,
    HALF_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    交错风格 RoPE：(x[2i], x[2i+1]) 配对旋转。
    每个 program 处理一个 (token, head) 对。
    """
    pid = tl.program_id(0)
    token_idx = pid // num_heads
    head_idx  = pid % num_heads

    pos = tl.load(POS_ptr + token_idx)

    # pair indices 0..HALF_DIM-1
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < HALF_DIM

    # cos/sin for each pair
    cos = tl.load(COS_ptr + pos * cos_stride + cols, mask=mask, other=0.0)
    sin = tl.load(SIN_ptr + pos * cos_stride + cols, mask=mask, other=0.0)

    # ---- Q ----
    q_base = Q_ptr + token_idx * q_stride_s + head_idx * q_stride_h
    # interleaved: even indices = 2*cols, odd = 2*cols+1
    q1 = tl.load(q_base + 2 * cols,     mask=mask, other=0.0)
    q2 = tl.load(q_base + 2 * cols + 1, mask=mask, other=0.0)
    tl.store(q_base + 2 * cols,     q1 * cos - q2 * sin, mask=mask)
    tl.store(q_base + 2 * cols + 1, q2 * cos + q1 * sin, mask=mask)

    # ---- K ----
    k_base = K_ptr + token_idx * k_stride_s + head_idx * k_stride_h
    k1 = tl.load(k_base + 2 * cols,     mask=mask, other=0.0)
    k2 = tl.load(k_base + 2 * cols + 1, mask=mask, other=0.0)
    tl.store(k_base + 2 * cols,     k1 * cos - k2 * sin, mask=mask)
    tl.store(k_base + 2 * cols + 1, k2 * cos + k1 * sin, mask=mask)


def rope_interleaved_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple:
    """
    Apply RoPE (interleaved / Llama style) to Q and K.
    """
    assert q.is_cuda
    q = q.clone()
    k = k.clone()
    seq_len, num_heads, head_dim = q.shape
    half_dim = head_dim // 2

    BLOCK_SIZE = triton.next_power_of_2(half_dim)
    total_programs = seq_len * num_heads

    rope_interleaved_kernel[(total_programs,)](
        q, k,
        cos_cache, sin_cache,
        positions,
        seq_len, num_heads, head_dim,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        cos_cache.stride(0),
        HALF_DIM=half_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return q, k
