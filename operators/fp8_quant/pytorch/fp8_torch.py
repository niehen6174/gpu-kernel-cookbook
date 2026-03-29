"""
FP8 量化 PyTorch Baseline

实现两种量化方案：
  Scheme A：Per-Tensor FP8   — 整个 tensor 共享一个 scale
  Scheme B：Per-Block FP8    — 128-element 分组量化，每组独立 scale

FP8 格式：float8_e4m3fn，max = 448.0
"""

import math
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

FP8_MAX = 448.0
FP8_DTYPE = torch.float8_e4m3fn


# ============================================================
# Scheme A：Per-Tensor FP8
# ============================================================

def fp8_per_tensor_quant(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-Tensor FP8 量化。

    Args:
        x: 任意形状的 float tensor（建议 FP16/BF16/FP32）

    Returns:
        q         : float8_e4m3fn，与 x 相同形状
        inv_scale : scalar float32 tensor（解量化 scale = amax / 448.0）

    量化公式：
        scale     = 448.0 / amax(|x|)
        q         = clamp(x * scale, -448, 448).to(fp8)
        inv_scale = 1.0 / scale = amax / 448.0   ← 存储的是解量化 scale
    """
    x_f32 = x.float()
    amax = x_f32.abs().amax().clamp(min=1e-12)
    scale = FP8_MAX / amax
    inv_scale = amax / FP8_MAX                      # 解量化 scale，存储
    q = (x_f32 * scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return q, inv_scale.to(torch.float32)


def fp8_per_tensor_dequant(q: torch.Tensor, inv_scale: torch.Tensor,
                            out_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """
    Per-Tensor FP8 解量化。

    Args:
        q         : float8_e4m3fn tensor
        inv_scale : scalar float32（amax / 448.0）
        out_dtype : 输出精度（默认 FP16）

    Returns:
        解量化后的 tensor，形状与 q 相同
    """
    return (q.float() * inv_scale.float()).to(out_dtype)


def fp8_per_tensor_gemm(
    a: torch.Tensor,
    a_inv_scale: torch.Tensor,
    b: torch.Tensor,
    b_inv_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Per-Tensor FP8 GEMM：out = a @ b.T，带 scale 解量化。

    Args:
        a           : (M, K) float8_e4m3fn
        a_inv_scale : scalar float32（a 的解量化 scale）
        b           : (N, K) float8_e4m3fn（权重，转置后与 a 匹配）
        b_inv_scale : scalar float32（b 的解量化 scale）
        out_dtype   : 输出精度（默认 BF16）

    Returns:
        out : (M, N)，dtype = out_dtype

    实现：
        1. 优先使用 torch._scaled_mm（SM90+ HW FP8 GEMM，输出 BF16）
        2. fallback：FP32 dequant 后做 FP32 GEMM
    """
    assert a.dtype == FP8_DTYPE, f"a must be {FP8_DTYPE}, got {a.dtype}"
    assert b.dtype == FP8_DTYPE, f"b must be {FP8_DTYPE}, got {b.dtype}"

    # 尝试 torch._scaled_mm（需要 SM90+ 并且两个 tensor 是 FP8）
    try:
        # _scaled_mm 要求：a (M,K) row-major, b (K,N) col-major（即 b.T 后 row-major）
        # 输入 b 是 (N,K) row-major，b.T 是 (K,N) 等同于 col-major (N,K)
        # scale_a, scale_b 必须是 CPU scalar float32 tensor（或 CUDA scalar）
        out = torch._scaled_mm(
            a, b.T,
            scale_a=a_inv_scale.reshape(1),
            scale_b=b_inv_scale.reshape(1),
            out_dtype=out_dtype,
        )
        return out
    except (RuntimeError, AttributeError):
        # fallback：FP32 dequant + FP32 GEMM
        a_f32 = a.float() * a_inv_scale.float()
        b_f32 = b.float() * b_inv_scale.float()
        out = a_f32 @ b_f32.T
        return out.to(out_dtype)


# ============================================================
# Scheme B：Per-Block FP8（group_size=128）
# ============================================================

def fp8_per_block_act_quant(
    x: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-Block FP8 激活量化（沿 K 维，每 group_size 个元素一组）。

    Args:
        x          : (M, K) float tensor
        group_size : 每组元素数（默认 128）

    Returns:
        q         : (M, K) float8_e4m3fn
        inv_scales: (M, K // group_size) float32（解量化 scale，每组一个）

    量化公式（每组 g）：
        inv_scale[i, g] = max(|x[i, g*gs:(g+1)*gs]|) / 448.0  ← 解量化 scale
        q[i, g*gs:]     = clamp(x[i, g*gs:] / inv_scale[i,g], -448, 448).to(fp8)
    """
    M, K = x.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    ngroups = K // group_size

    x_f32 = x.float()
    # reshape 为 (M, ngroups, group_size)
    x_g = x_f32.reshape(M, ngroups, group_size)

    # 每组的 amax → 解量化 scale
    amax_g = x_g.abs().amax(dim=-1).clamp(min=1e-12)   # (M, ngroups)
    inv_scales = amax_g / FP8_MAX                        # (M, ngroups) 解量化 scale

    # 量化：x / inv_scale = x * (1/inv_scale) = x * (448 / amax)
    q_f32 = x_g / inv_scales.unsqueeze(-1)               # (M, ngroups, group_size)
    q_f32 = q_f32.clamp(-FP8_MAX, FP8_MAX)
    q = q_f32.reshape(M, K).to(FP8_DTYPE)

    return q, inv_scales.to(torch.float32)


def fp8_per_block_weight_quant(
    w: torch.Tensor,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-Block FP8 权重量化（2D block，block_size×block_size）。

    Args:
        w          : (N, K) float tensor（权重矩阵）
        block_size : 2D block 大小（默认 128，即 128×128 块）

    Returns:
        q         : (N, K) float8_e4m3fn
        inv_scales: (N // block_size, K // block_size) float32（解量化 scale）

    量化公式（每个 2D 块 (n_blk, k_blk)）：
        inv_scale[n_blk, k_blk] = max(|w[n*bs:, k*bs:]|) / 448.0  ← 解量化 scale
        q[n*bs:, k*bs:] = clamp(w[n*bs:, k*bs:] / inv_scale, -448, 448).to(fp8)
    """
    N, K = w.shape
    assert N % block_size == 0, f"N={N} must be divisible by block_size={block_size}"
    assert K % block_size == 0, f"K={K} must be divisible by block_size={block_size}"

    n_blocks = N // block_size
    k_blocks = K // block_size

    w_f32 = w.float()
    # reshape 为 (n_blocks, block_size, k_blocks, block_size)
    w_blocks = w_f32.reshape(n_blocks, block_size, k_blocks, block_size)

    # 每个 2D block 的 amax → 解量化 scale (n_blocks, k_blocks)
    amax_blocks = w_blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    inv_scales = amax_blocks / FP8_MAX                    # (n_blocks, k_blocks)

    # 量化：w_block / inv_scale（广播到 (n_blocks, block_size, k_blocks, block_size)）
    q_f32 = w_blocks / inv_scales.unsqueeze(1).unsqueeze(3)
    q_f32 = q_f32.clamp(-FP8_MAX, FP8_MAX)
    q = q_f32.reshape(N, K).to(FP8_DTYPE)

    return q, inv_scales.to(torch.float32)


def fp8_per_block_gemm(
    a: torch.Tensor,
    a_inv_scales: torch.Tensor,
    b: torch.Tensor,
    b_inv_scales: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Per-Block FP8 GEMM（Python 参考实现，naive loop）。

    精确公式：
        out[i,j] = Σ_g  inv_scale_a[i,g] * inv_scale_b[j//128, g] * dot(q_a[i,g*128:], q_b[j,g*128:])

    Args:
        a           : (M, K) float8_e4m3fn
        a_inv_scales: (M, K // 128) float32
        b           : (N, K) float8_e4m3fn（权重，已量化）
        b_inv_scales: (N // 128, K // 128) float32
        out_dtype   : 输出精度（默认 BF16）

    Returns:
        out : (M, N) out_dtype
    """
    M, K = a.shape
    N = b.shape[0]
    group_size = 128  # per-block 固定 group size

    assert K % group_size == 0
    assert a_inv_scales.shape == (M, K // group_size)
    assert b_inv_scales.shape == (N // group_size, K // group_size)

    ngroups_k = K // group_size

    # 将 FP8 转换为 FP32 后做 GEMM（参考实现）
    a_f32 = a.float()
    b_f32 = b.float()

    # 方法：向量化的 per-group scale 乘法，然后矩阵乘
    # a_scaled[i, g*128:] = a_f32[i, g*128:] * a_inv_scales[i, g]
    a_scaled = a_f32.reshape(M, ngroups_k, group_size) * a_inv_scales.unsqueeze(-1)
    a_scaled = a_scaled.reshape(M, K)                  # (M, K) FP32

    # b_scaled[j, g*128:] = b_f32[j, g*128:] * b_inv_scales[j//128, g]
    # b_inv_scales: (N//128, K//128)，需要广播到 (N, K)
    n_blocks = N // group_size
    b_reshaped = b_f32.reshape(n_blocks, group_size, ngroups_k, group_size)
    # b_inv_scales: (n_blocks, ngroups_k) → 广播到 (n_blocks, group_size, ngroups_k, group_size)
    b_scaled = b_reshaped * b_inv_scales.unsqueeze(1).unsqueeze(3)
    b_scaled = b_scaled.reshape(N, K)                  # (N, K) FP32

    out = a_scaled @ b_scaled.T
    return out.to(out_dtype)


# ============================================================
# 精度分析工具
# ============================================================

def compute_quant_error(
    x_orig: torch.Tensor,
    x_dequant: torch.Tensor,
) -> Dict[str, float]:
    """
    计算量化误差指标。

    Args:
        x_orig    : 原始 float tensor
        x_dequant : 量化后解量化的 float tensor

    Returns:
        dict 包含：
          rmse          : 均方根误差
          max_abs_error : 最大绝对误差
          snr_db        : 信噪比（dB）
          cosine_sim    : 余弦相似度
    """
    x_o = x_orig.float()
    x_d = x_dequant.float()
    diff = x_o - x_d

    rmse = diff.pow(2).mean().sqrt().item()
    max_err = diff.abs().max().item()

    sig_norm = x_o.norm()
    noise_norm = diff.norm().clamp(min=1e-12)
    snr_db = 20 * math.log10((sig_norm / noise_norm).item())

    cos_sim = F.cosine_similarity(
        x_o.flatten().unsqueeze(0),
        x_d.flatten().unsqueeze(0),
    ).item()

    return {
        "rmse": rmse,
        "max_abs_error": max_err,
        "snr_db": snr_db,
        "cosine_sim": cos_sim,
    }
