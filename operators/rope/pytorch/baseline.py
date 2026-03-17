"""RoPE (Rotary Position Embedding) - PyTorch Baseline

数学定义 (GPT-NeoX / non-interleaved, rotate_half style):
    将 x 分为前半 x1 = x[..., :d/2] 和后半 x2 = x[..., d/2:]
    x'[..., :d/2] = x1 * cos - x2 * sin
    x'[..., d/2:] = x2 * cos + x1 * sin

    θ_i = pos / 10000^(2i/d)
    cos_cache[pos, i] = cos(θ_i),  shape (max_seq_len, head_dim//2)
    sin_cache[pos, i] = sin(θ_i),  shape (max_seq_len, head_dim//2)

参考: GPT-NeoX / Hugging Face transformers RoPE implementation
"""
import torch
import math


def build_cos_sin_cache(
    max_seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: str = "cuda",
) -> tuple:
    """
    预计算 cos/sin 缓存。
    返回: (cos_cache, sin_cache) 各形状 (max_seq_len, head_dim // 2)
    """
    half_dim = head_dim // 2
    # θ_i = 1 / (base^(2i/head_dim)), shape (half_dim,)
    inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, device=device).float() / half_dim))
    # positions: (max_seq_len,)
    positions = torch.arange(max_seq_len, device=device).float()
    # outer product → (max_seq_len, half_dim)
    freqs = torch.outer(positions, inv_freq)
    cos_cache = freqs.cos()
    sin_cache = freqs.sin()
    return cos_cache, sin_cache


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of x: [-x2, x1]"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple:
    """
    Apply RoPE to Q and K tensors (non-interleaved / GPT-NeoX style).

    Args:
        q: (seq_len, num_heads, head_dim) float32
        k: (seq_len, num_heads, head_dim) float32
        cos_cache: (max_seq_len, head_dim//2) precomputed cosines
        sin_cache: (max_seq_len, head_dim//2) precomputed sines
        positions: (seq_len,) int64 position indices

    Returns:
        (q_out, k_out) same shape as inputs
    """
    # Gather cos/sin for the given positions → (seq_len, head_dim//2)
    cos = cos_cache[positions]  # (seq_len, head_dim//2)
    sin = sin_cache[positions]  # (seq_len, head_dim//2)

    # Repeat to full head_dim: (seq_len, head_dim)
    cos = torch.cat([cos, cos], dim=-1)  # (seq_len, head_dim)
    sin = torch.cat([sin, sin], dim=-1)  # (seq_len, head_dim)

    # Broadcast over num_heads: (seq_len, 1, head_dim)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return q_out, k_out
