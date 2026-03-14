"""Attention - PyTorch Baseline"""
import torch
import torch.nn.functional as F
import math


def attention_pytorch(Q, K, V, causal=False):
    """
    标准 Scaled Dot-Product Attention。
    Q, K, V: (B, H, N, D)
    """
    B, H, N, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    # (B, H, N, N)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if causal:
        mask = torch.tril(torch.ones(N, N, device=Q.device)).bool()
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V)


def attention_pytorch_sdpa(Q, K, V, causal=False):
    """
    使用 PyTorch 2.0 的 scaled_dot_product_attention
    （内部可以调用 FlashAttention 或 math attention）
    """
    return F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
