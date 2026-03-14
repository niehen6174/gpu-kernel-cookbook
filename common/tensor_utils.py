"""
张量工具：生成随机测试数据，支持多种数据类型和设备
"""

import torch
from typing import Tuple


def rand_tensor(*shape, dtype=torch.float32, device="cuda") -> torch.Tensor:
    return torch.randn(*shape, dtype=dtype, device=device)


def rand_int_tensor(*shape, low=0, high=128, device="cuda") -> torch.Tensor:
    return torch.randint(low, high, shape, device=device)


def ones_tensor(*shape, dtype=torch.float32, device="cuda") -> torch.Tensor:
    return torch.ones(*shape, dtype=dtype, device=device)


def make_matmul_inputs(
    M: int, N: int, K: int, dtype=torch.float32, device="cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成 matmul 的输入：A(M,K), B(K,N)"""
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    return A, B


def make_attention_inputs(
    batch: int, heads: int, seq_len: int, head_dim: int,
    dtype=torch.float32, device="cuda"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """生成 attention 的 Q, K, V 输入"""
    Q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    K = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    V = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    return Q, K, V
