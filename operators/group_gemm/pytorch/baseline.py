"""
Group GEMM - PyTorch Baseline

Group GEMM 计算一批独立的矩阵乘法：
    C[g] = A[g] @ B[g]    g = 0, 1, ..., G-1

每个 group g 的矩阵大小可以不同：A[g]: (M_g, K_g)，B[g]: (K_g, N_g)

两种接口：
1. 等大小 group（batched）：所有 group 的 (M, K, N) 相同 → torch.bmm
2. 变大小 group（jagged）：每组大小不同 → 逐组调用 torch.matmul
"""

import torch
from typing import List


def group_gemm_pytorch_fixed(
    A: torch.Tensor,   # (G, M, K)
    B: torch.Tensor,   # (G, K, N)
) -> torch.Tensor:
    """
    Fixed-size group GEMM: 所有 group 的 (M, K, N) 相同。
    A: (G, M, K), B: (G, K, N) → C: (G, M, N)
    """
    return torch.bmm(A, B)


def group_gemm_pytorch_var(
    A_list: List[torch.Tensor],   # G 个 (M_g, K_g) 矩阵
    B_list: List[torch.Tensor],   # G 个 (K_g, N_g) 矩阵
) -> List[torch.Tensor]:
    """
    Variable-size group GEMM: 每个 group 的 (M, K, N) 可以不同。
    返回 G 个输出矩阵 C_g = A_g @ B_g。
    """
    return [torch.matmul(A, B) for A, B in zip(A_list, B_list)]
