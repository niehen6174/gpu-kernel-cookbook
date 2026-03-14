"""
Matrix Transpose - PyTorch Baseline
"""
import torch


def transpose_pytorch(A: torch.Tensor) -> torch.Tensor:
    """PyTorch 矩阵转置（调用 cuBLAS 底层）"""
    return A.T.contiguous()
