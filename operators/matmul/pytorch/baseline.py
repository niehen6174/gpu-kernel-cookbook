"""
Matmul - PyTorch Baseline

PyTorch 内部使用 cuBLAS（CUDA 11 以后也可以用 cuBLASLt + Tensor Core）。
"""
import torch


def matmul_pytorch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A, B)
