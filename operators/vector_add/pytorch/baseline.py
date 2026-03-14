"""
Vector Add - PyTorch Baseline

用于与 CUDA/Triton/CuTe kernel 做正确性和性能对比。
"""

import torch


def vector_add_pytorch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """PyTorch 原生向量加法（调用 cuBLAS/cuDNN 底层）"""
    return A + B


def vector_add_pytorch_inplace(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """in-place 版本，避免额外内存分配"""
    C = A.clone()
    C.add_(B)
    return C
