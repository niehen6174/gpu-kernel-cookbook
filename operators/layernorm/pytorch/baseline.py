"""LayerNorm - PyTorch Baseline"""
import torch
import torch.nn.functional as F


def layernorm_pytorch(X: torch.Tensor, weight=None, bias=None, eps=1e-5) -> torch.Tensor:
    N = X.shape[-1]
    normalized_shape = (N,)
    return F.layer_norm(X, normalized_shape, weight, bias, eps)
