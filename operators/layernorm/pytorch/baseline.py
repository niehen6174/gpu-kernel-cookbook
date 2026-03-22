"""LayerNorm - PyTorch Baseline"""
import torch
import torch.nn.functional as F


def layernorm_pytorch(X: torch.Tensor, weight=None, bias=None, eps=1e-5) -> torch.Tensor:
    N = X.shape[-1]
    normalized_shape = (N,)
    return F.layer_norm(X, normalized_shape, weight, bias, eps)


def fused_add_layernorm_pytorch(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
):
    """
    Fused residual add + LayerNorm (inplace on residual).
    Returns (normed_output, updated_residual).
      updated_residual = x + residual
      normed_output    = LayerNorm(updated_residual, weight, bias)
    """
    residual = x + residual
    N = residual.shape[-1]
    out = F.layer_norm(residual, (N,), weight, bias, eps)
    return out, residual
