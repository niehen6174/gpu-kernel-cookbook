"""RMSNorm - PyTorch Baseline

数学定义:
    rms(x) = sqrt( mean(x²) + eps )
    y_i    = x_i / rms(x) * weight_i

与 LayerNorm 的区别:
    - 不减均值（不中心化）
    - 更简单，LLM 中常用（LLaMA, Mistral 等）
"""
import torch


def rms_norm_pytorch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    RMSNorm: y = x / rms(x) * weight
    x: (..., N) float tensor
    weight: (N,) scale parameter
    """
    rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    return (x / rms * weight).to(x.dtype)


def fused_add_rms_norm_pytorch(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
):
    """
    Fused residual add + RMSNorm (inplace on residual).
    Returns (normed_output, updated_residual).
    """
    x = x + residual
    rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    return (x / rms * weight).to(x.dtype), x
