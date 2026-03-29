"""
SVDQuant baseline: 纯 FP16/BF16 GEMM（性能上限参考）

用途：
- 提供理想情况下的计算上限（cuBLAS FP16 matmul）
- 不做任何量化，直接用原始权重计算
"""

import torch


def matmul_fp16_baseline(
    x: torch.Tensor,       # (M, K) FP16/BF16
    W: torch.Tensor,       # (K, N) FP16/BF16（原始权重，转置前）
    bias: torch.Tensor | None = None,  # (N,)
) -> torch.Tensor:
    """
    纯 FP16 矩阵乘法基准，调用 cuBLAS（torch.matmul）。

    Returns:
        y: (M, N) FP16/BF16
    """
    y = torch.matmul(x, W)    # (M, K) x (K, N) -> (M, N)
    if bias is not None:
        y = y + bias
    return y


def svdquant_fp16_baseline(
    x: torch.Tensor,           # (M, K) FP16
    W: torch.Tensor,           # (K, N) FP16（原始全精度权重）
    lora_down: torch.Tensor,   # (K, rank) FP16
    lora_up: torch.Tensor,     # (rank, N) FP16
    smooth: torch.Tensor,      # (K,) FP16
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    FP16 baseline：执行 SVDQuant 等价计算，但不量化（用 FP16 权重）。

    这代表"量化误差为0"的理想情况，是数值正确性的上限参考。

    y = (x / smooth) @ W_smooth + (x @ lora_down) @ lora_up + bias
    其中 W_smooth = smooth * W（在权重预处理时完成）

    实际上，baseline 接受已经 smooth 处理好的权重 W_smooth：
    y = (x / smooth) @ W_smooth + (x @ lora_down) @ lora_up + bias

    Args:
        x:         (M, K) 输入激活
        W:         (K, N) smooth 后的残差权重（W_smooth - lora_down @ lora_up）
        lora_down: (K, rank) LoRA 下投影
        lora_up:   (rank, N) LoRA 上投影
        smooth:    (K,) 平滑因子
        bias:      (N,) 偏置（可选）

    Returns:
        y: (M, N)
    """
    # Step 1: 平滑激活
    x_smooth = x / smooth.unsqueeze(0)        # (M, K)

    # Step 2: 主 GEMM（残差部分）
    y_main = torch.matmul(x_smooth, W)         # (M, N)

    # Step 3: LoRA 部分（使用原始 x）
    lora_act = torch.matmul(x, lora_down)      # (M, rank)
    y_lora = torch.matmul(lora_act, lora_up)   # (M, N)

    # Step 4: 合并
    y = y_main + y_lora
    if bias is not None:
        y = y + bias
    return y
