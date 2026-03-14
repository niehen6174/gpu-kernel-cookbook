"""
LayerNorm - Triton 实现

Triton 的 LayerNorm 使用 online Welford 算法，一趟扫描计算 (mean, var)。

参考：Triton 官方教程中的 LayerNorm 实现
"""

import triton
import triton.language as tl
import torch


@triton.jit
def layernorm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N,
    stride,       # 每行的步长
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    每个 program 处理一行。
    X_ptr: (total_rows, N) 的起始地址
    """
    row = tl.program_id(0)
    X_row = X_ptr + row * stride
    Y_row = Y_ptr + row * stride

    # 加载整行
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)

    # 计算均值
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N

    # 计算方差（用 E[x^2] - E[x]^2 可能有精度问题，这里用 (x-mean)^2）
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N

    # 归一化
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = diff * inv_std  # (x - mean) / std

    # 仿射变换
    if W_ptr is not None:
        w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        x_norm = x_norm * w
    if B_ptr is not None:
        b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_norm = x_norm + b

    # 写回
    tl.store(Y_row + cols, x_norm, mask=mask)


def layernorm_triton(
    X: torch.Tensor,
    weight: torch.Tensor = None,
    bias: torch.Tensor = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Triton LayerNorm，对最后一维做归一化。
    X: (..., N) float32 CUDA tensor
    """
    assert X.is_cuda
    orig_shape = X.shape
    N = X.shape[-1]
    X_2d = X.reshape(-1, N)
    B = X_2d.shape[0]
    Y = torch.empty_like(X_2d)

    BLOCK_SIZE = triton.next_power_of_2(N)
    BLOCK_SIZE = min(BLOCK_SIZE, 65536)

    layernorm_kernel[(B,)](
        X_2d, weight, bias, Y,
        N,
        X_2d.stride(0),
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return Y.reshape(orig_shape)
