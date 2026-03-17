"""
RMSNorm - Triton 实现

V1: rms_norm_triton
    每个 program 处理一行，一趟计算 sum(x²)，rsqrt，应用 weight。

V2: fused_add_rms_norm_triton
    先将 residual 加到 x（inplace 更新 residual），再做 RMSNorm。
"""

import triton
import triton.language as tl
import torch


# -------------------------------------------------------------------------
# V1: RMSNorm
# -------------------------------------------------------------------------
@triton.jit
def rms_norm_kernel(
    X_ptr, W_ptr, Y_ptr,
    N,
    stride,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """每个 program 处理一行."""
    row = tl.program_id(0)
    X_row = X_ptr + row * stride
    Y_row = Y_ptr + row * stride

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # 加载行数据（FP32）
    x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)

    # 计算 sum(x²) / N + eps，再 rsqrt
    ss = tl.sum(x * x, axis=0)
    rms_inv = tl.rsqrt(ss / N + eps)

    # 加载 weight 并写出
    w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    y = x * rms_inv * w
    tl.store(Y_row + cols, y, mask=mask)


def rms_norm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Triton RMSNorm，对最后一维做归一化。
    x: (..., N) float32 CUDA tensor
    weight: (N,) float32
    """
    assert x.is_cuda
    orig_shape = x.shape
    N = x.shape[-1]
    x_2d = x.reshape(-1, N)
    B = x_2d.shape[0]
    y = torch.empty_like(x_2d)

    BLOCK_SIZE = triton.next_power_of_2(N)
    BLOCK_SIZE = min(BLOCK_SIZE, 65536)

    rms_norm_kernel[(B,)](
        x_2d, weight, y,
        N,
        x_2d.stride(0),
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y.reshape(orig_shape)


# -------------------------------------------------------------------------
# V2: Fused Add + RMSNorm
# -------------------------------------------------------------------------
@triton.jit
def fused_add_rms_norm_kernel(
    X_ptr, R_ptr, W_ptr, Y_ptr,
    N,
    stride,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    每个 program 处理一行。
    X_ptr: input x
    R_ptr: residual (inplace updated: residual = x + residual)
    W_ptr: weight
    Y_ptr: normed output
    """
    row = tl.program_id(0)
    X_row = X_ptr + row * stride
    R_row = R_ptr + row * stride
    Y_row = Y_ptr + row * stride

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R_row + cols, mask=mask, other=0.0).to(tl.float32)

    # fused residual add
    x = x + r

    # update residual inplace
    tl.store(R_row + cols, x, mask=mask)

    # RMSNorm
    ss = tl.sum(x * x, axis=0)
    rms_inv = tl.rsqrt(ss / N + eps)

    w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    y = x * rms_inv * w
    tl.store(Y_row + cols, y, mask=mask)


def fused_add_rms_norm_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple:
    """
    Fused residual add + RMSNorm.
    residual is updated inplace to x + residual.
    Returns (normed_output, updated_residual).
    """
    assert x.is_cuda
    orig_shape = x.shape
    N = x.shape[-1]
    x_2d = x.reshape(-1, N)
    r_2d = residual.reshape(-1, N).clone()  # inplace update on clone
    B = x_2d.shape[0]
    y = torch.empty_like(x_2d)

    BLOCK_SIZE = triton.next_power_of_2(N)
    BLOCK_SIZE = min(BLOCK_SIZE, 65536)

    fused_add_rms_norm_kernel[(B,)](
        x_2d, r_2d, weight, y,
        N,
        x_2d.stride(0),
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y.reshape(orig_shape), r_2d.reshape(orig_shape)
