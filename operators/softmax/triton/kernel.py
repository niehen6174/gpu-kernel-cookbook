"""
Softmax - Triton 实现

Triton 实现在线 softmax（online softmax），每个 program 处理一行。

Online Softmax 核心：
  在一趟循环中同时维护 (max, sum) 状态：
    当遇到更大的值 x_new > max_old 时：
      sum_new = sum_old * exp(max_old - x_new) + exp(0)
      max_new = x_new
    否则：
      sum_new = sum_old + exp(x_new - max_old)
      max_new = max_old

Triton 特点：
  - tl.reduce: 内置规约操作（max, sum），自动优化为 warp/block 级规约
  - 每个 program 处理完整一行（BLOCK_SIZE = row_len）
"""

import triton
import triton.language as tl
import torch


@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    B, N,               # B 行, N 列
    stride_b, stride_n, # 输入 strides
    BLOCK_SIZE: tl.constexpr,  # 必须 >= N，且是 2 的幂次
):
    """
    每个 program 处理一行（row = program_id）。
    BLOCK_SIZE 是处理一行所需的 tile 大小（需要 >= N）。
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # 加载整行数据（超出 N 的位置用 -inf 填充，不影响 max）
    row_start = input_ptr + row_idx * stride_b
    x = tl.load(row_start + col_offsets * stride_n, mask=mask, other=-float("inf"))

    # 数值稳定的 softmax：
    # Step 1: 找最大值
    x_max = tl.max(x, axis=0)  # scalar

    # Step 2: 计算 exp(x - max) 和
    x_shifted = x - x_max
    exp_x = tl.exp(x_shifted)
    exp_sum = tl.sum(tl.where(mask, exp_x, 0.0), axis=0)  # scalar

    # Step 3: 归一化
    y = exp_x / exp_sum

    # 写回（越界位置不写）
    row_out = output_ptr + row_idx * stride_b
    tl.store(row_out + col_offsets * stride_n, y, mask=mask)


def softmax_triton(X: torch.Tensor) -> torch.Tensor:
    """
    Triton softmax，对最后一维做 softmax。
    X: (B, N) float32 CUDA tensor
    """
    assert X.is_cuda and X.dim() == 2
    B, N = X.shape
    Y = torch.empty_like(X)

    # BLOCK_SIZE 必须是 2 的幂次，且 >= N
    BLOCK_SIZE = triton.next_power_of_2(N)
    # 如果 N 太大（如 >65536），可以分段处理，这里简化处理
    BLOCK_SIZE = min(BLOCK_SIZE, 65536)

    softmax_kernel[(B,)](
        X, Y,
        B, N,
        X.stride(0), X.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return Y


# -------------------------------------------------------------------------
# 更高效的版本：分块处理（适合 N 很大时）
# -------------------------------------------------------------------------
@triton.jit
def softmax_kernel_large_n(
    input_ptr, output_ptr,
    B, N,
    stride_b, stride_n,
    BLOCK_SIZE: tl.constexpr,
):
    """
    对 N 很大的情况，分多个 BLOCK 做扫描，然后汇总。
    每个 program 处理一行，但一行分多个 chunk 处理。

    这里使用 online softmax 算法：
      维护 (running_max, running_sum) 状态，一边扫描一边更新
    """
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * stride_b

    # Pass 1: 扫描整行，计算全局 (max, sum)
    running_max = -float("inf")
    running_sum = 0.0

    for col_start in range(0, N, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(row_start + offsets, mask=mask, other=-float("inf"))

        chunk_max = tl.max(x, axis=0)
        # online softmax update
        if chunk_max > running_max:
            # 校正 running_sum
            running_sum = running_sum * tl.exp(running_max - chunk_max) + tl.sum(
                tl.where(mask, tl.exp(x - chunk_max), 0.0), axis=0
            )
            running_max = chunk_max
        else:
            running_sum += tl.sum(tl.where(mask, tl.exp(x - running_max), 0.0), axis=0)

    # Pass 2: 写出
    row_out = output_ptr + row_idx * stride_b
    for col_start in range(0, N, BLOCK_SIZE):
        offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(row_start + offsets, mask=mask, other=0.0)
        y = tl.exp(x - running_max) / running_sum
        tl.store(row_out + offsets, y, mask=mask)


def softmax_triton_large(X: torch.Tensor) -> torch.Tensor:
    """适合 N 很大的 softmax"""
    assert X.is_cuda and X.dim() == 2
    B, N = X.shape
    Y = torch.empty_like(X)
    BLOCK_SIZE = 2048
    softmax_kernel_large_n[(B,)](
        X, Y, B, N, X.stride(0), X.stride(1), BLOCK_SIZE=BLOCK_SIZE
    )
    return Y
