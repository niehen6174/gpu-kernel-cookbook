"""
Matrix Transpose - Triton 实现

Triton 通过 2D program grid 处理矩阵转置。
每个 program 负责一个 BLOCK×BLOCK 的 tile。

Triton 自动处理：
  - Shared memory 分配和 bank conflict 避免
  - Coalesced 内存访问模式
  - 向量化加载/存储

关键技巧：
  - 使用 2D offset 计算处理矩形 tile
  - mask 防止越界
  - 转置通过 load 行/store 列（或反过来）实现
"""

import triton
import triton.language as tl
import torch


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}),
    ],
    key=["M", "N"],
)
@triton.jit
def transpose_kernel(
    input_ptr, output_ptr,
    M, N,               # 输入形状：M 行 N 列
    stride_im, stride_in,   # 输入 strides（通常是 N, 1）
    stride_om, stride_on,   # 输出 strides（通常是 M, 1）
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    输入:  (M, N) 矩阵
    输出:  (N, M) 矩阵（转置）

    每个 program 处理输入的一个 BLOCK_M × BLOCK_N 子块。
    """
    pid_m = tl.program_id(0)  # 对应输入的行方向
    pid_n = tl.program_id(1)  # 对应输入的列方向

    # 当前 block 的行/列起始偏移
    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    # 行/列偏移向量
    row_offsets = row_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    col_offsets = col_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # 2D mask：防止越界
    mask = (row_offsets[:, None] < M) & (col_offsets[None, :] < N)

    # 从输入加载 BLOCK_M × BLOCK_N 的 tile
    # input_ptr + row*stride_im + col*stride_in
    input_ptrs = input_ptr + row_offsets[:, None] * stride_im + col_offsets[None, :] * stride_in
    tile = tl.load(input_ptrs, mask=mask, other=0.0)  # shape: [BLOCK_M, BLOCK_N]

    # 转置：写到 output，行列互换
    # 输出中：第 col 行，第 row 列 = tile[row][col]
    # output_ptr + col*stride_om + row*stride_on
    output_ptrs = output_ptr + col_offsets[:, None] * stride_om + row_offsets[None, :] * stride_on
    # 注意：tile 的维度是 [BLOCK_M, BLOCK_N]，转置后变为 [BLOCK_N, BLOCK_M]
    tl.store(output_ptrs, tl.trans(tile), mask=tl.trans(mask))


def transpose_triton(A: torch.Tensor) -> torch.Tensor:
    """
    Triton matrix transpose.
    A: (M, N) float32 CUDA tensor
    Returns: (N, M) float32 CUDA tensor
    """
    assert A.is_cuda and A.dim() == 2
    M, N = A.shape
    B = torch.empty(N, M, dtype=A.dtype, device=A.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    transpose_kernel[grid](
        A, B,
        M, N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
    )
    return B
