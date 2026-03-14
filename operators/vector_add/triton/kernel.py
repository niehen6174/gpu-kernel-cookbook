"""
Vector Add - Triton 实现

Triton 是一种 Python DSL，可以编写比原始 CUDA 更简洁的 GPU kernel。
每个 Triton "program" 对应 CUDA 中的一个 block。
tl.load / tl.store 是 Triton 的内存访问原语，自动处理向量化。

关键概念：
  - tl.program_id(axis=0): 当前 program 的索引（相当于 blockIdx.x）
  - tl.arange(0, BLOCK_SIZE): 生成 [0, 1, ..., BLOCK_SIZE-1] 的偏移量向量
  - mask: 防止越界访问
  - BLOCK_SIZE: constexpr，编译期常量，Triton 会自动选择最优值
"""

import triton
import triton.language as tl
import torch


@triton.jit
def vector_add_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,  # 每个 program 处理的元素数，编译期确定
):
    # 获取当前 program 的 ID（相当于 blockIdx.x）
    pid = tl.program_id(axis=0)

    # 计算当前 program 负责处理的元素范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # shape: [BLOCK_SIZE]

    # 边界 mask，防止越界
    mask = offsets < n_elements

    # 从 DRAM 加载数据（masked load：越界位置填 0）
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    # 向量加法（逐元素）
    c = a + b

    # 写回 DRAM
    tl.store(c_ptr + offsets, c, mask=mask)


def vector_add_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton vector add 的 Python 入口。
    A, B: 1D float32 GPU tensor
    """
    assert A.shape == B.shape, "A and B must have the same shape"
    assert A.is_cuda and B.is_cuda, "Inputs must be on GPU"

    C = torch.empty_like(A)
    N = A.numel()

    # BLOCK_SIZE = 1024 时，每个 program 处理 1024 个元素
    # Triton 会自动选择最优的 BLOCK_SIZE（如果使用 autotune）
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)  # ceil_div

    vector_add_kernel[grid](
        A, B, C,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return C


# -------------------------------------------------------------------------
# 带 autotune 的版本：Triton 自动搜索最优超参数
# -------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
    ],
    key=["n_elements"],  # 根据 n_elements 缓存最优配置
)
@triton.jit
def vector_add_kernel_autotuned(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(c_ptr + offsets, a + b, mask=mask)


def vector_add_triton_autotuned(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    C = torch.empty_like(A)
    N = A.numel()
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    vector_add_kernel_autotuned[grid](A, B, C, N)
    return C
