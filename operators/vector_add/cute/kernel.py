"""
Vector Add - CuTe DSL 实现

CuTe DSL 是 NVIDIA CUTLASS 库提供的 Python DSL，
底层使用 CuTe（CUDA Templates for Efficient and Tunable Primitives）。
它通过 Layout 抽象来描述张量的内存布局，
使得 kernel 代码更接近数学语义。

关键概念：
  - cute.Tensor: 带 layout 信息的张量
  - cute.kernel: 装饰器，将 Python 函数编译为 GPU kernel
  - cute.jit: 装饰器，将 Python 函数编译为 JIT 代码（host 侧调度）
  - cute.arch.thread_idx() / block_idx() / block_dim(): 线程索引
"""

import cutlass.cute as cute
import torch


@cute.kernel
def vector_add_kernel(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    N: cute.Uint32,
):
    """CuTe kernel：每个 thread 处理一个元素"""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx
    if thread_idx < N:
        C[thread_idx] = A[thread_idx] + B[thread_idx]


@cute.jit
def vector_add_cute(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    N: cute.Uint32,
):
    """CuTe JIT host 函数：配置 grid/block 并启动 kernel"""
    num_threads_per_block = 256
    grid_dim = cute.ceil_div(cute.Int32(N), num_threads_per_block), 1, 1
    vector_add_kernel(A, B, C, N).launch(
        grid=grid_dim,
        block=(num_threads_per_block, 1, 1),
    )


def run_vector_add_cute(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """包装函数，接受 PyTorch tensor，转换为 CuTe tensor"""
    C = torch.empty_like(A)
    N = A.numel()
    # 将 PyTorch tensor 转换为 CuTe tensor（共享底层内存）
    A_cute = cute.from_dlpack(A)
    B_cute = cute.from_dlpack(B)
    C_cute = cute.from_dlpack(C)
    vector_add_cute(A_cute, B_cute, C_cute, N)
    return C
