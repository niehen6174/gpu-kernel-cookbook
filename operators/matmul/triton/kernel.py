"""
Matmul - Triton 实现

Triton 的矩阵乘法是其 "Hello World"，也是 Triton 教程的核心示例。

实现基于官方 Triton matmul tutorial，加入详细注释。

核心技术：
1. 2D program grid：pid_m × pid_n
2. Tiled 计算：BLOCK_M × BLOCK_N 的输出 tile
3. K 维度分块：每次迭代 BLOCK_K
4. tl.dot()：调用 GPU 的 tensor core（如果支持）

性能优化技巧：
- GROUP_SIZE_M: 分组 program，提升 L2 cache 命中率（super grouping）
- num_stages: pipeline 深度（prefetch stages）
- num_warps: 每个 block 的 warp 数
"""

import triton
import triton.language as tl
import torch


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_SIZE_M": 8},
            num_stages=5, num_warps=2
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,   # A 的 strides
    stride_bk, stride_bn,   # B 的 strides
    stride_cm, stride_cn,   # C 的 strides
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    计算 C = A @ B
    A: (M, K), B: (K, N), C: (M, N)

    每个 program 计算 C 的一个 BLOCK_M × BLOCK_N 子块。

    GROUP_SIZE_M 优化（L2 cache 利用率）：
      传统的 2D grid 中，相邻的 program (pid_m, pid_n) 和 (pid_m+1, pid_n)
      共享 B 的同一列 tile，但 A 的不同行 tile。
      如果按行优先 launch program，相邻 program 复用 A 的同一行 tile（L2 友好）。

      GROUP_SIZE_M 将 pid_m 方向分组，每组内的 program 先完成再换组，
      这样 B 的列 tile 有更高的 L2 重用率。

      具体映射：
        group_id = pid // (num_pid_n * GROUP_SIZE_M)
        within_group = pid % (num_pid_n * GROUP_SIZE_M)
        pid_m = group_id * GROUP_SIZE_M + within_group // num_pid_n
        pid_n = within_group % num_pid_n
    """
    # 计算当前 program 在 2D grid 中的位置
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # L2 cache 优化的 program 排布（swizzle）
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 初始化当前 block 要写的 C tile 的行列起始偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                    # [BLOCK_K]

    # A 和 B 的指针（2D 地址计算）
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # 累加器：存储当前 tile 的部分和
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 迭代 K 维度的所有 tile
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # 加载 A tile: [BLOCK_M, BLOCK_K]
        # mask: 防止 K 方向越界
        k_remaining = K - k * BLOCK_K
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, BLOCK_K]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # tl.dot: 调用 tensor core 或 FMA 指令进行矩阵乘
        acc += tl.dot(a, b)

        # 移动指针到下一个 K tile
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # 写回 C tile
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton 矩阵乘法。
    A: (M, K), B: (K, N) → C: (M, N)
    """
    assert A.is_cuda and B.is_cuda
    assert A.shape[1] == B.shape[0], f"Shape mismatch: A{A.shape}, B{B.shape}"
    M, K = A.shape
    K2, N = B.shape
    C = torch.empty(M, N, dtype=A.dtype, device=A.device)

    # 1D launch grid，每个 program 负责一个输出 tile
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C
