"""
Matmul - CuTe DSL 实现

数学定义：
    C[i, j] = sum_k A[i, k] * B[k, j]
    A: (M, K), B: (K, N) → C: (M, N)

V1: Naive — 1 thread per C element，串行 K 循环
    - grid = (ceil(N/BLOCK), ceil(M/BLOCK)), block = (BLOCK, BLOCK)
    - row = bidy*BLOCK+tidy, col = bidx*BLOCK+tidx
    - 直接串行 for k in range(K): acc += A[row,k]*B[k,col]

V2: Shared memory tiling — 分块加载 A/B 到 smem 再计算
    - TILE=32，block = (TILE, TILE) = 1024 线程
    - 每次迭代加载 [TILE, TILE] A-tile 和 B-tile 到 smem
    - 减少 HBM 访问次数，提升 L1/smem 复用率
    - 内层循环串行 for k in range(TILE): acc += sA[tidy,k]*sB[k,tidx]

性能关键：cute.compile() AOT 编译，按 (M, K, N) 键缓存。

注意：V1/V2 均用 FP32 串行 FMA，不使用 Tensor Core。
"""

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator
import torch

BLOCK = 32   # V1 naive block size
TILE  = 32   # V2 smem tile size


# -----------------------------------------------------------------------
# V1: Naive matmul — 1 thread per C element
# -----------------------------------------------------------------------
@cute.kernel
def matmul_v1_kernel(
    A: cute.Tensor,   # (M, K)
    B: cute.Tensor,   # (K, N)
    C: cute.Tensor,   # (M, N)
    M: cute.Int32,
    K: cute.Int32,
    N: cute.Int32,
):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    row = bidy * BLOCK + tidy
    col = bidx * BLOCK + tidx

    if row < M and col < N:
        acc = 0.0
        for k in range(K):
            acc = acc + A[row, k] * B[k, col]
        C[row, col] = acc


@cute.jit
def _launch_v1(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    M: cute.Int32,
    K: cute.Int32,
    N: cute.Int32,
):
    matmul_v1_kernel(A, B, C, M, K, N).launch(
        grid=((N + BLOCK - 1) // BLOCK,
              (M + BLOCK - 1) // BLOCK, 1),
        block=(BLOCK, BLOCK, 1),
    )


# -----------------------------------------------------------------------
# V2: Shared memory tiling matmul
# -----------------------------------------------------------------------
@cute.kernel
def matmul_v2_kernel(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    M: cute.Int32,
    K: cute.Int32,
    N: cute.Int32,
):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    smem = SmemAllocator()
    sA = smem.allocate_tensor(
        cute.Float32,
        cute.make_layout((TILE, TILE), stride=(TILE, 1)),
        16,
    )
    sB = smem.allocate_tensor(
        cute.Float32,
        cute.make_layout((TILE, TILE), stride=(TILE, 1)),
        16,
    )

    row = bidy * TILE + tidy   # C 行（对应 A 行）
    col = bidx * TILE + tidx   # C 列（对应 B 列）

    acc = 0.0

    # 沿 K 维分块
    for k_start in range(0, K, TILE):
        # 协作加载 A tile: A[row, k_start+tidx]
        if row < M and (k_start + tidx) < K:
            sA[tidy, tidx] = A[row, k_start + tidx]
        else:
            sA[tidy, tidx] = 0.0

        # 协作加载 B tile: B[k_start+tidy, col]
        if (k_start + tidy) < K and col < N:
            sB[tidy, tidx] = B[k_start + tidy, col]
        else:
            sB[tidy, tidx] = 0.0

        cute.arch.barrier()

        # 计算 A tile × B tile 的贡献
        for k in range(TILE):
            acc = acc + sA[tidy, k] * sB[k, tidx]

        cute.arch.barrier()

    if row < M and col < N:
        C[row, col] = acc


@cute.jit
def _launch_v2(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    M: cute.Int32,
    K: cute.Int32,
    N: cute.Int32,
):
    matmul_v2_kernel(A, B, C, M, K, N).launch(
        grid=((N + TILE - 1) // TILE,
              (M + TILE - 1) // TILE, 1),
        block=(TILE, TILE, 1),
    )


# -----------------------------------------------------------------------
# AOT 编译缓存（按 (M, K, N) 键）
# -----------------------------------------------------------------------
_compiled_v1: dict = {}
_compiled_v2: dict = {}


def _get_compiled(version: int, A, B, C):
    M, K = A.shape
    K2, N = B.shape
    key  = (M, K, N)
    if version == 1:
        if key not in _compiled_v1:
            _compiled_v1[key] = cute.compile(
                _launch_v1,
                from_dlpack(A), from_dlpack(B), from_dlpack(C),
                M, K, N,
            )
        return _compiled_v1[key]
    else:
        if key not in _compiled_v2:
            _compiled_v2[key] = cute.compile(
                _launch_v2,
                from_dlpack(A), from_dlpack(B), from_dlpack(C),
                M, K, N,
            )
        return _compiled_v2[key]


# -----------------------------------------------------------------------
# Python 包装
# -----------------------------------------------------------------------
def _run(version: int, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty(M, N, dtype=A.dtype, device=A.device)

    compiled = _get_compiled(version, A, B, C)
    compiled(from_dlpack(A), from_dlpack(B), from_dlpack(C), M, K, N)
    return C


def matmul_cutedsl_v1(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """CuTe DSL Matmul v1: naive 1-thread-per-element, serial K (AOT compiled)."""
    return _run(1, A, B)


def matmul_cutedsl_v2(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """CuTe DSL Matmul v2: shared memory tiling TILE=32 (AOT compiled)."""
    return _run(2, A, B)
