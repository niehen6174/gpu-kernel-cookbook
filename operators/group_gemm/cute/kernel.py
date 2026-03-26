"""
Group GEMM - CuTe DSL 实现

计算 G 组独立的矩阵乘法（所有 group 的 M, K, N 相同）：
    C[g, i, j] = sum_k A[g, i, k] * B[g, k, j]

V1: Naive — 每 thread 计算一个 C 元素，3D 索引 A[g, row, k]
    - block = (BLOCK, BLOCK), grid = (ceil(N/BLOCK), ceil(M/BLOCK), G)
    - thread 通过 blockIdx.z 确定 group_id

V2: Shared memory tiling — 分块加载 A/B tile 到 smem
    - TILE=32, block = (TILE, TILE)
    - grid = (ceil(N/TILE), ceil(M/TILE), G)
    - 减少 HBM 访问，提升 L1/smem 复用率

性能关键：cute.compile() AOT 编译，按 (G, M, K, N) 键缓存。
"""

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator
import torch

BLOCK = 32
TILE  = 32


# -----------------------------------------------------------------------
# V1: Naive group GEMM
# -----------------------------------------------------------------------
@cute.kernel
def group_gemm_v1_kernel(
    A: cute.Tensor,    # (G, M, K)
    B: cute.Tensor,    # (G, K, N)
    C: cute.Tensor,    # (G, M, N)
    G: cute.Int32,
    M: cute.Int32,
    K: cute.Int32,
    N: cute.Int32,
):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()

    group_id = bidz
    row = bidy * BLOCK + tidy
    col = bidx * BLOCK + tidx

    if row < M and col < N:
        acc = 0.0
        for k in range(K):
            acc = acc + A[group_id, row, k] * B[group_id, k, col]
        C[group_id, row, col] = acc


@cute.jit
def _launch_v1(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    G: cute.Int32,
    M: cute.Int32,
    K: cute.Int32,
    N: cute.Int32,
):
    group_gemm_v1_kernel(A, B, C, G, M, K, N).launch(
        grid=((N + BLOCK - 1) // BLOCK,
              (M + BLOCK - 1) // BLOCK,
              G),
        block=(BLOCK, BLOCK, 1),
    )


# -----------------------------------------------------------------------
# V2: Shared memory tiling group GEMM
# -----------------------------------------------------------------------
@cute.kernel
def group_gemm_v2_kernel(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    G: cute.Int32,
    M: cute.Int32,
    K: cute.Int32,
    N: cute.Int32,
):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()

    group_id = bidz

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

    row = bidy * TILE + tidy
    col = bidx * TILE + tidx

    acc = 0.0

    for k_start in range(0, K, TILE):
        # 协作加载 A[group_id, row, k_start+tidx]
        if row < M and (k_start + tidx) < K:
            sA[tidy, tidx] = A[group_id, row, k_start + tidx]
        else:
            sA[tidy, tidx] = 0.0

        # 协作加载 B[group_id, k_start+tidy, col]
        if (k_start + tidy) < K and col < N:
            sB[tidy, tidx] = B[group_id, k_start + tidy, col]
        else:
            sB[tidy, tidx] = 0.0

        cute.arch.barrier()

        for k in range(TILE):
            acc = acc + sA[tidy, k] * sB[k, tidx]

        cute.arch.barrier()

    if row < M and col < N:
        C[group_id, row, col] = acc


@cute.jit
def _launch_v2(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    G: cute.Int32,
    M: cute.Int32,
    K: cute.Int32,
    N: cute.Int32,
):
    group_gemm_v2_kernel(A, B, C, G, M, K, N).launch(
        grid=((N + TILE - 1) // TILE,
              (M + TILE - 1) // TILE,
              G),
        block=(TILE, TILE, 1),
    )


# -----------------------------------------------------------------------
# AOT 编译缓存（按 (G, M, K, N) 键）
# -----------------------------------------------------------------------
_compiled_v1: dict = {}
_compiled_v2: dict = {}


def _get_compiled(version: int, A, B, C):
    G, M, K = A.shape
    K2, N   = B.shape[1], B.shape[2]
    key     = (G, M, K, N)
    if version == 1:
        if key not in _compiled_v1:
            _compiled_v1[key] = cute.compile(
                _launch_v1,
                from_dlpack(A), from_dlpack(B), from_dlpack(C),
                G, M, K, N,
            )
        return _compiled_v1[key]
    else:
        if key not in _compiled_v2:
            _compiled_v2[key] = cute.compile(
                _launch_v2,
                from_dlpack(A), from_dlpack(B), from_dlpack(C),
                G, M, K, N,
            )
        return _compiled_v2[key]


# -----------------------------------------------------------------------
# Python 包装
# -----------------------------------------------------------------------
def _run(version: int, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.dim() == 3 and B.dim() == 3
    G, M, K  = A.shape
    G2, K2, N = B.shape
    assert G == G2 and K == K2
    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty(G, M, N, dtype=A.dtype, device=A.device)

    compiled = _get_compiled(version, A, B, C)
    compiled(from_dlpack(A), from_dlpack(B), from_dlpack(C), G, M, K, N)
    return C


def group_gemm_cutedsl_v1(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """CuTe DSL Group GEMM v1: naive 3D-indexed, serial K (AOT compiled)."""
    return _run(1, A, B)


def group_gemm_cutedsl_v2(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """CuTe DSL Group GEMM v2: shared memory tiling TILE=32 (AOT compiled)."""
    return _run(2, A, B)
