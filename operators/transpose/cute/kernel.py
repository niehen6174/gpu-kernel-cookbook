"""
Matrix Transpose - CuTe DSL 实现

数学定义：
    B[j, i] = A[i, j]    (A: M×N → B: N×M)

V1: Naive — 1 thread per element (2D block 32×32)
    - grid = (ceil(N/32), ceil(M/32)), block = (32, 32)
    - tidx = 列偏移, tidy = 行偏移
    - 直接 B[col, row] = A[row, col]（全局内存非合并写，教学用）

V2: Shared memory + bank-conflict 避免
    - TILE_DIM=32, BLOCK_ROWS=8 → block = (32, 8)，只 256 线程
    - smem layout stride=(1, TILE_DIM+1) 错开一行以避免 bank conflict
    - 先 coalesced 读 A tile 到 smem，再 coalesced 写 B tile
    - 每线程循环 TILE_DIM/BLOCK_ROWS = 4 次，覆盖整个 tile

性能关键：cute.compile() AOT 编译，按 (M, N) 键缓存。
"""

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator
import torch

TILE_DIM   = 32
BLOCK_ROWS = 8    # V2 block 的线程行数


# -----------------------------------------------------------------------
# V1: Naive transpose — 1 thread per element
# -----------------------------------------------------------------------
@cute.kernel
def transpose_v1_kernel(
    x: cute.Tensor,   # (M, N)
    y: cute.Tensor,   # (N, M)
):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    row = bidy * TILE_DIM + tidy
    col = bidx * TILE_DIM + tidx
    M   = x.shape[0]
    N   = x.shape[1]
    if row < M and col < N:
        y[col, row] = x[row, col]


@cute.jit
def _launch_v1(
    x: cute.Tensor,
    y: cute.Tensor,
    M: cute.Int32,
    N: cute.Int32,
):
    transpose_v1_kernel(x, y).launch(
        grid=((N + TILE_DIM - 1) // TILE_DIM,
              (M + TILE_DIM - 1) // TILE_DIM, 1),
        block=(TILE_DIM, TILE_DIM, 1),
    )


# -----------------------------------------------------------------------
# V2: Shared memory transpose（bank-conflict free）
# -----------------------------------------------------------------------
@cute.kernel
def transpose_v2_kernel(
    x: cute.Tensor,
    y: cute.Tensor,
):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    # smem tile: (TILE_DIM, TILE_DIM)，stride=(1, TILE_DIM+1) 消除 bank conflict
    smem = SmemAllocator()
    tile = smem.allocate_tensor(
        cute.Float32,
        cute.make_layout((TILE_DIM, TILE_DIM), stride=(1, TILE_DIM + 1)),
        16,
    )

    M = x.shape[0]
    N = x.shape[1]

    # ---- 读 A tile → smem（coalesced 读）----
    # 每线程处理 TILE_DIM/BLOCK_ROWS 行
    x_col = bidx * TILE_DIM + tidx
    x_row = bidy * TILE_DIM + tidy
    for i in range(0, TILE_DIM, BLOCK_ROWS):
        r = x_row + i
        if r < M and x_col < N:
            tile[tidy + i, tidx] = x[r, x_col]
    cute.arch.barrier()

    # ---- 写 B tile（coalesced 写）----
    # 转置坐标：B 的 (col_out, row_out) = smem 的 (tidx, tidy+i)
    y_col = bidy * TILE_DIM + tidx   # B 的列对应 A 的行块
    y_row = bidx * TILE_DIM + tidy   # B 的行对应 A 的列块
    for i in range(0, TILE_DIM, BLOCK_ROWS):
        r = y_row + i
        if r < N and y_col < M:
            y[r, y_col] = tile[tidx, tidy + i]


@cute.jit
def _launch_v2(
    x: cute.Tensor,
    y: cute.Tensor,
    M: cute.Int32,
    N: cute.Int32,
):
    transpose_v2_kernel(x, y).launch(
        grid=((N + TILE_DIM - 1) // TILE_DIM,
              (M + TILE_DIM - 1) // TILE_DIM, 1),
        block=(TILE_DIM, BLOCK_ROWS, 1),
    )


# -----------------------------------------------------------------------
# AOT 编译缓存（按 (M, N) 键）
# -----------------------------------------------------------------------
_compiled_v1: dict = {}
_compiled_v2: dict = {}


def _get_compiled(version: int, x, y):
    M, N = x.shape
    key  = (M, N)
    if version == 1:
        if key not in _compiled_v1:
            _compiled_v1[key] = cute.compile(
                _launch_v1,
                from_dlpack(x), from_dlpack(y), M, N,
            )
        return _compiled_v1[key]
    else:
        if key not in _compiled_v2:
            _compiled_v2[key] = cute.compile(
                _launch_v2,
                from_dlpack(x), from_dlpack(y), M, N,
            )
        return _compiled_v2[key]


# -----------------------------------------------------------------------
# Python 包装
# -----------------------------------------------------------------------
def _run(version: int, x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2
    M, N = x.shape
    x = x.contiguous()
    y = torch.empty(N, M, dtype=x.dtype, device=x.device)

    compiled = _get_compiled(version, x, y)
    compiled(from_dlpack(x), from_dlpack(y), M, N)
    return y


def transpose_cutedsl_v1(x: torch.Tensor) -> torch.Tensor:
    """CuTe DSL Transpose v1: naive 1-thread-per-element (AOT compiled)."""
    return _run(1, x)


def transpose_cutedsl_v2(x: torch.Tensor) -> torch.Tensor:
    """CuTe DSL Transpose v2: smem tiling + bank-conflict-free (AOT compiled)."""
    return _run(2, x)
