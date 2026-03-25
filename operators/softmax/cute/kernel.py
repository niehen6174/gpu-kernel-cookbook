"""
Softmax - CuTe DSL 实现

数学定义：
    y_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))

V1: Two-pass + shared memory tree reduce (grid=B, block=256)
    - Pass 1a: 各线程累积局部 max(x) → smem tree reduce → block max
    - Pass 1b: 各线程累积 sum(exp(x - max)) → smem tree reduce → block sum
    - Pass 2: y[i] = exp(x[i] - max) / sum

V2: Two-pass + warp shuffle reduce（减少 __syncthreads 次数）
    - 与 V1 相同算法，用 warp shuffle 代替 smem tree reduce
    - Warp 0 做第二级 reduce

性能关键：cute.compile() AOT 编译，按 (B, N) 键缓存。

关键 CuTe DSL 用法：
    - cute.arch.warp_reduction_max / warp_reduction_sum
    - cute.arch.shuffle_sync_bfly()
    - cute.arch.fmax()
    - cute.exp()
    - SmemAllocator: 分配 shared memory tensor
"""

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator
import torch

THREADS    = 256
NUM_WARPS  = THREADS // 32


# -----------------------------------------------------------------------
# V1: Two-pass + shared memory tree reduce
# -----------------------------------------------------------------------
@cute.kernel
def softmax_v1_kernel(
    x: cute.Tensor,   # (B, N)
    y: cute.Tensor,   # (B, N)
    N: cute.Int32,
):
    tidx, _, _  = cute.arch.thread_idx()
    bidx, _, _  = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()

    smem = SmemAllocator()
    s_max = smem.allocate_tensor(cute.Float32, cute.make_layout(THREADS), 16)
    s_sum = smem.allocate_tensor(cute.Float32, cute.make_layout(THREADS), 16)

    # Pass 1a: 每线程求局部 max
    local_max = -1.0e9
    for i in range(tidx, N, bdimx):
        local_max = cute.arch.fmax(local_max, x[bidx, i])
    s_max[tidx] = local_max
    cute.arch.barrier()

    # Smem tree reduce → block max
    stride = bdimx // 2
    while stride > 0:
        if tidx < stride:
            s_max[tidx] = cute.arch.fmax(s_max[tidx], s_max[tidx + stride])
        cute.arch.barrier()
        stride = stride // 2

    row_max = s_max[0]
    cute.arch.barrier()

    # Pass 1b: 每线程累积 sum(exp(x - max))
    local_sum = 0.0
    for i in range(tidx, N, bdimx):
        local_sum = local_sum + cute.exp(x[bidx, i] - row_max)
    s_sum[tidx] = local_sum
    cute.arch.barrier()

    # Smem tree reduce → block sum
    stride = bdimx // 2
    while stride > 0:
        if tidx < stride:
            s_sum[tidx] = s_sum[tidx] + s_sum[tidx + stride]
        cute.arch.barrier()
        stride = stride // 2

    row_sum = s_sum[0]
    cute.arch.barrier()

    # Pass 2: normalize
    for i in range(tidx, N, bdimx):
        y[bidx, i] = cute.exp(x[bidx, i] - row_max) / row_sum


@cute.jit
def _launch_v1(
    x: cute.Tensor,
    y: cute.Tensor,
    B: cute.Int32,
    N: cute.Int32,
):
    softmax_v1_kernel(x, y, N).launch(
        grid=(B, 1, 1),
        block=(THREADS, 1, 1),
    )


# -----------------------------------------------------------------------
# V2: Two-pass + warp shuffle reduce
# -----------------------------------------------------------------------
@cute.kernel
def softmax_v2_kernel(
    x: cute.Tensor,
    y: cute.Tensor,
    N: cute.Int32,
):
    tidx, _, _  = cute.arch.thread_idx()
    bidx, _, _  = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()
    lane    = tidx % 32
    warp_id = tidx // 32

    smem = SmemAllocator()
    s_max = smem.allocate_tensor(cute.Float32, cute.make_layout(NUM_WARPS), 16)
    s_sum = smem.allocate_tensor(cute.Float32, cute.make_layout(NUM_WARPS), 16)

    # Pass 1a: warp-level max reduce
    local_max = -1.0e9
    for i in range(tidx, N, bdimx):
        local_max = cute.arch.fmax(local_max, x[bidx, i])

    for mask in [16, 8, 4, 2, 1]:
        local_max = cute.arch.fmax(
            local_max,
            cute.arch.shuffle_sync_bfly(local_max, offset=mask),
        )

    if lane == 0:
        s_max[warp_id] = local_max
    cute.arch.barrier()

    # Warp 0 做第二级 reduce
    if warp_id == 0:
        val = s_max[lane] if lane < NUM_WARPS else -1.0e9
        for mask in [16, 8, 4, 2, 1]:
            val = cute.arch.fmax(val, cute.arch.shuffle_sync_bfly(val, offset=mask))
        if lane == 0:
            s_max[0] = val
    cute.arch.barrier()

    row_max = s_max[0]

    # Pass 1b: warp-level sum(exp) reduce
    local_sum = 0.0
    for i in range(tidx, N, bdimx):
        local_sum = local_sum + cute.exp(x[bidx, i] - row_max)

    for mask in [16, 8, 4, 2, 1]:
        local_sum = local_sum + cute.arch.shuffle_sync_bfly(local_sum, offset=mask)

    if lane == 0:
        s_sum[warp_id] = local_sum
    cute.arch.barrier()

    if warp_id == 0:
        val = s_sum[lane] if lane < NUM_WARPS else 0.0
        for mask in [16, 8, 4, 2, 1]:
            val = val + cute.arch.shuffle_sync_bfly(val, offset=mask)
        if lane == 0:
            s_sum[0] = val
    cute.arch.barrier()

    row_sum = s_sum[0]

    # Pass 2: normalize
    for i in range(tidx, N, bdimx):
        y[bidx, i] = cute.exp(x[bidx, i] - row_max) / row_sum


@cute.jit
def _launch_v2(
    x: cute.Tensor,
    y: cute.Tensor,
    B: cute.Int32,
    N: cute.Int32,
):
    softmax_v2_kernel(x, y, N).launch(
        grid=(B, 1, 1),
        block=(THREADS, 1, 1),
    )


# -----------------------------------------------------------------------
# AOT 编译缓存（按 (B, N) 键）
# -----------------------------------------------------------------------
_compiled_v1: dict = {}
_compiled_v2: dict = {}


def _get_compiled(version: int, x2, y):
    B, N = x2.shape
    key  = (B, N)
    if version == 1:
        if key not in _compiled_v1:
            _compiled_v1[key] = cute.compile(
                _launch_v1,
                from_dlpack(x2), from_dlpack(y),
                B, N,
            )
        return _compiled_v1[key]
    else:
        if key not in _compiled_v2:
            _compiled_v2[key] = cute.compile(
                _launch_v2,
                from_dlpack(x2), from_dlpack(y),
                B, N,
            )
        return _compiled_v2[key]


# -----------------------------------------------------------------------
# Python 包装
# -----------------------------------------------------------------------
def _run(version: int, x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    N  = x.shape[-1]
    x2 = x.reshape(-1, N).contiguous()
    B  = x2.shape[0]
    y  = torch.empty_like(x2)

    compiled = _get_compiled(version, x2, y)
    compiled(from_dlpack(x2), from_dlpack(y), B, N)
    return y.reshape(orig_shape)


def softmax_cutedsl_v1(x: torch.Tensor) -> torch.Tensor:
    """CuTe DSL Softmax v1: two-pass + smem tree reduce (AOT compiled)."""
    return _run(1, x)


def softmax_cutedsl_v2(x: torch.Tensor) -> torch.Tensor:
    """CuTe DSL Softmax v2: two-pass + warp shuffle reduce (AOT compiled)."""
    return _run(2, x)
