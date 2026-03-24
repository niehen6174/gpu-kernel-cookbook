"""
RMSNorm - CuTe DSL 实现

数学定义：
    rms(x) = sqrt( sum(x_i²) / N + eps )
    y_i    = x_i / rms(x) * weight_i

V1: Two-pass + shared memory tree reduce (grid=B, block=256)
    - blockIdx.x = 行号，避免 host-side for loop
    - Pass 1: 并行累积 sum(x²) → smem reduce → rms_inv
    - Pass 2: y[row, i] = x[row, i] * rms_inv * w[i]

V2: Two-pass + warp shuffle reduce（减少 __syncthreads 次数）

性能关键：使用 cute.compile() 提前将 @cute.jit 函数 AOT 编译，
          消除每次调用的 Python overhead（约 27ms → 0.19ms）。

关键 CuTe DSL 用法：
    - cute.Tensor: 带 layout 的 GPU 内存视图（from_dlpack 从 torch 转换）
    - SmemAllocator: 分配 shared memory tensor
    - cute.arch.barrier() = __syncthreads()
    - cute.arch.shuffle_sync_bfly() = __shfl_xor_sync
    - cute.rsqrt() = rsqrtf()
    - cute.compile(fn, *sample_args): AOT 编译，返回 compiled callable
"""

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator
import torch

THREADS = 256
NUM_WARPS = THREADS // 32

# 默认 benchmark shape，用于 AOT 编译的 sample args
_DEFAULT_B = 4096
_DEFAULT_N = 4096


# -----------------------------------------------------------------------
# V1: Two-pass + shared memory tree reduce
# -----------------------------------------------------------------------
@cute.kernel
def rms_norm_v1_kernel(
    x: cute.Tensor,   # (B, N)
    w: cute.Tensor,   # (N,)
    y: cute.Tensor,   # (B, N)
    N: cute.Int32,
    eps: cute.Float32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()

    smem = SmemAllocator()
    s = smem.allocate_tensor(cute.Float32, cute.make_layout(THREADS), 16)

    # Pass 1: 每线程累积本地 sum(x²)
    local_ss = 0.0
    for i in range(tidx, N, bdimx):
        v = x[bidx, i]
        local_ss += v * v
    s[tidx] = local_ss
    cute.arch.barrier()

    # Shared memory tree reduce
    stride = bdimx // 2
    while stride > 0:
        if tidx < stride:
            s[tidx] = s[tidx] + s[tidx + stride]
        cute.arch.barrier()
        stride = stride // 2

    rms_inv = cute.rsqrt(s[0] / N + eps)
    cute.arch.barrier()

    # Pass 2: y = x / rms * w
    for i in range(tidx, N, bdimx):
        y[bidx, i] = x[bidx, i] * rms_inv * w[i]


@cute.jit
def _launch_v1(
    x: cute.Tensor,
    w: cute.Tensor,
    y: cute.Tensor,
    B: cute.Int32,
    N: cute.Int32,
    eps: cute.Float32,
):
    rms_norm_v1_kernel(x, w, y, N, eps).launch(
        grid=(B, 1, 1),
        block=(THREADS, 1, 1),
    )


# -----------------------------------------------------------------------
# V2: Two-pass + warp shuffle reduce
# -----------------------------------------------------------------------
@cute.kernel
def rms_norm_v2_kernel(
    x: cute.Tensor,   # (B, N)
    w: cute.Tensor,   # (N,)
    y: cute.Tensor,   # (B, N)
    N: cute.Int32,
    eps: cute.Float32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()
    lane    = tidx % 32
    warp_id = tidx // 32

    smem = SmemAllocator()
    s = smem.allocate_tensor(cute.Float32, cute.make_layout(NUM_WARPS), 16)

    # Pass 1: 每线程累积本地 sum(x²)
    local_ss = 0.0
    for i in range(tidx, N, bdimx):
        v = x[bidx, i]
        local_ss += v * v

    # Warp shuffle reduce
    for mask in [16, 8, 4, 2, 1]:
        local_ss += cute.arch.shuffle_sync_bfly(local_ss, offset=mask)

    # Lane 0 写到 smem
    if lane == 0:
        s[warp_id] = local_ss
    cute.arch.barrier()

    # Warp 0 做第二级 reduce
    if warp_id == 0:
        val = s[lane] if lane < NUM_WARPS else 0.0
        for mask in [16, 8, 4, 2, 1]:
            val += cute.arch.shuffle_sync_bfly(val, offset=mask)
        if lane == 0:
            s[0] = val
    cute.arch.barrier()

    rms_inv = cute.rsqrt(s[0] / N + eps)

    # Pass 2
    for i in range(tidx, N, bdimx):
        y[bidx, i] = x[bidx, i] * rms_inv * w[i]


@cute.jit
def _launch_v2(
    x: cute.Tensor,
    w: cute.Tensor,
    y: cute.Tensor,
    B: cute.Int32,
    N: cute.Int32,
    eps: cute.Float32,
):
    rms_norm_v2_kernel(x, w, y, N, eps).launch(
        grid=(B, 1, 1),
        block=(THREADS, 1, 1),
    )


# -----------------------------------------------------------------------
# AOT 编译缓存（按 (B, N) 键缓存）
# cute.compile(fn, *sample_args) 固定 tensor layout（含 stride），
# 不同 shape 需要单独编译，故用 dict[(B,N)] 缓存。
# -----------------------------------------------------------------------
_compiled_v1: dict = {}
_compiled_v2: dict = {}

def _get_compiled(version: int, x2, weight, y):
    """懒编译：按 (B, N) shape 首次调用时 AOT 编译，之后直接复用。"""
    B, N = x2.shape
    key = (B, N)
    if version == 1:
        if key not in _compiled_v1:
            _compiled_v1[key] = cute.compile(
                _launch_v1,
                from_dlpack(x2), from_dlpack(weight), from_dlpack(y),
                B, N, float(1e-6),
            )
        return _compiled_v1[key]
    else:
        if key not in _compiled_v2:
            _compiled_v2[key] = cute.compile(
                _launch_v2,
                from_dlpack(x2), from_dlpack(weight), from_dlpack(y),
                B, N, float(1e-6),
            )
        return _compiled_v2[key]


# -----------------------------------------------------------------------
# Python 包装
# -----------------------------------------------------------------------
def _run(version: int, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig_shape = x.shape
    N = x.shape[-1]
    x2 = x.reshape(-1, N).contiguous()
    B = x2.shape[0]
    y = torch.empty_like(x2)

    tx = from_dlpack(x2)
    tw = from_dlpack(weight)
    ty = from_dlpack(y)

    compiled = _get_compiled(version, x2, weight, y)
    compiled(tx, tw, ty, B, N, eps)
    return y.reshape(orig_shape)


def rms_norm_cutedsl_v1(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """CuTe DSL RMSNorm v1: two-pass + smem tree reduce (AOT compiled)."""
    return _run(1, x, weight, eps)


def rms_norm_cutedsl_v2(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """CuTe DSL RMSNorm v2: two-pass + warp shuffle reduce (AOT compiled)."""
    return _run(2, x, weight, eps)
