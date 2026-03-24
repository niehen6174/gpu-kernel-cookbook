"""
RoPE - CuTe DSL 实现

数学定义 (GPT-NeoX / non-interleaved, rotate_half style):
    x1 = x[..., :half_dim],  x2 = x[..., half_dim:]
    x'[..., :half_dim] = x1 * cos - x2 * sin
    x'[..., half_dim:] = x2 * cos + x1 * sin

    cos_cache[pos, i], sin_cache[pos, i]: (max_seq_len, half_dim) precomputed

V1: 1 block per (token, head)
    - grid = (seq_len * num_heads, 1, 1), block = (HALF_DIM, 1, 1)
    - 每线程处理一个旋转维度，直接读写 q/k
    - bidx = token * num_heads + head

V2: 1 block per token（cos/sin 共享，rope_v3 CUDA 思路）
    - grid = (seq_len, 1, 1), block = (THREADS_V2, 1, 1)
    - 先把 cos/sin 装入 smem，所有 head 共享 → 32× 减少 HBM 读取
    - 每线程 tidx = head * half_dim + lane

性能关键：cute.compile() AOT 编译，按 (seq_len, num_heads, half_dim) 键缓存。

关键 CuTe DSL 用法：
    - 3D Tensor 索引: q[token, head, dim]
    - positions[i] 返回 int32 值，可作为 cos_cache[pos, j] 的索引
    - SmemAllocator: 分配 shared memory tensor
    - cute.arch.barrier() = __syncthreads()
"""

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator
import torch

# 编译时常量 —— 对应 head_dim=64 (half_dim=32)
# 支持的最大 half_dim；用于 smem 分配和 block 大小
HALF_DIM  = 32
THREADS_V2 = 1024   # 最大: 32 heads × 32 half_dim


# -----------------------------------------------------------------------
# V1: 1 block per (token, head), HALF_DIM threads per block
# -----------------------------------------------------------------------
@cute.kernel
def rope_v1_kernel(
    q: cute.Tensor,          # (seq_len, num_heads, head_dim)  inplace
    k: cute.Tensor,          # (seq_len, num_heads, head_dim)  inplace
    cos_cache: cute.Tensor,  # (max_seq_len, half_dim)
    sin_cache: cute.Tensor,  # (max_seq_len, half_dim)
    positions: cute.Tensor,  # (seq_len,) int32
    num_heads: cute.Int32,
    half_dim: cute.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()   # bidx = token_idx * num_heads + head_idx

    token_idx = bidx // num_heads
    head_idx  = bidx % num_heads

    if tidx < half_dim:
        pos = positions[token_idx]
        c = cos_cache[pos, tidx]
        s = sin_cache[pos, tidx]

        q1 = q[token_idx, head_idx, tidx]
        q2 = q[token_idx, head_idx, tidx + half_dim]
        q[token_idx, head_idx, tidx]            = q1 * c - q2 * s
        q[token_idx, head_idx, tidx + half_dim] = q2 * c + q1 * s

        k1 = k[token_idx, head_idx, tidx]
        k2 = k[token_idx, head_idx, tidx + half_dim]
        k[token_idx, head_idx, tidx]            = k1 * c - k2 * s
        k[token_idx, head_idx, tidx + half_dim] = k2 * c + k1 * s


@cute.jit
def _launch_v1(
    q: cute.Tensor,
    k: cute.Tensor,
    cos_cache: cute.Tensor,
    sin_cache: cute.Tensor,
    positions: cute.Tensor,
    seq_len: cute.Int32,
    num_heads: cute.Int32,
    half_dim: cute.Int32,
):
    rope_v1_kernel(q, k, cos_cache, sin_cache, positions, num_heads, half_dim).launch(
        grid=(seq_len * num_heads, 1, 1),
        block=(HALF_DIM, 1, 1),
    )


# -----------------------------------------------------------------------
# V2: 1 block per token，cos/sin 加载到 smem，所有 head 共享
# -----------------------------------------------------------------------
@cute.kernel
def rope_v2_kernel(
    q: cute.Tensor,
    k: cute.Tensor,
    cos_cache: cute.Tensor,
    sin_cache: cute.Tensor,
    positions: cute.Tensor,
    num_heads: cute.Int32,
    half_dim: cute.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()   # bidx = token_idx

    smem = SmemAllocator()
    s_cos = smem.allocate_tensor(cute.Float32, cute.make_layout(HALF_DIM), 16)
    s_sin = smem.allocate_tensor(cute.Float32, cute.make_layout(HALF_DIM), 16)

    pos = positions[bidx]

    # 协作加载 cos/sin 到 smem（前 half_dim 个线程）
    if tidx < half_dim:
        s_cos[tidx] = cos_cache[pos, tidx]
        s_sin[tidx] = sin_cache[pos, tidx]
    cute.arch.barrier()

    # 每个激活线程处理一个 (head, lane) 对
    total = num_heads * half_dim
    if tidx < total:
        head_idx = tidx // half_dim
        lane_idx = tidx % half_dim
        c   = s_cos[lane_idx]
        s_v = s_sin[lane_idx]

        q1 = q[bidx, head_idx, lane_idx]
        q2 = q[bidx, head_idx, lane_idx + half_dim]
        q[bidx, head_idx, lane_idx]            = q1 * c - q2 * s_v
        q[bidx, head_idx, lane_idx + half_dim] = q2 * c + q1 * s_v

        k1 = k[bidx, head_idx, lane_idx]
        k2 = k[bidx, head_idx, lane_idx + half_dim]
        k[bidx, head_idx, lane_idx]            = k1 * c - k2 * s_v
        k[bidx, head_idx, lane_idx + half_dim] = k2 * c + k1 * s_v


@cute.jit
def _launch_v2(
    q: cute.Tensor,
    k: cute.Tensor,
    cos_cache: cute.Tensor,
    sin_cache: cute.Tensor,
    positions: cute.Tensor,
    seq_len: cute.Int32,
    num_heads: cute.Int32,
    half_dim: cute.Int32,
):
    rope_v2_kernel(q, k, cos_cache, sin_cache, positions, num_heads, half_dim).launch(
        grid=(seq_len, 1, 1),
        block=(THREADS_V2, 1, 1),
    )


# -----------------------------------------------------------------------
# AOT 编译缓存（按 (seq_len, num_heads, half_dim) 键缓存）
# -----------------------------------------------------------------------
_compiled_v1: dict = {}
_compiled_v2: dict = {}


def _get_compiled(version: int, q, k, cos_cache, sin_cache, positions):
    seq_len, num_heads, head_dim = q.shape
    half_dim = head_dim // 2
    key = (seq_len, num_heads, half_dim)

    if version == 1:
        if key not in _compiled_v1:
            _compiled_v1[key] = cute.compile(
                _launch_v1,
                from_dlpack(q), from_dlpack(k),
                from_dlpack(cos_cache), from_dlpack(sin_cache),
                from_dlpack(positions),
                seq_len, num_heads, half_dim,
            )
        return _compiled_v1[key]
    else:
        if key not in _compiled_v2:
            _compiled_v2[key] = cute.compile(
                _launch_v2,
                from_dlpack(q), from_dlpack(k),
                from_dlpack(cos_cache), from_dlpack(sin_cache),
                from_dlpack(positions),
                seq_len, num_heads, half_dim,
            )
        return _compiled_v2[key]


# -----------------------------------------------------------------------
# Python 包装
# -----------------------------------------------------------------------
def _run(
    version: int,
    q: torch.Tensor,
    k: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple:
    q = q.clone().contiguous()
    k = k.clone().contiguous()
    cos_cache = cos_cache.contiguous()
    sin_cache = sin_cache.contiguous()
    pos32 = positions.to(torch.int32).contiguous()

    seq_len, num_heads, head_dim = q.shape
    half_dim = head_dim // 2

    compiled = _get_compiled(version, q, k, cos_cache, sin_cache, pos32)
    compiled(
        from_dlpack(q), from_dlpack(k),
        from_dlpack(cos_cache), from_dlpack(sin_cache),
        from_dlpack(pos32),
        seq_len, num_heads, half_dim,
    )
    return q, k


def rope_cutedsl_v1(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple:
    """CuTe DSL RoPE v1: 1 block per (token, head), HALF_DIM threads (AOT compiled)."""
    return _run(1, q, k, cos_cache, sin_cache, positions)


def rope_cutedsl_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple:
    """CuTe DSL RoPE v2: 1 block per token + smem cos/sin reuse (AOT compiled)."""
    return _run(2, q, k, cos_cache, sin_cache, positions)
