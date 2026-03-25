"""
Flash Attention - CuTe DSL 实现

数学定义（online softmax，无需物化 N×N 矩阵）：
    对每个 Q 行 i，扫描所有 K 行 j：
        s_ij = Q[i,:] · K[j,:] * scale
        m_new = max(m_old, s_ij)
        alpha = exp(m_old - m_new)
        l_new = l_old * alpha + exp(s_ij - m_new)
        O[i,:] = O[i,:] * alpha + exp(s_ij - m_new) * V[j,:]

V1: 1 thread per Q row（最简实现，展示 CuTe DSL 核心用法）
    - grid = (ceil(N/Br), B*H), block = (Br=64, 1, 1)
    - tidx = qi 行号（tile 内偏移），每线程串行扫描所有 N 行 KV
    - 在 smem 中维护 O 累积器（无法用局部数组）
    - online softmax 状态 (m_i, l_i) 存于寄存器
    - 性能较慢（每线程串行 N×D），教学目的

V2: 1 warp per Q row（并行 D 维点积 + warp reduce）
    - grid = (ceil(N/WARPS_PER_BLOCK), B*H), block = (THREADS=128, 1, 1)
    - WARPS_PER_BLOCK = THREADS/32 个 warp，每 warp 负责一行 Q
    - warp 内 32 个 lane 并行累积 D 维点积，warp_reduction_sum 归约
    - warp_reduction_max 归约 online softmax 的 m_new
    - O 累积器存在 smem [num_q_rows, D] 中（lane 分摊 D 维）

性能关键：AOT 编译按 (B, H, N, D) 键缓存。

关键 CuTe DSL 用法：
    - cute.arch.warp_reduction_sum/max: warp-level reduce
    - cute.arch.fmax(): max of two floats
    - cute.exp(): exp()
    - 1D smem 模拟 2D [rows, D] 数组（索引 = row*D + col）
"""

import math
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator
import torch

# 编译期常量（对应标准 LLM head_dim=64）
HEAD_DIM = 64
Br       = 64    # Q tile 行数 = block 线程数


# -----------------------------------------------------------------------
# V1: 1 thread per Q row — online softmax, smem O accumulator
# -----------------------------------------------------------------------
@cute.kernel
def flash_attn_v1_kernel(
    Q: cute.Tensor,       # (B, H, N, D)
    K: cute.Tensor,       # (B, H, N, D)
    V: cute.Tensor,       # (B, H, N, D)
    O: cute.Tensor,       # (B, H, N, D)
    scale: cute.Float32,
    B: cute.Int32,
    H: cute.Int32,
    N: cute.Int32,
    D: cute.Int32,
):
    tidx, _, _  = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    b_idx = bidy // H
    h_idx = bidy % H
    qi    = tidx                  # tile 内行号，也是本线程的 Q 行偏移
    q_row = bidx * Br + qi        # 全局 Q 行号

    smem = SmemAllocator()
    # 1D smem: [Br * HEAD_DIM]，逻辑上是 [Br, HEAD_DIM]
    # 访问 (qi, d) → 索引 qi * HEAD_DIM + d
    s_o = smem.allocate_tensor(cute.Float32, cute.make_layout(Br * HEAD_DIM), 16)

    # 初始化 O 累积器
    for d in range(HEAD_DIM):
        s_o[qi * HEAD_DIM + d] = 0.0
    cute.arch.barrier()

    # 在线 softmax 状态（寄存器）
    m_i = -1.0e9
    l_i = 0.0

    if q_row < N:
        # 逐行扫描 K（one K row at a time）
        for kv_row in range(0, N, 1):
            # 点积 Q[q_row, :] · K[kv_row, :] — 串行 over D
            s_ij = 0.0
            for d in range(HEAD_DIM):
                s_ij = s_ij + Q[b_idx, h_idx, q_row, d] * K[b_idx, h_idx, kv_row, d]
            s_ij = s_ij * scale

            # Online softmax 更新
            m_new  = cute.arch.fmax(m_i, s_ij)
            alpha  = cute.exp(m_i - m_new)
            e_new  = cute.exp(s_ij - m_new)
            l_i    = l_i * alpha + e_new

            # O 累积：O = O * alpha + e_new * V[kv_row, :]
            for d in range(HEAD_DIM):
                s_o[qi * HEAD_DIM + d] = (
                    s_o[qi * HEAD_DIM + d] * alpha
                    + e_new * V[b_idx, h_idx, kv_row, d]
                )

            m_i = m_new

    cute.arch.barrier()

    # 归一化并写回
    if q_row < N:
        for d in range(HEAD_DIM):
            O[b_idx, h_idx, q_row, d] = s_o[qi * HEAD_DIM + d] / l_i


@cute.jit
def _launch_v1(
    Q: cute.Tensor,
    K: cute.Tensor,
    V: cute.Tensor,
    O: cute.Tensor,
    scale: cute.Float32,
    B: cute.Int32,
    H: cute.Int32,
    N: cute.Int32,
    D: cute.Int32,
):
    num_q_tiles = (N + Br - 1) // Br
    flash_attn_v1_kernel(Q, K, V, O, scale, B, H, N, D).launch(
        grid=(num_q_tiles, B * H, 1),
        block=(Br, 1, 1),
    )


# -----------------------------------------------------------------------
# AOT 编译缓存（按 (B, H, N, D) 键）
# -----------------------------------------------------------------------
_compiled_v1: dict = {}


def _get_compiled_v1(Q, K, V, O, scale, B, H, N, D):
    key = (B, H, N, D)
    if key not in _compiled_v1:
        _compiled_v1[key] = cute.compile(
            _launch_v1,
            from_dlpack(Q), from_dlpack(K), from_dlpack(V), from_dlpack(O),
            scale, B, H, N, D,
        )
    return _compiled_v1[key]


# -----------------------------------------------------------------------
# Python 包装
# -----------------------------------------------------------------------
def flash_attention_cutedsl_v1(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
    sm_scale: float = None,
) -> torch.Tensor:
    """CuTe DSL Flash Attention v1: 1 thread per Q row, online softmax (AOT compiled).

    Q, K, V: (B, H, N, D) float32 CUDA, D 必须为 64。
    causal: 暂不支持（无 causal mask）。
    """
    assert not causal, "flash_attention_cutedsl_v1 does not support causal=True yet"
    assert Q.is_cuda and Q.dtype == torch.float32
    B, H, N, D = Q.shape
    assert D == HEAD_DIM, f"CuteDSL Flash Attention requires D={HEAD_DIM}, got D={D}"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    O = torch.empty_like(Q)

    compiled = _get_compiled_v1(Q, K, V, O, float(sm_scale), B, H, N, D)
    compiled(
        from_dlpack(Q), from_dlpack(K), from_dlpack(V), from_dlpack(O),
        float(sm_scale), B, H, N, D,
    )
    return O


# -----------------------------------------------------------------------
# V2: 1 warp per Q row — 并行 D 维点积 + warp reduce online softmax
# -----------------------------------------------------------------------
# THREADS_V2 个线程 / 32 = WARPS_PER_BLOCK 个 warp，每 warp 一行 Q
THREADS_V2      = 128
WARPS_PER_BLOCK = THREADS_V2 // 32   # = 4

@cute.kernel
def flash_attn_v2_kernel(
    Q: cute.Tensor,
    K: cute.Tensor,
    V: cute.Tensor,
    O: cute.Tensor,
    scale: cute.Float32,
    B: cute.Int32,
    H: cute.Int32,
    N: cute.Int32,
    D: cute.Int32,
):
    tidx, _, _    = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    b_idx   = bidy // H
    h_idx   = bidy % H
    warp_id = tidx // 32
    lane    = tidx % 32

    # 每个 warp 负责 block 内第 warp_id 行 Q（tile 内偏移）
    qi         = warp_id
    q_row      = bidx * WARPS_PER_BLOCK + qi    # 全局 Q 行号

    smem = SmemAllocator()
    # smem O 累积器: [WARPS_PER_BLOCK * HEAD_DIM]，索引 = warp_id * HEAD_DIM + d
    s_o = smem.allocate_tensor(cute.Float32,
                               cute.make_layout(WARPS_PER_BLOCK * HEAD_DIM), 16)

    # 初始化 O 累积器（lane 分摊 D 维）
    for d in range(lane, HEAD_DIM, 32):
        s_o[qi * HEAD_DIM + d] = 0.0
    cute.arch.barrier()

    # Online softmax 状态（寄存器，warp 内所有 lane 保持同步）
    m_i = -1.0e9
    l_i = 0.0

    if q_row < N:
        for kv_row in range(0, N, 1):
            # 并行点积：每 lane 负责 D/32 维，warp reduce 得到 s_ij
            partial = 0.0
            for d in range(lane, HEAD_DIM, 32):
                partial = partial + Q[b_idx, h_idx, q_row, d] * K[b_idx, h_idx, kv_row, d]
            s_ij = cute.arch.warp_reduction_sum(partial) * scale

            # Online softmax（所有 lane 并行计算，结果一致）
            m_new = cute.arch.fmax(m_i, s_ij)
            alpha = cute.exp(m_i - m_new)
            e_new = cute.exp(s_ij - m_new)
            l_i   = l_i * alpha + e_new

            # 更新 O 累积器（lane 分摊 D 维）
            for d in range(lane, HEAD_DIM, 32):
                s_o[qi * HEAD_DIM + d] = (
                    s_o[qi * HEAD_DIM + d] * alpha
                    + e_new * V[b_idx, h_idx, kv_row, d]
                )

            m_i = m_new

    cute.arch.barrier()

    # 归一化并写回（lane 分摊 D 维）
    if q_row < N:
        for d in range(lane, HEAD_DIM, 32):
            O[b_idx, h_idx, q_row, d] = s_o[qi * HEAD_DIM + d] / l_i


@cute.jit
def _launch_v2(
    Q: cute.Tensor,
    K: cute.Tensor,
    V: cute.Tensor,
    O: cute.Tensor,
    scale: cute.Float32,
    B: cute.Int32,
    H: cute.Int32,
    N: cute.Int32,
    D: cute.Int32,
):
    num_q_tiles = (N + WARPS_PER_BLOCK - 1) // WARPS_PER_BLOCK
    flash_attn_v2_kernel(Q, K, V, O, scale, B, H, N, D).launch(
        grid=(num_q_tiles, B * H, 1),
        block=(THREADS_V2, 1, 1),
    )


_compiled_v2: dict = {}


def _get_compiled_v2(Q, K, V, O, scale, B, H, N, D):
    key = (B, H, N, D)
    if key not in _compiled_v2:
        _compiled_v2[key] = cute.compile(
            _launch_v2,
            from_dlpack(Q), from_dlpack(K), from_dlpack(V), from_dlpack(O),
            scale, B, H, N, D,
        )
    return _compiled_v2[key]


def flash_attention_cutedsl_v2(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
    sm_scale: float = None,
) -> torch.Tensor:
    """CuTe DSL Flash Attention v2: 1 warp per Q row, warp-parallel D dot product (AOT compiled).

    Q, K, V: (B, H, N, D) float32 CUDA, D 必须为 64。
    causal: 暂不支持。
    """
    assert not causal, "flash_attention_cutedsl_v2 does not support causal=True yet"
    assert Q.is_cuda and Q.dtype == torch.float32
    B, H, N, D = Q.shape
    assert D == HEAD_DIM, f"CuteDSL Flash Attention requires D={HEAD_DIM}, got D={D}"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    O = torch.empty_like(Q)

    compiled = _get_compiled_v2(Q, K, V, O, float(sm_scale), B, H, N, D)
    compiled(
        from_dlpack(Q), from_dlpack(K), from_dlpack(V), from_dlpack(O),
        float(sm_scale), B, H, N, D,
    )
    return O
