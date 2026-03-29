"""
SageAttention V2 CuTe DSL 实现 (Hopper SM90a)

V2 相对 V1 的升级：
1. Q: per-block INT8 → per-warp INT8（WARPQ=16，每个 BLOCK_M=64 含 4 个 warp scale）
2. K: 不变（per-block INT8 + K Smoothing）
3. V: FP16 → FP8 per-channel（round-trip dequantize 回 FP16 后传入 kernel）
   + V Smoothing（均值在 Python epilogue 恢复）

FP8 V WGMMA 在 CuTe DSL 中不可行，故采用量化噪声模拟方案：
V 在 Python 预处理时经 FP8 round-trip 后 dequantize 回 FP16，kernel 仍使用 FP16 PV GEMM。
V mean 恢复在 kernel 输出后由 Python epilogue 完成。

Kernel 核心改动（相对 kernel.py）：
- gQScale: (B*H*N_Q_BLOCKS,) → (B*H*N_Q_BLOCKS, WARPS_PER_BLOCK)
- q_scale_val: gQScale[bidx] → gQScale[bidx, tidx // 32]  (per-warp)

性能与 V1 相同（相同 smem 布局），量化步骤略慢（多 V FP8 round-trip）。

设计参考: SageAttention V2 (arXiv:2501.01005)
"""

import os
import math
import torch

if "CUTE_DSL_ARCH" not in os.environ:
    os.environ["CUTE_DSL_ARCH"] = "sm_90a"

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warpgroup, cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils

# ============================================================
# 静态超参数
# ============================================================
BLOCK_M      = 64    # Q tile M-dim per CTA
BLOCK_N      = 64    # KV tile N-dim (WGMMA m64n64)
HEAD_DIM     = 64    # head dimension (fixed)
NUM_THREADS  = 128   # 1 warpgroup = 128 threads
WARPQ        = 16    # Q rows per warp (V2: per-warp quantization)
WARPS_PER_BLOCK = BLOCK_M // WARPQ   # = 4

# acc_S per-thread fragment (BLOCK_M=64, BLOCK_N=64, m64n64 WGMMA INT8)
# 128 threads × 32 elems = 64×64 total, 2 M-rows per thread
ACC_FRAG_SIZE    = 32
ACC_ROWS_PER_THR = 2

# flat element counts
TILE_ELEMS_K = BLOCK_N * HEAD_DIM   # int8: 64×64 = 4096
TILE_ELEMS_V = BLOCK_N * HEAD_DIM   # fp16: 64×64 = 4096
Q_ELEMS      = BLOCK_M * HEAD_DIM   # int8: 64×64 = 4096

LOG2_E = 1.4426950408889634

# ============================================================
# 量化（GPU Triton kernels + V2 helpers）
# ============================================================
from .quant import (
    smooth_and_quant_k_gpu,
    quant_per_block_int8,
    quant_q_per_warp_int8_gpu,
    quant_v_per_channel_fp8,
)


# ============================================================
# CuTe DSL Kernel V2 (per-warp Q scale, FP16 V after FP8 round-trip)
# ============================================================

@cute.kernel
def _sage_kernel_v2(
    gQ:      cute.Tensor,   # (B*H*N_Q_BLOCKS,  BLOCK_M,  HEAD_DIM) int8
    gK:      cute.Tensor,   # (B*H*N_KV_BLOCKS, BLOCK_N,  HEAD_DIM) int8
    gV:      cute.Tensor,   # (B*H*N_KV_BLOCKS, HEAD_DIM, BLOCK_N)  fp16
    gO:      cute.Tensor,   # (B*H*N_Q_BLOCKS,  BLOCK_M,  HEAD_DIM) fp32
    gQScale: cute.Tensor,   # (B*H*N_Q_BLOCKS, WARPS_PER_BLOCK) fp32  ← 2D (V2)
    gKScale: cute.Tensor,   # (B*H, N_KV_BLOCKS) fp32
    n_kv_blocks: cute.Int32,
    n_q_blocks:  cute.Int32,
    mma_qk:      cute.TiledMma,
    mma_pv:      cute.TiledMma,   # FP16×FP16→FP32
    sQ_layout:   cute.ComposedLayout,
    sK_layout:   cute.ComposedLayout,   # 2 stages
    sV_layout:   cute.ComposedLayout,   # FP16
    copy_QK:       cute.TiledCopy,
    copy_V:        cute.TiledCopy,
    copy_QK_async: cute.TiledCopy,
    copy_V_async:  cute.TiledCopy,   # FP16
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    smem = SmemAllocator()

    @cute.struct
    class SS:
        sQ: cute.struct.Align[cute.struct.MemRange[cutlass.Int8,   cute.cosize(sQ_layout)], 128]
        sK: cute.struct.Align[cute.struct.MemRange[cutlass.Int8,   cute.cosize(sK_layout)], 128]
        sV: cute.struct.Align[cute.struct.MemRange[cutlass.Float16, cute.cosize(sV_layout)], 128]

    storage = smem.allocate(SS)
    sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
    sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
    sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)

    sQ_s0 = cute.slice_(sQ, (None, None, 0))
    sK_s0 = cute.slice_(sK, (None, None, 0))
    sK_s1 = cute.slice_(sK, (None, None, 1))
    sV_s0 = cute.slice_(sV, (None, None, 0))

    head_flat = bidx // n_q_blocks

    thr_QK       = copy_QK.get_slice(tidx)
    thr_V        = copy_V.get_slice(tidx)
    thr_QK_async = copy_QK_async.get_slice(tidx)
    thr_V_async  = copy_V_async.get_slice(tidx)

    tQsQ          = thr_QK.partition_D(sQ_s0)
    tKsK_async_s0 = thr_QK_async.partition_D(sK_s0)
    tKsK_async_s1 = thr_QK_async.partition_D(sK_s1)
    tVsV_async    = thr_V_async.partition_D(sV_s0)

    gQ_base = gQ.iterator
    gK_base = gK.iterator
    gV_base = gV.iterator
    gO_base = gO.iterator

    q_ptr      = (gQ_base + bidx * Q_ELEMS).align(128)
    k_ptr_base = gK_base + head_flat * n_kv_blocks * TILE_ELEMS_K
    v_ptr_base = gV_base + head_flat * n_kv_blocks * TILE_ELEMS_V
    o_ptr      = gO_base + bidx * Q_ELEMS

    gQ_cta = cute.make_tensor(q_ptr, cute.make_layout((BLOCK_M, HEAD_DIM), stride=(HEAD_DIM, 1)))
    gO_cta = cute.make_tensor(o_ptr, cute.make_layout((BLOCK_M, HEAD_DIM), stride=(HEAD_DIM, 1)))

    # --- Load Q (vectorized sync) ---
    cute.copy(copy_QK, thr_QK.partition_S(gQ_cta), tQsQ)

    # --- Prologue: prefetch K[0] → stage 0, K[1] → stage 1 ---
    k0_ptr = (k_ptr_base + 0).align(128)
    gK_0 = cute.make_tensor(k0_ptr, cute.make_layout((BLOCK_N, HEAD_DIM), stride=(HEAD_DIM, 1)))
    cute.copy(copy_QK_async, thr_QK_async.partition_S(gK_0), tKsK_async_s0)
    cute.arch.cp_async_commit_group()   # group A: K[0] → stage 0

    if n_kv_blocks > 1:
        k1_ptr = (k_ptr_base + TILE_ELEMS_K).align(128)
        gK_1 = cute.make_tensor(k1_ptr, cute.make_layout((BLOCK_N, HEAD_DIM), stride=(HEAD_DIM, 1)))
        cute.copy(copy_QK_async, thr_QK_async.partition_S(gK_1), tKsK_async_s1)
        cute.arch.cp_async_commit_group()  # group B: K[1] → stage 1
        cute.arch.cp_async_wait_group(1)   # wait K[0] only
    else:
        cute.arch.cp_async_wait_group(0)
    cute.arch.barrier()

    # --- MMA fragments ---
    thr_qk = mma_qk.get_slice(tidx)
    tCrQ   = mma_qk.make_fragment_A(thr_qk.partition_A(sQ))
    tCrK   = mma_qk.make_fragment_B(thr_qk.partition_B(sK))
    acc_S  = cute.make_fragment(thr_qk.partition_shape_C((BLOCK_M, BLOCK_N)), cutlass.Int32)

    thr_pv = mma_pv.get_slice(tidx)
    rP    = cute.make_fragment(thr_pv.partition_shape_A((BLOCK_M, BLOCK_N)), cutlass.Float16)
    tCrV  = mma_pv.make_fragment_B(thr_pv.partition_B(sV))
    acc_O = cute.make_fragment(thr_pv.partition_shape_C((BLOCK_M, HEAD_DIM)), cutlass.Float32)
    acc_O.fill(0.0)

    row_max = cute.make_fragment(ACC_ROWS_PER_THR, cutlass.Float32)
    row_sum = cute.make_fragment(ACC_ROWS_PER_THR, cutlass.Float32)
    row_max.fill(-cutlass.Float32.inf)
    row_sum.fill(cutlass.Float32(0.0))

    # V2: per-warp Q scale — warp index = tidx // 32
    warp_idx = tidx // 32
    q_scale_val = gQScale[bidx, warp_idx]

    # ---- Main KV loop ----
    for n_tile in range(n_kv_blocks):
        cur_stage = n_tile % 2

        # QK GEMM (INT8×INT8→INT32, m64n64)
        acc_S.fill(0)
        mma_qk.set(warpgroup.Field.ACCUMULATE, False)
        warpgroup.fence()
        for k in cutlass.range_constexpr(cute.size(tCrQ, mode=[2])):
            cute.gemm(mma_qk, acc_S, tCrQ[None, None, k, 0], tCrK[None, None, k, cur_stage], acc_S)
            mma_qk.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.commit_group()
        warpgroup.wait_group(0)

        # Issue V[n] async → group A (gV is fp16, layout (HEAD_DIM, BLOCK_N))
        v_ptr = (v_ptr_base + n_tile * TILE_ELEMS_V).align(128)
        gV_n = cute.make_tensor(v_ptr, cute.make_layout((HEAD_DIM, BLOCK_N), stride=(BLOCK_N, 1)))
        cute.copy(copy_V_async, thr_V_async.partition_S(gV_n), tVsV_async)
        cute.arch.cp_async_commit_group()  # group A: V[n]

        # Issue K[n+2] → cur_stage smem → group B
        if n_tile + 2 < n_kv_blocks:
            k_n2_ptr = (k_ptr_base + (n_tile + 2) * TILE_ELEMS_K).align(128)
            gK_n2 = cute.make_tensor(k_n2_ptr, cute.make_layout((BLOCK_N, HEAD_DIM), stride=(HEAD_DIM, 1)))
            if cur_stage == 0:
                cute.copy(copy_QK_async, thr_QK_async.partition_S(gK_n2), tKsK_async_s0)
            else:
                cute.copy(copy_QK_async, thr_QK_async.partition_S(gK_n2), tKsK_async_s1)
            cute.arch.cp_async_commit_group()  # group B: K[n+2]

        # Online softmax (register-only, overlaps cp.async)
        k_scale_val = gKScale[head_flat, n_tile]
        dequant = q_scale_val * k_scale_val

        row_max_new = cute.make_fragment(ACC_ROWS_PER_THR, cutlass.Float32)
        row_max_new.fill(-cutlass.Float32.inf)
        for i in cutlass.range_constexpr(ACC_FRAG_SIZE):
            r = (i // 2) % ACC_ROWS_PER_THR
            val = cutlass.Float32(acc_S[i]) * dequant
            if val > row_max_new[r]: row_max_new[r] = val
        for r in cutlass.range_constexpr(ACC_ROWS_PER_THR):
            row_max_new[r] = cute.arch.fmax(row_max_new[r],
                cute.arch.shuffle_sync_bfly(row_max_new[r], offset=2, mask=-1, mask_and_clamp=31))
            row_max_new[r] = cute.arch.fmax(row_max_new[r],
                cute.arch.shuffle_sync_bfly(row_max_new[r], offset=1, mask=-1, mask_and_clamp=31))
            row_max_new[r] = cute.arch.fmax(row_max[r], row_max_new[r])

        row_sum_new = cute.make_fragment(ACC_ROWS_PER_THR, cutlass.Float32)
        row_sum_new.fill(cutlass.Float32(0.0))
        for i in cutlass.range_constexpr(ACC_FRAG_SIZE):
            r = (i // 2) % ACC_ROWS_PER_THR
            p_val = cute.math.exp2(cutlass.Float32(acc_S[i]) * dequant - row_max_new[r], fastmath=True)
            rP[i] = cutlass.Float16(p_val)
            row_sum_new[r] = row_sum_new[r] + p_val
        for r in cutlass.range_constexpr(ACC_ROWS_PER_THR):
            row_sum_new[r] = row_sum_new[r] + cute.arch.shuffle_sync_bfly(
                row_sum_new[r], offset=2, mask=-1, mask_and_clamp=31)
            row_sum_new[r] = row_sum_new[r] + cute.arch.shuffle_sync_bfly(
                row_sum_new[r], offset=1, mask=-1, mask_and_clamp=31)

        for i in cutlass.range_constexpr(cute.size(acc_O)):
            r = (i // 2) % ACC_ROWS_PER_THR
            acc_O[i] = acc_O[i] * cute.math.exp2(row_max[r] - row_max_new[r], fastmath=True)

        for r in cutlass.range_constexpr(ACC_ROWS_PER_THR):
            alpha = cute.math.exp2(row_max[r] - row_max_new[r], fastmath=True)
            row_sum[r] = row_sum[r] * alpha + row_sum_new[r]
            row_max[r] = row_max_new[r]

        # Wait V (group A); allow K[n+2] (group B) to overlap PV GEMM
        if n_tile + 2 < n_kv_blocks:
            cute.arch.cp_async_wait_group(1)
        else:
            cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # PV GEMM (rs-mode: rP in registers)
        mma_pv.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.fence()
        for k in cutlass.range_constexpr(cute.size(rP, mode=[2])):
            cute.gemm(mma_pv, acc_O, rP[None, None, k], tCrV[None, None, k, 0], acc_O)
        warpgroup.commit_group()
        warpgroup.wait_group(0)

        # Wait K[n+2] after PV GEMM
        if n_tile + 2 < n_kv_blocks:
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

    # Epilogue: normalize + store FP32
    tOgO = thr_pv.partition_C(gO_cta)
    for i in cutlass.range_constexpr(cute.size(acc_O)):
        r = (i // 2) % ACC_ROWS_PER_THR
        acc_O[i] = acc_O[i] / (row_sum[r] + cutlass.Float32(1e-6))
    cute.basic_copy(acc_O, tOgO)


@cute.jit
def _sage_jit_v2(
    gQ:      cute.Tensor,
    gK:      cute.Tensor,
    gV:      cute.Tensor,
    gO:      cute.Tensor,
    gQScale: cute.Tensor,   # 2D: (B*H*N_Q_BLOCKS, WARPS_PER_BLOCK)
    gKScale: cute.Tensor,
    n_kv_blocks: cute.Int32,
    n_q_blocks:  cute.Int32,
    n_cta:       cute.Int32,
):
    # QK GEMM: INT8×INT8→INT32, tiler=(BLOCK_M=64, BLOCK_N=64)
    mma_qk = sm90_utils.make_trivial_tiled_mma(
        cutlass.Int8, cutlass.Int8,
        utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        cutlass.Int32, (1, 1, 1),
        tiler_mn=(BLOCK_M, BLOCK_N),
        a_source=warpgroup.OperandSource.SMEM,
    )
    # PV GEMM: FP16×FP16→FP32, rs-mode
    mma_pv = sm90_utils.make_trivial_tiled_mma(
        cutlass.Float16, cutlass.Float16,
        utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        cutlass.Float32, (1, 1, 1),
        tiler_mn=(BLOCK_M, HEAD_DIM),
        a_source=warpgroup.OperandSource.RMEM,
    )

    sQ_layout = sm90_utils.make_smem_layout_a(
        utils.LayoutEnum.ROW_MAJOR, (BLOCK_M, BLOCK_N, HEAD_DIM), cutlass.Int8, 1)
    sK_layout = sm90_utils.make_smem_layout_b(
        utils.LayoutEnum.ROW_MAJOR, (BLOCK_M, BLOCK_N, HEAD_DIM), cutlass.Int8, 2)
    sV_layout = sm90_utils.make_smem_layout_b(
        utils.LayoutEnum.ROW_MAJOR, (BLOCK_M, HEAD_DIM, BLOCK_N), cutlass.Float16, 1)

    # 128-bit vectorized copies
    # Q/K: INT8, BLOCK_N=64 cols → 16 elems/128bit → 4 col_groups → 32 threads_M
    copy_atom_QK = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Int8, num_bits_per_copy=128)
    copy_QK = cute.make_tiled_copy_tv(copy_atom_QK,
        cute.make_layout((32, 4), stride=(4, 1)), cute.make_layout((1, 16)))
    # V: FP16, BLOCK_N=64 cols → 8 elems/128bit → 8 col_groups → 16 threads_M
    copy_atom_V = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16, num_bits_per_copy=128)
    copy_V = cute.make_tiled_copy_tv(copy_atom_V,
        cute.make_layout((16, 8), stride=(8, 1)), cute.make_layout((1, 8)))

    copy_atom_QK_async = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.Int8, num_bits_per_copy=128)
    copy_QK_async = cute.make_tiled_copy_tv(copy_atom_QK_async,
        cute.make_layout((32, 4), stride=(4, 1)), cute.make_layout((1, 16)))
    copy_atom_V_async = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.Float16, num_bits_per_copy=128)
    copy_V_async = cute.make_tiled_copy_tv(copy_atom_V_async,
        cute.make_layout((16, 8), stride=(8, 1)), cute.make_layout((1, 8)))

    _sage_kernel_v2(
        gQ, gK, gV, gO, gQScale, gKScale,
        n_kv_blocks, n_q_blocks,
        mma_qk, mma_pv,
        sQ_layout, sK_layout, sV_layout,
        copy_QK, copy_V, copy_QK_async, copy_V_async,
    ).launch(grid=(n_cta, 1, 1), block=(NUM_THREADS, 1, 1))


# ============================================================
# AOT 编译缓存
# ============================================================
_compiled_kernels_v2: dict = {}


def _get_compiled_v2(B, H, N, D, kv_len, q_tiles, k_tiles, v_tiles, out,
                     q_scale_2d, k_scale_2d, N_KV_BLOCKS, N_Q_BLOCKS, N_CTAs):
    key = (B, H, N, D, kv_len)
    if key not in _compiled_kernels_v2:
        _compiled_kernels_v2[key] = cute.compile(
            _sage_jit_v2,
            from_dlpack(q_tiles, assumed_align=128),
            from_dlpack(k_tiles, assumed_align=128),
            from_dlpack(v_tiles, assumed_align=128),
            from_dlpack(out),
            from_dlpack(q_scale_2d),   # 2D: (B*H*N_Q_BLOCKS, WARPS_PER_BLOCK)
            from_dlpack(k_scale_2d),
            cute.Int32(N_KV_BLOCKS),
            cute.Int32(N_Q_BLOCKS),
            cute.Int32(N_CTAs),
        )
    return _compiled_kernels_v2[key]


# ============================================================
# Public API
# ============================================================
def sageattn_cutedsl_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale: float = None,
    smooth_k: bool = True,
    smooth_v: bool = True,
) -> torch.Tensor:
    """
    SageAttention V2 CuTe DSL 前向传播 (SM90a WGMMA).

    V2 特性:
    - Q: per-warp INT8 量化（WARPQ=16，每 BLOCK_M=64 有 4 个 scale）
    - K: per-block INT8 + K Smoothing（同 V1）
    - V: FP8 per-channel 量化（round-trip dequantize 回 FP16）+ V Smoothing

    参数:
        q, k, v  : (B, H, N, D) float16, CUDA
        is_causal: 因果 mask（暂未实现）
        sm_scale : softmax 缩放（默认 D^{-0.5}）
        smooth_k : K Smoothing（默认 True）
        smooth_v : V Smoothing（默认 True）

    返回:
        out: (B, H, N, D) float16
    """
    assert q.is_cuda and q.dtype == torch.float16
    B, H, N, D = q.shape
    assert D == HEAD_DIM, f"sageattn_cutedsl_v2: 只支持 head_dim={HEAD_DIM}, got {D}"
    assert N % BLOCK_M == 0, f"N={N} 必须是 BLOCK_M={BLOCK_M} 的整数倍"
    assert N % BLOCK_N == 0, f"N={N} 必须是 BLOCK_N={BLOCK_N} 的整数倍"
    assert N % WARPQ == 0, f"N={N} 必须是 WARPQ={WARPQ} 的整数倍"

    if sm_scale is None:
        sm_scale = D ** -0.5

    # ------------------------------------------------------------------
    # K 量化（同 V1: per-block INT8 + K Smoothing）
    # ------------------------------------------------------------------
    if smooth_k:
        k_int8, k_scale, km = smooth_and_quant_k_gpu(k, block_sz=BLOCK_N)
    else:
        k_int8, k_scale = quant_per_block_int8(k, sm_scale=1.0, block_sz=BLOCK_N)

    # ------------------------------------------------------------------
    # V FP8 per-channel 量化 → dequantize 回 FP16（含量化噪声）
    # ------------------------------------------------------------------
    v_fp16, v_scale, vm = quant_v_per_channel_fp8(v, smooth_v=smooth_v)
    # v_fp16: (B, H, N, D) fp16, ready for FP16 PV GEMM
    # vm    : (B, H, 1, D) fp16 or None, add back in Python epilogue

    # ------------------------------------------------------------------
    # Q per-warp INT8 量化
    # ------------------------------------------------------------------
    q_int8, q_scale_2d = quant_q_per_warp_int8_gpu(q, sm_scale, BLOCK_M=BLOCK_M, WARPQ=WARPQ)
    # q_scale_2d: (B, H, N // BLOCK_M, WARPS_PER_BLOCK)

    N_Q_BLOCKS  = N // BLOCK_M
    N_KV_BLOCKS = N // BLOCK_N
    N_CTAs      = B * H * N_Q_BLOCKS

    # ------------------------------------------------------------------
    # Tile-contiguous 重排
    # ------------------------------------------------------------------
    # K: (B, H, N_KV_BLOCKS, BLOCK_N, D) → (B*H, N_KV_BLOCKS, BLOCK_N, D)
    k_tiles = k_int8.view(B, H, N_KV_BLOCKS, BLOCK_N, D)
    k_tiles = k_tiles.reshape(B * H, N_KV_BLOCKS, BLOCK_N, D).contiguous()

    # V (FP16 after FP8 round-trip): transposed (HEAD_DIM, BLOCK_N) layout
    v_tiles = v_fp16.view(B, H, N_KV_BLOCKS, BLOCK_N, D)
    v_tiles = v_tiles.permute(0, 1, 2, 4, 3)               # (B, H, N_KV_BLOCKS, D, BLOCK_N)
    v_tiles = v_tiles.reshape(B * H, N_KV_BLOCKS, D, BLOCK_N).contiguous()

    # Q: (B, H, N_Q_BLOCKS, BLOCK_M, D) → (B*H*N_Q_BLOCKS, BLOCK_M, D)
    q_tiles = q_int8.view(B, H, N_Q_BLOCKS, BLOCK_M, D)
    q_tiles = q_tiles.reshape(B * H * N_Q_BLOCKS, BLOCK_M, D).contiguous()

    # Q scale 2D: (B, H, N_Q_BLOCKS, WARPS_PER_BLOCK) → (B*H*N_Q_BLOCKS, WARPS_PER_BLOCK)
    q_scale_bh = q_scale_2d.reshape(B * H * N_Q_BLOCKS, WARPS_PER_BLOCK).contiguous()

    # K scale 2D: (B, H, N_KV_BLOCKS) → (B*H, N_KV_BLOCKS)
    k_scale_2d = k_scale.reshape(B * H, N_KV_BLOCKS).contiguous()

    # Output buffer: (B*H*N_Q_BLOCKS, BLOCK_M, D) fp32
    out_buf = torch.empty(B * H * N_Q_BLOCKS, BLOCK_M, D, dtype=torch.float32, device=q.device)

    # ------------------------------------------------------------------
    # Kernel launch
    # ------------------------------------------------------------------
    compiled = _get_compiled_v2(
        B, H, N, D, N,
        q_tiles, k_tiles, v_tiles, out_buf,
        q_scale_bh, k_scale_2d,
        N_KV_BLOCKS, N_Q_BLOCKS, N_CTAs,
    )
    compiled(
        from_dlpack(q_tiles, assumed_align=128),
        from_dlpack(k_tiles, assumed_align=128),
        from_dlpack(v_tiles, assumed_align=128),
        from_dlpack(out_buf),
        from_dlpack(q_scale_bh),
        from_dlpack(k_scale_2d),
        cute.Int32(N_KV_BLOCKS),
        cute.Int32(N_Q_BLOCKS),
        cute.Int32(N_CTAs),
    )
    torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Python epilogue: V mean recovery (V Smoothing)
    # ------------------------------------------------------------------
    out = out_buf.view(B, H, N, D)      # (B, H, N, D) fp32
    if smooth_v and vm is not None:
        out = out + vm.to(torch.float32)   # vm: (B, H, 1, D) fp16 → fp32, broadcasts over N

    return out.to(torch.float16)
