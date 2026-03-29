"""
SageAttention CuTe DSL 实现 (Hopper SM90a) — v3 优化版

v3 相对 v2 (kernel.py) 的关键优化：
1. BLOCK_M: 64 → 128（CTA 处理 2× 的 Q 行）
   - N_CTAs 减半 → kernel launch overhead 减少 50%
   - 每个 KV tile 被 2 倍的 Q 行复用（单 CTA 内）

2. Split-warpgroup 设计（取代 buggy atom_layout=(2,1,1)）:
   - 两个独立 (1,1,1) warpgroup pipeline，各处理 HALF_M=64 行
   - WG0 (tidx 0-127)  → sQ0 (rows 0-63)
   - WG1 (tidx 128-255) → sQ1 (rows 64-127)
   - sQ0/sQ1: 各自独立 smem 区域，避免 WGMMA descriptor 竞争
   - sK/sV:   所有 256 线程协作加载，WG 共享
   - 关键: cute.arch.barrier() 在 if-block 外执行（需要所有256线程）
            per-WG MMA 和 softmax 在 if wg_idx == 0/1: 内执行

3. K Double Buffering 保持不变（已在 v2 验证有效）
4. rs-mode PV GEMM 保持不变
5. 128-bit 向量化 copy 保持不变

per-WG warp reduce 分析（与 v2 相同）：
  每个 WG: 128 threads, HALF_M=64, BLOCK_N=64
  (1,1,1) m64n64 WGMMA: 128 threads × 32 elems/thread = 64×64 total
  2 M-rows per thread, 16 N-cols per row → 4 threads/row
  → warp-reduce: 2 步 (offset 2, 1)

设计参考: SageAttention (arXiv:2410.02367, official kernel CTA_Q=128)
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
BLOCK_M      = 128   # Q tile M-dim per CTA (doubled from v2)
HALF_M       = 64    # per-warpgroup M-dim (= BLOCK_M / 2)
BLOCK_N      = 64    # KV tile N-dim (WGMMA m64n64)
HEAD_DIM     = 64    # head dimension (fixed)
NUM_THREADS  = 256   # 2 warpgroups × 128 threads
THREADS_PER_WG = 128

# acc_S per-thread fragment for (1,1,1) m64n64 WGMMA:
# 128 threads × 32 elems = 64×64 total, 2 M-rows per thread
ACC_FRAG_SIZE    = 32
ACC_ROWS_PER_THR = 2

# flat element counts
TILE_ELEMS_K = BLOCK_N * HEAD_DIM   # int8: 64×64 = 4096
TILE_ELEMS_V = BLOCK_N * HEAD_DIM   # fp16: 64×64 = 4096
HALF_Q_ELEMS = HALF_M * HEAD_DIM    # int8: 64×64 = 4096  (per-WG)
Q_ELEMS      = BLOCK_M * HEAD_DIM   # int8: 128×64 = 8192

LOG2_E = 1.4426950408889634

# ============================================================
# 量化（GPU Triton kernels）
# ============================================================
from .quant import quant_per_block_int8, smooth_and_quant_k_gpu


def quant_q_per_block_gpu_bm128(q: torch.Tensor, sm_scale: float):
    """Q per-block INT8 with block_sz=128 for BLOCK_M=128."""
    return quant_per_block_int8(q, sm_scale=sm_scale * LOG2_E, block_sz=BLOCK_M)


# ============================================================
# CuTe DSL Kernel
# ============================================================

@cute.kernel
def _sage_kernel_v3(
    gQ:      cute.Tensor,   # (B*H*N_Q_BLOCKS,  BLOCK_M,  HEAD_DIM) int8
    gK:      cute.Tensor,   # (B*H*N_KV_BLOCKS, BLOCK_N,  HEAD_DIM) int8
    gV:      cute.Tensor,   # (B*H*N_KV_BLOCKS, HEAD_DIM, BLOCK_N)  fp16
    gO:      cute.Tensor,   # (B*H*N_Q_BLOCKS,  BLOCK_M,  HEAD_DIM) fp32
    gQScale: cute.Tensor,   # (B*H*N_Q_BLOCKS,) fp32
    gKScale: cute.Tensor,   # (B*H, N_KV_BLOCKS) fp32
    n_kv_blocks: cute.Int32,
    n_q_blocks:  cute.Int32,
    mma_qk:      cute.TiledMma,
    mma_pv:      cute.TiledMma,   # FP16×FP16→FP32, rs-mode
    sQ0_layout:  cute.ComposedLayout,   # HALF_M=64 rows, 1 stage
    sQ1_layout:  cute.ComposedLayout,   # HALF_M=64 rows, 1 stage
    sK_layout:   cute.ComposedLayout,   # BLOCK_N rows, 2 stages
    sV_layout:   cute.ComposedLayout,   # HEAD_DIM rows (fp16), 1 stage
    copy_q:        cute.TiledCopy,      # 128-thread Q copy (per-WG)
    copy_K_async:  cute.TiledCopy,      # 256-thread K async (cp.async)
    copy_V_async:  cute.TiledCopy,      # 256-thread V async (cp.async, FP16)
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    smem = SmemAllocator()

    @cute.struct
    class SS:
        sQ0: cute.struct.Align[cute.struct.MemRange[cutlass.Int8,    cute.cosize(sQ0_layout)], 128]
        sQ1: cute.struct.Align[cute.struct.MemRange[cutlass.Int8,    cute.cosize(sQ1_layout)], 128]
        sK:  cute.struct.Align[cute.struct.MemRange[cutlass.Int8,    cute.cosize(sK_layout)],  128]
        sV:  cute.struct.Align[cute.struct.MemRange[cutlass.Float16, cute.cosize(sV_layout)],  128]

    storage = smem.allocate(SS)
    sQ0 = storage.sQ0.get_tensor(sQ0_layout.outer, swizzle=sQ0_layout.inner)
    sQ1 = storage.sQ1.get_tensor(sQ1_layout.outer, swizzle=sQ1_layout.inner)
    sK  = storage.sK.get_tensor(sK_layout.outer,   swizzle=sK_layout.inner)
    sV  = storage.sV.get_tensor(sV_layout.outer,   swizzle=sV_layout.inner)

    sQ0_s0 = cute.slice_(sQ0, (None, None, 0))
    sQ1_s0 = cute.slice_(sQ1, (None, None, 0))
    sK_s0  = cute.slice_(sK,  (None, None, 0))
    sK_s1  = cute.slice_(sK,  (None, None, 1))
    sV_s0  = cute.slice_(sV,  (None, None, 0))

    # per-warpgroup indices
    wg_idx  = cute.arch.make_warp_uniform(tidx // THREADS_PER_WG)
    wg_tidx = tidx % THREADS_PER_WG

    head_flat = bidx // n_q_blocks

    # ----------------------------------------------------------------
    # Load Q: per-WG 128-thread copy
    # WG0 → rows 0..HALF_M-1 → sQ0
    # WG1 → rows HALF_M..BLOCK_M-1 → sQ1
    # ----------------------------------------------------------------
    thr_q  = copy_q.get_slice(wg_tidx)
    gQ_base = gQ.iterator + bidx * Q_ELEMS
    q0_ptr = gQ_base.align(128)
    q1_ptr = (gQ_base + HALF_Q_ELEMS).align(128)

    if wg_idx == 0:
        gQ0 = cute.make_tensor(q0_ptr, cute.make_layout((HALF_M, HEAD_DIM), stride=(HEAD_DIM, 1)))
        cute.copy(copy_q, thr_q.partition_S(gQ0), thr_q.partition_D(sQ0_s0))
    if wg_idx == 1:
        gQ1 = cute.make_tensor(q1_ptr, cute.make_layout((HALF_M, HEAD_DIM), stride=(HEAD_DIM, 1)))
        cute.copy(copy_q, thr_q.partition_S(gQ1), thr_q.partition_D(sQ1_s0))

    # ----------------------------------------------------------------
    # K/V async copy handles (all 256 threads, using tidx)
    # ----------------------------------------------------------------
    thr_K_async = copy_K_async.get_slice(tidx)
    thr_V_async = copy_V_async.get_slice(tidx)
    tKsK_async_s0 = thr_K_async.partition_D(sK_s0)
    tKsK_async_s1 = thr_K_async.partition_D(sK_s1)
    tVsV_async    = thr_V_async.partition_D(sV_s0)

    gK_base = gK.iterator
    gV_base = gV.iterator
    k_ptr_base = gK_base + head_flat * n_kv_blocks * TILE_ELEMS_K
    v_ptr_base = gV_base + head_flat * n_kv_blocks * TILE_ELEMS_V

    # ----------------------------------------------------------------
    # Prologue: K[0] → stage 0, K[1] → stage 1 (all 256 threads)
    # ----------------------------------------------------------------
    k0_ptr = (k_ptr_base + 0).align(128)
    gK_0 = cute.make_tensor(k0_ptr, cute.make_layout((BLOCK_N, HEAD_DIM), stride=(HEAD_DIM, 1)))
    cute.copy(copy_K_async, thr_K_async.partition_S(gK_0), tKsK_async_s0)
    cute.arch.cp_async_commit_group()   # group A: K[0] → stage 0

    if n_kv_blocks > 1:
        k1_ptr = (k_ptr_base + TILE_ELEMS_K).align(128)
        gK_1 = cute.make_tensor(k1_ptr, cute.make_layout((BLOCK_N, HEAD_DIM), stride=(HEAD_DIM, 1)))
        cute.copy(copy_K_async, thr_K_async.partition_S(gK_1), tKsK_async_s1)
        cute.arch.cp_async_commit_group()   # group B: K[1] → stage 1
        cute.arch.cp_async_wait_group(1)    # wait K[0]
    else:
        cute.arch.cp_async_wait_group(0)
    cute.arch.barrier()   # Q (sync copy) and K[0] are ready

    # ----------------------------------------------------------------
    # Per-WG MMA fragments (defined outside if-blocks for proper scoping)
    # Each thread has its own copy; WG0 uses Q0, WG1 uses Q1.
    # ----------------------------------------------------------------
    thr_qk = mma_qk.get_slice(wg_tidx)
    tCrQ0  = mma_qk.make_fragment_A(thr_qk.partition_A(sQ0))
    tCrQ1  = mma_qk.make_fragment_A(thr_qk.partition_A(sQ1))
    tCrK   = mma_qk.make_fragment_B(thr_qk.partition_B(sK))
    acc_S  = cute.make_fragment(thr_qk.partition_shape_C((HALF_M, BLOCK_N)), cutlass.Int32)

    thr_pv = mma_pv.get_slice(wg_tidx)
    rP     = cute.make_fragment(thr_pv.partition_shape_A((HALF_M, BLOCK_N)), cutlass.Float16)
    tCrV   = mma_pv.make_fragment_B(thr_pv.partition_B(sV))
    acc_O  = cute.make_fragment(thr_pv.partition_shape_C((HALF_M, HEAD_DIM)), cutlass.Float32)
    acc_O.fill(0.0)

    row_max = cute.make_fragment(ACC_ROWS_PER_THR, cutlass.Float32)
    row_sum = cute.make_fragment(ACC_ROWS_PER_THR, cutlass.Float32)
    row_max.fill(-cutlass.Float32.inf)
    row_sum.fill(cutlass.Float32(0.0))

    q_scale_val = gQScale[bidx]

    # ----------------------------------------------------------------
    # Main KV loop — barriers OUTSIDE if-blocks, MMA INSIDE
    # ----------------------------------------------------------------
    for n_tile in range(n_kv_blocks):
        cur_stage = n_tile % 2

        # --- WG0: QK GEMM using sQ0 ---
        if wg_idx == 0:
            acc_S.fill(0)
            mma_qk.set(warpgroup.Field.ACCUMULATE, False)
            warpgroup.fence()
            for k in cutlass.range_constexpr(cute.size(tCrQ0, mode=[2])):
                cute.gemm(mma_qk, acc_S, tCrQ0[None, None, k, 0], tCrK[None, None, k, cur_stage], acc_S)
                mma_qk.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()
            warpgroup.wait_group(0)

        # --- WG1: QK GEMM using sQ1 ---
        if wg_idx == 1:
            acc_S.fill(0)
            mma_qk.set(warpgroup.Field.ACCUMULATE, False)
            warpgroup.fence()
            for k in cutlass.range_constexpr(cute.size(tCrQ1, mode=[2])):
                cute.gemm(mma_qk, acc_S, tCrQ1[None, None, k, 0], tCrK[None, None, k, cur_stage], acc_S)
                mma_qk.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()
            warpgroup.wait_group(0)

        # --- ALL: Issue V[n] async BEFORE softmax to maximize overlap
        # V goes to sV (not sK), so safe to issue even while WG1 may still be in GEMM.
        # After this commit_group, group A = V[n].
        v_ptr = (v_ptr_base + n_tile * TILE_ELEMS_V).align(128)
        gV_n = cute.make_tensor(v_ptr, cute.make_layout((HEAD_DIM, BLOCK_N), stride=(BLOCK_N, 1)))
        cute.copy(copy_V_async, thr_V_async.partition_S(gV_n), tVsV_async)
        cute.arch.cp_async_commit_group()   # group A: V[n]

        # --- WG0/WG1: Online softmax (overlaps V cp.async)
        # Both WGs read from their own acc_S (in registers), no smem access here.
        # Key: K[n+2] is NOT issued yet — wait until after V-wait barrier (see below)
        # so that sK[cur_stage] is safe to overwrite only after both WGs finish reading it.
        if wg_idx == 0:
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

        if wg_idx == 1:
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

        # --- ALL: Wait V ready (group A), then issue K[n+2] → cur_stage
        # At this barrier, both WGs have finished their WGMMA (warpgroup.wait_group(0)
        # called in each branch above), so sK[cur_stage] is safe to overwrite.
        cute.arch.cp_async_wait_group(0)   # wait V (only group A is in flight, wait all)
        cute.arch.barrier()   # all 256 threads: V visible AND both WGs done reading sK

        # --- ALL: Issue K[n+2] async → overwrites sK[cur_stage] (safe after barrier) ---
        if n_tile + 2 < n_kv_blocks:
            k_n2_ptr = (k_ptr_base + (n_tile + 2) * TILE_ELEMS_K).align(128)
            gK_n2 = cute.make_tensor(k_n2_ptr, cute.make_layout((BLOCK_N, HEAD_DIM), stride=(HEAD_DIM, 1)))
            if cur_stage == 0:
                cute.copy(copy_K_async, thr_K_async.partition_S(gK_n2), tKsK_async_s0)
            else:
                cute.copy(copy_K_async, thr_K_async.partition_S(gK_n2), tKsK_async_s1)
            cute.arch.cp_async_commit_group()   # group A: K[n+2]

        # --- WG0: PV GEMM (rs-mode), overlaps K[n+2] cp.async ---
        if wg_idx == 0:
            mma_pv.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.fence()
            for k in cutlass.range_constexpr(cute.size(rP, mode=[2])):
                cute.gemm(mma_pv, acc_O, rP[None, None, k], tCrV[None, None, k, 0], acc_O)
            warpgroup.commit_group()
            warpgroup.wait_group(0)

        # --- WG1: PV GEMM (rs-mode), overlaps K[n+2] cp.async ---
        if wg_idx == 1:
            mma_pv.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.fence()
            for k in cutlass.range_constexpr(cute.size(rP, mode=[2])):
                cute.gemm(mma_pv, acc_O, rP[None, None, k], tCrV[None, None, k, 0], acc_O)
            warpgroup.commit_group()
            warpgroup.wait_group(0)

        # --- ALL: Wait K[n+2] after PV GEMM ---
        if n_tile + 2 < n_kv_blocks:
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()   # all 256 threads: K[n+2] visible in smem

    # ----------------------------------------------------------------
    # Epilogue: normalize + store FP32
    # WG0 → rows 0..HALF_M-1; WG1 → rows HALF_M..BLOCK_M-1
    # ----------------------------------------------------------------
    gO_base = gO.iterator + bidx * Q_ELEMS

    if wg_idx == 0:
        o0_ptr = gO_base
        gO0 = cute.make_tensor(o0_ptr, cute.make_layout((HALF_M, HEAD_DIM), stride=(HEAD_DIM, 1)))
        tOgO = thr_pv.partition_C(gO0)
        for i in cutlass.range_constexpr(cute.size(acc_O)):
            r = (i // 2) % ACC_ROWS_PER_THR
            acc_O[i] = acc_O[i] / (row_sum[r] + cutlass.Float32(1e-6))
        cute.basic_copy(acc_O, tOgO)

    if wg_idx == 1:
        o1_ptr = gO_base + HALF_M * HEAD_DIM
        gO1 = cute.make_tensor(o1_ptr, cute.make_layout((HALF_M, HEAD_DIM), stride=(HEAD_DIM, 1)))
        tOgO = thr_pv.partition_C(gO1)
        for i in cutlass.range_constexpr(cute.size(acc_O)):
            r = (i // 2) % ACC_ROWS_PER_THR
            acc_O[i] = acc_O[i] / (row_sum[r] + cutlass.Float32(1e-6))
        cute.basic_copy(acc_O, tOgO)


@cute.jit
def _sage_jit_v3(
    gQ:      cute.Tensor,
    gK:      cute.Tensor,
    gV:      cute.Tensor,
    gO:      cute.Tensor,
    gQScale: cute.Tensor,
    gKScale: cute.Tensor,
    n_kv_blocks: cute.Int32,
    n_q_blocks:  cute.Int32,
    n_cta:       cute.Int32,
):
    # QK GEMM: INT8×INT8→INT32, (1,1,1), per-warpgroup HALF_M=64 rows
    mma_qk = sm90_utils.make_trivial_tiled_mma(
        cutlass.Int8, cutlass.Int8,
        utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        cutlass.Int32, (1, 1, 1),
        tiler_mn=(HALF_M, BLOCK_N),
        a_source=warpgroup.OperandSource.SMEM,
    )
    # PV GEMM: FP16×FP16→FP32, rs-mode, per-warpgroup HALF_M=64 rows
    mma_pv = sm90_utils.make_trivial_tiled_mma(
        cutlass.Float16, cutlass.Float16,
        utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        cutlass.Float32, (1, 1, 1),
        tiler_mn=(HALF_M, HEAD_DIM),
        a_source=warpgroup.OperandSource.RMEM,
    )

    # sQ0, sQ1: (HALF_M=64, BLOCK_N=64, HEAD_DIM=64) with 1 stage
    sQ0_layout = sm90_utils.make_smem_layout_a(
        utils.LayoutEnum.ROW_MAJOR, (HALF_M, BLOCK_N, HEAD_DIM), cutlass.Int8, 1)
    sQ1_layout = sm90_utils.make_smem_layout_a(
        utils.LayoutEnum.ROW_MAJOR, (HALF_M, BLOCK_N, HEAD_DIM), cutlass.Int8, 1)
    # sK: BLOCK_N rows, 2 stages
    sK_layout = sm90_utils.make_smem_layout_b(
        utils.LayoutEnum.ROW_MAJOR, (HALF_M, BLOCK_N, HEAD_DIM), cutlass.Int8, 2)
    # sV: HEAD_DIM rows (fp16), 1 stage
    sV_layout = sm90_utils.make_smem_layout_b(
        utils.LayoutEnum.ROW_MAJOR, (HALF_M, HEAD_DIM, BLOCK_N), cutlass.Float16, 1)

    # 128-thread Q copy (per-WG, wg_tidx 0-127)
    # INT8, HALF_M=64 rows × HEAD_DIM=64 cols = 4096 bytes
    # (32,4) × (1,16): 32×4=128 threads, 16 INT8 per thread = 128-bit
    ca_int8 = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Int8, num_bits_per_copy=128)
    copy_q = cute.make_tiled_copy_tv(ca_int8,
        cute.make_layout((32, 4), stride=(4, 1)), cute.make_layout((1, 16)))

    # K async copy: 256 threads, INT8
    # (64,4) × (1,16): 64×4=256 threads, 16 INT8 per thread = 128-bit
    ca_k_async = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.Int8, num_bits_per_copy=128)
    copy_K_async = cute.make_tiled_copy_tv(ca_k_async,
        cute.make_layout((64, 4), stride=(4, 1)), cute.make_layout((1, 16)))

    # V async copy: 256 threads, FP16
    # (32,8) × (1,8): 32×8=256 threads, 8 FP16 per thread = 128-bit
    ca_v_async = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.Float16, num_bits_per_copy=128)
    copy_V_async = cute.make_tiled_copy_tv(ca_v_async,
        cute.make_layout((32, 8), stride=(8, 1)), cute.make_layout((1, 8)))

    _sage_kernel_v3(
        gQ, gK, gV, gO, gQScale, gKScale,
        n_kv_blocks, n_q_blocks,
        mma_qk, mma_pv,
        sQ0_layout, sQ1_layout, sK_layout, sV_layout,
        copy_q, copy_K_async, copy_V_async,
    ).launch(grid=(n_cta, 1, 1), block=(NUM_THREADS, 1, 1))


# ============================================================
# AOT 编译缓存
# ============================================================
_compiled_kernels: dict = {}


def _get_compiled(B, H, N, D, kv_len, q_int8, k_int8_tiles, v_tiles, out,
                  q_scale_flat, k_scale_2d, N_KV_BLOCKS, N_Q_BLOCKS, N_CTAs):
    key = (B, H, N, D, kv_len)
    if key not in _compiled_kernels:
        _compiled_kernels[key] = cute.compile(
            _sage_jit_v3,
            from_dlpack(q_int8, assumed_align=128),
            from_dlpack(k_int8_tiles, assumed_align=128),
            from_dlpack(v_tiles, assumed_align=128),
            from_dlpack(out),
            from_dlpack(q_scale_flat),
            from_dlpack(k_scale_2d),
            cute.Int32(N_KV_BLOCKS),
            cute.Int32(N_Q_BLOCKS),
            cute.Int32(N_CTAs),
        )
    return _compiled_kernels[key]


# ============================================================
# Public API
# ============================================================
def sageattn_cutedsl_v3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale: float = None,
    smooth_k: bool = True,
) -> torch.Tensor:
    """
    SageAttention CuTe DSL v3 前向传播 (SM90a WGMMA, BLOCK_M=128).

    v3 优化: BLOCK_M 从 64 增大到 128，减少 N_CTAs，提升 K/V tile 复用率。
    使用 split-warpgroup (1,1,1)×2 设计代替 buggy atom_layout=(2,1,1)。

    参数:
        q, k, v: (B, H, N, D) float16, CUDA
        is_causal: 因果 mask (暂未实现)
        sm_scale: softmax 缩放 (默认 D^{-0.5})
        smooth_k: K Smoothing
    """
    assert q.is_cuda and q.dtype == torch.float16
    B, H, N, D = q.shape
    assert D == HEAD_DIM, f"sageattn_cutedsl_v3: 只支持 head_dim={HEAD_DIM}, got {D}"
    assert N % BLOCK_M == 0, f"N={N} 必须是 BLOCK_M={BLOCK_M} 的整数倍"
    assert N % BLOCK_N == 0, f"N={N} 必须是 BLOCK_N={BLOCK_N} 的整数倍"

    if sm_scale is None:
        sm_scale = D ** -0.5

    # K 量化
    if smooth_k:
        k_int8, k_scale, km = smooth_and_quant_k_gpu(k, block_sz=BLOCK_N)
    else:
        k_int8, k_scale = quant_per_block_int8(k, sm_scale=1.0, block_sz=BLOCK_N)

    # Q 量化 (block_sz=128)
    q_int8, q_scale = quant_q_per_block_gpu_bm128(q, sm_scale)

    N_Q_BLOCKS  = N // BLOCK_M
    N_KV_BLOCKS = N // BLOCK_N
    N_CTAs      = B * H * N_Q_BLOCKS

    # Tile-contiguous 重排
    k_tiles = k_int8.view(B, H, N_KV_BLOCKS, BLOCK_N, D)
    k_tiles = k_tiles.reshape(B * H, N_KV_BLOCKS, BLOCK_N, D).contiguous()

    # V: transposed (HEAD_DIM, BLOCK_N) layout for ROW_MAJOR K-major smem
    v_tiles = v.view(B, H, N_KV_BLOCKS, BLOCK_N, D)
    v_tiles = v_tiles.permute(0, 1, 2, 4, 3)   # (B, H, N_KV_BLOCKS, D, BLOCK_N)
    v_tiles = v_tiles.reshape(B * H, N_KV_BLOCKS, D, BLOCK_N).contiguous()

    q_tiles = q_int8.view(B, H, N_Q_BLOCKS, BLOCK_M, D)
    q_tiles = q_tiles.reshape(B * H * N_Q_BLOCKS, BLOCK_M, D).contiguous()

    out = torch.empty(B * H * N_Q_BLOCKS, BLOCK_M, D, dtype=torch.float32, device=q.device)
    q_scale_flat = q_scale.reshape(B * H * N_Q_BLOCKS).contiguous()
    k_scale_2d   = k_scale.reshape(B * H, N_KV_BLOCKS).contiguous()

    compiled = _get_compiled(
        B, H, N, D, N,
        q_tiles, k_tiles, v_tiles, out,
        q_scale_flat, k_scale_2d,
        N_KV_BLOCKS, N_Q_BLOCKS, N_CTAs,
    )
    compiled(
        from_dlpack(q_tiles, assumed_align=128),
        from_dlpack(k_tiles, assumed_align=128),
        from_dlpack(v_tiles, assumed_align=128),
        from_dlpack(out),
        from_dlpack(q_scale_flat),
        from_dlpack(k_scale_2d),
        cute.Int32(N_KV_BLOCKS),
        cute.Int32(N_Q_BLOCKS),
        cute.Int32(N_CTAs),
    )
    torch.cuda.synchronize()

    return out.view(B, H, N, D).to(torch.float32)
