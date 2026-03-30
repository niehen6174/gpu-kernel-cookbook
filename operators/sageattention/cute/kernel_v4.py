"""
SageAttention CuTe DSL 实现 (Hopper SM90a) — v4: FP8 V GMEM 带宽优化

v4 相对 v3 的优化：
  V GMEM 带宽减半: FP16(8KB/tile) → FP8(4KB/tile) = 2× GMEM 节省
  V 以 FP8 E4M3FN 量化存储，cp.async 加载 FP8 字节到 sV_f8 (4KB smem)
  sV_f8 → sV_f16 在线转换（256 线程协作，vectorized .to(Float32).to(Float16)）
  FP16 sV_f16 → FP16 WGMMA（与 v3 相同的 PV GEMM 路径）

内存使用变化（vs v3）：
  v3: sQ0(4KB) + sQ1(4KB) + sK(8KB×2stage) + sV(8KB) = 24KB
  v4: sQ0(4KB) + sQ1(4KB) + sK(8KB×2stage) + sV_f8(4KB) + sV_f16(8KB) = 28KB
  smem 增加 4KB 换取 GMEM 带宽减半（H20: 228KB 共享内存，28KB 仍绰绰有余）

V dequant 融合（P-fused-vscale, 精确）：
  gV 量化: v_fp8 = round(V_real / v_scale)，范围 [-448, 448]
  softmax 时: rP[i] = Float16(exp(...) * v_scale)
  PV GEMM:  = rP × V_fp8_f16 = (P × v_scale) × (V_real / v_scale) = P × V_real  ✓

CuTe DSL 关键约束：
  - FP8 dlpack 限制：from_dlpack(fp8_tensor) 不支持 → 以 int8 传入，内核内 recast
  - FP8 scalar 赋值约束：nvgpu.cvt_fptrunc 需 ≥4 元素对齐 vector
    → 不能逐元素填充 FP8 fragment
  - FP8→FP16 单步: .to(Float16) 会生成 arith.extf(f16→f16) 错误
    → 需要 .to(Float32).to(Float16) 两步（F32 作为中间类型）
  - Int8 async copy 需 assumed_align=128（MLIR 静态对齐验证）
    → from_dlpack(v_int8, assumed_align=128)

相比 v3 代码改动摘要：
  - SS.sV dtype: Float16 → Float8E4M3FN；新增 SS.sV_f16 (Float16)
  - gV: Float16 → Int8 (FP8 bit patterns)；新增 gVScale
  - copy_V_async: Float16 atom → Int8 atom；新增 copy_f8_reg / copy_f16_reg
  - sV_f8 → sV_f16 转换步骤：在 cp_async_wait 之后、PV GEMM 之前
  - rP[i] = Float16(p_val * v_scale_val)（融合 V dequant 到 softmax）
  - Python wrapper: 新增 quant_v_per_tile_fp8_gpu + v_fp8_int8 / v_scale 传入
"""

import os
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
# 静态超参数（与 v3 相同）
# ============================================================
BLOCK_M        = 128
HALF_M         = 64
BLOCK_N        = 64
HEAD_DIM       = 64
NUM_THREADS    = 256
THREADS_PER_WG = 128

ACC_FRAG_SIZE    = 32
ACC_ROWS_PER_THR = 2

TILE_ELEMS_K   = BLOCK_N * HEAD_DIM    # int8: 4096 bytes
TILE_ELEMS_V   = HEAD_DIM * BLOCK_N    # fp8: 4096 bytes (1 byte/elem, half of FP16)
HALF_Q_ELEMS   = HALF_M * HEAD_DIM     # 4096 bytes
Q_ELEMS        = BLOCK_M * HEAD_DIM    # 8192 bytes

LOG2_E = 1.4426950408889634

# ============================================================
# 量化
# ============================================================
from .quant import quant_per_block_int8, smooth_and_quant_k_gpu, quant_v_per_tile_fp8_gpu


def quant_q_per_block_gpu_bm128(q: torch.Tensor, sm_scale: float):
    return quant_per_block_int8(q, sm_scale=sm_scale * LOG2_E, block_sz=BLOCK_M)


# ============================================================
# CuTe DSL Kernel
# ============================================================

@cute.kernel
def _sage_kernel_v4(
    gQ:      cute.Tensor,   # (B*H*N_Q_BLOCKS, BLOCK_M,  HEAD_DIM) int8
    gK:      cute.Tensor,   # (B*H, N_KV_BLOCKS, BLOCK_N, HEAD_DIM) int8
    gV:      cute.Tensor,   # (B*H, N_KV_BLOCKS, HEAD_DIM, BLOCK_N) int8 (FP8 E4M3FN bit pattern)
    gO:      cute.Tensor,   # (B*H*N_Q_BLOCKS, BLOCK_M,  HEAD_DIM) fp32
    gQScale: cute.Tensor,   # (B*H*N_Q_BLOCKS,) fp32
    gKScale: cute.Tensor,   # (B*H, N_KV_BLOCKS) fp32
    gVScale: cute.Tensor,   # (B*H, N_KV_BLOCKS) fp32
    n_kv_blocks: cute.Int32,
    n_q_blocks:  cute.Int32,
    mma_qk:      cute.TiledMma,           # INT8×INT8→INT32, smem-mode A
    mma_pv:      cute.TiledMma,           # FP16×FP16→FP32, rs-mode A
    sQ0_layout:  cute.ComposedLayout,
    sQ1_layout:  cute.ComposedLayout,
    sK_layout:   cute.ComposedLayout,
    sV_f8_layout:  cute.Layout,           # FP8 V smem flat layout (4KB, no swizzle)
    sV_f16_layout: cute.ComposedLayout,   # FP16 V smem for WGMMA (8KB, swizzled)
    copy_q:          cute.TiledCopy,        # 128-thread Q smem copy
    copy_K_async:    cute.TiledCopy,        # 256-thread K async (cp.async, Int8)
    copy_V_async:    cute.TiledCopy,        # 256-thread V async (cp.async, Int8 carrying FP8)
    copy_f8_reg:     cute.TiledCopy,        # FP8 smem→reg (64-bit, 8 FP8/thr), 128-thread
    copy_f16_reg:    cute.TiledCopy,        # FP16 reg→smem (128-bit, 8 FP16/thr), 128-thread
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    smem = SmemAllocator()

    @cute.struct
    class SS:
        sQ0:    cute.struct.Align[cute.struct.MemRange[cutlass.Int8,         cute.cosize(sQ0_layout)],   128]
        sQ1:    cute.struct.Align[cute.struct.MemRange[cutlass.Int8,         cute.cosize(sQ1_layout)],   128]
        sK:     cute.struct.Align[cute.struct.MemRange[cutlass.Int8,         cute.cosize(sK_layout)],    128]
        sV_f8:    cute.struct.Align[cute.struct.MemRange[cutlass.Float8E4M3FN, cute.cosize(sV_f8_layout)],   128]
        sV_f16_0: cute.struct.Align[cute.struct.MemRange[cutlass.Float16,      cute.cosize(sV_f16_layout)], 128]
        sV_f16_1: cute.struct.Align[cute.struct.MemRange[cutlass.Float16,      cute.cosize(sV_f16_layout)], 128]

    storage = smem.allocate(SS)
    sQ0    = storage.sQ0.get_tensor(sQ0_layout.outer,       swizzle=sQ0_layout.inner)
    sQ1    = storage.sQ1.get_tensor(sQ1_layout.outer,       swizzle=sQ1_layout.inner)
    sK     = storage.sK.get_tensor(sK_layout.outer,         swizzle=sK_layout.inner)
    # sV_f8 uses flat (no-swizzle) layout — it's only an intermediate buffer, not for WGMMA
    # This ensures both write (recast Int8) and read (FP8 copy) use the same linear address map
    sV_f8  = storage.sV_f8.get_tensor(sV_f8_layout)
    sV_f16_0 = storage.sV_f16_0.get_tensor(sV_f16_layout.outer, swizzle=sV_f16_layout.inner)
    sV_f16_1 = storage.sV_f16_1.get_tensor(sV_f16_layout.outer, swizzle=sV_f16_layout.inner)

    sQ0_s0    = cute.slice_(sQ0,    (None, None, 0))
    sQ1_s0    = cute.slice_(sQ1,    (None, None, 0))
    sK_s0     = cute.slice_(sK,     (None, None, 0))
    sK_s1     = cute.slice_(sK,     (None, None, 1))
    # sV_f8 is 2D flat: no stage slice needed
    sV_f8_s0  = sV_f8   # (HEAD_DIM, BLOCK_N) flat
    sV_f16_s0_wg0 = cute.slice_(sV_f16_0, (None, None, 0))   # WG0's private FP16 V buffer
    sV_f16_s0_wg1 = cute.slice_(sV_f16_1, (None, None, 0))   # WG1's private FP16 V buffer

    wg_idx  = cute.arch.make_warp_uniform(tidx // THREADS_PER_WG)
    wg_tidx = tidx % THREADS_PER_WG

    head_flat = bidx // n_q_blocks

    # ----------------------------------------------------------------
    # Load Q (per-WG 128-thread copy, sync)
    # ----------------------------------------------------------------
    thr_q   = copy_q.get_slice(wg_tidx)
    gQ_base = gQ.iterator + bidx * Q_ELEMS
    q0_ptr  = gQ_base.align(128)
    q1_ptr  = (gQ_base + HALF_Q_ELEMS).align(128)

    if wg_idx == 0:
        gQ0 = cute.make_tensor(q0_ptr, cute.make_layout((HALF_M, HEAD_DIM), stride=(HEAD_DIM, 1)))
        cute.copy(copy_q, thr_q.partition_S(gQ0), thr_q.partition_D(sQ0_s0))
    if wg_idx == 1:
        gQ1 = cute.make_tensor(q1_ptr, cute.make_layout((HALF_M, HEAD_DIM), stride=(HEAD_DIM, 1)))
        cute.copy(copy_q, thr_q.partition_S(gQ1), thr_q.partition_D(sQ1_s0))

    # ----------------------------------------------------------------
    # K/V async copy handles (all 256 threads)
    # gV is Int8 (FP8 bit patterns) with assumed_align=128
    # ----------------------------------------------------------------
    thr_K_async = copy_K_async.get_slice(tidx)
    thr_V_async = copy_V_async.get_slice(tidx)
    tKsK_async_s0 = thr_K_async.partition_D(sK_s0)
    tKsK_async_s1 = thr_K_async.partition_D(sK_s1)
    # V async writes to sV_f8 via recast (Int8 atom writes to Int8 view of flat FP8 smem)
    # sV_f8 is flat (no swizzle), so recast produces consistent linear addressing
    # The FP8→FP16 conversion reads from the same flat sV_f8 → no swizzle mismatch
    sV_f8_as_i8   = cute.recast_tensor(sV_f8_s0, cutlass.Int8)
    tVsVf8_async  = thr_V_async.partition_D(sV_f8_as_i8)

    # FP8→FP16 smem-to-smem conversion handles (per-WG: each WG owns its sV_f16 buffer)
    # Each WG reads ALL sV_f8 elements (128 threads × 32 elements = 4096) into its own sV_f16.
    # This ensures warpgroup.fence() before WGMMA covers ALL stores to that WG's sV_f16.
    thr_f8_reg_wg  = copy_f8_reg.get_slice(wg_tidx)    # 128-thread partition
    thr_f16_reg_wg = copy_f16_reg.get_slice(wg_tidx)   # 128-thread partition
    # Pre-allocate conversion fragment partitions and registers OUTSIDE the loop
    gV_f8_src_pre      = thr_f8_reg_wg.partition_S(sV_f8_s0)           # same for both WGs (read-only)
    gV_f16_dst_pre_wg0 = thr_f16_reg_wg.partition_D(sV_f16_s0_wg0)    # WG0 writes to sV_f16_0
    gV_f16_dst_pre_wg1 = thr_f16_reg_wg.partition_D(sV_f16_s0_wg1)    # WG1 writes to sV_f16_1
    reg_f8_pre  = cute.make_fragment_like(gV_f8_src_pre)
    reg_f16_pre = cute.make_fragment_like(gV_f16_dst_pre_wg0)

    gK_base    = gK.iterator
    gV_base    = gV.iterator        # Int8 base pointer
    k_ptr_base = gK_base + head_flat * n_kv_blocks * TILE_ELEMS_K
    v_ptr_base = gV_base + head_flat * n_kv_blocks * TILE_ELEMS_V

    # ----------------------------------------------------------------
    # Prologue: K[0] → stage 0, K[1] → stage 1
    # ----------------------------------------------------------------
    k0_ptr = (k_ptr_base + 0).align(128)
    gK_0 = cute.make_tensor(k0_ptr, cute.make_layout((BLOCK_N, HEAD_DIM), stride=(HEAD_DIM, 1)))
    cute.copy(copy_K_async, thr_K_async.partition_S(gK_0), tKsK_async_s0)
    cute.arch.cp_async_commit_group()

    if n_kv_blocks > 1:
        k1_ptr = (k_ptr_base + TILE_ELEMS_K).align(128)
        gK_1 = cute.make_tensor(k1_ptr, cute.make_layout((BLOCK_N, HEAD_DIM), stride=(HEAD_DIM, 1)))
        cute.copy(copy_K_async, thr_K_async.partition_S(gK_1), tKsK_async_s1)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(1)
    else:
        cute.arch.cp_async_wait_group(0)
    cute.arch.barrier()

    # ----------------------------------------------------------------
    # Per-WG MMA fragments
    # ----------------------------------------------------------------
    thr_qk = mma_qk.get_slice(wg_tidx)
    tCrQ0  = mma_qk.make_fragment_A(thr_qk.partition_A(sQ0))
    tCrQ1  = mma_qk.make_fragment_A(thr_qk.partition_A(sQ1))
    tCrK   = mma_qk.make_fragment_B(thr_qk.partition_B(sK))
    acc_S  = cute.make_fragment(thr_qk.partition_shape_C((HALF_M, BLOCK_N)), cutlass.Int32)

    thr_pv = mma_pv.get_slice(wg_tidx)
    # rP stores P * v_scale (fused V dequant, PV GEMM = P × V_real)
    rP     = cute.make_fragment(thr_pv.partition_shape_A((HALF_M, BLOCK_N)), cutlass.Float16)
    # tCrV_wg0/wg1 read from each WG's private sV_f16 (written only by that WG's threads)
    # This ensures warpgroup.fence() fully covers all stores visible to that WG's WGMMA.
    tCrV_wg0 = mma_pv.make_fragment_B(thr_pv.partition_B(sV_f16_0))
    tCrV_wg1 = mma_pv.make_fragment_B(thr_pv.partition_B(sV_f16_1))
    acc_O  = cute.make_fragment(thr_pv.partition_shape_C((HALF_M, HEAD_DIM)), cutlass.Float32)
    acc_O.fill(0.0)

    row_max = cute.make_fragment(ACC_ROWS_PER_THR, cutlass.Float32)
    row_sum = cute.make_fragment(ACC_ROWS_PER_THR, cutlass.Float32)
    row_max.fill(-cutlass.Float32.inf)
    row_sum.fill(cutlass.Float32(0.0))

    q_scale_val = gQScale[bidx]

    # ----------------------------------------------------------------
    # Main KV loop
    # ----------------------------------------------------------------
    for n_tile in range(n_kv_blocks):
        cur_stage = n_tile % 2

        # --- WG0: QK GEMM ---
        if wg_idx == 0:
            acc_S.fill(0)
            mma_qk.set(warpgroup.Field.ACCUMULATE, False)
            warpgroup.fence()
            for k in cutlass.range_constexpr(cute.size(tCrQ0, mode=[2])):
                cute.gemm(mma_qk, acc_S, tCrQ0[None, None, k, 0], tCrK[None, None, k, cur_stage], acc_S)
                mma_qk.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()
            warpgroup.wait_group(0)

        # --- WG1: QK GEMM ---
        if wg_idx == 1:
            acc_S.fill(0)
            mma_qk.set(warpgroup.Field.ACCUMULATE, False)
            warpgroup.fence()
            for k in cutlass.range_constexpr(cute.size(tCrQ1, mode=[2])):
                cute.gemm(mma_qk, acc_S, tCrQ1[None, None, k, 0], tCrK[None, None, k, cur_stage], acc_S)
                mma_qk.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()
            warpgroup.wait_group(0)

        # --- ALL: Issue V[n] FP8 async copy → sV_f8 (overlaps softmax) ---
        v_ptr = (v_ptr_base + n_tile * TILE_ELEMS_V).align(128)
        gV_n  = cute.make_tensor(v_ptr, cute.make_layout((HEAD_DIM, BLOCK_N), stride=(BLOCK_N, 1)))
        cute.copy(copy_V_async, thr_V_async.partition_S(gV_n), tVsVf8_async)
        cute.arch.cp_async_commit_group()

        # --- WG0: Online softmax (overlaps V cp.async)
        # rP[i] = Float16(exp2(...) * v_scale_val) — fused V dequant
        if wg_idx == 0:
            k_scale_val = gKScale[head_flat, n_tile]
            v_scale_val = gVScale[head_flat, n_tile]
            dequant     = q_scale_val * k_scale_val

            row_max_new = cute.make_fragment(ACC_ROWS_PER_THR, cutlass.Float32)
            row_max_new.fill(-cutlass.Float32.inf)
            for i in cutlass.range_constexpr(ACC_FRAG_SIZE):
                r   = (i // 2) % ACC_ROWS_PER_THR
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
                r     = (i // 2) % ACC_ROWS_PER_THR
                p_val = cute.math.exp2(cutlass.Float32(acc_S[i]) * dequant - row_max_new[r], fastmath=True)
                rP[i] = cutlass.Float16(p_val * v_scale_val)
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
                alpha      = cute.math.exp2(row_max[r] - row_max_new[r], fastmath=True)
                row_sum[r] = row_sum[r] * alpha + row_sum_new[r]
                row_max[r] = row_max_new[r]

        # --- WG1: Online softmax ---
        if wg_idx == 1:
            k_scale_val = gKScale[head_flat, n_tile]
            v_scale_val = gVScale[head_flat, n_tile]
            dequant     = q_scale_val * k_scale_val

            row_max_new = cute.make_fragment(ACC_ROWS_PER_THR, cutlass.Float32)
            row_max_new.fill(-cutlass.Float32.inf)
            for i in cutlass.range_constexpr(ACC_FRAG_SIZE):
                r   = (i // 2) % ACC_ROWS_PER_THR
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
                r     = (i // 2) % ACC_ROWS_PER_THR
                p_val = cute.math.exp2(cutlass.Float32(acc_S[i]) * dequant - row_max_new[r], fastmath=True)
                rP[i] = cutlass.Float16(p_val * v_scale_val)
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
                alpha      = cute.math.exp2(row_max[r] - row_max_new[r], fastmath=True)
                row_sum[r] = row_sum[r] * alpha + row_sum_new[r]
                row_max[r] = row_max_new[r]

        # --- ALL: Wait V[n] FP8 ready ---
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()   # sV_f8 visible to all 256 threads

        # --- WG0: FP8 smem → sV_f16_0 (WG0's private FP16 V buffer) ---
        # Each WG independently writes ALL 4096 elements of its own sV_f16 buffer.
        # warpgroup.fence() directly after the conversion with no cross-thread code
        # in between ensures all WG0 stores are visible to WG0's WGMMA.
        if wg_idx == 0:
            cute.copy(copy_f8_reg, gV_f8_src_pre, reg_f8_pre)
            reg_f16_pre.store(reg_f8_pre.load().to(cutlass.Float32).to(cutlass.Float16))
            cute.copy(copy_f16_reg, reg_f16_pre, gV_f16_dst_pre_wg0)

        # --- WG1: FP8 smem → sV_f16_1 (WG1's private FP16 V buffer) ---
        if wg_idx == 1:
            cute.copy(copy_f8_reg, gV_f8_src_pre, reg_f8_pre)
            reg_f16_pre.store(reg_f8_pre.load().to(cutlass.Float32).to(cutlass.Float16))
            cute.copy(copy_f16_reg, reg_f16_pre, gV_f16_dst_pre_wg1)

        # Barrier: ensure both WGs have written sV_f16_0 and sV_f16_1 before WGMMA.
        # The per-WG warpgroup.fence() inside each GEMM block then provides the
        # WGMMA-proxy ordering for that WG's stores to its own buffer.
        cute.arch.barrier()

        # --- ALL: Issue K[n+2] async (overlaps PV GEMM) ---
        if n_tile + 2 < n_kv_blocks:
            k_n2_ptr = (k_ptr_base + (n_tile + 2) * TILE_ELEMS_K).align(128)
            gK_n2    = cute.make_tensor(k_n2_ptr, cute.make_layout((BLOCK_N, HEAD_DIM), stride=(HEAD_DIM, 1)))
            if cur_stage == 0:
                cute.copy(copy_K_async, thr_K_async.partition_S(gK_n2), tKsK_async_s0)
            else:
                cute.copy(copy_K_async, thr_K_async.partition_S(gK_n2), tKsK_async_s1)
            cute.arch.cp_async_commit_group()

        # --- WG0: PV GEMM (rs-mode), rP × sV_f16_0 ---
        if wg_idx == 0:
            mma_pv.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.fence()   # orders WG0's own stores to sV_f16_0
            for k in cutlass.range_constexpr(cute.size(rP, mode=[2])):
                cute.gemm(mma_pv, acc_O, rP[None, None, k], tCrV_wg0[None, None, k, 0], acc_O)
            warpgroup.commit_group()
            warpgroup.wait_group(0)

        # --- WG1: PV GEMM ---
        if wg_idx == 1:
            mma_pv.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.fence()   # orders WG1's own stores to sV_f16_1
            for k in cutlass.range_constexpr(cute.size(rP, mode=[2])):
                cute.gemm(mma_pv, acc_O, rP[None, None, k], tCrV_wg1[None, None, k, 0], acc_O)
            warpgroup.commit_group()
            warpgroup.wait_group(0)

        # --- ALL: Wait K[n+2] (and barrier to synchronize WGs for next iteration) ---
        if n_tile + 2 < n_kv_blocks:
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
        else:
            # No K[n+2] was issued, but we still need to synchronize both WGs
            # before the next iteration uses sV_f16_{0,1} for conversion again.
            cute.arch.barrier()

    # ----------------------------------------------------------------
    # Epilogue: normalize + store FP32
    # ----------------------------------------------------------------
    gO_base = gO.iterator + bidx * Q_ELEMS

    if wg_idx == 0:
        o0_ptr = gO_base
        gO0    = cute.make_tensor(o0_ptr, cute.make_layout((HALF_M, HEAD_DIM), stride=(HEAD_DIM, 1)))
        tOgO   = thr_pv.partition_C(gO0)
        for i in cutlass.range_constexpr(cute.size(acc_O)):
            r        = (i // 2) % ACC_ROWS_PER_THR
            acc_O[i] = acc_O[i] / (row_sum[r] + cutlass.Float32(1e-6))
        cute.basic_copy(acc_O, tOgO)

    if wg_idx == 1:
        o1_ptr = gO_base + HALF_M * HEAD_DIM
        gO1    = cute.make_tensor(o1_ptr, cute.make_layout((HALF_M, HEAD_DIM), stride=(HEAD_DIM, 1)))
        tOgO   = thr_pv.partition_C(gO1)
        for i in cutlass.range_constexpr(cute.size(acc_O)):
            r        = (i // 2) % ACC_ROWS_PER_THR
            acc_O[i] = acc_O[i] / (row_sum[r] + cutlass.Float32(1e-6))
        cute.basic_copy(acc_O, tOgO)


@cute.jit
def _sage_jit_v4(
    gQ:      cute.Tensor,
    gK:      cute.Tensor,
    gV:      cute.Tensor,   # Int8 (FP8 bit patterns), assumed_align=128
    gO:      cute.Tensor,
    gQScale: cute.Tensor,
    gKScale: cute.Tensor,
    gVScale: cute.Tensor,
    n_kv_blocks: cute.Int32,
    n_q_blocks:  cute.Int32,
    n_cta:       cute.Int32,
):
    # QK GEMM: INT8×INT8→INT32, (1,1,1) smem-mode A
    mma_qk = sm90_utils.make_trivial_tiled_mma(
        cutlass.Int8, cutlass.Int8,
        utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        cutlass.Int32, (1, 1, 1),
        tiler_mn=(HALF_M, BLOCK_N),
        a_source=warpgroup.OperandSource.SMEM,
    )
    # PV GEMM: FP16×FP16→FP32, rs-mode A (P from registers, V from sV_f16 smem)
    mma_pv = sm90_utils.make_trivial_tiled_mma(
        cutlass.Float16, cutlass.Float16,
        utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        cutlass.Float32, (1, 1, 1),
        tiler_mn=(HALF_M, HEAD_DIM),
        a_source=warpgroup.OperandSource.RMEM,
    )

    sQ0_layout = sm90_utils.make_smem_layout_a(
        utils.LayoutEnum.ROW_MAJOR, (HALF_M, BLOCK_N, HEAD_DIM), cutlass.Int8, 1)
    sQ1_layout = sm90_utils.make_smem_layout_a(
        utils.LayoutEnum.ROW_MAJOR, (HALF_M, BLOCK_N, HEAD_DIM), cutlass.Int8, 1)
    sK_layout = sm90_utils.make_smem_layout_b(
        utils.LayoutEnum.ROW_MAJOR, (HALF_M, BLOCK_N, HEAD_DIM), cutlass.Int8, 2)
    # sV_f8: flat (no-swizzle) FP8 layout — 4KB intermediate buffer
    # Using a flat layout ensures write (Int8 recast, no swizzle) and
    # read (FP8 copy, uses tensor's own layout) have consistent address mapping.
    # Swizzle is only needed for sV_f16 which is consumed by WGMMA.
    sV_f8_layout = cute.make_layout(
        (HEAD_DIM, BLOCK_N), stride=(BLOCK_N, 1))
    # sV_f16: FP16 smem for WGMMA, 1 stage (8KB, WGMMA-compatible swizzled)
    sV_f16_layout = sm90_utils.make_smem_layout_b(
        utils.LayoutEnum.ROW_MAJOR, (HALF_M, HEAD_DIM, BLOCK_N), cutlass.Float16, 1)

    # Q copy: 128-thread, INT8, 128-bit/thread
    ca_int8 = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Int8, num_bits_per_copy=128)
    copy_q = cute.make_tiled_copy_tv(ca_int8,
        cute.make_layout((32, 4), stride=(4, 1)), cute.make_layout((1, 16)))

    # K async copy: 256-thread, INT8, 128-bit/thread
    ca_k_async = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.Int8, num_bits_per_copy=128)
    copy_K_async = cute.make_tiled_copy_tv(ca_k_async,
        cute.make_layout((64, 4), stride=(4, 1)), cute.make_layout((1, 16)))

    # V async copy: 256-thread, INT8, 128-bit/thread
    # gV tile shape: (HEAD_DIM, BLOCK_N) = (64, 64)
    # Thread layout (64, 4):(4,1) × value (1, 16):
    #   64 thread-rows × 4 thread-cols × 16 cols/thr = 64 rows × 64 cols = 4096 elements/pass
    # NOTE: use same layout as K async (which also handles a 64×64 tile)
    ca_v_async = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.Int8, num_bits_per_copy=128)
    copy_V_async = cute.make_tiled_copy_tv(ca_v_async,
        cute.make_layout((64, 4), stride=(4, 1)), cute.make_layout((1, 16)))

    # FP8 smem→reg: 128-thread (per-WG), 64-bit = 8 FP8/thread
    # 128 threads × 8 FP8 × 4 passes = 4096 elements — covers full (64,64) tile per WG
    # Thread layout (16, 8):(8,1) × value (1,8): tiler (16,64), needs 4 passes for (64,64)
    ca_f8_reg = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float8E4M3FN, num_bits_per_copy=64)
    copy_f8_reg = cute.make_tiled_copy_tv(ca_f8_reg,
        cute.make_layout((16, 8), stride=(8, 1)), cute.make_layout((1, 8)))

    # FP16 reg→smem: 128-thread (per-WG), 128-bit = 8 FP16/thread
    # Matches 8 FP8 after FP8→FP16 extension
    ca_f16_reg = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16, num_bits_per_copy=128)
    copy_f16_reg = cute.make_tiled_copy_tv(ca_f16_reg,
        cute.make_layout((16, 8), stride=(8, 1)), cute.make_layout((1, 8)))

    _sage_kernel_v4(
        gQ, gK, gV, gO, gQScale, gKScale, gVScale,
        n_kv_blocks, n_q_blocks,
        mma_qk, mma_pv,
        sQ0_layout, sQ1_layout, sK_layout, sV_f8_layout, sV_f16_layout,
        copy_q, copy_K_async, copy_V_async, copy_f8_reg, copy_f16_reg,
    ).launch(grid=(n_cta, 1, 1), block=(NUM_THREADS, 1, 1))


# ============================================================
# AOT 编译缓存
# ============================================================
_compiled_kernels: dict = {}


def _get_compiled(B, H, N, D, kv_len,
                  q_tiles, k_tiles, v_fp8_int8, out,
                  q_scale_flat, k_scale_2d, v_scale_2d,
                  N_KV_BLOCKS, N_Q_BLOCKS, N_CTAs):
    key = (B, H, N, D, kv_len)
    if key not in _compiled_kernels:
        _compiled_kernels[key] = cute.compile(
            _sage_jit_v4,
            from_dlpack(q_tiles,      assumed_align=128),
            from_dlpack(k_tiles,      assumed_align=128),
            from_dlpack(v_fp8_int8,   assumed_align=128),
            from_dlpack(out),
            from_dlpack(q_scale_flat),
            from_dlpack(k_scale_2d),
            from_dlpack(v_scale_2d),
            cute.Int32(N_KV_BLOCKS),
            cute.Int32(N_Q_BLOCKS),
            cute.Int32(N_CTAs),
        )
    return _compiled_kernels[key]


# ============================================================
# Public API
# ============================================================
def sageattn_cutedsl_v4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale: float = None,
    smooth_k: bool = True,
) -> torch.Tensor:
    """
    SageAttention CuTe DSL v4 前向传播 (SM90a WGMMA, FP8 V GMEM).

    v4 优化: V 以 FP8 E4M3FN 量化后从 GMEM 加载（2× 带宽节省），
    在 smem 中扩展为 FP16 后参与 FP16×FP16 WGMMA。

    参数:
        q, k, v: (B, H, N, D) float16, CUDA
        is_causal: 因果 mask (暂未实现)
        sm_scale: softmax 缩放 (默认 D^{-0.5})
        smooth_k: K Smoothing
    """
    assert q.is_cuda and q.dtype == torch.float16
    B, H, N, D = q.shape
    assert D == HEAD_DIM, f"sageattn_cutedsl_v4: 只支持 head_dim={HEAD_DIM}, got {D}"
    assert N % BLOCK_M == 0, f"N={N} 必须是 BLOCK_M={BLOCK_M} 的整数倍"
    assert N % BLOCK_N == 0, f"N={N} 必须是 BLOCK_N={BLOCK_N} 的整数倍"

    if sm_scale is None:
        sm_scale = D ** -0.5

    # K 量化 (INT8)
    if smooth_k:
        k_int8, k_scale, km = smooth_and_quant_k_gpu(k, block_sz=BLOCK_N)
    else:
        k_int8, k_scale = quant_per_block_int8(k, sm_scale=1.0, block_sz=BLOCK_N)

    # Q 量化 (INT8, block_sz=128)
    q_int8, q_scale = quant_q_per_block_gpu_bm128(q, sm_scale)

    N_Q_BLOCKS  = N // BLOCK_M
    N_KV_BLOCKS = N // BLOCK_N
    N_CTAs      = B * H * N_Q_BLOCKS

    # K tiles: (B*H, N_KV_BLOCKS, BLOCK_N, HEAD_DIM)
    k_tiles = k_int8.view(B, H, N_KV_BLOCKS, BLOCK_N, D)
    k_tiles = k_tiles.reshape(B * H, N_KV_BLOCKS, BLOCK_N, D).contiguous()

    # V tiles: (B*H, N_KV_BLOCKS, HEAD_DIM, BLOCK_N) — transposed for ROW_MAJOR smem layout
    # quant_v_per_tile_fp8_gpu expects (B*H, N_KV_BLOCKS, D, BLOCK_N) fp16
    v_pre_tile = v.view(B, H, N_KV_BLOCKS, BLOCK_N, D)
    v_pre_tile = v_pre_tile.permute(0, 1, 2, 4, 3)   # (B, H, N_KV_BLOCKS, D, BLOCK_N)
    v_pre_tile = v_pre_tile.reshape(B * H, N_KV_BLOCKS, D, BLOCK_N).contiguous()

    # V 量化 (FP8 E4M3FN, per-tile)
    v_fp8_int8_raw, v_scale = quant_v_per_tile_fp8_gpu(v_pre_tile, block_n=BLOCK_N)
    # v_fp8_int8_raw is (B*H, N_KV_BLOCKS, D, BLOCK_N) int8 (FP8 bit patterns)
    v_fp8_int8 = v_fp8_int8_raw.contiguous()

    # Q tiles: (B*H*N_Q_BLOCKS, BLOCK_M, HEAD_DIM)
    q_tiles = q_int8.view(B, H, N_Q_BLOCKS, BLOCK_M, D)
    q_tiles = q_tiles.reshape(B * H * N_Q_BLOCKS, BLOCK_M, D).contiguous()

    out = torch.empty(B * H * N_Q_BLOCKS, BLOCK_M, D, dtype=torch.float32, device=q.device)
    q_scale_flat = q_scale.reshape(B * H * N_Q_BLOCKS).contiguous()
    k_scale_2d   = k_scale.reshape(B * H, N_KV_BLOCKS).contiguous()
    v_scale_2d   = v_scale.reshape(B * H, N_KV_BLOCKS).contiguous()

    compiled = _get_compiled(
        B, H, N, D, N,
        q_tiles, k_tiles, v_fp8_int8, out,
        q_scale_flat, k_scale_2d, v_scale_2d,
        N_KV_BLOCKS, N_Q_BLOCKS, N_CTAs,
    )
    compiled(
        from_dlpack(q_tiles,      assumed_align=128),
        from_dlpack(k_tiles,      assumed_align=128),
        from_dlpack(v_fp8_int8,   assumed_align=128),
        from_dlpack(out),
        from_dlpack(q_scale_flat),
        from_dlpack(k_scale_2d),
        from_dlpack(v_scale_2d),
        cute.Int32(N_KV_BLOCKS),
        cute.Int32(N_Q_BLOCKS),
        cute.Int32(N_CTAs),
    )
    torch.cuda.synchronize()

    return out.view(B, H, N, D).to(torch.float32)
