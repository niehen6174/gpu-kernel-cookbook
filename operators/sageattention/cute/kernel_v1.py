"""
SageAttention V1 — CuTe DSL 实现 (Hopper SM90a)

算法（对标 Triton v1）:
  1. (可选) K Smoothing: k_smooth = k - mean(k, dim=seq)
  2. Q/K per-block INT8 量化
     - Q scale 融合 sm_scale * log2(e)
     - K scale = max(|k_block|) / 127
  3. WGMMA QK GEMM: INT8 × INT8 → INT32
  4. Online softmax (exp2 路径)
  5. ss-mode PV GEMM: P (FP16, smem) × V (FP16, smem) → FP32 acc_O
     - ss-mode 完全避开 rs-mode layout 转换复杂度，先保证正确性

设计:
  - BLOCK_M = 64, BLOCK_N = 64, HEAD_DIM = 64
  - 1 warpgroup (128 线程) per CTA
  - flash-style: 一个 CTA 负责一个 Q tile × 遍历所有 KV tiles
  - N_CTAs = B * H * N_Q_BLOCKS

正确性参考: operators/sageattention/triton/kernel_v1.py
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
BLOCK_M     = 64    # Q tile (M-dim per CTA)
BLOCK_N     = 64    # KV tile (N-dim)
NUM_THREADS = 128   # 1 warpgroup

# acc_S fragment 参数 (BLOCK_M=64, BLOCK_N=64, warpgroup WGMMA)
ACC_FRAG_SIZE    = 32   # cute.size(acc_S) per thread
ACC_ROWS_PER_THR = 2    # M-rows per thread

# tile 元素数
TILE_ELEMS_K = BLOCK_N * BLOCK_M   # int8: 64×64
TILE_ELEMS_V = BLOCK_M * BLOCK_N   # fp16: 64×64
Q_ELEMS      = BLOCK_M * BLOCK_M   # int8: 64×64 (D=64)

LOG2_E = 1.4426950408889634


# ============================================================
# Python 层量化（CPU 版，保留用于调试 / 对比）
# ============================================================
def _smooth_and_quant_k_cpu(k: torch.Tensor) -> tuple:
    """K Smoothing + per-block INT8 量化（CPU Python 循环版，仅用于调试）。"""
    B, H, N, D = k.shape
    km = k.float().mean(dim=2, keepdim=True)
    k_s = (k.float() - km).to(torch.float16)

    nblocks = (N + BLOCK_N - 1) // BLOCK_N
    k_int8  = torch.empty_like(k_s, dtype=torch.int8)
    k_scale = torch.empty((B, H, nblocks), dtype=torch.float32, device=k.device)

    for b in range(B):
        for h in range(H):
            for bn in range(nblocks):
                s = bn * BLOCK_N
                e = min(s + BLOCK_N, N)
                blk = k_s[b, h, s:e, :].float()
                sc  = blk.abs().max() / 127.0 + 1e-7
                k_int8[b, h, s:e, :] = (
                    (blk / sc + 0.5 * blk.sign()).clamp(-128, 127).to(torch.int8)
                )
                k_scale[b, h, bn] = sc

    return k_int8, k_scale, km.to(torch.float16)


def _quant_q_per_block_cpu(q: torch.Tensor, sm_scale: float) -> tuple:
    """Q per-block INT8 量化（CPU Python 循环版，仅用于调试）。"""
    B, H, N, D = q.shape
    nblocks = (N + BLOCK_M - 1) // BLOCK_M
    q_int8  = torch.empty_like(q, dtype=torch.int8)
    q_scale = torch.empty((B, H, nblocks), dtype=torch.float32, device=q.device)

    scale_factor = sm_scale * LOG2_E

    for b in range(B):
        for h in range(H):
            for bm in range(nblocks):
                s = bm * BLOCK_M
                e = min(s + BLOCK_M, N)
                blk = q[b, h, s:e, :].float() * scale_factor
                sc  = blk.abs().max() / 127.0 + 1e-7
                q_int8[b, h, s:e, :] = (
                    (blk / sc + 0.5 * blk.sign()).clamp(-128, 127).to(torch.int8)
                )
                q_scale[b, h, bm] = sc

    return q_int8, q_scale


# GPU 量化 (Triton kernel)
from .quant import quant_q_per_block_gpu, smooth_and_quant_k_gpu


# ============================================================
# CuTe DSL Kernel (ss-mode PV GEMM)
# ============================================================
@cute.kernel
def _sage_v1_kernel(
    gQ:      cute.Tensor,       # (B*H*N_Q_BLOCKS, BLOCK_M, D) int8, tile-contiguous
    gK:      cute.Tensor,       # (B*H, N_KV_BLOCKS, BLOCK_N, D) int8
    gV:      cute.Tensor,       # (B*H, N_KV_BLOCKS, D, BLOCK_N) fp16 (transposed tile)
    gO:      cute.Tensor,       # (B*H*N_Q_BLOCKS, BLOCK_M, D) fp32
    gQScale: cute.Tensor,       # (B*H*N_Q_BLOCKS,) fp32
    gKScale: cute.Tensor,       # (B*H, N_KV_BLOCKS) fp32
    n_kv_blocks: cute.Int32,
    n_q_blocks:  cute.Int32,
    mma_qk:      cute.TiledMma,
    mma_pv:      cute.TiledMma,
    sQ_layout:   cute.ComposedLayout,
    sK_layout:   cute.ComposedLayout,
    sV_layout:   cute.ComposedLayout,
    sP_layout:   cute.ComposedLayout,
    copy_QK:     cute.TiledCopy,
    copy_V:      cute.TiledCopy,
    copy_QK_async: cute.TiledCopy,
    copy_V_async:  cute.TiledCopy,
    copy_P_store:  cute.TiledCopy,   # make_tiled_copy_C(atom, mma_qk) — C → smem
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    smem = SmemAllocator()

    @cute.struct
    class SS:
        sQ: cute.struct.Align[cute.struct.MemRange[cutlass.Int8,    cute.cosize(sQ_layout)], 128]
        sK: cute.struct.Align[cute.struct.MemRange[cutlass.Int8,    cute.cosize(sK_layout)], 128]
        sV: cute.struct.Align[cute.struct.MemRange[cutlass.Float16, cute.cosize(sV_layout)], 128]
        sP: cute.struct.Align[cute.struct.MemRange[cutlass.Float16, cute.cosize(sP_layout)], 128]

    storage = smem.allocate(SS)
    sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
    sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
    sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
    sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)

    # 取第 0 个 pipeline stage（无 double buffering，单 smem）
    sQ_s0 = cute.slice_(sQ, (None, None, 0))
    sK_s0 = cute.slice_(sK, (None, None, 0))
    sV_s0 = cute.slice_(sV, (None, None, 0))
    sP_s0 = cute.slice_(sP, (None, None, 0))

    # head_flat = (b * H + h)，用于索引 KV 和 K-scale
    head_flat = bidx // n_q_blocks

    # --- Copy 线程切片 ---
    thr_QK       = copy_QK.get_slice(tidx)
    thr_V        = copy_V.get_slice(tidx)
    thr_QK_async = copy_QK_async.get_slice(tidx)
    thr_V_async  = copy_V_async.get_slice(tidx)
    thr_P_store  = copy_P_store.get_slice(tidx)

    tQsQ       = thr_QK.partition_D(sQ_s0)
    tKsK_async = thr_QK_async.partition_D(sK_s0)
    tVsV_async = thr_V_async.partition_D(sV_s0)

    # --- 全局 tensor 指针 ---
    gQ_base = gQ.iterator
    gK_base = gK.iterator
    gV_base = gV.iterator
    gO_base = gO.iterator

    q_ptr      = (gQ_base + bidx * Q_ELEMS).align(128)
    k_ptr_base = gK_base + head_flat * n_kv_blocks * TILE_ELEMS_K
    v_ptr_base = gV_base + head_flat * n_kv_blocks * TILE_ELEMS_V
    o_ptr      = gO_base + bidx * Q_ELEMS

    gQ_cta = cute.make_tensor(q_ptr, cute.make_layout((BLOCK_M, BLOCK_M), stride=(BLOCK_M, 1)))
    gO_cta = cute.make_tensor(o_ptr, cute.make_layout((BLOCK_M, BLOCK_M), stride=(BLOCK_M, 1)))

    # --- 加载 Q（同步，vectorized）---
    cute.copy(copy_QK, thr_QK.partition_S(gQ_cta), tQsQ)

    # --- 异步加载 K[0] → smem，wait 完成 ---
    k0_ptr = (k_ptr_base + 0).align(128)
    gK_0 = cute.make_tensor(k0_ptr, cute.make_layout((BLOCK_N, BLOCK_M), stride=(BLOCK_M, 1)))
    cute.copy(copy_QK_async, thr_QK_async.partition_S(gK_0), tKsK_async)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.barrier()

    # --- MMA fragments ---
    thr_qk = mma_qk.get_slice(tidx)
    tCrQ   = mma_qk.make_fragment_A(thr_qk.partition_A(sQ))
    tCrK   = mma_qk.make_fragment_B(thr_qk.partition_B(sK))
    acc_S  = cute.make_fragment(thr_qk.partition_shape_C((BLOCK_M, BLOCK_N)), cutlass.Int32)

    # convert_layout_acc_frgA: 将 acc_S layout 变换为 PV GEMM A-fragment layout
    # (Flash Attention 正确 ss-mode 模式)
    acc_S_layout = cute.make_layout(thr_qk.partition_shape_C((BLOCK_M, BLOCK_N)))
    l = cute.logical_divide(acc_S_layout, ((None, None, 2), None, None))
    frgA_layout = cute.make_layout(
        ((l.shape[0][0], l.shape[0][1], l.shape[0][2][0]), l.shape[1], (l.shape[0][2][1], l.shape[2])),
        stride=((l.stride[0][0], l.stride[0][1], l.stride[0][2][0]), l.stride[1], (l.stride[0][2][1], l.stride[2])),
    )
    # tOrP 使用 frgA_layout（与 copy_P_store 的 retile 兼容）
    tOrP = cute.make_fragment(frgA_layout, cutlass.Float16)
    # copy_P_store 目标: sP smem 的 thread 分区
    tPsP = thr_P_store.partition_D(sP_s0)

    thr_pv = mma_pv.get_slice(tidx)
    # ss-mode: tCrP 是 sP smem 的引用（rank 3，无 stage 维）
    tCrP   = mma_pv.make_fragment_A(thr_pv.partition_A(sP_s0))
    # tCrV 是 sV smem 的引用（rank 4，有 stage 维）
    tCrV   = mma_pv.make_fragment_B(thr_pv.partition_B(sV))
    acc_O  = cute.make_fragment(thr_pv.partition_shape_C((BLOCK_M, BLOCK_M)), cutlass.Float32)
    acc_O.fill(0.0)

    # --- Online softmax 状态 ---
    row_max = cute.make_fragment(ACC_ROWS_PER_THR, cutlass.Float32)
    row_sum = cute.make_fragment(ACC_ROWS_PER_THR, cutlass.Float32)
    row_max.fill(-cutlass.Float32.inf)
    row_sum.fill(cutlass.Float32(0.0))

    q_scale_val = gQScale[bidx]

    # ============================================================
    # Main KV loop
    # ============================================================
    for n_tile in range(n_kv_blocks):
        # ---- QK GEMM (INT8 × INT8 → INT32) ----
        acc_S.fill(0)
        mma_qk.set(warpgroup.Field.ACCUMULATE, False)
        warpgroup.fence()
        for k in cutlass.range_constexpr(cute.size(tCrQ, mode=[2])):
            cute.gemm(mma_qk, acc_S, tCrQ[None, None, k, 0], tCrK[None, None, k, 0], acc_S)
            mma_qk.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.commit_group()
        warpgroup.wait_group(0)

        # ---- 异步加载 V[n] ----
        v_ptr = (v_ptr_base + n_tile * TILE_ELEMS_V).align(128)
        gV_n  = cute.make_tensor(v_ptr, cute.make_layout((BLOCK_M, BLOCK_N), stride=(BLOCK_N, 1)))
        cute.copy(copy_V_async, thr_V_async.partition_S(gV_n), tVsV_async)
        cute.arch.cp_async_commit_group()  # group A: V[n]

        # ---- 异步加载 K[n+1] ----
        if n_tile + 1 < n_kv_blocks:
            k_next_ptr = (k_ptr_base + (n_tile + 1) * TILE_ELEMS_K).align(128)
            gK_next = cute.make_tensor(
                k_next_ptr, cute.make_layout((BLOCK_N, BLOCK_M), stride=(BLOCK_M, 1))
            )
            cute.copy(copy_QK_async, thr_QK_async.partition_S(gK_next), tKsK_async)
            cute.arch.cp_async_commit_group()  # group B: K[n+1]

        # ---- Online Softmax + 填充 sP ----
        k_scale_val = gKScale[head_flat, n_tile]
        dequant = q_scale_val * k_scale_val

        # 通过 frgA_layout 视图读取 acc_S：与 tOrP 使用相同的坐标映射
        # tOrP_acc[i] 与 tOrP[i] 对应相同的 (m,n) 坐标
        tOrP_acc = cute.make_tensor(acc_S.iterator, frgA_layout)

        # pass 1: 求新行最大值（通过 tOrP_acc 读取 acc_S）
        row_max_new = cute.make_fragment(ACC_ROWS_PER_THR, cutlass.Float32)
        row_max_new.fill(-cutlass.Float32.inf)
        for i in cutlass.range_constexpr(ACC_FRAG_SIZE):
            r = (i // 2) % ACC_ROWS_PER_THR
            val = cutlass.Float32(tOrP_acc[i]) * dequant
            if val > row_max_new[r]:
                row_max_new[r] = val

        # warp-level reduce max（4 lanes × 2 lanes，共 128 线程 = 4 warps/warpgroup）
        for r in cutlass.range_constexpr(ACC_ROWS_PER_THR):
            row_max_new[r] = cute.arch.fmax(
                row_max_new[r],
                cute.arch.shuffle_sync_bfly(row_max_new[r], offset=2, mask=-1, mask_and_clamp=31)
            )
            row_max_new[r] = cute.arch.fmax(
                row_max_new[r],
                cute.arch.shuffle_sync_bfly(row_max_new[r], offset=1, mask=-1, mask_and_clamp=31)
            )
            row_max_new[r] = cute.arch.fmax(row_max[r], row_max_new[r])

        # pass 2: exp2，累积 row_sum，写入 tOrP
        # tOrP[i] 和 tOrP_acc[i] 使用同一 frgA_layout，坐标一致
        row_sum_new = cute.make_fragment(ACC_ROWS_PER_THR, cutlass.Float32)
        row_sum_new.fill(cutlass.Float32(0.0))

        for i in cutlass.range_constexpr(ACC_FRAG_SIZE):
            r = (i // 2) % ACC_ROWS_PER_THR
            p_val = cute.math.exp2(
                cutlass.Float32(tOrP_acc[i]) * dequant - row_max_new[r], fastmath=True
            )
            tOrP[i] = cutlass.Float16(p_val)
            row_sum_new[r] = row_sum_new[r] + p_val

        # warp-level reduce sum
        for r in cutlass.range_constexpr(ACC_ROWS_PER_THR):
            row_sum_new[r] = row_sum_new[r] + cute.arch.shuffle_sync_bfly(
                row_sum_new[r], offset=2, mask=-1, mask_and_clamp=31
            )
            row_sum_new[r] = row_sum_new[r] + cute.arch.shuffle_sync_bfly(
                row_sum_new[r], offset=1, mask=-1, mask_and_clamp=31
            )

        # 用 alpha 缩放 acc_O（rescale 旧累积值）
        for i in cutlass.range_constexpr(cute.size(acc_O)):
            r_o = (i // 2) % ACC_ROWS_PER_THR
            acc_O[i] = acc_O[i] * cute.math.exp2(
                row_max[r_o] - row_max_new[r_o], fastmath=True
            )

        # 更新 row_max / row_sum
        for r in cutlass.range_constexpr(ACC_ROWS_PER_THR):
            alpha = cute.math.exp2(row_max[r] - row_max_new[r], fastmath=True)
            row_sum[r] = row_sum[r] * alpha + row_sum_new[r]
            row_max[r] = row_max_new[r]

        # ---- 把 tOrP 写入 sP smem（ss-mode 所需）----
        # retile 将 QK MMA C-fragment 重整到 copy TV layout
        # 使用 thread slice thr_P_store（与 Flash Attention ss-mode 一致）
        tPrP = thr_P_store.retile(tOrP)
        cute.copy(thr_P_store, tPrP, tPsP)

        # ---- 等待 V[n] 到达 smem ----
        if n_tile + 1 < n_kv_blocks:
            cute.arch.cp_async_wait_group(1)  # 允许 K_next 继续
        else:
            cute.arch.cp_async_wait_group(0)
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta
        )
        cute.arch.barrier()

        # ---- PV GEMM (ss-mode) ----
        # tCrP rank=3 (sP_s0 smem view): [None, None, k]
        # tCrV rank=4 (sV staged smem): [None, None, k, 0]
        mma_atom_pv = cute.make_mma_atom(mma_pv.op)
        mma_atom_pv.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.fence()
        for k in cutlass.range_constexpr(cute.size(tCrP, mode=[2])):
            cute.gemm(mma_atom_pv, acc_O, tCrP[None, None, k], tCrV[None, None, k, 0], acc_O)
            mma_atom_pv.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.commit_group()
        warpgroup.wait_group(0)

        # ---- 等待 K[n+1] 完成 ----
        if n_tile + 1 < n_kv_blocks:
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

    # ============================================================
    # Epilogue: 归一化 + 写出 FP32
    # ============================================================
    tOgO = thr_pv.partition_C(gO_cta)
    for i in cutlass.range_constexpr(cute.size(acc_O)):
        r = (i // 2) % ACC_ROWS_PER_THR
        acc_O[i] = acc_O[i] / (row_sum[r] + cutlass.Float32(1e-6))
    cute.basic_copy(acc_O, tOgO)


@cute.jit
def _sage_v1_jit(
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
    HEAD_DIM = BLOCK_M   # D=64

    # QK GEMM: INT8 × INT8 → INT32
    mma_qk = sm90_utils.make_trivial_tiled_mma(
        cutlass.Int8, cutlass.Int8,
        utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        cutlass.Int32, (1, 1, 1),
        tiler_mn=(BLOCK_M, BLOCK_N),
        a_source=warpgroup.OperandSource.SMEM,
    )
    # PV GEMM: FP16 × FP16 → FP32 (ss-mode: A from smem)
    mma_pv = sm90_utils.make_trivial_tiled_mma(
        cutlass.Float16, cutlass.Float16,
        utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        cutlass.Float32, (1, 1, 1),
        tiler_mn=(BLOCK_M, HEAD_DIM),
        a_source=warpgroup.OperandSource.SMEM,  # ss-mode
    )

    sQ_layout = sm90_utils.make_smem_layout_a(
        utils.LayoutEnum.ROW_MAJOR, (BLOCK_M, BLOCK_N, HEAD_DIM), cutlass.Int8, 1)
    sK_layout = sm90_utils.make_smem_layout_b(
        utils.LayoutEnum.ROW_MAJOR, (BLOCK_M, BLOCK_N, HEAD_DIM), cutlass.Int8, 1)
    sV_layout = sm90_utils.make_smem_layout_b(
        utils.LayoutEnum.ROW_MAJOR, (BLOCK_M, HEAD_DIM, BLOCK_N), cutlass.Float16, 1)
    # sP layout: P 是 (BLOCK_M, BLOCK_N) FP16，作为 PV GEMM 的 A 矩阵
    # make_smem_layout_a(layout, (M, N, K), dtype, stages)
    # PV GEMM: A=P(BLOCK_M×BLOCK_N), B=V(BLOCK_N×HEAD_DIM), C=O(BLOCK_M×HEAD_DIM)
    # 所以 mma_tiler_mnk = (BLOCK_M, HEAD_DIM, BLOCK_N), stages=1
    sP_layout = sm90_utils.make_smem_layout_a(
        utils.LayoutEnum.ROW_MAJOR, (BLOCK_M, HEAD_DIM, BLOCK_N), cutlass.Float16, 1)

    # 128-bit vectorized 同步拷贝 (Q)
    # INT8 128-bit = 16 elems; for (64M, 64K): K-groups=64/16=4, M-threads=128/4=32
    # T=(32,4):(4,1): 32 threads along M, 4 copy-groups along K
    copy_atom_QK = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Int8, num_bits_per_copy=128)
    copy_QK = cute.make_tiled_copy_tv(
        copy_atom_QK,
        cute.make_layout((32, 4), stride=(4, 1)),
        cute.make_layout((1, 16)))
    copy_atom_V = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Float16, num_bits_per_copy=128)
    # FP16 128-bit = 8 elems; for (64M, 64N): N-groups=64/8=8, M-threads=128/8=16
    copy_V = cute.make_tiled_copy_tv(
        copy_atom_V,
        cute.make_layout((16, 8), stride=(8, 1)),
        cute.make_layout((1, 8)))

    # cp.async 异步拷贝 (K, V pipeline)
    copy_atom_QK_async = cute.make_copy_atom(
        cpasync.CopyG2SOp(), cutlass.Int8, num_bits_per_copy=128)
    copy_QK_async = cute.make_tiled_copy_tv(
        copy_atom_QK_async,
        cute.make_layout((32, 4), stride=(4, 1)),
        cute.make_layout((1, 16)))
    copy_atom_V_async = cute.make_copy_atom(
        cpasync.CopyG2SOp(), cutlass.Float16, num_bits_per_copy=128)
    copy_V_async = cute.make_tiled_copy_tv(
        copy_atom_V_async,
        cute.make_layout((16, 8), stride=(8, 1)),
        cute.make_layout((1, 8)))

    # P 写回 smem 的拷贝：make_tiled_copy_C(atom, mma_qk)
    # — 用 mma_qk 构造，因为 P 的形状是 QK GEMM 的 C (BLOCK_M × BLOCK_N)
    # — retile(tOrP) 将 C-fragment 重整为 copy 的 TV tile layout
    # — num_bits_per_copy=32: C-fragment 内层连续 2 个 FP16（4 bytes）
    copy_atom_P_store = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Float16, num_bits_per_copy=32)
    copy_P_store = cute.make_tiled_copy_C(copy_atom_P_store, mma_qk)

    _sage_v1_kernel(
        gQ, gK, gV, gO, gQScale, gKScale,
        n_kv_blocks, n_q_blocks,
        mma_qk, mma_pv,
        sQ_layout, sK_layout, sV_layout, sP_layout,
        copy_QK, copy_V, copy_QK_async, copy_V_async,
        copy_P_store,
    ).launch(grid=(n_cta, 1, 1), block=(NUM_THREADS, 1, 1))


# ============================================================
# AOT 编译缓存
# ============================================================
_compiled_kernels: dict = {}


def _get_compiled(B, H, N, D, q_tiles, k_tiles, v_tiles, out,
                  q_scale_flat, k_scale_2d, N_KV_BLOCKS, N_Q_BLOCKS, N_CTAs):
    key = (B, H, N, D)
    if key not in _compiled_kernels:
        _compiled_kernels[key] = cute.compile(
            _sage_v1_jit,
            from_dlpack(q_tiles,     assumed_align=128),
            from_dlpack(k_tiles,     assumed_align=128),
            from_dlpack(v_tiles,     assumed_align=128),
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
def sageattn_v1_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale: float = None,
    smooth_k: bool = True,
) -> torch.Tensor:
    """
    SageAttention V1 CuTe DSL 前向传播 (SM90a, ss-mode PV GEMM)。

    参数:
        q, k, v: (B, H, N, D) float16, CUDA, D=64
        is_causal: 暂不支持（留作后续扩展）
        sm_scale: 默认 1/sqrt(D)
        smooth_k: K Smoothing

    返回:
        O: (B, H, N, D) float16
    """
    assert q.is_cuda and q.dtype == torch.float16
    B, H, N, D = q.shape
    assert D == 64, f"当前只支持 head_dim=64，got {D}"
    assert N % BLOCK_M == 0, f"N={N} 必须是 BLOCK_M={BLOCK_M} 的倍数"
    assert not is_causal, "is_causal 暂未实现"

    if sm_scale is None:
        sm_scale = D ** -0.5

    # K Smoothing + 量化 (GPU Triton kernel)
    if smooth_k:
        k_int8, k_scale, km = smooth_and_quant_k_gpu(k)
    else:
        # No-smooth path: quantize K without subtracting mean
        from .quant import quant_per_block_int8
        k_int8, k_scale = quant_per_block_int8(k, sm_scale=1.0)
        km = None

    # Q 量化 (GPU Triton kernel)
    q_int8, q_scale = quant_q_per_block_gpu(q, sm_scale)

    N_Q_BLOCKS  = N // BLOCK_M
    N_KV_BLOCKS = N // BLOCK_N
    N_CTAs      = B * H * N_Q_BLOCKS

    # Tile-contiguous 重排
    # K: (B, H, N, D) → (B*H, N_KV_BLOCKS, BLOCK_N, D) int8
    k_tiles = k_int8.view(B, H, N_KV_BLOCKS, BLOCK_N, D)
    k_tiles = k_tiles.reshape(B * H, N_KV_BLOCKS, BLOCK_N, D).contiguous()

    # V: (B, H, N, D) → (B*H, N_KV_BLOCKS, D, BLOCK_N) fp16 (转置 tile)
    v_tiles = v.view(B, H, N_KV_BLOCKS, BLOCK_N, D)
    v_tiles = v_tiles.permute(0, 1, 2, 4, 3)
    v_tiles = v_tiles.reshape(B * H, N_KV_BLOCKS, D, BLOCK_N).contiguous()

    # Q: (B, H, N, D) → (B*H*N_Q_BLOCKS, BLOCK_M, D) int8
    q_tiles = q_int8.view(B, H, N_Q_BLOCKS, BLOCK_M, D)
    q_tiles = q_tiles.reshape(B * H * N_Q_BLOCKS, BLOCK_M, D).contiguous()

    # Output: (B*H*N_Q_BLOCKS, BLOCK_M, D) fp32
    out = torch.empty(B * H * N_Q_BLOCKS, BLOCK_M, D, dtype=torch.float32, device=q.device)

    # Scale tensors
    q_scale_flat = q_scale.reshape(B * H * N_Q_BLOCKS).contiguous()
    k_scale_2d   = k_scale.reshape(B * H, N_KV_BLOCKS).contiguous()

    # 编译 + 运行
    compiled = _get_compiled(
        B, H, N, D,
        q_tiles, k_tiles, v_tiles, out,
        q_scale_flat, k_scale_2d,
        N_KV_BLOCKS, N_Q_BLOCKS, N_CTAs,
    )
    compiled(
        from_dlpack(q_tiles,     assumed_align=128),
        from_dlpack(k_tiles,     assumed_align=128),
        from_dlpack(v_tiles,     assumed_align=128),
        from_dlpack(out),
        from_dlpack(q_scale_flat),
        from_dlpack(k_scale_2d),
        cute.Int32(N_KV_BLOCKS),
        cute.Int32(N_Q_BLOCKS),
        cute.Int32(N_CTAs),
    )
    torch.cuda.synchronize()

    return out.view(B, H, N, D).to(torch.float16)
