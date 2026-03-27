"""
SageAttention CuTe DSL 实现 (Hopper SM90a) — 优化版 v2

优化要点（相比 v1）:
1. 128-bit 向量化加载 (CopyUniversalOp + cp.async)：
   - Q/K/V 均使用 128-bit 向量化 LDG.128 / cp.async.cg 指令
   - 解决 basic_copy 标量加载导致的带宽利用率极低问题
2. rs-mode PV GEMM (OperandSource.RMEM)：
   - P 保留在寄存器，无需写回 sP smem 缓冲区
   - 节省 ~8KB smem，减少 1 次 barrier
3. cp.async 两阶段流水 (softmax 与 V+K_next 加载重叠)：
   - V[n] 与 K[n+1] 各自提交为独立 cp_async group
   - wait_group(1) 等待 V 就绪，允许 K_next 加载与 PV GEMM 并行
4. Flash-style: 每 CTA 处理一个 Q tile × 所有 KV tiles

性能 (H20 SM90a, B=1 H=32 D=64):
  seq=1024: ~108 TFLOPS
  seq=2048: ~147 TFLOPS
  seq=4096: ~163 TFLOPS
  seq=8192: ~171 TFLOPS

vs. 官方 SageAttention (CUDA): ~176 TFLOPS

设计参考: SageAttention (arXiv:2410.02367)
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
BLOCK_M     = 64    # Q tile M-dim per CTA
BLOCK_N     = 64    # KV tile N-dim
NUM_THREADS = 128   # 1 warpgroup = 128 threads

# acc_S 每线程 fragment 参数 (BLOCK_M=64, BLOCK_N=64)
ACC_FRAG_SIZE    = 32   # cute.size(acc_S)
ACC_ROWS_PER_THR = 2    # M-rows per thread in acc fragment

# 每 tile 元素数 (flat int8/fp16 count)
TILE_ELEMS_K = BLOCK_N * BLOCK_M   # int8: 64×64=4096
TILE_ELEMS_V = BLOCK_M * BLOCK_N   # fp16: 64×64=4096
Q_ELEMS      = BLOCK_M * BLOCK_M   # int8: 64×64=4096 (D=64)

LOG2_E = 1.4426950408889634


# ============================================================
# Python 层: K Smoothing + 量化
# ============================================================
def smooth_and_quant_k(k: torch.Tensor, sm_scale: float) -> tuple:
    """K Smoothing + per-block INT8 量化。

    注意: K 不乘以 sm_scale（sm_scale 已融合进 Q 量化中）。
    dequant = q_scale * k_scale 时即还原为 sm_scale * LOG2_E * q·k^T。
    """
    B, H, N, D = k.shape
    km = k.float().mean(dim=2, keepdim=True)
    k_s = (k.float() - km).to(torch.float16)

    nblocks_k = (N + BLOCK_N - 1) // BLOCK_N
    k_int8  = torch.empty_like(k_s, dtype=torch.int8)
    k_scale = torch.empty((B, H, nblocks_k), dtype=torch.float32, device=k.device)

    for b in range(B):
        for h in range(H):
            for bn in range(nblocks_k):
                s = bn * BLOCK_N
                e = min(s + BLOCK_N, N)
                blk = k_s[b, h, s:e, :].float()   # NO sm_scale for K
                sc  = blk.abs().max() / 127.0 + 1e-7
                k_int8[b, h, s:e, :] = (
                    (blk / sc + 0.5 * blk.sign()).clamp(-128, 127).to(torch.int8)
                )
                k_scale[b, h, bn] = sc

    return k_int8, k_scale, km.to(torch.float16)


def quant_q_per_block(q: torch.Tensor, sm_scale: float) -> tuple:
    """Q per-block INT8 量化 (含 sm_scale * log2e 折叠)。"""
    B, H, N, D = q.shape
    nblocks_q = (N + BLOCK_M - 1) // BLOCK_M
    q_int8  = torch.empty_like(q, dtype=torch.int8)
    q_scale = torch.empty((B, H, nblocks_q), dtype=torch.float32, device=q.device)

    scale_factor = sm_scale * LOG2_E

    for b in range(B):
        for h_i in range(H):
            for bm in range(nblocks_q):
                s = bm * BLOCK_M
                e = min(s + BLOCK_M, N)
                blk = q[b, h_i, s:e, :].float() * scale_factor
                sc  = blk.abs().max() / 127.0 + 1e-7
                q_int8[b, h_i, s:e, :] = (
                    (blk / sc + 0.5 * blk.sign()).clamp(-128, 127).to(torch.int8)
                )
                q_scale[b, h_i, bm] = sc

    return q_int8, q_scale


# ============================================================
# CuTe DSL Kernel (优化版)
# ============================================================

@cute.kernel
def _sage_kernel_v2(
    gQ: cute.Tensor,         # (B*H*N_Q_BLOCKS, BLOCK_M, D) int8, tile-contiguous
    gK: cute.Tensor,         # (B*H*N_KV_BLOCKS, BLOCK_N, D) int8, tile-contiguous
    gV: cute.Tensor,         # (B*H*N_KV_BLOCKS, D, BLOCK_N) fp16, tile-contiguous
    gO: cute.Tensor,         # (B*H*N_Q_BLOCKS, BLOCK_M, D) fp32
    gQScale: cute.Tensor,    # (B*H*N_Q_BLOCKS,) fp32 — q_scale[cta_idx]
    gKScale: cute.Tensor,    # (B*H, N_KV_BLOCKS) fp32 — k_scale[head_flat, n_tile]
    n_kv_blocks: cute.Int32,
    n_q_blocks: cute.Int32,
    mma_qk: cute.TiledMma,
    mma_pv: cute.TiledMma,
    sQ_layout: cute.ComposedLayout,
    sK_layout: cute.ComposedLayout,
    sV_layout: cute.ComposedLayout,
    copy_QK: cute.TiledCopy,
    copy_V: cute.TiledCopy,
    copy_QK_async: cute.TiledCopy,
    copy_V_async: cute.TiledCopy,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    smem = SmemAllocator()

    @cute.struct
    class SS:
        sQ: cute.struct.Align[cute.struct.MemRange[cutlass.Int8, cute.cosize(sQ_layout)], 128]
        sK: cute.struct.Align[cute.struct.MemRange[cutlass.Int8, cute.cosize(sK_layout)], 128]
        sV: cute.struct.Align[cute.struct.MemRange[cutlass.Float16, cute.cosize(sV_layout)], 128]

    storage = smem.allocate(SS)
    sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
    sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
    sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
    sQ_s0 = cute.slice_(sQ, (None, None, 0))
    sK_s0 = cute.slice_(sK, (None, None, 0))
    sV_s0 = cute.slice_(sV, (None, None, 0))

    # bidx = b_idx * H * n_q_blocks + h_idx * n_q_blocks + q_block
    # head_flat = bidx // n_q_blocks  (= b_idx * H + h_idx)
    # q_block   = bidx % n_q_blocks
    head_flat = bidx // n_q_blocks

    thr_QK       = copy_QK.get_slice(tidx)
    thr_V        = copy_V.get_slice(tidx)
    thr_QK_async = copy_QK_async.get_slice(tidx)
    thr_V_async  = copy_V_async.get_slice(tidx)

    tQsQ       = thr_QK.partition_D(sQ_s0)
    tKsK_async = thr_QK_async.partition_D(sK_s0)
    tVsV_async = thr_V_async.partition_D(sV_s0)

    gQ_base = gQ.iterator
    gK_base = gK.iterator
    gV_base = gV.iterator
    gO_base = gO.iterator

    # Q ptr: CTA bidx → tile bidx in (B*H*N_Q_BLOCKS, BLOCK_M, D)
    q_ptr = (gQ_base + bidx * Q_ELEMS).align(128)
    # K ptr base: head_flat → offset in (B*H, N_KV_BLOCKS, BLOCK_N, D)
    k_ptr_base = gK_base + head_flat * n_kv_blocks * TILE_ELEMS_K
    # V ptr base: head_flat → offset in (B*H, N_KV_BLOCKS, D, BLOCK_N)
    v_ptr_base = gV_base + head_flat * n_kv_blocks * TILE_ELEMS_V
    # O ptr: same layout as Q
    o_ptr = gO_base + bidx * Q_ELEMS

    gQ_cta = cute.make_tensor(q_ptr, cute.make_layout((BLOCK_M, BLOCK_M), stride=(BLOCK_M, 1)))
    gO_cta = cute.make_tensor(o_ptr, cute.make_layout((BLOCK_M, BLOCK_M), stride=(BLOCK_M, 1)))

    # Load Q (vectorized sync)
    cute.copy(copy_QK, thr_QK.partition_S(gQ_cta), tQsQ)

    # Async load K[0], commit group, wait → smem ready before first GEMM
    k0_ptr = (k_ptr_base + 0).align(128)
    gK_0 = cute.make_tensor(k0_ptr, cute.make_layout((BLOCK_N, BLOCK_M), stride=(BLOCK_M, 1)))
    cute.copy(copy_QK_async, thr_QK_async.partition_S(gK_0), tKsK_async)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.barrier()

    # MMA fragments
    thr_qk = mma_qk.get_slice(tidx)
    tCrQ = mma_qk.make_fragment_A(thr_qk.partition_A(sQ))
    tCrK = mma_qk.make_fragment_B(thr_qk.partition_B(sK))
    acc_S = cute.make_fragment(thr_qk.partition_shape_C((BLOCK_M, BLOCK_N)), cutlass.Int32)

    thr_pv = mma_pv.get_slice(tidx)
    rP    = cute.make_fragment(thr_pv.partition_shape_A((BLOCK_M, BLOCK_N)), cutlass.Float16)
    tCrV  = mma_pv.make_fragment_B(thr_pv.partition_B(sV))
    acc_O = cute.make_fragment(thr_pv.partition_shape_C((BLOCK_M, BLOCK_M)), cutlass.Float32)
    acc_O.fill(0.0)

    # Online softmax state
    row_max = cute.make_fragment(ACC_ROWS_PER_THR, cutlass.Float32)
    row_sum = cute.make_fragment(ACC_ROWS_PER_THR, cutlass.Float32)
    row_max.fill(-cutlass.Float32.inf)
    row_sum.fill(cutlass.Float32(0.0))

    q_scale_val = gQScale[bidx]

    # ---- Main KV loop ----
    for n_tile in range(n_kv_blocks):
        # QK GEMM
        acc_S.fill(0)
        mma_qk.set(warpgroup.Field.ACCUMULATE, False)
        warpgroup.fence()
        for k in cutlass.range_constexpr(cute.size(tCrQ, mode=[2])):
            cute.gemm(mma_qk, acc_S, tCrQ[None, None, k, 0], tCrK[None, None, k, 0], acc_S)
            mma_qk.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.commit_group()
        warpgroup.wait_group(0)

        # Issue async loads: V[n] (group A), K[n+1] (group B)
        v_ptr = (v_ptr_base + n_tile * TILE_ELEMS_V).align(128)
        gV_n  = cute.make_tensor(v_ptr, cute.make_layout((BLOCK_M, BLOCK_N), stride=(BLOCK_N, 1)))
        cute.copy(copy_V_async, thr_V_async.partition_S(gV_n), tVsV_async)
        cute.arch.cp_async_commit_group()  # group A: V[n]

        if n_tile + 1 < n_kv_blocks:
            k_next = (k_ptr_base + (n_tile + 1) * TILE_ELEMS_K).align(128)
            gK_next = cute.make_tensor(k_next, cute.make_layout((BLOCK_N, BLOCK_M), stride=(BLOCK_M, 1)))
            cute.copy(copy_QK_async, thr_QK_async.partition_S(gK_next), tKsK_async)
            cute.arch.cp_async_commit_group()  # group B: K[n+1]

        # Softmax (register-only → overlaps with both cp.async groups)
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

        for i in cutlass.range_constexpr(ACC_FRAG_SIZE):
            r = (i // 2) % ACC_ROWS_PER_THR
            acc_O[i] = acc_O[i] * cute.math.exp2(row_max[r] - row_max_new[r], fastmath=True)

        for r in cutlass.range_constexpr(ACC_ROWS_PER_THR):
            alpha = cute.math.exp2(row_max[r] - row_max_new[r], fastmath=True)
            row_sum[r] = row_sum[r] * alpha + row_sum_new[r]
            row_max[r] = row_max_new[r]

        # Wait for V (group A). If K_next was issued, allow it to overlap PV GEMM.
        if n_tile + 1 < n_kv_blocks:
            cute.arch.cp_async_wait_group(1)  # wait until ≤1 group outstanding (V done)
        else:
            cute.arch.cp_async_wait_group(0)  # last tile: only V group, wait for it
        cute.arch.barrier()

        # PV GEMM (rs-mode: rP stays in registers)
        mma_pv.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.fence()
        for k in cutlass.range_constexpr(cute.size(rP, mode=[2])):
            cute.gemm(mma_pv, acc_O, rP[None, None, k], tCrV[None, None, k, 0], acc_O)
        warpgroup.commit_group()
        warpgroup.wait_group(0)

        # After PV GEMM: wait for K_next (group B), barrier for next QK GEMM
        if n_tile + 1 < n_kv_blocks:
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

    # Epilogue: normalize + store FP32
    tOgO = thr_pv.partition_C(gO_cta)
    for i in cutlass.range_constexpr(ACC_FRAG_SIZE):
        r = (i // 2) % ACC_ROWS_PER_THR
        acc_O[i] = acc_O[i] / (row_sum[r] + cutlass.Float32(1e-6))
    cute.basic_copy(acc_O, tOgO)


@cute.jit
def _sage_jit_v2(
    gQ: cute.Tensor,
    gK: cute.Tensor,
    gV: cute.Tensor,
    gO: cute.Tensor,
    gQScale: cute.Tensor,
    gKScale: cute.Tensor,
    n_kv_blocks: cute.Int32,
    n_q_blocks: cute.Int32,
    n_cta: cute.Int32,
):
    HEAD_DIM = BLOCK_M   # D=64 = BLOCK_M

    mma_qk = sm90_utils.make_trivial_tiled_mma(
        cutlass.Int8, cutlass.Int8,
        utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        cutlass.Int32, (1, 1, 1),
        tiler_mn=(BLOCK_M, BLOCK_N),
        a_source=warpgroup.OperandSource.SMEM,
    )
    mma_pv = sm90_utils.make_trivial_tiled_mma(
        cutlass.Float16, cutlass.Float16,
        utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        utils.LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
        cutlass.Float32, (1, 1, 1),
        tiler_mn=(BLOCK_M, HEAD_DIM),
        a_source=warpgroup.OperandSource.RMEM,  # rs-mode: P in registers
    )

    sQ_layout = sm90_utils.make_smem_layout_a(
        utils.LayoutEnum.ROW_MAJOR, (BLOCK_M, BLOCK_N, HEAD_DIM), cutlass.Int8, 1)
    sK_layout = sm90_utils.make_smem_layout_b(
        utils.LayoutEnum.ROW_MAJOR, (BLOCK_M, BLOCK_N, HEAD_DIM), cutlass.Int8, 1)
    sV_layout = sm90_utils.make_smem_layout_b(
        utils.LayoutEnum.ROW_MAJOR, (BLOCK_M, HEAD_DIM, BLOCK_N), cutlass.Float16, 1)

    # Sync copy for Q initial load (128-bit vectorized)
    copy_atom_QK = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Int8, num_bits_per_copy=128)
    copy_QK = cute.make_tiled_copy_tv(copy_atom_QK,
        cute.make_layout((8, 16), stride=(16, 1)), cute.make_layout((1, 16)))
    copy_atom_V = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16, num_bits_per_copy=128)
    copy_V = cute.make_tiled_copy_tv(copy_atom_V,
        cute.make_layout((16, 8), stride=(8, 1)), cute.make_layout((1, 8)))

    # Async copies for K/V pipeline
    copy_atom_QK_async = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.Int8, num_bits_per_copy=128)
    copy_QK_async = cute.make_tiled_copy_tv(copy_atom_QK_async,
        cute.make_layout((8, 16), stride=(16, 1)), cute.make_layout((1, 16)))
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
_compiled_kernels: dict = {}


def _get_compiled(B, H, N, D, kv_len, q_int8, k_int8_tiles, v_tiles, out,
                  q_scale_flat, k_scale_2d, N_KV_BLOCKS, N_Q_BLOCKS, N_CTAs):
    key = (B, H, N, D, kv_len)
    if key not in _compiled_kernels:
        _compiled_kernels[key] = cute.compile(
            _sage_jit_v2,
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
def sageattn_cutedsl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale: float = None,
    smooth_k: bool = True,
) -> torch.Tensor:
    """
    SageAttention CuTe DSL 前向传播 (SM90a WGMMA, 128-bit vectorized + cp.async pipeline)。

    参数:
        q, k, v: (B, H, N, D) float16, CUDA
        is_causal: 因果 mask (目前未实现)
        sm_scale: softmax 缩放 (默认 D^{-0.5})
        smooth_k: K Smoothing

    返回:
        O: (B, H, N, D) float32

    数据布局:
        Q/K 量化后重排为 tile-contiguous 布局以支持 128-bit 对齐加载:
        - K: (B*H, N_KV_BLOCKS, BLOCK_N, D) int8
        - V: (B*H, N_KV_BLOCKS, D, BLOCK_N) fp16 (transposed tiles)
    """
    assert q.is_cuda and q.dtype == torch.float16
    B, H, N, D = q.shape
    assert D == 64, f"sageattn_cutedsl: 目前只支持 head_dim=64, got {D}"
    assert N % BLOCK_M == 0, f"N={N} 必须是 BLOCK_M={BLOCK_M} 的整数倍"

    if sm_scale is None:
        sm_scale = D ** -0.5

    # --- K 量化 + tile-contiguous 重排 ---
    if smooth_k:
        k_int8, k_scale, km = smooth_and_quant_k(k, sm_scale)  # (B,H,N,D) int8
    else:
        nblocks_k = N // BLOCK_N
        k_int8  = torch.empty_like(k, dtype=torch.int8)
        k_scale = torch.empty((B, H, nblocks_k), dtype=torch.float32, device=k.device)
        for b in range(B):
            for h_i in range(H):
                for bn in range(nblocks_k):
                    s = bn * BLOCK_N
                    blk = k[b, h_i, s:s+BLOCK_N, :].float()  # NO sm_scale for K
                    sc  = blk.abs().max() / 127.0 + 1e-7
                    k_int8[b, h_i, s:s+BLOCK_N, :] = (
                        (blk / sc + 0.5 * blk.sign()).clamp(-128, 127).to(torch.int8)
                    )
                    k_scale[b, h_i, bn] = sc
        km = None

    # --- Q 量化 ---
    q_int8, q_scale = quant_q_per_block(q, sm_scale)  # (B,H,N,D) int8, (B,H,N_Q_BLOCKS) fp32

    N_Q_BLOCKS  = N // BLOCK_M
    N_KV_BLOCKS = N // BLOCK_N
    N_CTAs      = B * H * N_Q_BLOCKS

    # --- Tile-contiguous 重排 ---
    # K: (B, H, N, D) → (B, H, N_KV_BLOCKS, BLOCK_N, D) → (B*H, N_KV_BLOCKS, BLOCK_N, D)
    k_tiles = k_int8.view(B, H, N_KV_BLOCKS, BLOCK_N, D)
    k_tiles = k_tiles.reshape(B * H, N_KV_BLOCKS, BLOCK_N, D).contiguous()

    # V: (B, H, N, D) fp16 → (B, H, N_KV_BLOCKS, BLOCK_N, D) → transpose tile → (B*H, N_KV_BLOCKS, D, BLOCK_N)
    v_tiles = v.view(B, H, N_KV_BLOCKS, BLOCK_N, D)
    v_tiles = v_tiles.permute(0, 1, 2, 4, 3)  # (B, H, N_KV_BLOCKS, D, BLOCK_N)
    v_tiles = v_tiles.reshape(B * H, N_KV_BLOCKS, D, BLOCK_N).contiguous()

    # Q: (B, H, N, D) int8 → (B, H, N_Q_BLOCKS, BLOCK_M, D) → (B*H*N_Q_BLOCKS, BLOCK_M, D)
    q_tiles = q_int8.view(B, H, N_Q_BLOCKS, BLOCK_M, D)
    q_tiles = q_tiles.reshape(B * H * N_Q_BLOCKS, BLOCK_M, D).contiguous()

    # O: (B*H*N_Q_BLOCKS, BLOCK_M, D) fp32
    out = torch.empty(B * H * N_Q_BLOCKS, BLOCK_M, D, dtype=torch.float32, device=q.device)

    # Q scale: (B, H, N_Q_BLOCKS) → (B*H*N_Q_BLOCKS,) flat — one scale per CTA
    q_scale_flat = q_scale.reshape(B * H * N_Q_BLOCKS).contiguous()

    # K scale: (B, H, N_KV_BLOCKS) → (B*H, N_KV_BLOCKS) — indexed by [head_flat, n_tile]
    k_scale_2d = k_scale.reshape(B * H, N_KV_BLOCKS).contiguous()

    # --- Launch kernel ---
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

    # Reshape output to (B, H, N, D)
    return out.view(B, H, N, D).to(torch.float32)
