"""
SageAttention V2 — Triton 实现（论文 SageAttention2: Ultra Efficient...）

相比 V1 的核心升级：

1. Q: per-warp INT8 量化（更细粒度，每个 warp 的 Q tile 独立 scale）
   - V1：每 BLKQ=128 个 token 一个 scale（per-block）
   - V2：每 WARPQ=16 个 token 一个 scale（per-warp within block）
   - 更细粒度 → 量化误差更小，尤其对 Q 的 outlier 有效

2. K: 仍然 per-block INT8（BLKK=64）+ K Smoothing

3. V: FP8 量化（E4M3，per-channel，不是 per-block）
   - 每个 head_dim 位置独立 scale（即 V 沿 seq dim 量化）
   - V Smoothing：v_smooth = v - mean(v, dim=seq)
     → 同 K Smoothing，降低离群值影响
   - 注意：V 在 PV 乘积后需要加回均值贡献：out += p_sum * vm

4. PV 乘积：
   - P (fp16) @ V_fp8 → fp32 acc（Triton 的 dot 可接受 fp8 输入）
   - 每列乘以 v_scale（dequantize）
   - 加回 V 均值：out_final = out_norm + (l_i_normalized) * vm

设计决策（本实现）：
- 用 Triton 模拟 per-warp INT8 Q：通过额外的 scale 向量（WARPQ 粒度）
- FP8 V：使用 torch.float8_e4m3fn，在 Triton 中以 fp8 加载
- V smoothing 在 Python 层完成，内核中加回均值
"""

import torch
import triton
import triton.language as tl
import math


# ============================================================
# Q Per-warp INT8 量化
# WARPQ=16: 每 16 个 token 一个 scale（per-warp，相比 V1 的 per-block 更细）
# ============================================================
@triton.jit
def _quant_q_per_warp_int8(
    X_ptr, O_ptr, Scale_ptr,
    stride_xb, stride_xh, stride_xn,
    stride_ob, stride_oh, stride_on,
    stride_sb, stride_sh,
    seq_len,
    sm_scale,
    C: tl.constexpr,
    BLKQ: tl.constexpr,     # 外层 block（128）
    WARPQ: tl.constexpr,    # warp 内粒度（16）
):
    """
    per-warp INT8 量化：将 BLKQ 分成 BLKQ//WARPQ 个 warp-tile，
    每个 warp-tile 独立计算 max/scale。
    """
    off_blk  = tl.program_id(0) // (BLKQ // WARPQ)
    off_warp = tl.program_id(0) %  (BLKQ // WARPQ)
    off_h    = tl.program_id(1)
    off_b    = tl.program_id(2)

    offs_n = off_blk * BLKQ + off_warp * WARPQ + tl.arange(0, WARPQ)
    offs_c = tl.arange(0, C)

    x_ptrs = X_ptr + off_b * stride_xb + off_h * stride_xh \
             + offs_n[:, None] * stride_xn + offs_c[None, :]
    o_ptrs = O_ptr + off_b * stride_ob + off_h * stride_oh \
             + offs_n[:, None] * stride_on + offs_c[None, :]
    s_ptr  = Scale_ptr + off_b * stride_sb + off_h * stride_sh \
             + off_blk * (BLKQ // WARPQ) + off_warp

    x = tl.load(x_ptrs, mask=offs_n[:, None] < seq_len, other=0.0)
    x = x.to(tl.float32) * sm_scale

    scale = tl.max(tl.abs(x)) / 127.0 + 1e-7
    x_q   = x / scale
    x_q  += 0.5 * tl.where(x_q >= 0, 1.0, -1.0)
    x_q   = x_q.to(tl.int8)

    tl.store(o_ptrs, x_q, mask=offs_n[:, None] < seq_len)
    tl.store(s_ptr, scale)


def quant_q_per_warp_int8(q: torch.Tensor, sm_scale: float,
                           BLKQ: int = 128, WARPQ: int = 16) -> tuple:
    """Q per-warp INT8 量化（HND 布局）"""
    B, H, seq, D = q.shape
    q_int8  = torch.empty_like(q, dtype=torch.int8)
    nwarps  = (triton.cdiv(seq, BLKQ)) * (BLKQ // WARPQ)
    q_scale = torch.empty((B, H, nwarps), dtype=torch.float32, device=q.device)

    grid = (nwarps, H, B)
    _quant_q_per_warp_int8[grid](
        q, q_int8, q_scale,
        q.stride(0), q.stride(1), q.stride(2),
        q_int8.stride(0), q_int8.stride(1), q_int8.stride(2),
        q_scale.stride(0), q_scale.stride(1),
        seq, sm_scale,
        C=D, BLKQ=BLKQ, WARPQ=WARPQ,
    )
    return q_int8, q_scale


# ============================================================
# K Per-block INT8（与 V1 相同，复用）
# ============================================================
@triton.jit
def _quant_per_block_int8(
    X_ptr, O_ptr, Scale_ptr,
    stride_xb, stride_xh, stride_xn,
    stride_ob, stride_oh, stride_on,
    stride_sb, stride_sh,
    seq_len, sm_scale,
    C: tl.constexpr, BLK: tl.constexpr,
):
    off_blk = tl.program_id(0)
    off_h   = tl.program_id(1)
    off_b   = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_c = tl.arange(0, C)

    x = tl.load(X_ptr + off_b * stride_xb + off_h * stride_xh
                + offs_n[:, None] * stride_xn + offs_c[None, :],
                mask=offs_n[:, None] < seq_len, other=0.0)
    x = x.to(tl.float32) * sm_scale
    scale = tl.max(tl.abs(x)) / 127.0 + 1e-7
    x_q   = (x / scale + 0.5 * tl.where(x / scale >= 0, 1.0, -1.0)).to(tl.int8)

    tl.store(O_ptr + off_b * stride_ob + off_h * stride_oh
             + offs_n[:, None] * stride_on + offs_c[None, :],
             x_q, mask=offs_n[:, None] < seq_len)
    tl.store(Scale_ptr + off_b * stride_sb + off_h * stride_sh + off_blk, scale)


def quant_k_per_block_int8(k: torch.Tensor, BLKK: int = 64) -> tuple:
    """K per-block INT8 量化（HND 布局）"""
    B, H, seq, D = k.shape
    k_int8  = torch.empty_like(k, dtype=torch.int8)
    nblocks = triton.cdiv(seq, BLKK)
    k_scale = torch.empty((B, H, nblocks), dtype=torch.float32, device=k.device)

    _quant_per_block_int8[(nblocks, H, B)](
        k, k_int8, k_scale,
        k.stride(0), k.stride(1), k.stride(2),
        k_int8.stride(0), k_int8.stride(1), k_int8.stride(2),
        k_scale.stride(0), k_scale.stride(1),
        seq, 1.0, C=D, BLK=BLKK,
    )
    return k_int8, k_scale


# ============================================================
# V Per-channel FP8 量化（在 Python 层完成）
# ============================================================
def quant_v_per_channel_fp8(v: torch.Tensor, smooth_v: bool = True):
    """
    V per-channel FP8 量化。

    SageAttention V2 中 V 的量化方式（论文 Appendix）：
    - 先 V Smoothing：v_smooth = v - mean(v, dim=seq)，保存 vm = mean(v, dim=seq)
    - 对 v_smooth 做 per-channel（沿 head_dim）量化到 FP8 E4M3
    - per-channel：每个 head_dim 位置的 scale = max(|v_smooth[:,i]|) / 448

    返回：
        v_fp8:  (B, H, kv_len, D)，torch.float8_e4m3fn
        v_scale: (B, H, D)，每个 head_dim channel 的 scale
        vm:     (B, H, 1, D) or None，V 的序列均值（用于恢复）
    """
    B, H, seq, D = v.shape

    if smooth_v:
        vm = v.float().mean(dim=2, keepdim=True)   # (B, H, 1, D)
        v_smooth = (v.float() - vm).to(torch.float16)
    else:
        vm = None
        v_smooth = v.half()

    # Per-channel scale: max(|v_smooth|) / 448 along seq dim
    v_max = v_smooth.float().abs().amax(dim=2)     # (B, H, D)
    v_scale = v_max / 448.0 + 1e-7                 # (B, H, D)

    v_fp8 = (v_smooth.float() / v_scale.unsqueeze(2)).clamp(-448, 448)
    v_fp8 = v_fp8.to(torch.float8_e4m3fn)          # (B, H, kv_len, D)

    return v_fp8, v_scale, vm


# ============================================================
# SageAttention V2 主内核
# INT8 Q (per-warp) + INT8 K (per-block) + FP8 V (per-channel)
# ============================================================
@triton.jit
def _sage_v2_kernel(
    Q_int8, Q_scale,    # Q INT8 + per-warp scale (B,H,nwarps_total)
    K_int8, K_scale,    # K INT8 + per-block scale (B,H,nblocks_k)
    V_fp8, V_scale,     # V FP8  + per-channel scale (B,H,D)
    V_mean,             # V 均值 (B,H,D)，可为 None（通过 HAS_VM 控制）
    Out,
    stride_qb, stride_qh, stride_qn,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_on,
    B, H_qo, H_kv,
    qo_len, kv_len,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WARPQ:   tl.constexpr,     # per-warp Q 粒度
    CAUSAL:  tl.constexpr,
    HAS_VM:  tl.constexpr,     # 是否有 V mean（V smoothing）
):
    start_m = tl.program_id(0)
    off_h   = tl.program_id(1)
    off_b   = tl.program_id(2)

    num_kv_groups = H_qo // H_kv
    off_kv_h = off_h // num_kv_groups

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # 加载 Q (INT8) [BLOCK_M, HEAD_DIM]
    Q_base = Q_int8 + off_b * stride_qb + off_h * stride_qh
    q = tl.load(Q_base + offs_m[:, None] * stride_qn + offs_d[None, :],
                mask=offs_m[:, None] < qo_len, other=0)

    # Q per-warp scale：每 WARPQ 行一个 scale
    # scale shape: (nwarps_total,)，其中 nwarps_total = cdiv(qo_len, WARPQ)
    # start_m * (BLOCK_M//WARPQ) + warp_within_block
    warp_per_block = BLOCK_M // WARPQ
    q_scale_base = Q_scale + (off_b * H_qo + off_h) * tl.cdiv(qo_len, WARPQ)

    # K/V 基址
    K_base = K_int8 + off_b * stride_kb + off_kv_h * stride_kh
    K_scale_base = K_scale + (off_b * H_kv + off_kv_h) * tl.cdiv(kv_len, BLOCK_N)
    V_base = V_fp8  + off_b * stride_vb + off_kv_h * stride_vh
    V_scale_base = V_scale + (off_b * H_kv + off_kv_h) * HEAD_DIM

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    hi = (start_m + 1) * BLOCK_M if CAUSAL else kv_len

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n  = start_n + tl.arange(0, BLOCK_N)

        # 加载 K (INT8) 为 [HEAD_DIM, BLOCK_N]（转置访问）
        k = tl.load(K_base + offs_d[:, None] * 1 + offs_n[None, :] * stride_kn,
                    mask=offs_n[None, :] < kv_len, other=0)   # [D, BN]
        k_scale = tl.load(K_scale_base + start_n // BLOCK_N)

        # QK: [BM, D] @ [D, BN] → [BM, BN]
        # 使用 per-warp Q scale：对 BLOCK_M 内的每个 WARPQ 行乘不同 scale
        # 简化：使用 Q 的 per-block scale（取 BLOCK_M 内的均值 scale）
        # 完整 per-warp 实现需要分段乘 scale，这里使用一个近似：
        # 对每个 warp block 内的行乘对应 scale，然后加权求和
        # 由于 Triton 不支持向量化的 scale apply，我们使用循环（编译时展开）
        qk_raw = tl.dot(q, k).to(tl.float32)  # [BM, BN], INT32

        # Apply per-warp Q scale（每 WARPQ 行对应一个 scale）
        # 构造 scale 向量 [BLOCK_M, 1]
        warp_idx = tl.arange(0, BLOCK_M) // WARPQ   # [BM]，每行所属 warp
        q_scales_vec = tl.load(q_scale_base + start_m * warp_per_block + warp_idx,
                               mask=offs_m < qo_len, other=1.0)  # [BM]
        qk = qk_raw * q_scales_vec[:, None] * k_scale   # [BM, BN]

        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
        qk = tl.where(offs_n[None, :] < kv_len, qk, float("-inf"))

        # Online softmax
        m_ij  = tl.maximum(m_i, tl.max(qk, axis=1))
        p     = tl.math.exp2(qk - m_ij[:, None])
        l_ij  = tl.sum(p, axis=1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i   = l_i * alpha + l_ij
        acc   = acc * alpha[:, None]
        m_i   = m_ij

        # 加载 V (FP8) [BN, D]
        v_fp8 = tl.load(V_base + offs_n[:, None] * stride_vn + offs_d[None, :],
                        mask=offs_n[:, None] < kv_len, other=0.0)  # [BN, D]

        # PV: P [BM, BN] @ V [BN, D] → [BM, D]
        # FP8 V 在 dot 中会自动上转为 fp32
        acc += tl.dot(p.to(tl.float16), v_fp8.to(tl.float16), out_dtype=tl.float32)

    # 归一化
    acc = acc / l_i[:, None]

    # V scale dequantize：每个 head_dim 位置乘以对应 scale
    v_scales = tl.load(V_scale_base + offs_d)   # [D]
    acc = acc * v_scales[None, :]

    # V mean 恢复：out += l_i_norm_sum * vm
    # 由于 acc 已经归一化（除以 l_i），而 softmax 权重 sum = 1
    # 只需 acc += vm（均值向量，相当于 sum_j p_j * vm = vm）
    if HAS_VM:
        vm = tl.load(V_mean + (off_b * H_kv + off_kv_h) * HEAD_DIM + offs_d)  # [D]
        acc = acc + vm[None, :]

    # 写回
    O_base = Out + off_b * stride_ob + off_h * stride_oh
    tl.store(O_base + offs_m[:, None] * stride_on + offs_d[None, :],
             acc.to(Out.type.element_ty),
             mask=offs_m[:, None] < qo_len)


# ============================================================
# Python 包装
# ============================================================
def sageattn_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale: float = None,
    smooth_k: bool = True,
    smooth_v: bool = True,
) -> torch.Tensor:
    """
    SageAttention V2 前向传播。

    相对 V1 的改进：
    - Q: per-warp INT8（WARPQ=16，比 V1 的 per-block 128 精度更高）
    - V: FP8 E4M3 量化（per-channel）+ V Smoothing
    - 更高的量化精度 → 在相同速度下误差更小

    参数:
        q, k, v: (B, H, N, D) float16，HND 布局
        is_causal: 是否使用因果 mask
        sm_scale: softmax 缩放
        smooth_k: 是否 K Smoothing（减去序列均值）
        smooth_v: 是否 V Smoothing（减去序列均值，用 FP8 量化）
    """
    assert q.is_cuda and q.dtype == torch.float16
    B, H_qo, qo_len, D = q.shape
    _, H_kv, kv_len, _ = k.shape

    if sm_scale is None:
        sm_scale = D ** -0.5

    # K Smoothing
    if smooth_k:
        km = k.mean(dim=2, keepdim=True)
        k  = k - km

    # V FP8 量化 + V Smoothing
    v_fp8, v_scale, vm = quant_v_per_channel_fp8(v, smooth_v=smooth_v)

    # Q per-warp INT8 量化
    BLKQ, WARPQ = 128, 16
    q_int8, q_scale = quant_q_per_warp_int8(q, sm_scale=sm_scale * 1.44269504,
                                             BLKQ=BLKQ, WARPQ=WARPQ)

    # K per-block INT8 量化
    BLKK = 64
    k_int8, k_scale = quant_k_per_block_int8(k, BLKK)

    # V mean 指针（若有）
    if smooth_v and vm is not None:
        vm_flat = vm.squeeze(2).contiguous()   # (B, H_kv, D)
    else:
        vm_flat = torch.zeros(B, H_kv, D, dtype=torch.float32, device=q.device)

    out = torch.empty_like(q)

    BLOCK_M = 128
    BLOCK_N = 64
    assert D in (64, 128), f"SageAttention v2 只支持 head_dim=64 或 128"

    grid = (triton.cdiv(qo_len, BLOCK_M), H_qo, B)
    _sage_v2_kernel[grid](
        q_int8, q_scale,
        k_int8, k_scale,
        v_fp8, v_scale.contiguous(),
        vm_flat,
        out,
        q_int8.stride(0), q_int8.stride(1), q_int8.stride(2),
        k_int8.stride(0), k_int8.stride(1), k_int8.stride(2),
        v_fp8.stride(0),  v_fp8.stride(1),  v_fp8.stride(2),
        out.stride(0),    out.stride(1),    out.stride(2),
        B, H_qo, H_kv,
        qo_len, kv_len,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        WARPQ=WARPQ,
        CAUSAL=is_causal,
        HAS_VM=smooth_v,
        num_warps=4 if D == 64 else 8,
        num_stages=2,
    )
    return out
