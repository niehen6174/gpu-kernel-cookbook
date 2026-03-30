"""
FP8 量化 Triton Kernel — WGMMA 版本

关键设计：
  - FP8 指针直接以 float8e4nv 类型传入（不做 uint8 转换）
  - tl.dot(fp8_a, tl.trans(fp8_b)) → Triton 自动发射 wgmma.mma_async（SM90+）
  - 不再做 .to(tl.float16) 的 cast，避免退化成 HMMA

实现方案：
  - Per-Tensor FP8 GEMM：K-loop 后统一乘 scalar scale
  - Per-Block FP8 GEMM：每个 K-tile 乘对应 per-group scale，再累加

Triton 版本要求：>= 3.0（float8e4nv dtype 支持）
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

FP8_MAX = 448.0
FP8_DTYPE = torch.float8_e4m3fn


# ============================================================
# Part 1：Per-Tensor FP8 量化（Python 层，直接复用）
# ============================================================

def triton_fp8_per_tensor_quant(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-Tensor FP8 量化（Python 实现，用于配合 Triton GEMM）。

    Returns:
        q         : float8_e4m3fn，与 x 相同形状
        inv_scale : scalar float32（解量化 scale = amax / 448.0）
    """
    x_f32 = x.float()
    amax = x_f32.abs().amax().clamp(min=1e-12)
    inv_scale = (amax / FP8_MAX).to(torch.float32)
    scale = FP8_MAX / amax
    q = (x_f32 * scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return q, inv_scale


# ============================================================
# Part 2：Per-Block FP8 激活量化（Python 层）
# ============================================================

def triton_fp8_per_block_act_quant(
    x: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-Block FP8 激活量化（沿 K 维 group_size 分组）。

    Returns:
        q         : (M, K) float8_e4m3fn
        inv_scales: (M, K // group_size) float32
    """
    M, K = x.shape
    assert K % group_size == 0
    ngroups = K // group_size

    x_g = x.float().reshape(M, ngroups, group_size)
    amax_g = x_g.abs().amax(dim=-1).clamp(min=1e-12)
    inv_scales = amax_g / FP8_MAX
    q_g = x_g / inv_scales.unsqueeze(-1)
    q = q_g.clamp(-FP8_MAX, FP8_MAX).reshape(M, K).to(FP8_DTYPE)
    return q, inv_scales.to(torch.float32)


# ============================================================
# Part 3：Per-Tensor FP8 GEMM Kernel（WGMMA）
# ============================================================

@triton.autotune(
    configs=[
        # FP8 WGMMA 最优：BLOCK_M=128, BLOCK_N=256, BLOCK_K=64
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,
                       'num_warps': 8, 'num_stages': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,
                       'num_warps': 8, 'num_stages': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64,
                       'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128,
                       'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,
                       'num_warps': 4, 'num_stages': 3}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fp8_per_tensor_gemm_wgmma_kernel(
    a_ptr,          # (M, K) float8e4nv
    b_ptr,          # (N, K) float8e4nv（B 以行优先存储，每行长 K）
    out_ptr,        # (M, N) bfloat16
    inv_scale_a,    # scalar float32：inv_scale_a = amax_a / 448
    inv_scale_b,    # scalar float32：inv_scale_b = amax_b / 448
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Per-Tensor FP8 GEMM：out = (a @ b.T) * inv_scale_a * inv_scale_b

    关键：
      - a_ptr / b_ptr 为 float8e4nv 指针，tl.load 直接得到 float8e4nv tensor
      - tl.dot(a_tile, tl.trans(b_tile)) → wgmma.mma_async（SM90+）
      - scalar scale 在 K-loop 结束后统一乘（最高效）
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k_start * BLOCK_K + tl.arange(0, BLOCK_K)

        # 直接加载 FP8（float8e4nv），不做任何类型转换
        a_tile = tl.load(
            a_ptr + m_offs[:, None] * K + k_offs[None, :],
            mask=(m_offs[:, None] < M) & (k_offs[None, :] < K),
            other=0.0,
        )
        b_tile = tl.load(
            b_ptr + n_offs[:, None] * K + k_offs[None, :],
            mask=(n_offs[:, None] < N) & (k_offs[None, :] < K),
            other=0.0,
        )

        # WGMMA：FP8 dot → FP32 accumulator
        # tl.dot 看到 float8e4nv 输入，SM90+ 自动发射 wgmma.mma_async
        acc = tl.dot(a_tile, tl.trans(b_tile), acc=acc, out_dtype=tl.float32)

    # 应用标量 scale（一次乘法，比 per-element 更高效）
    alpha = inv_scale_a.to(tl.float32) * inv_scale_b.to(tl.float32)
    acc = acc * alpha

    # 写输出（BF16）
    out_ptrs = out_ptr + m_offs[:, None] * N + n_offs[None, :]
    tl.store(
        out_ptrs,
        acc.to(tl.bfloat16),
        mask=(m_offs[:, None] < M) & (n_offs[None, :] < N),
    )


def triton_fp8_per_tensor_gemm(
    a: torch.Tensor,
    a_inv_scale: torch.Tensor,
    b: torch.Tensor,
    b_inv_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Per-Tensor FP8 GEMM（Triton WGMMA 实现）。

    Args:
        a           : (M, K) float8_e4m3fn
        a_inv_scale : scalar float32
        b           : (N, K) float8_e4m3fn
        b_inv_scale : scalar float32
        out_dtype   : 输出精度（默认 bfloat16）

    Returns:
        out : (M, N) out_dtype
    """
    assert a.dtype == torch.float8_e4m3fn, f"a.dtype must be float8_e4m3fn, got {a.dtype}"
    assert b.dtype == torch.float8_e4m3fn, f"b.dtype must be float8_e4m3fn, got {b.dtype}"

    M, K = a.shape
    N = b.shape[0]

    # 输出直接为 BF16（kernel 内部 acc=FP32，store 为 BF16）
    out = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)

    # 确保 contiguous
    a = a.contiguous()
    b = b.contiguous()

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    _fp8_per_tensor_gemm_wgmma_kernel[grid](
        a, b, out,
        a_inv_scale.float().item(),
        b_inv_scale.float().item(),
        M, N, K,
    )

    if out_dtype != torch.bfloat16:
        out = out.to(out_dtype)
    return out


# ============================================================
# Part 4：Per-Block FP8 GEMM Kernel（WGMMA + per-group scale）
# ============================================================

@triton.autotune(
    configs=[
        # BLOCK_K = GROUP_SIZE = 128，FP8 WGMMA 每个 K-tile 一个 group
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128,
                       'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128,
                       'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128,
                       'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128,
                       'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128,
                       'num_warps': 4, 'num_stages': 3}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fp8_per_block_gemm_wgmma_kernel(
    # A: (M, K) FP8
    a_ptr,
    a_inv_scales_ptr,  # (M, K // GROUP_SIZE) float32
    # B: (N, K) FP8（行优先，行长 K）
    b_ptr,
    b_inv_scales_ptr,  # (N // GROUP_SIZE, K // GROUP_SIZE) float32
    # 输出
    out_ptr,           # (M, N) bfloat16
    # 维度
    M, N, K,
    GROUP_SIZE: tl.constexpr,   # = BLOCK_K = 128
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,      # 必须 == GROUP_SIZE
):
    """
    Per-Block FP8 GEMM：
      out[i,j] = Σ_g inv_scale_a[i,g] * inv_scale_b[j//128,g]
                   * raw_dot(q_a[i,g*128:], q_b[j,g*128:])

    策略：
      - BLOCK_K = GROUP_SIZE = 128，每 K-tile 恰好对应一个 group
      - 对每个 K-tile：raw = tl.dot(fp8_a, fp8_b.T)（WGMMA）
      - 然后：acc += raw * a_scale[:, None] * b_scale[None, :]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offs = m_start + tl.arange(0, BLOCK_M)
    n_offs = n_start + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    ngroups_k = K // GROUP_SIZE

    for k_tile in range(ngroups_k):
        k_offs = k_tile * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

        # 加载 A tile：(BLOCK_M, GROUP_SIZE) float8e4nv
        a_tile = tl.load(
            a_ptr + m_offs[:, None] * K + k_offs[None, :],
            mask=(m_offs[:, None] < M) & (k_offs[None, :] < K),
            other=0.0,
        )
        # 加载 B tile：(BLOCK_N, GROUP_SIZE) float8e4nv
        b_tile = tl.load(
            b_ptr + n_offs[:, None] * K + k_offs[None, :],
            mask=(n_offs[:, None] < N) & (k_offs[None, :] < K),
            other=0.0,
        )

        # WGMMA：raw FP8 dot（不带任何 scale）
        raw = tl.dot(a_tile, tl.trans(b_tile), out_dtype=tl.float32)

        # 加载当前 K-group 的 scale
        # a_inv_scales: (M, K//GROUP_SIZE)，取 [:, k_tile]
        a_scale = tl.load(
            a_inv_scales_ptr + m_offs * ngroups_k + k_tile,
            mask=m_offs < M,
            other=1.0,
        ).to(tl.float32)  # (BLOCK_M,)

        # b_inv_scales: (N//GROUP_SIZE, K//GROUP_SIZE)
        # b 的 N-block：n_offs // GROUP_SIZE
        b_block_n = n_offs // GROUP_SIZE
        b_scale = tl.load(
            b_inv_scales_ptr + b_block_n * ngroups_k + k_tile,
            mask=n_offs < N,
            other=1.0,
        ).to(tl.float32)  # (BLOCK_N,)

        # 应用 per-group scale，累加到 acc
        acc = acc + raw * a_scale[:, None] * b_scale[None, :]

    # 写输出（BF16）
    out_ptrs = out_ptr + m_offs[:, None] * N + n_offs[None, :]
    tl.store(
        out_ptrs,
        acc.to(tl.bfloat16),
        mask=(m_offs[:, None] < M) & (n_offs[None, :] < N),
    )


def triton_fp8_per_block_gemm(
    a: torch.Tensor,
    a_inv_scales: torch.Tensor,
    b: torch.Tensor,
    b_inv_scales: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Per-Block FP8 GEMM（Triton WGMMA 实现）。

    Args:
        a           : (M, K) float8_e4m3fn
        a_inv_scales: (M, K // 128) float32
        b           : (N, K) float8_e4m3fn
        b_inv_scales: (N // 128, K // 128) float32
        out_dtype   : 输出精度（默认 bfloat16）

    Returns:
        out : (M, N) out_dtype
    """
    assert a.dtype == torch.float8_e4m3fn
    assert b.dtype == torch.float8_e4m3fn

    M, K = a.shape
    N = b.shape[0]
    GROUP_SIZE = 128

    assert K % GROUP_SIZE == 0, f"K={K} must be divisible by GROUP_SIZE={GROUP_SIZE}"
    assert a_inv_scales.shape == (M, K // GROUP_SIZE), \
        f"a_inv_scales.shape={a_inv_scales.shape}, expected ({M}, {K // GROUP_SIZE})"
    assert b_inv_scales.shape == (N // GROUP_SIZE, K // GROUP_SIZE), \
        f"b_inv_scales.shape={b_inv_scales.shape}, expected ({N // GROUP_SIZE}, {K // GROUP_SIZE})"

    out = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)

    a = a.contiguous()
    b = b.contiguous()
    a_inv_scales = a_inv_scales.contiguous()
    b_inv_scales = b_inv_scales.contiguous()

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    _fp8_per_block_gemm_wgmma_kernel[grid](
        a, a_inv_scales,
        b, b_inv_scales,
        out,
        M, N, K,
        GROUP_SIZE=GROUP_SIZE,
    )

    if out_dtype != torch.bfloat16:
        out = out.to(out_dtype)
    return out
