"""
FP8 量化 Triton Kernel

实现两种量化方案的 Triton 版本：
  - Per-Tensor FP8 量化 + GEMM
  - Per-Block FP8 量化 + GEMM（group_size=128）

Triton FP8 GEMM 采用标准 tiled GEMM，在 tile 内处理 scale。
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

FP8_MAX = 448.0
FP8_DTYPE = torch.float8_e4m3fn


# ============================================================
# Part 1：Per-Tensor FP8 量化 Kernel
# ============================================================

@triton.jit
def _fp8_per_tensor_quant_kernel(
    x_ptr,          # 输入 (M, K)，FP16/BF16/FP32
    q_ptr,          # 输出 (M, K)，FP8
    inv_scale_ptr,  # 输出 scalar（解量化 scale），float32
    numel: tl.constexpr,
    fp8_max: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Per-Tensor FP8 量化（二次 pass：第一次求 amax，第二次量化）。

    注意：这里是量化 pass（amax 已由外部提前计算好，通过 inv_scale_ptr 传入 scale）。
    实际工作：x * (fp8_max / amax)，clamp，转 FP8。
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    inv_scale = tl.load(inv_scale_ptr)   # 解量化 scale = amax / fp8_max
    scale = fp8_max / inv_scale          # 量化 scale

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    q_f32 = x * scale
    q_f32 = tl.minimum(tl.maximum(q_f32, -fp8_max), fp8_max)
    # 存储为 float8e4m3：Triton 支持 tl.float8e4m3 cast
    tl.store(q_ptr + offs, q_f32.to(tl.float8e4m3fnuz), mask=mask)


def triton_fp8_per_tensor_quant(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-Tensor FP8 量化（Triton 实现）。

    Args:
        x : 任意形状 float tensor

    Returns:
        q         : float8_e4m3fn，与 x 相同形状
        inv_scale : scalar float32（解量化 scale = amax / 448.0）
    """
    x_f32 = x.float()
    amax = x_f32.abs().amax().clamp(min=1e-12)
    inv_scale = (amax / FP8_MAX).to(torch.float32)

    q = torch.empty_like(x, dtype=FP8_DTYPE)
    numel = x.numel()

    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)

    _fp8_per_tensor_quant_kernel[grid](
        x_f32.contiguous().flatten(),
        q.flatten(),
        inv_scale,
        numel=numel,
        fp8_max=FP8_MAX,
        BLOCK=BLOCK,
    )
    return q, inv_scale


# ============================================================
# Part 2：Per-Block FP8 激活量化 Kernel（group_size=128）
# ============================================================

@triton.jit
def _fp8_per_block_act_quant_kernel(
    x_ptr,          # (M, K) float
    q_ptr,          # (M, K) float8
    inv_scales_ptr, # (M, K // GROUP_SIZE) float32
    M, K,
    GROUP_SIZE: tl.constexpr,
    fp8_max: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    Per-Block FP8 激活量化（沿 K 维分组）。

    每个 program 处理一行（pid_m = row index），
    对该行的每个 group 求 amax，量化，写 scale。
    """
    pid_m = tl.program_id(0) * BLOCK_M + tl.program_id(1)
    if pid_m >= M:
        return

    ngroups = K // GROUP_SIZE

    for g in tl.static_range(0, 1):  # 展开技巧，实际用 runtime loop
        pass

    for g in range(ngroups):
        k_start = g * GROUP_SIZE
        k_offs = k_start + tl.arange(0, GROUP_SIZE)
        mask = k_offs < K

        x_g = tl.load(x_ptr + pid_m * K + k_offs, mask=mask, other=0.0).to(tl.float32)

        amax = tl.max(tl.abs(x_g), axis=0).to(tl.float32)
        amax = tl.maximum(amax, 1e-12)
        inv_scale = amax / fp8_max         # 解量化 scale
        scale = fp8_max / amax             # 量化 scale

        q_g = x_g * scale
        q_g = tl.minimum(tl.maximum(q_g, -fp8_max), fp8_max)
        tl.store(q_ptr + pid_m * K + k_offs,
                 q_g.to(tl.float8e4m3fnuz), mask=mask)
        tl.store(inv_scales_ptr + pid_m * ngroups + g, inv_scale)


def triton_fp8_per_block_act_quant(
    x: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-Block FP8 激活量化（Triton 实现）。

    Args:
        x          : (M, K) float tensor
        group_size : 每组元素数（默认 128）

    Returns:
        q         : (M, K) float8_e4m3fn
        inv_scales: (M, K // group_size) float32
    """
    M, K = x.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    ngroups = K // group_size

    q = torch.empty_like(x, dtype=FP8_DTYPE)
    inv_scales = torch.empty(M, ngroups, dtype=torch.float32, device=x.device)

    grid = (M,)
    _fp8_per_block_act_quant_kernel[grid](
        x.float().contiguous(),
        q,
        inv_scales,
        M, K,
        GROUP_SIZE=group_size,
        fp8_max=FP8_MAX,
        BLOCK_M=1,
    )
    return q, inv_scales


# ============================================================
# Part 3：Per-Block FP8 GEMM Kernel（Triton tiled GEMM）
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'num_warps': 8, 'num_stages': 3}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fp8_per_block_gemm_kernel(
    # A: (M, K) FP8
    a_ptr,
    a_inv_scales_ptr,  # (M, K // GROUP_SIZE) float32
    # B: (N, K) FP8
    b_ptr,
    b_inv_scales_ptr,  # (N // GROUP_SIZE, K // GROUP_SIZE) float32
    # 输出
    out_ptr,           # (M, N) out_dtype
    # 维度
    M, N, K,
    GROUP_SIZE: tl.constexpr,
    fp8_max: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # 必须等于 GROUP_SIZE（128）
):
    """
    Per-Block FP8 GEMM：
      out[i,j] = Σ_g inv_scale_a[i,g] * inv_scale_b[j//128,g] * dot(qa[i,g*128:], qb[j,g*128:])

    BLOCK_K == GROUP_SIZE，每个 K-tile 对应一个 group。
    在 tile 内用 FP16 tl.dot，先 dequant 再 dot。
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

        # 加载 A tile: (BLOCK_M, GROUP_SIZE) FP8 → dequant 为 FP16
        a_ptrs = a_ptr + m_offs[:, None] * K + k_offs[None, :]
        a_tile = tl.load(a_ptrs,
                         mask=(m_offs[:, None] < M) & (k_offs[None, :] < K),
                         other=0.0)
        # 加载 A scale: (BLOCK_M,)
        a_scale_ptrs = a_inv_scales_ptr + m_offs * ngroups_k + k_tile
        a_scale = tl.load(a_scale_ptrs, mask=m_offs < M, other=1.0).to(tl.float32)
        # dequant A
        a_dq = a_tile.to(tl.float32) * a_scale[:, None]  # (BLOCK_M, GROUP_SIZE)

        # 加载 B tile: (BLOCK_N, GROUP_SIZE) FP8 → dequant 为 FP16
        b_ptrs = b_ptr + n_offs[:, None] * K + k_offs[None, :]
        b_tile = tl.load(b_ptrs,
                         mask=(n_offs[:, None] < N) & (k_offs[None, :] < K),
                         other=0.0)
        # 加载 B scale: (BLOCK_N,)
        # b_inv_scales: (N//GROUP_SIZE, K//GROUP_SIZE)，b_block_row = n_offs // GROUP_SIZE
        b_block_n = n_offs // GROUP_SIZE     # 每个 n 对应的 block row（整数除法）
        b_scale_ptrs = b_inv_scales_ptr + b_block_n * ngroups_k + k_tile
        b_scale = tl.load(b_scale_ptrs, mask=n_offs < N, other=1.0).to(tl.float32)
        # dequant B
        b_dq = b_tile.to(tl.float32) * b_scale[:, None]  # (BLOCK_N, GROUP_SIZE)

        # tl.dot: (BLOCK_M, GROUP_SIZE) x (BLOCK_N, GROUP_SIZE).T -> (BLOCK_M, BLOCK_N)
        acc += tl.dot(a_dq.to(tl.float16), tl.trans(b_dq.to(tl.float16)))

    # 存储输出
    out_ptrs = out_ptr + m_offs[:, None] * N + n_offs[None, :]
    tl.store(out_ptrs, acc.to(tl.float16),
             mask=(m_offs[:, None] < M) & (n_offs[None, :] < N))


def triton_fp8_per_block_gemm(
    a: torch.Tensor,
    a_inv_scales: torch.Tensor,
    b: torch.Tensor,
    b_inv_scales: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Per-Block FP8 GEMM（Triton 实现）。

    Args:
        a           : (M, K) float8_e4m3fn
        a_inv_scales: (M, K // 128) float32
        b           : (N, K) float8_e4m3fn
        b_inv_scales: (N // 128, K // 128) float32
        out_dtype   : 输出精度

    Returns:
        out : (M, N) out_dtype
    """
    M, K = a.shape
    N = b.shape[0]
    GROUP_SIZE = 128

    assert K % GROUP_SIZE == 0
    assert a_inv_scales.shape == (M, K // GROUP_SIZE)
    assert b_inv_scales.shape == (N // GROUP_SIZE, K // GROUP_SIZE)

    # Triton 输出为 FP16（kernel 内部 acc 是 FP32，存储为 FP16）
    out = torch.empty(M, N, dtype=torch.float16, device=a.device)

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    _fp8_per_block_gemm_kernel[grid](
        a, a_inv_scales,
        b, b_inv_scales,
        out,
        M, N, K,
        GROUP_SIZE=GROUP_SIZE,
        fp8_max=FP8_MAX,
    )

    if out_dtype != torch.float16:
        out = out.to(out_dtype)
    return out


# ============================================================
# Part 4：Per-Tensor FP8 GEMM Kernel（Triton，scalar scale）
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'num_warps': 8, 'num_stages': 3}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fp8_per_tensor_gemm_kernel(
    a_ptr,              # (M, K) FP8
    b_ptr,              # (N, K) FP8
    out_ptr,            # (M, N) FP16
    inv_scale_a,        # scalar float32
    inv_scale_b,        # scalar float32
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Per-Tensor FP8 GEMM：
      out[i,j] = inv_scale_a * inv_scale_b * dot(a[i,:], b[j,:])

    scalar scale 在 tile loop 外乘一次（效率更高）。
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k_start * BLOCK_K + tl.arange(0, BLOCK_K)

        a_tile = tl.load(a_ptr + m_offs[:, None] * K + k_offs[None, :],
                         mask=(m_offs[:, None] < M) & (k_offs[None, :] < K),
                         other=0.0)
        b_tile = tl.load(b_ptr + n_offs[:, None] * K + k_offs[None, :],
                         mask=(n_offs[:, None] < N) & (k_offs[None, :] < K),
                         other=0.0)

        # FP8 → FP16 进行 dot
        a_fp16 = a_tile.to(tl.float16)
        b_fp16 = b_tile.to(tl.float16)
        acc += tl.dot(a_fp16, tl.trans(b_fp16))

    # 乘 scalar scale（在 tile 结果上统一乘）
    alpha = inv_scale_a.to(tl.float32) * inv_scale_b.to(tl.float32)
    acc = acc * alpha

    out_ptrs = out_ptr + m_offs[:, None] * N + n_offs[None, :]
    tl.store(out_ptrs, acc.to(tl.float16),
             mask=(m_offs[:, None] < M) & (n_offs[None, :] < N))


def triton_fp8_per_tensor_gemm(
    a: torch.Tensor,
    a_inv_scale: torch.Tensor,
    b: torch.Tensor,
    b_inv_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Per-Tensor FP8 GEMM（Triton 实现）。

    Args:
        a           : (M, K) float8_e4m3fn
        a_inv_scale : scalar float32（解量化 scale）
        b           : (N, K) float8_e4m3fn
        b_inv_scale : scalar float32
        out_dtype   : 输出精度

    Returns:
        out : (M, N) out_dtype
    """
    M, K = a.shape
    N = b.shape[0]

    out = torch.empty(M, N, dtype=torch.float16, device=a.device)

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    _fp8_per_tensor_gemm_kernel[grid](
        a, b, out,
        a_inv_scale.float().item(),
        b_inv_scale.float().item(),
        M, N, K,
    )

    if out_dtype != torch.float16:
        out = out.to(out_dtype)
    return out
