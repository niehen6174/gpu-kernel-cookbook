"""
FP8 量化 CUTLASS 3.x SM90 — Python JIT 入口

功能：
1. FP8 Per-Tensor 量化 + GEMM（V1：scalar epilogue）
2. FP8 Per-Block  量化 + GEMM（V2：EVT per-row/col 近似）

JIT 编译：首次约 60-120s，之后缓存。

使用方式：
    from operators.fp8_quant.cutlass_fp8.kernel import (
        fp8_per_tensor_forward,
        fp8_per_block_forward,
        FP8_V1_AVAILABLE,
        FP8_V2_AVAILABLE,
    )
"""

import os
import torch
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# CUTLASS 路径
# ---------------------------------------------------------------------------

_CUTLASS_ROOT = "/usr/local/app/leowhzhang/worksapce/cuda_learn/nunchaku/third_party/cutlass"
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 共用 CUDA 编译标志（与 W8A8 kernel.py 保持一致）
# ---------------------------------------------------------------------------

_COMMON_CUDA_FLAGS = [
    "-std=c++17", "-O3",
    "--expt-relaxed-constexpr", "--expt-extended-lambda",
    # SM90a：WGMMA 需要 sm_90a，sm_90 不够
    "-gencode", "arch=compute_90a,code=sm_90a",
    f"-I{_CUTLASS_ROOT}/include",
    f"-I{_CUTLASS_ROOT}/tools/util/include",
    "-DCUTLASS_ARCH_MMA_SM90_SUPPORTED",
    "-DCUTE_ARCH_MMA_SM90A_ENABLED",
    # 消除 CUDA half 相关宏冲突
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-Xcompiler", "-fPIC",
]

FP8_MAX = 448.0

# ---------------------------------------------------------------------------
# V1：Per-Tensor FP8 GEMM（lazy JIT load）
# ---------------------------------------------------------------------------

_ext_v1 = None
FP8_V1_AVAILABLE = False


def _load_extension_v1():
    """触发 CUTLASS FP8 V1 扩展的 JIT 编译（幂等）"""
    global _ext_v1, FP8_V1_AVAILABLE
    if _ext_v1 is not None:
        return _ext_v1

    from torch.utils.cpp_extension import load
    try:
        _ext_v1 = load(
            name="gemm_fp8_sm90_v1",
            sources=[os.path.join(_THIS_DIR, "gemm_fp8_sm90_v1.cu")],
            extra_cuda_cflags=_COMMON_CUDA_FLAGS,
            extra_cflags=["-std=c++17", "-O3"],
            extra_ldflags=["-L/lib64", "-lcuda"],
            verbose=True,
        )
        FP8_V1_AVAILABLE = True
        print("[FP8-V1] CUTLASS SM90 FP8 Per-Tensor GEMM 编译成功")
    except Exception as e:
        print(f"[FP8-V1] JIT 编译失败: {e}")
        FP8_V1_AVAILABLE = False
    return _ext_v1


# ---------------------------------------------------------------------------
# V2：Per-Block FP8 GEMM（lazy JIT load）
# ---------------------------------------------------------------------------

_ext_v2 = None
FP8_V2_AVAILABLE = False


def _load_extension_v2():
    """触发 CUTLASS FP8 V2 扩展的 JIT 编译（幂等）"""
    global _ext_v2, FP8_V2_AVAILABLE
    if _ext_v2 is not None:
        return _ext_v2

    from torch.utils.cpp_extension import load
    try:
        _ext_v2 = load(
            name="gemm_fp8_sm90_v2",
            sources=[os.path.join(_THIS_DIR, "gemm_fp8_sm90_v2.cu")],
            extra_cuda_cflags=_COMMON_CUDA_FLAGS,
            extra_cflags=["-std=c++17", "-O3"],
            extra_ldflags=["-L/lib64", "-lcuda"],
            verbose=True,
        )
        FP8_V2_AVAILABLE = True
        print("[FP8-V2] CUTLASS SM90 FP8 Per-Block GEMM 编译成功")
    except Exception as e:
        print(f"[FP8-V2] JIT 编译失败: {e}")
        FP8_V2_AVAILABLE = False
    return _ext_v2


# ---------------------------------------------------------------------------
# 量化函数（Python 层，供 V1/V2 使用）
# ---------------------------------------------------------------------------

def fp8_per_tensor_quant(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-Tensor FP8 量化。

    Returns:
        q         : (M, K) float8_e4m3fn
        inv_scale : scalar float32（解量化 scale = amax / 448.0）
    """
    x_f32 = x.float()
    amax = x_f32.abs().amax().clamp(min=1e-12)
    scale = FP8_MAX / amax
    inv_scale = amax / FP8_MAX
    q = (x_f32 * scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, inv_scale.to(torch.float32)


def fp8_per_block_act_quant(
    x: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-Block FP8 激活量化（沿 K 维分组）。

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
    q = q_g.clamp(-FP8_MAX, FP8_MAX).reshape(M, K).to(torch.float8_e4m3fn)
    return q, inv_scales.to(torch.float32)


def fp8_per_block_weight_quant(
    w: torch.Tensor,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-Block FP8 权重量化（2D block）。

    Returns:
        q         : (N, K) float8_e4m3fn
        inv_scales: (N // block_size, K // block_size) float32
    """
    N, K = w.shape
    assert N % block_size == 0 and K % block_size == 0
    n_blocks = N // block_size
    k_blocks = K // block_size

    w_blocks = w.float().reshape(n_blocks, block_size, k_blocks, block_size)
    amax_blocks = w_blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    inv_scales = amax_blocks / FP8_MAX
    q_f32 = w_blocks / inv_scales.unsqueeze(1).unsqueeze(3)
    q = q_f32.clamp(-FP8_MAX, FP8_MAX).reshape(N, K).to(torch.float8_e4m3fn)
    return q, inv_scales.to(torch.float32)


def prepare_fp8_block_weights(
    w: torch.Tensor,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    权重预处理（量化 + 缓存），供 fp8_per_block_forward 使用。

    Args:
        w          : (N, K) float tensor（原始权重）
        block_size : 量化 block size

    Returns:
        q_w       : (N, K) float8_e4m3fn
        inv_scales: (N // block_size, K // block_size) float32
    """
    return fp8_per_block_weight_quant(w, block_size)


# ---------------------------------------------------------------------------
# GEMM 封装（调用 CUTLASS V1/V2 或 fallback）
# ---------------------------------------------------------------------------

def fp8_per_tensor_gemm_cutlass(
    a: torch.Tensor,
    b: torch.Tensor,
    inv_scale_a: torch.Tensor,
    inv_scale_b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Per-Tensor FP8 GEMM（CUTLASS V1）。

    Args:
        a           : (M, K) float8_e4m3fn
        b           : (N, K) float8_e4m3fn（权重）
        inv_scale_a : scalar float32
        inv_scale_b : scalar float32
        bias        : (N,) bfloat16 or None

    Returns:
        out : (M, N) bfloat16
    """
    M, K = a.shape
    N = b.shape[0]

    out = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
    alpha = inv_scale_a.float().item() * inv_scale_b.float().item()

    ext = _load_extension_v1()
    if FP8_V1_AVAILABLE:
        ext.fp8_per_tensor_gemm_v1(a, b, out, alpha, bias)
    else:
        # Fallback: torch._scaled_mm or FP32
        try:
            result = torch._scaled_mm(
                a, b.T,
                scale_a=inv_scale_a.reshape(1),
                scale_b=inv_scale_b.reshape(1),
                out_dtype=torch.bfloat16,
            )
            out.copy_(result)
        except (RuntimeError, AttributeError):
            a_f32 = a.float() * inv_scale_a.float()
            b_f32 = b.float() * inv_scale_b.float()
            out = (a_f32 @ b_f32.T).to(torch.bfloat16)

    if bias is not None:
        out = out + bias.to(torch.bfloat16)
    return out


def fp8_per_block_gemm_cutlass(
    a: torch.Tensor,
    a_inv_scales: torch.Tensor,
    b: torch.Tensor,
    b_inv_scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Per-Block FP8 GEMM（CUTLASS V2，EVT per-row/col 近似）。

    Args:
        a           : (M, K) float8_e4m3fn
        a_inv_scales: (M, K // 128) float32
        b           : (N, K) float8_e4m3fn
        b_inv_scales: (N // 128, K // 128) float32
        bias        : (N,) bfloat16 or None

    Returns:
        out : (M, N) bfloat16
    """
    M, K = a.shape
    N = b.shape[0]
    GROUP_SIZE = 128

    # 计算 per-row/col 近似 scale（mean 近似）
    act_scale_row = a_inv_scales.float().mean(dim=-1)    # (M,)
    wgt_scale_col = b_inv_scales.float().mean(dim=0)     # (K // 128,) → 需要 per-col

    # b_inv_scales: (N // GROUP_SIZE, K // GROUP_SIZE)
    # per-col wgt scale: 对每个 N 方向 block 取 mean over K groups
    wgt_scale_col_n = b_inv_scales.float().mean(dim=-1)  # (N // GROUP_SIZE,)
    # 广播到 (N,)：每 GROUP_SIZE 个 N 元素共享同一个 scale
    wgt_scale_full = wgt_scale_col_n.repeat_interleave(GROUP_SIZE)  # (N,)

    out = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)

    ext = _load_extension_v2()
    if FP8_V2_AVAILABLE:
        bias_bf16 = bias.to(torch.bfloat16) if bias is not None else None
        ext.fp8_per_block_gemm_v2(
            a, b, out,
            act_scale_row.contiguous(),
            wgt_scale_full.contiguous(),
            bias_bf16,
        )
    else:
        # Fallback: PyTorch per-block GEMM
        from operators.fp8_quant.pytorch.fp8_torch import fp8_per_block_gemm
        out = fp8_per_block_gemm(a, a_inv_scales, b, b_inv_scales, out_dtype=torch.bfloat16)
        if bias is not None:
            out = out + bias.to(torch.bfloat16)

    return out


# ---------------------------------------------------------------------------
# 高层 forward（含在线量化）
# ---------------------------------------------------------------------------

def fp8_per_tensor_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Per-Tensor FP8 前向（含在线量化）。

    Args:
        x    : (M, K) float tensor（激活）
        w    : (N, K) float tensor（权重）
        bias : (N,) float tensor or None

    Returns:
        out : (M, N) bfloat16
    """
    q_a, inv_scale_a = fp8_per_tensor_quant(x)
    q_b, inv_scale_b = fp8_per_tensor_quant(w)
    return fp8_per_tensor_gemm_cutlass(q_a, q_b, inv_scale_a, inv_scale_b, bias)


def fp8_per_block_forward(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_inv_scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Per-Block FP8 前向（权重已预量化）。

    Args:
        x            : (M, K) float tensor（激活，在线量化）
        w_q          : (N, K) float8_e4m3fn（预量化权重）
        w_inv_scales : (N // 128, K // 128) float32（权重解量化 scale）
        bias         : (N,) float tensor or None
        group_size   : 激活量化 group size

    Returns:
        out : (M, N) bfloat16
    """
    q_a, inv_scales_a = fp8_per_block_act_quant(x, group_size)
    bias_bf16 = bias.to(torch.bfloat16) if bias is not None else None
    return fp8_per_block_gemm_cutlass(q_a, inv_scales_a, w_q, w_inv_scales, bias_bf16)
