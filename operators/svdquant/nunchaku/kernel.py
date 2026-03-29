"""
Nunchaku INT4 MMA 实现

调用 nunchaku 的 svdq_gemm_w4a4_cuda + svdq_quantize_w4a4_act_fuse_lora_cuda，
利用 Hopper WGMMA INT4 Tensor Core 指令实现真 W4A4 GEMM。

张量格式转换说明：
    量化权重: (K, N) int8  → (N, K//2) int8 packed
    权重 scales: (K//64, N) FP16  → 直接用
    lora_up: (R, N) FP16  → (N, R) FP16  (.T.contiguous())
"""

import sys
import os
import torch

# -------------------------------------------------------------------------
# 尝试导入 nunchaku，不可用时设 NUNCHAKU_AVAILABLE = False
# -------------------------------------------------------------------------

# 尝试从本地源码目录加载（未 pip install 时走此路径）
_NUNCHAKU_SRC = os.path.join(os.path.dirname(__file__), "../../../../nunchaku")
if os.path.isdir(_NUNCHAKU_SRC) and _NUNCHAKU_SRC not in sys.path:
    sys.path.insert(0, _NUNCHAKU_SRC)

try:
    from nunchaku.ops.gemm import svdq_gemm_w4a4_cuda
    from nunchaku.ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda
    NUNCHAKU_AVAILABLE = True
except ImportError:
    NUNCHAKU_AVAILABLE = False

from operators.svdquant.pytorch.svdquant_torch import int4_pack_uint8


# -------------------------------------------------------------------------
# 权重预处理辅助
# -------------------------------------------------------------------------

def prepare_nunchaku_weights(q_w: torch.Tensor, lora_up: torch.Tensor):
    """
    预处理权重，消除每次 forward 的转换开销（在 benchmark setup 阶段调用一次）。

    参数：
        q_w    : (K, N) int8  — 量化权重（我们的格式）
        lora_up: (R, N) FP16  — LoRA up-projection

    返回：
        wgt_packed : (N, K//2) int8  — nunchaku 期望的 packed 格式
        lora_up_T  : (N, R)   FP16  — nunchaku 期望的转置格式
    """
    # (K, N) int8 → transpose → (N, K) int8 → pack → (N, K//2) uint8 → view int8
    wgt_packed = int4_pack_uint8(q_w.T.contiguous()).view(torch.int8)  # (N, K//2)
    lora_up_T = lora_up.T.contiguous()                                  # (N, R)
    return wgt_packed, lora_up_T


# -------------------------------------------------------------------------
# 在线版（每次 forward 含权重转换，用于正确性测试）
# -------------------------------------------------------------------------

def svdquant_forward_nunchaku(
    x,
    q_w,
    wscales,
    lora_down,
    lora_up,
    smooth,
    bias=None,
    group_size=64,
):
    """
    Nunchaku INT4 W4A4 GEMM 前向（在线权重转换版）。

    每次调用时都会执行 q_w 的格式转换，适合正确性验证但不适合性能 benchmark。

    参数：
        x        : (M, K) FP16  — 输入激活
        q_w      : (K, N) int8  — 量化权重（我们的格式）
        wscales  : (K//64, N) FP16  — 权重 scales
        lora_down: (K, R) FP16  — LoRA down-projection
        lora_up  : (R, N) FP16  — LoRA up-projection
        smooth   : (K,) FP16    — smooth factor
        bias     : (N,) FP16 or None
        group_size: int (default 64)

    返回：
        out : (M, N) FP16
    """
    if not NUNCHAKU_AVAILABLE:
        raise RuntimeError("nunchaku 未安装，无法使用 svdquant_forward_nunchaku")

    M, K = x.shape
    _, N = q_w.shape

    # 权重格式转换：(K, N) int8 → (N, K//2) int8 packed
    wgt = int4_pack_uint8(q_w.T.contiguous()).view(torch.int8)  # (N, K//2)
    lu_T = lora_up.T.contiguous()                                # (N, R)

    # Step 1: 激活量化 + LoRA down（fused nunchaku kernel）
    #   输出: act (M_pad, K//2) uint8
    #         ascales (K//64, M_pad) FP16
    #         lora_act (M_pad, R) FP32
    act, ascales, lora_act = svdq_quantize_w4a4_act_fuse_lora_cuda(
        input=x,
        lora_down=lora_down,
        smooth=smooth,
    )
    M_pad = act.shape[0]

    # Step 2: INT4 W4A4 GEMM + LoRA up（nunchaku Hopper MMA kernel）
    out = torch.empty(M_pad, N, dtype=torch.float16, device=x.device)
    svdq_gemm_w4a4_cuda(
        act=act.view(torch.int8),            # (M_pad, K//2) int8
        wgt=wgt,                             # (N, K//2) int8
        out=out,                             # (M_pad, N) FP16
        ascales=ascales,                     # (K//64, M_pad) FP16
        wscales=wscales,                     # (K//64, N) FP16
        lora_act_in=lora_act,                # (M_pad, R) FP32
        lora_up=lu_T,                        # (N, R) FP16
        bias=bias.to(torch.float16) if bias is not None else None,
    )

    return out[:M]


# -------------------------------------------------------------------------
# 缓存版（预处理权重，用于 benchmark 性能测量）
# -------------------------------------------------------------------------

def svdquant_forward_nunchaku_cached(
    x,
    wgt_packed,
    wscales,
    lora_down,
    lora_up_T,
    smooth,
    bias=None,
):
    """
    Nunchaku INT4 W4A4 GEMM 前向（预缓存权重版）。

    wgt_packed 和 lora_up_T 已在 setup 阶段由 prepare_nunchaku_weights() 预处理，
    消除了每次 forward 的格式转换开销，适合 benchmark 性能测量。

    参数：
        x         : (M, K) FP16  — 输入激活
        wgt_packed: (N, K//2) int8  — 已打包的量化权重
        wscales   : (K//64, N) FP16  — 权重 scales
        lora_down : (K, R) FP16  — LoRA down-projection
        lora_up_T : (N, R) FP16  — LoRA up-projection（已转置）
        smooth    : (K,) FP16    — smooth factor
        bias      : (N,) FP16 or None

    返回：
        out : (M, N) FP16
    """
    if not NUNCHAKU_AVAILABLE:
        raise RuntimeError("nunchaku 未安装，无法使用 svdquant_forward_nunchaku_cached")

    M, K = x.shape
    N = wgt_packed.shape[0]

    # Step 1: 激活量化 + LoRA down（fused nunchaku kernel）
    act, ascales, lora_act = svdq_quantize_w4a4_act_fuse_lora_cuda(
        input=x,
        lora_down=lora_down,
        smooth=smooth,
    )
    M_pad = act.shape[0]

    # Step 2: INT4 W4A4 GEMM + LoRA up（nunchaku Hopper MMA kernel）
    out = torch.empty(M_pad, N, dtype=torch.float16, device=x.device)
    svdq_gemm_w4a4_cuda(
        act=act.view(torch.int8),            # (M_pad, K//2) int8
        wgt=wgt_packed,                      # (N, K//2) int8
        out=out,                             # (M_pad, N) FP16
        ascales=ascales,                     # (K//64, M_pad) FP16
        wscales=wscales,                     # (K//64, N) FP16
        lora_act_in=lora_act,                # (M_pad, R) FP32
        lora_up=lora_up_T,                   # (N, R) FP16
        bias=bias.to(torch.float16) if bias is not None else None,
    )

    return out[:M]
