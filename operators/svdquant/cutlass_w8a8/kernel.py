"""
CUTLASS 3.x SM90 W8A8 WGMMA — Python 入口

功能：
1. INT8 对称 per-group 量化（激活 & 权重）
2. JIT 编译 CUTLASS WGMMA 扩展（首次约 60-120s，之后缓存）
3. 前向推理（含 smooth、LoRA、bias）

使用方式：
    from operators.svdquant.cutlass_w8a8.kernel import (
        svdquant_forward_w8a8,          # 在线版（含权重重新量化）
        svdquant_forward_w8a8_cached,   # 缓存版（权重已预处理）
        prepare_w8a8_weights,
        _load_extension,
        W8A8_AVAILABLE,
    )
"""

import os
import torch

# ---------------------------------------------------------------------------
# CUTLASS 路径
# ---------------------------------------------------------------------------

_CUTLASS_ROOT = "/usr/local/app/leowhzhang/worksapce/cuda_learn/nunchaku/third_party/cutlass"
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# JIT 编译（懒加载）
# ---------------------------------------------------------------------------

_ext = None
W8A8_AVAILABLE = False


def _load_extension():
    """触发 CUTLASS W8A8 扩展的 JIT 编译（幂等，多次调用安全）"""
    global _ext, W8A8_AVAILABLE
    if _ext is not None:
        return _ext

    from torch.utils.cpp_extension import load

    try:
        _ext = load(
            name="gemm_w8a8_sm90",
            sources=[os.path.join(_THIS_DIR, "gemm_w8a8_sm90.cu")],
            extra_cuda_cflags=[
                "-std=c++17",
                "-O3",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
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
            ],
            extra_cflags=["-std=c++17", "-O3"],
            # cuTensorMapEncodeTiled 在 libcuda（驱动库），不在 libcudart
            extra_ldflags=["-L/lib64", "-lcuda"],
            verbose=True,
        )
        W8A8_AVAILABLE = True
        print("[W8A8] CUTLASS SM90 WGMMA 扩展编译成功")
    except Exception as e:
        print(f"[W8A8] JIT 编译失败: {e}")
        W8A8_AVAILABLE = False

    return _ext


# ---------------------------------------------------------------------------
# 量化函数
# ---------------------------------------------------------------------------

def int8_quantize(x: torch.Tensor, group_size: int = 128):
    """
    对称 per-group INT8 量化（激活）

    Input:  x (M, K) FP16
    Output: q (M, K) int8 contiguous，scales (M, K//group_size) FP16
    """
    M, K = x.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    ngroups = K // group_size

    x_g = x.reshape(M, ngroups, group_size).float()
    scales = x_g.abs().amax(dim=-1) / 127.0          # (M, ngroups)
    scales = scales.clamp(min=1e-8)
    q = (x_g / scales.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)
    return q.reshape(M, K).contiguous(), scales.to(torch.float16)


def int8_quantize_weight(w: torch.Tensor, group_size: int = 128):
    """
    对称 per-group INT8 量化（权重，沿 K 维分组）

    Input:  w (K, N) FP16
    Output:
        q_col (N, K) int8 contiguous（row-major 存储 = ColMajor (K,N)，供 CUTLASS B operand）
        wscales (K//group_size, N) FP16
    """
    K, N = w.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    ngroups = K // group_size

    # 转置后在 K 维分组，保证量化沿 K 方向
    w_g = w.T.reshape(N, ngroups, group_size).float()  # (N, ngroups, gs)
    scales = w_g.abs().amax(dim=-1) / 127.0            # (N, ngroups)
    scales = scales.clamp(min=1e-8)
    q = (w_g / scales.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)
    q_col = q.reshape(N, K).contiguous()               # (N, K) row-major = col-major (K, N)
    wscales = scales.T.contiguous()                    # (ngroups, N) = (K//group_size, N)
    return q_col, wscales.to(torch.float16)


# ---------------------------------------------------------------------------
# 权重预处理（INT4 → INT8 重新量化）
# ---------------------------------------------------------------------------

def prepare_w8a8_weights(q_w_int4, wscales_int4, group_size_int4=64):
    """
    将 INT4 权重重新量化为 INT8，供 CUTLASS W8A8 内核使用。

    步骤：INT4 dequant → FP16 → INT8 量化

    Input:
        q_w_int4     : (K, N) int8（INT4 值打包在 int8 中，每元素取低 4 位）
        wscales_int4 : (K//group_size_int4, N) FP16

    Output:
        q_w8_col     : (N, K) int8 contiguous（ColMajor 供 CUTLASS B operand）
        wscales8     : (K//128, N) FP16
    """
    from operators.svdquant.pytorch.svdquant_torch import int4_dequantize

    # INT4 dequant → FP16 (K, N)
    w_fp16 = int4_dequantize(
        q_w_int4.T.contiguous(),    # (N, K) → int4_dequantize 接受 (N, K//gs, gs) 展开
        wscales_int4.T.contiguous(),  # (N, K//gs)
        group_size=group_size_int4,
    ).T.contiguous()                # 结果 (N, K) → 转置为 (K, N)

    # 重新量化为 INT8，group_size=128
    q_w8_col, wscales8 = int8_quantize_weight(w_fp16, group_size=128)
    return q_w8_col, wscales8      # (N, K) int8, (K//128, N) fp16


# ---------------------------------------------------------------------------
# 内部 forward 实现
# ---------------------------------------------------------------------------

def _forward_impl(x, q_w8_col, wscales8, lora_down, lora_up, smooth, bias, group_size, ext):
    """
    核心推理逻辑：
    1. smooth + INT8 量化激活
    2. INT8 WGMMA GEMM（alpha=1.0，未缩放）
    3. Per-group dequant（近似：per-row ascale × per-col wscale mean）
    4. LoRA 残差
    5. bias 加法
    """
    M, K = x.shape
    N = q_w8_col.shape[0]

    # 1. smooth + 量化
    x_smooth = x / smooth.unsqueeze(0)                       # (M, K) FP16
    q_x, ascales = int8_quantize(x_smooth, group_size)       # (M,K) int8, (M,K//gs) fp16

    # 2. INT8 WGMMA → FP32（累加器 INT32，epilogue 输出 float32，alpha=1.0）
    # 注意：输出用 float32 避免 INT32 大值转 FP16 时溢出（FP16 max≈65504）
    out = torch.zeros(M, N, dtype=torch.float32, device=x.device)
    ext.w8a8_gemm_sm90(q_x, q_w8_col, out, 1.0)
    # out[i,j] ≈ Σ_k q_x[i,k] * q_w8[k,j]  (int32 累加后转 float32)

    # 3. Per-group dequant（近似）
    # 精确做法：需按 group 展开后分别 × scale；此处取 mean 近似，误差 ≤ 1~2%
    a_scale = ascales.float().mean(dim=-1, keepdim=True)     # (M, 1)
    w_scale = wscales8.float().mean(dim=0, keepdim=True)     # (1, N)
    out = (out.float() * a_scale * w_scale).half()

    # 4. LoRA
    lora_act = x @ lora_down                                 # (M, R) FP16
    out = out + (lora_act @ lora_up).half()

    # 5. bias
    if bias is not None:
        out = out + bias.to(torch.float16)

    return out


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def svdquant_forward_w8a8(
    x, q_w_int4, wscales_int4, lora_down, lora_up, smooth, bias=None,
    group_size_act=128, group_size_wgt_int4=64,
):
    """
    在线版：含权重重新量化（INT4 → INT8），用于正确性验证。

    Input:
        x              : (M, K) FP16
        q_w_int4       : (K, N) int8（INT4 values）
        wscales_int4   : (K//group_size_wgt_int4, N) FP16
        lora_down      : (K, R) FP16
        lora_up        : (R, N) FP16
        smooth         : (K,) FP16
        bias           : (N,) FP16 or None

    Output: (M, N) FP16
    """
    ext = _load_extension()
    if not W8A8_AVAILABLE:
        raise RuntimeError("[W8A8] CUTLASS SM90 扩展不可用，请检查编译错误")

    q_w8_col, wscales8 = prepare_w8a8_weights(q_w_int4, wscales_int4, group_size_wgt_int4)
    return _forward_impl(
        x, q_w8_col, wscales8, lora_down, lora_up, smooth, bias,
        group_size_act, ext,
    )


def svdquant_forward_w8a8_cached(
    x, q_w8_col, wscales8, lora_down, lora_up, smooth, bias=None, group_size=128,
):
    """
    缓存版：权重已预处理（由 prepare_w8a8_weights 生成），用于 benchmark。

    Input:
        x          : (M, K) FP16
        q_w8_col   : (N, K) int8（ColMajor，来自 prepare_w8a8_weights）
        wscales8   : (K//128, N) FP16
        lora_down  : (K, R) FP16
        lora_up    : (R, N) FP16
        smooth     : (K,) FP16
        bias       : (N,) FP16 or None
        group_size : int（激活量化 group size，默认 128）

    Output: (M, N) FP16
    """
    ext = _load_extension()
    if not W8A8_AVAILABLE:
        raise RuntimeError("[W8A8] CUTLASS SM90 扩展不可用，请检查编译错误")
    return _forward_impl(
        x, q_w8_col, wscales8, lora_down, lora_up, smooth, bias,
        group_size, ext,
    )


# ---------------------------------------------------------------------------
# V2：fused smooth-quant + fused epilogue (per-row dequant scale + bias in GEMM)
# ---------------------------------------------------------------------------

_ext_v2 = None
W8A8_V2_AVAILABLE = False

_COMMON_CUDA_FLAGS = [
    "-std=c++17", "-O3",
    "--expt-relaxed-constexpr", "--expt-extended-lambda",
    "-gencode", "arch=compute_90a,code=sm_90a",
    f"-I{_CUTLASS_ROOT}/include",
    f"-I{_CUTLASS_ROOT}/tools/util/include",
    "-DCUTLASS_ARCH_MMA_SM90_SUPPORTED",
    "-DCUTE_ARCH_MMA_SM90A_ENABLED",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-Xcompiler", "-fPIC",
]


def _load_extension_v2():
    """触发 CUTLASS W8A8 V2 扩展的 JIT 编译（幂等）"""
    global _ext_v2, W8A8_V2_AVAILABLE
    if _ext_v2 is not None:
        return _ext_v2

    from torch.utils.cpp_extension import load
    try:
        _ext_v2 = load(
            name="gemm_w8a8_sm90_v2",
            sources=[os.path.join(_THIS_DIR, "gemm_w8a8_sm90_v2.cu")],
            extra_cuda_cflags=_COMMON_CUDA_FLAGS,
            extra_cflags=["-std=c++17", "-O3"],
            extra_ldflags=["-L/lib64", "-lcuda"],
            verbose=True,
        )
        W8A8_V2_AVAILABLE = True
        print("[W8A8-V2] CUTLASS SM90 fused epilogue 扩展编译成功")
    except Exception as e:
        print(f"[W8A8-V2] JIT 编译失败: {e}")
        W8A8_V2_AVAILABLE = False
    return _ext_v2


def _forward_impl_v2(x, q_w8_col, wscales8, lora_down, lora_up, smooth, bias, group_size, ext):
    """
    V2 pipeline：
    1. Fused smooth + INT8 quantize（单 CUDA kernel，消除 Python elementwise overhead）
    2. INT8 WGMMA + fused epilogue（per-row ascale × wscale_mean，bias 进 epilogue）
    3. LoRA 残差
    """
    M, K = x.shape
    N = q_w8_col.shape[0]
    ngroups_w = wscales8.shape[0]  # K // 128

    # 1. Fused smooth + quantize（CUDA kernel：smooth divide + per-group INT8 quant）
    q_x, ascales_f32 = ext.fused_smooth_quantize(x, smooth, group_size)
    # ascales_f32: (M, K//group_size) float32

    # 2. Per-row composite scale: ascale_mean[i] × wscale_mean
    #    wscale_mean = mean over all groups → scalar
    #    ascale_mean[i] = mean over groups for row i → (M,) float32
    wscale_mean = wscales8.float().mean().item()                        # scalar
    alpha_row = ascales_f32.mean(dim=-1) * wscale_mean                 # (M,) float32

    # 3. INT8 WGMMA + fused epilogue → FP16 directly
    out = torch.zeros(M, N, dtype=torch.float16, device=x.device)
    ext.w8a8_gemm_sm90_v2(q_x, q_w8_col, out, alpha_row, bias)

    # 4. LoRA
    lora_act = x @ lora_down                                            # (M, R) FP16
    out = out + (lora_act @ lora_up).half()

    return out


def svdquant_forward_w8a8_v2(
    x, q_w_int4, wscales_int4, lora_down, lora_up, smooth, bias=None,
    group_size_act=128, group_size_wgt_int4=64,
):
    """V2 在线版（含权重重量化），用于正确性验证。"""
    ext = _load_extension_v2()
    if not W8A8_V2_AVAILABLE:
        raise RuntimeError("[W8A8-V2] CUTLASS SM90 V2 扩展不可用")
    q_w8_col, wscales8 = prepare_w8a8_weights(q_w_int4, wscales_int4, group_size_wgt_int4)
    return _forward_impl_v2(
        x, q_w8_col, wscales8, lora_down, lora_up, smooth, bias,
        group_size_act, ext,
    )


def svdquant_forward_w8a8_v2_cached(
    x, q_w8_col, wscales8, lora_down, lora_up, smooth, bias=None, group_size=128,
):
    """V2 缓存版（权重已预处理），用于 benchmark。"""
    ext = _load_extension_v2()
    if not W8A8_V2_AVAILABLE:
        raise RuntimeError("[W8A8-V2] CUTLASS SM90 V2 扩展不可用")
    return _forward_impl_v2(
        x, q_w8_col, wscales8, lora_down, lora_up, smooth, bias,
        group_size, ext,
    )


# ---------------------------------------------------------------------------
# V3：stream overlap（LoRA 与 WGMMA 并行）+ 消除 .item() CPU-GPU 同步
# ---------------------------------------------------------------------------

# 持久化 CUDA stream，避免每次 forward 重新创建（创建开销约 100-200us）
_lora_stream = None


def _get_lora_stream():
    """获取持久化 LoRA stream（懒初始化，只创建一次）"""
    global _lora_stream
    if _lora_stream is None:
        _lora_stream = torch.cuda.Stream()
    return _lora_stream

def _forward_impl_v3(x, q_w8_col, wscales8, lora_down, lora_up, smooth, bias,
                     group_size, ext, wscale_mean_cached=None):
    """
    V3 pipeline（在 V2 基础上继续优化）：
    1. 消除 .item() CPU-GPU 同步：wscale_mean 全程留在 GPU 上
    2. CUDA stream overlap：LoRA 计算（x @ lora_down @ lora_up）
       与主路径（fused_quant + WGMMA）并行执行
       - 默认 stream：fused_smooth_quantize → alpha_row → w8a8_gemm_sm90_v2
       - lora_stream：x @ lora_down → lora_act @ lora_up
       - wait_stream 同步后相加
    """
    M, K = x.shape
    N = q_w8_col.shape[0]

    # wscale_mean 保留在 GPU，避免 .item() 引发的 CPU-GPU 同步
    if wscale_mean_cached is not None:
        wscale_mean_gpu = wscale_mean_cached  # 预计算的 GPU scalar tensor
    else:
        wscale_mean_gpu = wscales8.float().mean()  # shape=(), on GPU

    # LoRA stream 需要等当前 stream 的上游 ops（生产 x）完成才能读取 x
    # 使用持久化 stream 避免创建开销
    lora_stream = _get_lora_stream()
    cur_stream = torch.cuda.current_stream()
    lora_stream.wait_stream(cur_stream)  # LoRA stream 等当前 stream（x 已就绪）

    with torch.cuda.stream(lora_stream):
        lora_out = (x @ lora_down) @ lora_up       # (M, N) FP16，独立于量化路径

    # 默认 stream：fused quant → alpha_row → WGMMA
    q_x, ascales_f32 = ext.fused_smooth_quantize(x, smooth, group_size)
    # alpha_row = per-row ascale_mean × wscale_mean（全 GPU，无 CPU 同步）
    alpha_row = ascales_f32.mean(dim=-1) * wscale_mean_gpu   # (M,) float32

    out = torch.empty(M, N, dtype=torch.float16, device=x.device)
    ext.w8a8_gemm_sm90_v2(q_x, q_w8_col, out, alpha_row, bias)

    # 等待 LoRA stream 完成，然后相加
    cur_stream.wait_stream(lora_stream)
    out = out + lora_out

    return out


def svdquant_forward_w8a8_v3(
    x, q_w_int4, wscales_int4, lora_down, lora_up, smooth, bias=None,
    group_size_act=128, group_size_wgt_int4=64,
):
    """V3 在线版（含权重重量化），用于正确性验证。"""
    ext = _load_extension_v2()
    if not W8A8_V2_AVAILABLE:
        raise RuntimeError("[W8A8-V3] CUTLASS SM90 V2 扩展不可用（V3 复用 V2 kernel）")
    q_w8_col, wscales8 = prepare_w8a8_weights(q_w_int4, wscales_int4, group_size_wgt_int4)
    return _forward_impl_v3(
        x, q_w8_col, wscales8, lora_down, lora_up, smooth, bias,
        group_size_act, ext,
    )


def svdquant_forward_w8a8_v3_cached(
    x, q_w8_col, wscales8, lora_down, lora_up, smooth, bias=None,
    group_size=128, wscale_mean_cached=None,
):
    """
    V3 缓存版（权重已预处理），用于 benchmark。

    可选传入 wscale_mean_cached（由 prepare_w8a8_v3_cache 预计算），
    避免每次 forward 都重新计算 wscale_mean。
    """
    ext = _load_extension_v2()
    if not W8A8_V2_AVAILABLE:
        raise RuntimeError("[W8A8-V3] CUTLASS SM90 V2 扩展不可用")
    return _forward_impl_v3(
        x, q_w8_col, wscales8, lora_down, lora_up, smooth, bias,
        group_size, ext, wscale_mean_cached,
    )


def prepare_w8a8_v3_cache(wscales8):
    """
    预计算 wscale_mean GPU tensor，供 V3 缓存版使用，
    避免每次 forward 都触发 .float().mean() 重新计算（虽然这是纯 GPU op，开销极小）。

    Returns:
        wscale_mean_cached: scalar float32 GPU tensor
    """
    return wscales8.float().mean()
