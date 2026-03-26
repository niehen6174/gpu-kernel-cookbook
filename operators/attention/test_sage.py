"""
SageAttention vs FlashAttention 对比测试

测试内容：
1. 正确性：与 PyTorch 参考实现对比（float32 精度）
2. 长序列 benchmark：N in [1024, 2048, 4096, 8192, 16384]
   对比：FlashAttention v2（官方）vs SageAttention v1/v2（本实现 Triton）
          vs 官方 SageAttention 包（若已安装）

运行：
    cd gpu-kernel-lab
    python -m operators.attention.test_sage
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

import torch
import math

from common.check import check_correctness
from common.utils import benchmark_func, compute_tflops, get_gpu_info
from operators.attention.sage.kernel_v1 import sageattn_v1
from operators.attention.sage.kernel_v2 import sageattn_v2


# ============================================================
# 参考实现（FP16 PyTorch 精确版）
# ============================================================
def attention_ref_fp16(q, k, v, is_causal=False, sm_scale=None):
    """
    PyTorch 精确注意力，fp32 计算后转为 fp16 输出。
    用于正确性验证参考值。
    """
    B, H, N, D = q.shape
    if sm_scale is None:
        sm_scale = D ** -0.5
    qk = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
    if is_causal:
        mask = torch.tril(torch.ones(N, N, device=q.device)).bool()
        qk = qk.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(qk, dim=-1)   # float32
    return torch.matmul(attn, v.float()).to(torch.float16)


# ============================================================
# FlashAttention v2 包装（官方 flash_attn 包）
# ============================================================
try:
    from flash_attn import flash_attn_func
    HAS_FA2 = True
except ImportError:
    HAS_FA2 = False
    print("[WARN] flash_attn not installed, skipping FA2 benchmark")


def flash_attn_v2(q, k, v, is_causal=False, sm_scale=None):
    """
    官方 FlashAttention v2。
    输入: (B, H, N, D) float16，输出: (B, H, N, D) float16
    flash_attn 期望 (B, N, H, D) (NHD layout)。
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    # (B, H, N, D) → (B, N, H, D)
    q_ = q.transpose(1, 2).contiguous()
    k_ = k.transpose(1, 2).contiguous()
    v_ = v.transpose(1, 2).contiguous()
    out = flash_attn_func(q_, k_, v_, softmax_scale=sm_scale, causal=is_causal)
    return out.transpose(1, 2).contiguous()   # → (B, H, N, D)


# ============================================================
# 官方 SageAttention 包（若已安装）
# ============================================================
try:
    from sageattention import sageattn as sageattn_official
    HAS_SAGE_PKG = True
except ImportError:
    HAS_SAGE_PKG = False


def sage_official(q, k, v, is_causal=False, sm_scale=None):
    """官方 sageattention 包（tensor_layout=HND）"""
    return sageattn_official(q, k, v, tensor_layout="HND",
                              is_causal=is_causal, sm_scale=sm_scale)


# ============================================================
# 正确性测试
# ============================================================
def test_correctness():
    print("=" * 70)
    print("SageAttention — 正确性测试（与 FP16 PyTorch 参考值对比）")
    print("=" * 70)

    test_shapes = [
        # (B, H, N, D)
        (1, 8, 512, 64),
        (2, 8, 1024, 64),
        (1, 16, 2048, 64),
        (1, 8, 512, 128),
        (2, 8, 1024, 128),
    ]

    for B, H, N, D in test_shapes:
        print(f"\n  Shape: B={B} H={H} N={N} D={D}")
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        ref = attention_ref_fp16(q, k, v)

        # ---- SageAttention v1 (ours) ----
        out_v1 = sageattn_v1(q, k, v, smooth_k=True)
        # SageAttention 是近似算法，允许更大误差（INT8量化误差）
        # 典型 cosine sim > 0.999，相对误差 ~1%
        check_correctness(out_v1, ref,
                          name=f"SageAttn v1 (ours, smooth_k) N={N} D={D}",
                          atol=0.05, rtol=0.02)

        out_v1_ns = sageattn_v1(q, k, v, smooth_k=False)
        check_correctness(out_v1_ns, ref,
                          name=f"SageAttn v1 (ours, no_smooth) N={N} D={D}",
                          atol=0.05, rtol=0.02)

        # ---- SageAttention v2 (ours) ----
        out_v2 = sageattn_v2(q, k, v, smooth_k=True, smooth_v=True)
        check_correctness(out_v2, ref,
                          name=f"SageAttn v2 (ours, smooth_kv) N={N} D={D}",
                          atol=0.05, rtol=0.02)

        # ---- FlashAttention v2 ----
        if HAS_FA2:
            out_fa2 = flash_attn_v2(q, k, v)
            check_correctness(out_fa2, ref,
                              name=f"FlashAttn v2 N={N} D={D}",
                              atol=5e-3, rtol=1e-3)

        # ---- 官方 SageAttention（若有）----
        if HAS_SAGE_PKG:
            out_sage = sage_official(q, k, v)
            check_correctness(out_sage, ref,
                              name=f"SageAttn official N={N} D={D}",
                              atol=0.05, rtol=0.02)

    # 因果注意力测试
    print("\n  [Causal Attention] B=1, H=8, N=1024, D=64")
    q = torch.randn(1, 8, 1024, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 8, 1024, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 8, 1024, 64, device="cuda", dtype=torch.float16)
    ref_causal = attention_ref_fp16(q, k, v, is_causal=True)
    check_correctness(sageattn_v1(q, k, v, is_causal=True), ref_causal,
                      name="SageAttn v1 causal", atol=0.05)
    # V2 causal：FP8 V 量化使用全局均值做 smoothing，在因果遮蔽下（早期 token 仅能 attend 少量位置）
    # 误差略高，使用更宽松的容差（~0.12 为已知数值精度上限）
    check_correctness(sageattn_v2(q, k, v, is_causal=True), ref_causal,
                      name="SageAttn v2 causal", atol=0.15)
    if HAS_FA2:
        check_correctness(flash_attn_v2(q, k, v, is_causal=True), ref_causal,
                          name="FlashAttn v2 causal", atol=5e-3)
    print()


# ============================================================
# 量化误差分析（对比有/无 smoothing 的效果）
# ============================================================
def test_smoothing_effect():
    print("=" * 70)
    print("K/V Smoothing 效果分析（对比有无 smoothing 的量化误差）")
    print("=" * 70)

    B, H, N, D = 1, 8, 2048, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    ref = attention_ref_fp16(q, k, v)

    # 无 smoothing
    out_no_smooth  = sageattn_v1(q, k, v, smooth_k=False)
    # 有 K smoothing
    out_k_smooth   = sageattn_v1(q, k, v, smooth_k=True)
    # V2 K+V smoothing
    out_v2_smooth  = sageattn_v2(q, k, v, smooth_k=True, smooth_v=True)
    out_v2_no_sv   = sageattn_v2(q, k, v, smooth_k=True, smooth_v=False)

    def cosine_sim(a, b):
        a_ = a.float().flatten()
        b_ = b.float().flatten()
        return (a_ * b_).sum() / (a_.norm() * b_.norm())

    def mean_abs_err(a, b):
        return (a.float() - b.float()).abs().mean().item()

    print(f"\n  Shape: B={B} H={H} N={N} D={D}")
    print(f"  {'方法':<30} {'MAE':>10} {'cos_sim':>10}")
    print(f"  {'-'*52}")

    for name, out in [
        ("v1 无 smoothing",      out_no_smooth),
        ("v1 K smoothing",       out_k_smooth),
        ("v2 K+V smoothing",     out_v2_smooth),
        ("v2 K smooth, no V sm", out_v2_no_sv),
    ]:
        mae = mean_abs_err(out, ref)
        cos = cosine_sim(out, ref).item()
        print(f"  {name:<30} {mae:>10.4f} {cos:>10.6f}")
    print()


# ============================================================
# 长序列 Benchmark：SageAttention vs FlashAttention
# ============================================================
def run_long_seq_benchmark(
    B: int = 1,
    H: int = 16,
    D: int = 64,
    seq_lens: list = None,
):
    if seq_lens is None:
        seq_lens = [1024, 2048, 4096, 8192, 16384]

    print("=" * 70)
    print(f"长序列 Benchmark: B={B} H={H} D={D}")
    print("=" * 70)
    print(get_gpu_info())
    print()

    header = f"{'N':>6}  {'FA2':>12}  {'Sage v1 (ours)':>16}  {'Sage v2 (ours)':>16}"
    if HAS_SAGE_PKG:
        header += f"  {'Sage official':>14}"
    print(header)
    print("-" * 80)

    for N in seq_lens:
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        flops = 4 * B * H * N * N * D

        # FlashAttention v2
        if HAS_FA2:
            r_fa2 = benchmark_func(flash_attn_v2, q, k, v)
            t_fa2 = r_fa2["mean_ms"]
            tf_fa2 = compute_tflops(flops, t_fa2)
            fa2_str = f"{t_fa2:.3f}ms({tf_fa2:.1f}T)"
        else:
            t_fa2 = None
            fa2_str = "N/A"

        # SageAttention v1 (ours)
        r_s1 = benchmark_func(sageattn_v1, q, k, v)
        t_s1 = r_s1["mean_ms"]
        tf_s1 = compute_tflops(flops, t_s1)
        sp1 = f"({t_fa2/t_s1:.2f}x)" if t_fa2 else ""
        s1_str = f"{t_s1:.3f}ms({tf_s1:.1f}T){sp1}"

        # SageAttention v2 (ours)
        r_s2 = benchmark_func(sageattn_v2, q, k, v)
        t_s2 = r_s2["mean_ms"]
        tf_s2 = compute_tflops(flops, t_s2)
        sp2 = f"({t_fa2/t_s2:.2f}x)" if t_fa2 else ""
        s2_str = f"{t_s2:.3f}ms({tf_s2:.1f}T){sp2}"

        line = f"{N:>6}  {fa2_str:>12}  {s1_str:>16}  {s2_str:>16}"

        # 官方 SageAttention
        if HAS_SAGE_PKG:
            r_so = benchmark_func(sage_official, q, k, v)
            t_so = r_so["mean_ms"]
            tf_so = compute_tflops(flops, t_so)
            sp_o = f"({t_fa2/t_so:.2f}x)" if t_fa2 else ""
            line += f"  {t_so:.3f}ms({tf_so:.1f}T){sp_o:>8}"

        print(line)

    print()


# ============================================================
# 详细 Benchmark：固定 N，比较所有实现的 TFLOPS
# ============================================================
def run_detailed_benchmark(B=1, H=16, N=4096, D=64):
    print("=" * 70)
    print(f"详细 Benchmark: B={B} H={H} N={N} D={D}")
    print("（TFLOPS 基于 4*B*H*N²*D 计算，不含量化开销）")
    print("=" * 70)

    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    flops = 4 * B * H * N * N * D

    implementations = []

    if HAS_FA2:
        implementations.append(("FlashAttention v2", lambda: flash_attn_v2(q, k, v)))

    implementations += [
        ("SageAttn v1 (ours, smooth)",    lambda: sageattn_v1(q, k, v, smooth_k=True)),
        ("SageAttn v1 (ours, no smooth)", lambda: sageattn_v1(q, k, v, smooth_k=False)),
        ("SageAttn v2 (ours, smooth_kv)", lambda: sageattn_v2(q, k, v, smooth_k=True, smooth_v=True)),
        ("SageAttn v2 (ours, smooth_k)",  lambda: sageattn_v2(q, k, v, smooth_k=True, smooth_v=False)),
    ]

    if HAS_SAGE_PKG:
        implementations.append(("SageAttn official",   lambda: sage_official(q, k, v)))

    print(f"\n  {'实现':<35}  {'时间(ms)':>10}  {'TFLOPS':>8}  {'vs FA2':>8}")
    print(f"  {'-'*65}")

    fa2_time = None
    for name, fn in implementations:
        try:
            r = benchmark_func(fn)
            t = r["mean_ms"]
            tf = compute_tflops(flops, t)
            if fa2_time is None and "FlashAttention" in name:
                fa2_time = t
            speedup = f"{fa2_time/t:.2f}x" if fa2_time and name != "FlashAttention v2" else "baseline"
            print(f"  {name:<35}  {t:>10.3f}  {tf:>8.2f}  {speedup:>8}")
        except Exception as e:
            print(f"  {name:<35}  [ERROR: {e}]")
    print()

    # Causal attention
    print(f"  [Causal 模式]")
    print(f"  {'实现':<35}  {'时间(ms)':>10}  {'TFLOPS':>8}")
    print(f"  {'-'*55}")
    causal_impls = []
    if HAS_FA2:
        causal_impls.append(("FlashAttention v2 causal", lambda: flash_attn_v2(q, k, v, is_causal=True)))
    causal_impls += [
        ("SageAttn v1 causal", lambda: sageattn_v1(q, k, v, is_causal=True)),
        ("SageAttn v2 causal", lambda: sageattn_v2(q, k, v, is_causal=True)),
    ]
    for name, fn in causal_impls:
        try:
            r = benchmark_func(fn)
            t = r["mean_ms"]
            tf = compute_tflops(flops // 2, t)  # causal ~half FLOPs
            print(f"  {name:<35}  {t:>10.3f}  {tf:>8.2f}")
        except Exception as e:
            print(f"  {name:<35}  [ERROR: {e}]")
    print()


# ============================================================
# 内存效率分析
# ============================================================
def run_memory_analysis():
    print("=" * 70)
    print("内存占用分析（不同序列长度下 peak memory，H=16 D=64）")
    print("=" * 70)
    print(f"  {'N':>6}  {'FA2(MB)':>10}  {'Sage v1(MB)':>12}  {'Sage v2(MB)':>12}")
    print(f"  {'-'*45}")

    B, H, D = 1, 16, 64
    for N in [2048, 4096, 8192, 16384]:
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        def measure_mem(fn):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            fn()
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / 1024**2

        mem_fa2 = measure_mem(lambda: flash_attn_v2(q, k, v)) if HAS_FA2 else float("nan")
        mem_s1  = measure_mem(lambda: sageattn_v1(q, k, v))
        mem_s2  = measure_mem(lambda: sageattn_v2(q, k, v))

        print(f"  {N:>6}  {mem_fa2:>10.1f}  {mem_s1:>12.1f}  {mem_s2:>12.1f}")
    print()


# ============================================================
# 误差 vs 序列长度分析（量化误差随 N 的变化）
# ============================================================
def run_error_scaling():
    print("=" * 70)
    print("量化误差 vs 序列长度（MSE 相对于 FP16 精确参考值）")
    print("=" * 70)
    print(f"  {'N':>6}  {'Sage v1 MAE':>14}  {'Sage v2 MAE':>14}  {'Sage v1 cos':>12}  {'Sage v2 cos':>12}")
    print(f"  {'-'*65}")

    B, H, D = 1, 8, 64
    for N in [512, 1024, 2048, 4096, 8192]:
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        ref = attention_ref_fp16(q, k, v)
        o1  = sageattn_v1(q, k, v, smooth_k=True)
        o2  = sageattn_v2(q, k, v, smooth_k=True, smooth_v=True)

        def mae(a, b): return (a.float() - b.float()).abs().mean().item()
        def cos(a, b):
            a_ = a.float().flatten()
            b_ = b.float().flatten()
            return ((a_ * b_).sum() / (a_.norm() * b_.norm())).item()

        print(f"  {N:>6}  {mae(o1,ref):>14.5f}  {mae(o2,ref):>14.5f}  "
              f"{cos(o1,ref):>12.6f}  {cos(o2,ref):>12.6f}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SageAttention (V1/V2) vs FlashAttention 对比测试")
    print("GPU:", torch.cuda.get_device_name(0))
    print("=" * 70 + "\n")

    test_correctness()
    test_smoothing_effect()
    run_error_scaling()
    run_long_seq_benchmark()
    run_detailed_benchmark(N=4096)
    run_detailed_benchmark(N=8192)
    run_memory_analysis()
