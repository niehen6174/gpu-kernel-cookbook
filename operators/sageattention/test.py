"""
SageAttention vs FlashAttention 对比测试

测试内容：
1. 正确性：与 PyTorch 参考实现对比（float32 精度）
2. 长序列 benchmark：N in [1024, 2048, 4096, 8192, 16384]
   对比：FlashAttention v2（官方）vs SageAttention v1/v2（本实现 Triton）
          vs 官方 SageAttention 包（若已安装）
          vs CuTe DSL v2/v3（BLOCK_M=64/128, split-warpgroup）

运行：
    cd gpu-kernel-lab
    python -m operators.sageattention.test
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

import torch
import math

from common.check import check_correctness
from common.utils import benchmark_func, compute_tflops, get_gpu_info
from operators.sageattention.triton.kernel_v1 import sageattn_v1
from operators.sageattention.triton.kernel_v2 import sageattn_v2

# CuTe DSL kernels (SM90a Hopper only)
try:
    from operators.sageattention.cute.kernel import sageattn_cutedsl
    from operators.sageattention.cute.kernel_v3 import sageattn_cutedsl_v3
    HAS_CUTE = True
except Exception as e:
    HAS_CUTE = False
    print(f"[WARN] CuTe DSL kernels not available: {e}")


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
    attn = torch.softmax(qk, dim=-1)
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
    q_ = q.transpose(1, 2).contiguous()
    k_ = k.transpose(1, 2).contiguous()
    v_ = v.transpose(1, 2).contiguous()
    out = flash_attn_func(q_, k_, v_, softmax_scale=sm_scale, causal=is_causal)
    return out.transpose(1, 2).contiguous()


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
        (1, 8,  512,  64),
        (2, 8, 1024,  64),
        (1, 16, 2048, 64),
        (1, 8,  512, 128),
        (2, 8, 1024, 128),
    ]

    for B, H, N, D in test_shapes:
        print(f"\n  Shape: B={B} H={H} N={N} D={D}")
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        ref = attention_ref_fp16(q, k, v)

        # SageAttention v1 (ours) — INT8 QK，近似算法，atol=0.05
        check_correctness(sageattn_v1(q, k, v, smooth_k=True), ref,
                          name=f"SageAttn v1 (smooth_k) N={N} D={D}", atol=0.05, rtol=0.02)
        check_correctness(sageattn_v1(q, k, v, smooth_k=False), ref,
                          name=f"SageAttn v1 (no smooth) N={N} D={D}", atol=0.05, rtol=0.02)

        # SageAttention v2 (ours) — INT8 QK + FP8 V
        check_correctness(sageattn_v2(q, k, v, smooth_k=True, smooth_v=True), ref,
                          name=f"SageAttn v2 (smooth_kv) N={N} D={D}", atol=0.05, rtol=0.02)

        # FlashAttention v2（精确，atol=5e-3）
        if HAS_FA2:
            check_correctness(flash_attn_v2(q, k, v), ref,
                              name=f"FlashAttn v2 N={N} D={D}", atol=5e-3, rtol=1e-3)

        # 官方 SageAttention
        if HAS_SAGE_PKG:
            check_correctness(sage_official(q, k, v), ref,
                              name=f"SageAttn official N={N} D={D}", atol=0.05, rtol=0.02)

        # CuTe DSL v2/v3 (仅支持 D=64, SM90a)
        if HAS_CUTE and D == 64:
            check_correctness(sageattn_cutedsl(q, k, v, smooth_k=True).to(torch.float16), ref,
                              name=f"CuTe DSL v2 N={N} D={D}", atol=0.05, rtol=0.02)
            check_correctness(sageattn_cutedsl_v3(q, k, v, smooth_k=True).to(torch.float16), ref,
                              name=f"CuTe DSL v3 N={N} D={D}", atol=0.05, rtol=0.02)

    # 因果注意力测试
    print("\n  [Causal Attention] B=1 H=8 N=1024 D=64")
    q = torch.randn(1, 8, 1024, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 8, 1024, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 8, 1024, 64, device="cuda", dtype=torch.float16)
    ref_causal = attention_ref_fp16(q, k, v, is_causal=True)

    check_correctness(sageattn_v1(q, k, v, is_causal=True), ref_causal,
                      name="SageAttn v1 causal", atol=0.05)
    # V2 causal：FP8 V 全局均值 smoothing 在因果遮蔽下误差略高
    check_correctness(sageattn_v2(q, k, v, is_causal=True), ref_causal,
                      name="SageAttn v2 causal", atol=0.15)
    if HAS_FA2:
        check_correctness(flash_attn_v2(q, k, v, is_causal=True), ref_causal,
                          name="FlashAttn v2 causal", atol=5e-3)
    print()


# ============================================================
# K/V Smoothing 效果分析
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

    def cosine_sim(a, b):
        a_ = a.float().flatten()
        b_ = b.float().flatten()
        return (a_ * b_).sum() / (a_.norm() * b_.norm())

    def mae(a, b):
        return (a.float() - b.float()).abs().mean().item()

    print(f"\n  Shape: B={B} H={H} N={N} D={D}")
    print(f"  {'方法':<30} {'MAE':>10} {'cos_sim':>10}")
    print(f"  {'-'*52}")

    for name, out in [
        ("v1 无 smoothing",      sageattn_v1(q, k, v, smooth_k=False)),
        ("v1 K smoothing",       sageattn_v1(q, k, v, smooth_k=True)),
        ("v2 K+V smoothing",     sageattn_v2(q, k, v, smooth_k=True, smooth_v=True)),
        ("v2 K smooth only",     sageattn_v2(q, k, v, smooth_k=True, smooth_v=False)),
    ]:
        print(f"  {name:<30} {mae(out, ref):>10.4f} {cosine_sim(out, ref).item():>10.6f}")
    print()


# ============================================================
# 量化误差 vs 序列长度
# ============================================================
def run_error_scaling():
    print("=" * 70)
    print("量化误差 vs 序列长度（MAE / cos_sim 相对于 FP16 参考值）")
    print("=" * 70)
    print(f"  {'N':>6}  {'Sage v1 MAE':>14}  {'Sage v2 MAE':>14}  "
          f"{'Sage v1 cos':>12}  {'Sage v2 cos':>12}")
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
            a_ = a.float().flatten(); b_ = b.float().flatten()
            return ((a_ * b_).sum() / (a_.norm() * b_.norm())).item()

        print(f"  {N:>6}  {mae(o1,ref):>14.5f}  {mae(o2,ref):>14.5f}  "
              f"{cos(o1,ref):>12.6f}  {cos(o2,ref):>12.6f}")
    print()


# ============================================================
# 长序列 Benchmark：SageAttention vs FlashAttention
# ============================================================
def run_long_seq_benchmark(B=1, H=16, D=64, seq_lens=None):
    if seq_lens is None:
        seq_lens = [1024, 2048, 4096, 8192, 16384]

    print("=" * 70)
    print(f"长序列 Benchmark: B={B} H={H} D={D}")
    print("=" * 70)
    print(get_gpu_info())
    print()

    header = f"{'N':>6}  {'FA2':>16}  {'Sage v1 (ours)':>20}  {'Sage v2 (ours)':>20}"
    if HAS_SAGE_PKG:
        header += f"  {'Sage official':>18}"
    print(header)
    print("-" * 85)

    for N in seq_lens:
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        flops = 4 * B * H * N * N * D

        if HAS_FA2:
            r = benchmark_func(flash_attn_v2, q, k, v)
            t_fa2 = r["mean_ms"]
            fa2_str = f"{t_fa2:.3f}ms ({compute_tflops(flops, t_fa2):.1f}T)"
        else:
            t_fa2 = None
            fa2_str = "N/A"

        r1 = benchmark_func(sageattn_v1, q, k, v)
        t1 = r1["mean_ms"]
        sp1 = f"{t_fa2/t1:.2f}x" if t_fa2 else ""
        s1_str = f"{t1:.3f}ms ({compute_tflops(flops, t1):.1f}T) {sp1}"

        r2 = benchmark_func(sageattn_v2, q, k, v)
        t2 = r2["mean_ms"]
        sp2 = f"{t_fa2/t2:.2f}x" if t_fa2 else ""
        s2_str = f"{t2:.3f}ms ({compute_tflops(flops, t2):.1f}T) {sp2}"

        line = f"{N:>6}  {fa2_str:>16}  {s1_str:>20}  {s2_str:>20}"

        if HAS_SAGE_PKG:
            ro = benchmark_func(sage_official, q, k, v)
            t_o = ro["mean_ms"]
            sp_o = f"{t_fa2/t_o:.2f}x" if t_fa2 else ""
            line += f"  {t_o:.3f}ms ({compute_tflops(flops, t_o):.1f}T) {sp_o}"

        print(line)
    print()


# ============================================================
# 详细 Benchmark
# ============================================================
def run_detailed_benchmark(B=1, H=16, N=4096, D=64):
    print("=" * 70)
    print(f"详细 Benchmark: B={B} H={H} N={N} D={D}")
    print("（TFLOPS 基于 4*B*H*N²*D，不含量化开销）")
    print("=" * 70)

    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    flops = 4 * B * H * N * N * D

    impls = []
    if HAS_FA2:
        impls.append(("FlashAttention v2",            lambda: flash_attn_v2(q, k, v)))
    impls += [
        ("SageAttn v1 (smooth_k)",         lambda: sageattn_v1(q, k, v, smooth_k=True)),
        ("SageAttn v1 (no smooth)",        lambda: sageattn_v1(q, k, v, smooth_k=False)),
        ("SageAttn v2 (smooth_kv)",        lambda: sageattn_v2(q, k, v, smooth_k=True, smooth_v=True)),
        ("SageAttn v2 (smooth_k only)",    lambda: sageattn_v2(q, k, v, smooth_k=True, smooth_v=False)),
    ]
    if HAS_CUTE and D == 64:
        impls += [
            ("CuTe DSL v2 (BLOCK_M=64)",   lambda: sageattn_cutedsl(q, k, v, smooth_k=True)),
            ("CuTe DSL v3 (BLOCK_M=128)",  lambda: sageattn_cutedsl_v3(q, k, v, smooth_k=True)),
        ]
    if HAS_SAGE_PKG:
        impls.append(("SageAttn official",            lambda: sage_official(q, k, v)))

    print(f"\n  {'实现':<35}  {'时间(ms)':>10}  {'TFLOPS':>8}  {'vs FA2':>8}")
    print(f"  {'-'*65}")

    fa2_t = None
    for name, fn in impls:
        try:
            r = benchmark_func(fn)
            t = r["mean_ms"]
            tf = compute_tflops(flops, t)
            if fa2_t is None and "FlashAttention" in name:
                fa2_t = t
            sp = f"{fa2_t/t:.2f}x" if fa2_t and "FlashAttention" not in name else "baseline"
            print(f"  {name:<35}  {t:>10.3f}  {tf:>8.2f}  {sp:>8}")
        except Exception as e:
            print(f"  {name:<35}  [ERROR: {e}]")
    print()

    # Causal
    print(f"  [Causal 模式]")
    print(f"  {'实现':<35}  {'时间(ms)':>10}  {'TFLOPS':>8}")
    print(f"  {'-'*55}")
    causal = []
    if HAS_FA2:
        causal.append(("FlashAttention v2 causal",    lambda: flash_attn_v2(q, k, v, is_causal=True)))
    causal += [
        ("SageAttn v1 causal",             lambda: sageattn_v1(q, k, v, is_causal=True)),
        ("SageAttn v2 causal",             lambda: sageattn_v2(q, k, v, is_causal=True)),
    ]
    for name, fn in causal:
        try:
            r = benchmark_func(fn)
            t = r["mean_ms"]
            tf = compute_tflops(flops // 2, t)
            print(f"  {name:<35}  {t:>10.3f}  {tf:>8.2f}")
        except Exception as e:
            print(f"  {name:<35}  [ERROR: {e}]")
    print()


# ============================================================
# 内存占用分析
# ============================================================
def run_memory_analysis():
    print("=" * 70)
    print("内存占用分析（peak memory，H=16 D=64）")
    print("=" * 70)
    print(f"  {'N':>6}  {'FA2(MB)':>10}  {'Sage v1(MB)':>12}  {'Sage v2(MB)':>12}")
    print(f"  {'-'*45}")

    B, H, D = 1, 16, 64
    for N in [2048, 4096, 8192, 16384]:
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        def measure(fn):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            fn()
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / 1024**2

        mem_fa2 = measure(lambda: flash_attn_v2(q, k, v)) if HAS_FA2 else float("nan")
        mem_s1  = measure(lambda: sageattn_v1(q, k, v))
        mem_s2  = measure(lambda: sageattn_v2(q, k, v))
        print(f"  {N:>6}  {mem_fa2:>10.1f}  {mem_s1:>12.1f}  {mem_s2:>12.1f}")
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
