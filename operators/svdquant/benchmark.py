"""
SVDQuant Benchmark

比较各版本实现的性能：
1. FP16 baseline（cuBLAS，理论上限）
2. PyTorch SVDQuant（参考实现）
3. Triton SVDQuant（优化版）

运行方式：
    python operators/svdquant/benchmark.py
    python operators/svdquant/benchmark.py --warmup 5 --repeat 50
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

import torch

from common.utils import benchmark_func, compute_tflops, get_gpu_info
from operators.svdquant.pytorch.svdquant_torch import (
    create_svdquant_params,
    svdquant_forward_torch,
)
from operators.svdquant.pytorch.baseline import svdquant_fp16_baseline, matmul_fp16_baseline


# -------------------------------------------------------------------------
# 工具函数
# -------------------------------------------------------------------------

def make_tensors(M, K, N, rank=32, group_size=64, dtype=torch.float16, device="cuda"):
    """生成标准测试张量"""
    torch.manual_seed(42)
    x = torch.randn(M, K, dtype=dtype, device=device) * 0.5
    W = torch.randn(K, N, dtype=dtype, device=device) * 0.02
    smooth = torch.rand(K, dtype=dtype, device=device) * 0.5 + 0.5
    bias = torch.randn(N, dtype=dtype, device=device) * 0.01
    params = create_svdquant_params(W, rank=rank, group_size=group_size, smooth=smooth)
    return x, W, smooth, bias, params


def compute_gemm_flops(M, K, N, rank):
    """
    计算 SVDQuant 前向的理论 FLOPs：
    - 主 GEMM: 2*M*K*N
    - LoRA down: 2*M*K*rank
    - LoRA up: 2*M*rank*N
    总计: 2*M*(K*N + K*rank + rank*N) ≈ 2*M*K*(N + rank) + 2*M*rank*N
    """
    gemm_flops = 2 * M * K * N
    lora_down_flops = 2 * M * K * rank
    lora_up_flops = 2 * M * rank * N
    return gemm_flops + lora_down_flops + lora_up_flops


def compute_memory_bytes(M, K, N, rank, group_size=64, dtype_bytes=2):
    """
    计算理论内存访问量（bytes）：

    FP16 baseline:
        read: x(M*K) + W(K*N), write: y(M*N) = (M*K + K*N + M*N) * 2

    SVDQuant:
        read: x(M*K*2) + Q_x(M*K//2) + Q_W(K*N//2)
              + ascales(M*K/group_size*2) + wscales(K/group_size*N*2)
              + lora_down(K*rank*2) + lora_up(rank*N*2)
              + smooth(K*2)
        write: y(M*N*2)
    """
    fp16_bytes = (M * K + K * N + M * N) * dtype_bytes

    quant_x_bytes = M * (K // 2)           # INT4 quantized x
    quant_w_bytes = (K // 2) * N           # INT4 quantized W
    ascales_bytes = M * (K // group_size) * dtype_bytes
    wscales_bytes = (K // group_size) * N * dtype_bytes
    lora_bytes    = (K * rank + rank * N) * dtype_bytes
    smooth_bytes  = K * dtype_bytes
    x_read_bytes  = M * K * dtype_bytes
    y_write_bytes = M * N * dtype_bytes

    svdq_bytes = (quant_x_bytes + quant_w_bytes + ascales_bytes + wscales_bytes +
                  lora_bytes + smooth_bytes + x_read_bytes + y_write_bytes)

    return {
        "fp16": fp16_bytes,
        "svdquant": svdq_bytes,
        "compression_ratio": fp16_bytes / svdq_bytes,
    }


# -------------------------------------------------------------------------
# Benchmark 函数
# -------------------------------------------------------------------------

def run_benchmark(M, K, N, rank=32, group_size=64, warmup=10, repeat=100):
    """对给定矩阵尺寸运行完整 benchmark"""
    x, W, smooth, bias, params = make_tensors(M, K, N, rank=rank, group_size=group_size)

    # 预处理 FP16 baseline 需要的权重
    from operators.svdquant.pytorch.svdquant_torch import int4_dequantize
    W_residual_dq = int4_dequantize(
        params["q_w"].T.contiguous(),
        params["wscales"].T.contiguous(),
        group_size=group_size,
    ).T.contiguous()

    flops = compute_gemm_flops(M, K, N, rank)
    mem_info = compute_memory_bytes(M, K, N, rank, group_size)

    results = {}

    # ---- FP16 Baseline (cuBLAS) ----
    def fp16_fn():
        return matmul_fp16_baseline(x, W, bias)

    r = benchmark_func(fp16_fn, warmup=warmup, repeat=repeat)
    results["fp16_baseline"] = {
        "mean_ms": r["mean_ms"],
        "min_ms": r["min_ms"],
        "tflops": compute_tflops(flops, r["mean_ms"]),
        "speedup": 1.0,   # baseline
    }

    # ---- SVDQuant FP16 Baseline (带 SVD 分解，无量化) ----
    def svdq_fp16_fn():
        return svdquant_fp16_baseline(
            x, W_residual_dq, params["lora_down"], params["lora_up"], smooth, bias
        )

    r = benchmark_func(svdq_fp16_fn, warmup=warmup, repeat=repeat)
    results["svdquant_fp16_nosim"] = {
        "mean_ms": r["mean_ms"],
        "min_ms": r["min_ms"],
        "tflops": compute_tflops(flops, r["mean_ms"]),
        "speedup": results["fp16_baseline"]["mean_ms"] / r["mean_ms"],
    }

    # ---- PyTorch SVDQuant (INT4 模拟) ----
    def torch_svdq_fn():
        return svdquant_forward_torch(
            x, params["q_w"], params["wscales"],
            params["lora_down"], params["lora_up"],
            smooth, bias, group_size=group_size,
        )

    r = benchmark_func(torch_svdq_fn, warmup=warmup, repeat=repeat)
    results["svdquant_pytorch"] = {
        "mean_ms": r["mean_ms"],
        "min_ms": r["min_ms"],
        "tflops": compute_tflops(flops, r["mean_ms"]),
        "speedup": results["fp16_baseline"]["mean_ms"] / r["mean_ms"],
    }

    # ---- Triton SVDQuant ----
    try:
        from operators.svdquant.triton.kernel import svdquant_forward_triton

        def triton_svdq_fn():
            return svdquant_forward_triton(
                x, params["q_w"], params["wscales"],
                params["lora_down"], params["lora_up"],
                smooth, bias, group_size=group_size,
            )

        # warmup（Triton JIT 需要首次编译）
        triton_svdq_fn()
        torch.cuda.synchronize()

        r = benchmark_func(triton_svdq_fn, warmup=warmup, repeat=repeat)
        results["svdquant_triton"] = {
            "mean_ms": r["mean_ms"],
            "min_ms": r["min_ms"],
            "tflops": compute_tflops(flops, r["mean_ms"]),
            "speedup": results["fp16_baseline"]["mean_ms"] / r["mean_ms"],
        }
    except Exception as e:
        results["svdquant_triton"] = {"error": str(e)}

    # ---- Triton Optimized SVDQuant（单 K-loop + autotune）----
    try:
        from operators.svdquant.triton.kernel import svdquant_forward_triton_opt

        def triton_opt_fn():
            return svdquant_forward_triton_opt(
                x, params["q_w"], params["wscales"],
                params["lora_down"], params["lora_up"],
                smooth, bias, group_size=group_size,
            )

        # warmup + autotune
        triton_opt_fn()
        torch.cuda.synchronize()

        r = benchmark_func(triton_opt_fn, warmup=warmup, repeat=repeat)
        results["svdquant_triton_opt"] = {
            "mean_ms": r["mean_ms"],
            "min_ms": r["min_ms"],
            "tflops": compute_tflops(flops, r["mean_ms"]),
            "speedup": results["fp16_baseline"]["mean_ms"] / r["mean_ms"],
        }
    except Exception as e:
        results["svdquant_triton_opt"] = {"error": str(e)}

    # ---- CuTe V2 ----
    try:
        from operators.svdquant.cute.kernel import svdquant_forward_cute, CUTE_AVAILABLE as _CUTE_OK
        if _CUTE_OK:
            def cute_v2_fn():
                return svdquant_forward_cute(
                    x, params["q_w"], params["wscales"],
                    params["lora_down"], params["lora_up"],
                    smooth, bias, group_size=group_size, version="v2",
                )
            cute_v2_fn()
            torch.cuda.synchronize()
            r = benchmark_func(cute_v2_fn, warmup=warmup, repeat=repeat)
            results["svdquant_cute_v2"] = {
                "mean_ms": r["mean_ms"],
                "min_ms": r["min_ms"],
                "tflops": compute_tflops(flops, r["mean_ms"]),
                "speedup": results["fp16_baseline"]["mean_ms"] / r["mean_ms"],
            }
        else:
            results["svdquant_cute_v2"] = {"error": "cutlass.cute not available"}
    except Exception as e:
        results["svdquant_cute_v2"] = {"error": str(e)}

    # ---- CuTe V3（cuBLAS GEMM + CuTe LoRA epilogue）----
    try:
        from operators.svdquant.cute.kernel import svdquant_forward_cute_v3, CUTE_AVAILABLE as _CUTE_OK3
        if _CUTE_OK3:
            def cute_v3_fn():
                return svdquant_forward_cute_v3(
                    x, params["q_w"], params["wscales"],
                    params["lora_down"], params["lora_up"],
                    smooth, bias, group_size=group_size,
                )
            cute_v3_fn()
            torch.cuda.synchronize()
            r = benchmark_func(cute_v3_fn, warmup=warmup, repeat=repeat)
            results["svdquant_cute_v3"] = {
                "mean_ms": r["mean_ms"],
                "min_ms": r["min_ms"],
                "tflops": compute_tflops(flops, r["mean_ms"]),
                "speedup": results["fp16_baseline"]["mean_ms"] / r["mean_ms"],
            }
        else:
            results["svdquant_cute_v3"] = {"error": "cutlass.cute not available"}
    except Exception as e:
        results["svdquant_cute_v3"] = {"error": str(e)}

    # ---- Nunchaku INT4 MMA ----
    try:
        from operators.svdquant.nunchaku.kernel import (
            svdquant_forward_nunchaku_cached,
            prepare_nunchaku_weights,
            NUNCHAKU_AVAILABLE,
        )
        if NUNCHAKU_AVAILABLE:
            # 预处理权重（不计入 benchmark）
            wgt_packed, lu_T = prepare_nunchaku_weights(params["q_w"], params["lora_up"])

            def nunchaku_fn():
                return svdquant_forward_nunchaku_cached(
                    x, wgt_packed, params["wscales"],
                    params["lora_down"], lu_T, smooth, bias,
                )

            # warmup
            nunchaku_fn()
            torch.cuda.synchronize()

            r = benchmark_func(nunchaku_fn, warmup=warmup, repeat=repeat)
            results["svdquant_nunchaku"] = {
                "mean_ms": r["mean_ms"],
                "min_ms": r["min_ms"],
                "tflops": compute_tflops(flops, r["mean_ms"]),
                "speedup": results["fp16_baseline"]["mean_ms"] / r["mean_ms"],
            }
        else:
            results["svdquant_nunchaku"] = {"error": "nunchaku not available"}
    except Exception as e:
        results["svdquant_nunchaku"] = {"error": str(e)}

    # ---- CUTLASS W8A8 SM90 WGMMA ----
    try:
        import operators.svdquant.cutlass_w8a8.kernel as _w8a8_mod
        from operators.svdquant.cutlass_w8a8.kernel import (
            svdquant_forward_w8a8_cached,
            prepare_w8a8_weights,
        )
        _w8a8_mod._load_extension()   # 触发 JIT 编译（首次约 60-120s，之后缓存）
        if _w8a8_mod.W8A8_AVAILABLE:
            q_w8_col, wscales8 = prepare_w8a8_weights(
                params["q_w"], params["wscales"], group_size_int4=group_size)

            def w8a8_fn():
                return svdquant_forward_w8a8_cached(
                    x, q_w8_col, wscales8,
                    params["lora_down"], params["lora_up"],
                    smooth, bias,
                )

            # warmup
            w8a8_fn()
            torch.cuda.synchronize()

            r = benchmark_func(w8a8_fn, warmup=warmup, repeat=repeat)
            results["svdquant_w8a8_sm90"] = {
                "mean_ms": r["mean_ms"],
                "min_ms":  r["min_ms"],
                "tflops":  compute_tflops(flops, r["mean_ms"]),
                "speedup": results["fp16_baseline"]["mean_ms"] / r["mean_ms"],
            }
        else:
            results["svdquant_w8a8_sm90"] = {"error": "W8A8 not available"}
    except Exception as e:
        results["svdquant_w8a8_sm90"] = {"error": str(e)}

    # ---- CUTLASS W8A8 V2 (fused smooth-quant + fused epilogue) ----
    try:
        import operators.svdquant.cutlass_w8a8.kernel as _w8a8_mod
        from operators.svdquant.cutlass_w8a8.kernel import (
            svdquant_forward_w8a8_v2_cached,
            prepare_w8a8_weights,
        )
        _w8a8_mod._load_extension_v2()
        if _w8a8_mod.W8A8_V2_AVAILABLE:
            q_w8_col, wscales8 = prepare_w8a8_weights(
                params["q_w"], params["wscales"], group_size_int4=group_size)

            def w8a8_v2_fn():
                return svdquant_forward_w8a8_v2_cached(
                    x, q_w8_col, wscales8,
                    params["lora_down"], params["lora_up"],
                    smooth, bias,
                )

            w8a8_v2_fn()
            torch.cuda.synchronize()

            r = benchmark_func(w8a8_v2_fn, warmup=warmup, repeat=repeat)
            results["svdquant_w8a8_v2"] = {
                "mean_ms": r["mean_ms"],
                "min_ms":  r["min_ms"],
                "tflops":  compute_tflops(flops, r["mean_ms"]),
                "speedup": results["fp16_baseline"]["mean_ms"] / r["mean_ms"],
            }
        else:
            results["svdquant_w8a8_v2"] = {"error": "W8A8 V2 not available"}
    except Exception as e:
        results["svdquant_w8a8_v2"] = {"error": str(e)}

    # ---- CUTLASS W8A8 V3 (V2 + stream overlap LoRA + no .item() sync) ----
    try:
        import operators.svdquant.cutlass_w8a8.kernel as _w8a8_mod
        from operators.svdquant.cutlass_w8a8.kernel import (
            svdquant_forward_w8a8_v3_cached,
            prepare_w8a8_weights,
            prepare_w8a8_v3_cache,
        )
        _w8a8_mod._load_extension_v2()
        if _w8a8_mod.W8A8_V2_AVAILABLE:
            q_w8_col, wscales8 = prepare_w8a8_weights(
                params["q_w"], params["wscales"], group_size_int4=group_size)
            wscale_mean_cached = prepare_w8a8_v3_cache(wscales8)

            def w8a8_v3_fn():
                return svdquant_forward_w8a8_v3_cached(
                    x, q_w8_col, wscales8,
                    params["lora_down"], params["lora_up"],
                    smooth, bias,
                    wscale_mean_cached=wscale_mean_cached,
                )

            w8a8_v3_fn()
            torch.cuda.synchronize()

            r = benchmark_func(w8a8_v3_fn, warmup=warmup, repeat=repeat)
            results["svdquant_w8a8_v3"] = {
                "mean_ms": r["mean_ms"],
                "min_ms":  r["min_ms"],
                "tflops":  compute_tflops(flops, r["mean_ms"]),
                "speedup": results["fp16_baseline"]["mean_ms"] / r["mean_ms"],
            }
        else:
            results["svdquant_w8a8_v3"] = {"error": "W8A8 V3 not available (V2 ext needed)"}
    except Exception as e:
        results["svdquant_w8a8_v3"] = {"error": str(e)}

    return results, mem_info


# -------------------------------------------------------------------------
# 打印函数
# -------------------------------------------------------------------------

def print_results(config_key, results, mem_info):
    print(f"\n  Config: {config_key}")
    print(f"  Memory: FP16={mem_info['fp16']/1024**2:.1f}MB, "
          f"SVDQuant={mem_info['svdquant']/1024**2:.1f}MB, "
          f"Compression={mem_info['compression_ratio']:.2f}x")
    print(f"  {'Impl':<28} {'Mean(ms)':>10} {'Min(ms)':>10} {'TFLOPS':>10} {'Speedup':>10}")
    print(f"  {'-'*68}")
    for impl, r in results.items():
        if "error" in r:
            print(f"  {impl:<28} ERROR: {r['error'][:50]}")
        else:
            print(f"  {impl:<28} {r['mean_ms']:>10.4f} {r['min_ms']:>10.4f} "
                  f"{r['tflops']:>10.3f} {r['speedup']:>10.2f}x")


# -------------------------------------------------------------------------
# 主函数
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SVDQuant Benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=100, help="Repeat iterations")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--group-size", type=int, default=64, help="Quantization group size")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA GPU available!")
        return

    print(get_gpu_info())
    print(f"\nSVDQuant Benchmark")
    print(f"rank={args.rank}, group_size={args.group_size}")
    print(f"warmup={args.warmup}, repeat={args.repeat}")
    print("="*80)

    configs = [
        (64,  512,  512),    # 小
        (256, 2048, 2048),   # 中
        (1024, 4096, 4096),  # 大（LLM 典型尺寸）
    ]

    all_results = {}
    for M, K, N in configs:
        config_key = f"M={M},K={K},N={N}"
        print(f"\nBenchmarking {config_key}...")
        try:
            results, mem_info = run_benchmark(
                M, K, N,
                rank=args.rank,
                group_size=args.group_size,
                warmup=args.warmup,
                repeat=args.repeat,
            )
            all_results[config_key] = results
            print_results(config_key, results, mem_info)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("Summary: speedup vs PyTorch SVDQuant")
    cols = [
        ("svdquant_triton",     "Triton (orig)"),
        ("svdquant_triton_opt", "Triton (opt)"),
        ("svdquant_cute_v2",    "CuTe V2"),
        ("svdquant_cute_v3",    "CuTe V3"),
        ("svdquant_nunchaku",   "Nunchaku"),
        ("svdquant_w8a8_sm90",  "W8A8 V1"),
        ("svdquant_w8a8_v2",    "W8A8 V2"),
        ("svdquant_w8a8_v3",    "W8A8 V3"),
    ]
    print(f"  {'Config':<28} " + " ".join(f"{label:>18}" for _, label in cols))
    for config_key, results in all_results.items():
        p = results.get("svdquant_pytorch", {})
        parts = []
        for key, _ in cols:
            r = results.get(key, {})
            if "error" not in r and "error" not in p and "mean_ms" in r and "mean_ms" in p:
                rel = p["mean_ms"] / r["mean_ms"]
                parts.append(f"{rel:>17.2f}x")
            elif "error" in r:
                parts.append(f"{'ERR':>18}")
            else:
                parts.append(f"{'N/A':>18}")
        print(f"  {config_key:<28} " + " ".join(parts))


if __name__ == "__main__":
    main()
