"""
统一 Benchmark 入口

运行方式：
    python benchmarks/benchmark.py                     # 运行所有算子
    python benchmarks/benchmark.py --op matmul         # 只运行指定算子
    python benchmarks/benchmark.py --save              # 保存结果到 results/
"""

import sys
import os
import argparse
import json
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))

import torch
from common.utils import benchmark_func, compute_bandwidth, compute_tflops, get_gpu_info


def benchmark_vector_add():
    from operators.vector_add.pytorch.baseline import vector_add_pytorch
    from operators.vector_add.triton.kernel import vector_add_triton

    results = {}
    for N in [1024*1024, 1024*1024*16, 1024*1024*64]:
        A = torch.randn(N, device="cuda")
        B = torch.randn(N, device="cuda")
        bytes_accessed = N * 3 * 4

        key = f"N={N//1024//1024}M"
        results[key] = {}

        for name, fn in [("pytorch", lambda: vector_add_pytorch(A, B)),
                          ("triton",  lambda: vector_add_triton(A, B))]:
            r = benchmark_func(fn)
            results[key][name] = {
                "mean_ms": r["mean_ms"],
                "bw_gbs": compute_bandwidth(bytes_accessed, r["mean_ms"]),
            }

    return results


def benchmark_transpose():
    from operators.transpose.pytorch.baseline import transpose_pytorch
    from operators.transpose.triton.kernel import transpose_triton

    results = {}
    for M, N in [(1024, 1024), (4096, 4096), (8192, 8192)]:
        A = torch.randn(M, N, device="cuda")
        bytes_accessed = M * N * 4 * 2

        key = f"{M}x{N}"
        results[key] = {}
        for name, fn in [("pytorch", lambda: transpose_pytorch(A)),
                          ("triton",  lambda: transpose_triton(A))]:
            r = benchmark_func(fn)
            results[key][name] = {
                "mean_ms": r["mean_ms"],
                "bw_gbs": compute_bandwidth(bytes_accessed, r["mean_ms"]),
            }
    return results


def benchmark_softmax():
    from operators.softmax.pytorch.baseline import softmax_pytorch
    from operators.softmax.triton.kernel import softmax_triton

    results = {}
    for B, N in [(1024, 512), (4096, 2048), (4096, 8192)]:
        X = torch.randn(B, N, device="cuda")
        bytes_accessed = B * N * 3 * 4

        key = f"B={B},N={N}"
        results[key] = {}
        for name, fn in [("pytorch", lambda: softmax_pytorch(X)),
                          ("triton",  lambda: softmax_triton(X))]:
            r = benchmark_func(fn)
            results[key][name] = {
                "mean_ms": r["mean_ms"],
                "bw_gbs": compute_bandwidth(bytes_accessed, r["mean_ms"]),
            }
    return results


def benchmark_layernorm():
    from operators.layernorm.pytorch.baseline import layernorm_pytorch
    from operators.layernorm.triton.kernel import layernorm_triton

    results = {}
    for B, N in [(4096, 512), (4096, 1024), (4096, 4096)]:
        X = torch.randn(B, N, device="cuda")
        W = torch.ones(N, device="cuda")
        b = torch.zeros(N, device="cuda")
        bytes_accessed = B * N * 4 * 4

        key = f"B={B},N={N}"
        results[key] = {}
        for name, fn in [("pytorch", lambda: layernorm_pytorch(X, W, b)),
                          ("triton",  lambda: layernorm_triton(X, W, b))]:
            r = benchmark_func(fn)
            results[key][name] = {
                "mean_ms": r["mean_ms"],
                "bw_gbs": compute_bandwidth(bytes_accessed, r["mean_ms"]),
            }
    return results


def benchmark_matmul():
    from operators.matmul.pytorch.baseline import matmul_pytorch
    from operators.matmul.triton.kernel import matmul_triton

    results = {}
    for M, K, N in [(512, 512, 512), (2048, 2048, 2048), (4096, 4096, 4096)]:
        A = torch.randn(M, K, device="cuda")
        B = torch.randn(K, N, device="cuda")
        flops = 2 * M * N * K

        key = f"{M}x{K}x{N}"
        results[key] = {}
        for name, fn in [("pytorch (cuBLAS)", lambda: matmul_pytorch(A, B)),
                          ("triton",           lambda: matmul_triton(A, B))]:
            r = benchmark_func(fn)
            results[key][name] = {
                "mean_ms": r["mean_ms"],
                "tflops": compute_tflops(flops, r["mean_ms"]),
            }
    return results


def benchmark_attention():
    from operators.attention.pytorch.baseline import attention_pytorch
    from operators.attention.triton.kernel import flash_attention_triton

    results = {}
    for B, H, N, D in [(4, 8, 512, 64), (4, 8, 1024, 64), (1, 8, 4096, 64)]:
        Q = torch.randn(B, H, N, D, device="cuda")
        K = torch.randn(B, H, N, D, device="cuda")
        V = torch.randn(B, H, N, D, device="cuda")
        flops = 4 * B * H * N * N * D

        key = f"B={B},H={H},N={N},D={D}"
        results[key] = {}
        for name, fn in [("pytorch (naive)",  lambda: attention_pytorch(Q, K, V)),
                          ("triton flash",     lambda: flash_attention_triton(Q, K, V))]:
            r = benchmark_func(fn)
            results[key][name] = {
                "mean_ms": r["mean_ms"],
                "tflops": compute_tflops(flops, r["mean_ms"]),
            }
    return results


BENCHMARKS = {
    "vector_add": benchmark_vector_add,
    "transpose":  benchmark_transpose,
    "softmax":    benchmark_softmax,
    "layernorm":  benchmark_layernorm,
    "matmul":     benchmark_matmul,
    "attention":  benchmark_attention,
}


def print_results(op_name, results):
    print(f"\n{'='*60}")
    print(f"  {op_name.upper()}")
    print(f"{'='*60}")
    for config, impls in results.items():
        print(f"\n  Config: {config}")
        baseline_ms = None
        for impl, metrics in impls.items():
            ms = metrics["mean_ms"]
            if baseline_ms is None:
                baseline_ms = ms
            speedup = baseline_ms / ms
            extra = ""
            if "bw_gbs" in metrics:
                extra = f"  BW={metrics['bw_gbs']:.1f} GB/s"
            elif "tflops" in metrics:
                extra = f"  {metrics['tflops']:.2f} TFLOPS"
            print(f"    {impl:<20} {ms:.4f} ms  speedup={speedup:.2f}x{extra}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=str, default="all",
                        help="Operator to benchmark (all, vector_add, transpose, ...)")
    parser.add_argument("--save", action="store_true",
                        help="Save results to results/ directory")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA GPU available!")
        return

    print(get_gpu_info())

    ops_to_run = list(BENCHMARKS.keys()) if args.op == "all" else [args.op]
    all_results = {}

    for op in ops_to_run:
        if op not in BENCHMARKS:
            print(f"Unknown operator: {op}")
            continue
        print(f"\nBenchmarking {op}...")
        try:
            results = BENCHMARKS[op]()
            all_results[op] = results
            print_results(op, results)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    if args.save:
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"results/benchmark_{timestamp}.json"
        with open(fname, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {fname}")


if __name__ == "__main__":
    main()
