"""
Softmax 正确性测试 + Benchmark

运行方式：
    python test.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

import torch
import ctypes

from common.check import check_correctness
from common.utils import benchmark_func, compute_bandwidth, get_gpu_info
from operators.softmax.pytorch.baseline import softmax_pytorch
from operators.softmax.triton.kernel import softmax_triton, softmax_triton_large


def load_cuda_lib():
    lib_path = os.path.join(os.path.dirname(__file__), "cuda/softmax.so")
    if not os.path.exists(lib_path):
        print(f"[SKIP] CUDA lib not found: {lib_path}  (run: cd cuda && bash build.sh)")
        return None
    lib = ctypes.CDLL(lib_path)
    for fn in ["softmax_cuda_v1", "softmax_cuda_v2", "softmax_cuda_v3"]:
        getattr(lib, fn).argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int
        ]
    return lib


def run_cuda(lib, fn, X):
    B, N = X.shape
    Y = torch.empty_like(X)
    getattr(lib, fn)(X.data_ptr(), Y.data_ptr(), ctypes.c_int(B), ctypes.c_int(N))
    return Y


def test_correctness():
    print("=" * 60)
    print("Correctness Test")
    print("=" * 60)
    for B, N in [(1, 1024), (64, 512), (128, 2048)]:
        print(f"\n  Shape: ({B}, {N})")
        X = torch.randn(B, N, device="cuda")
        ref = softmax_pytorch(X)

        out = softmax_triton(X)
        check_correctness(out, ref, name=f"Triton ({B}x{N})")

        lib = load_cuda_lib()
        if lib:
            for v in ["softmax_cuda_v1", "softmax_cuda_v2", "softmax_cuda_v3"]:
                out = run_cuda(lib, v, X)
                check_correctness(out, ref, name=f"CUDA {v.split('_')[-1]} ({B}x{N})")

        try:
            from operators.softmax.cutlass.wrapper import softmax_cutlass_v1, softmax_cutlass_v2
            check_correctness(softmax_cutlass_v1(X), ref, name=f"CuTe v1 ({B}x{N})")
            check_correctness(softmax_cutlass_v2(X), ref, name=f"CuTe v2 ({B}x{N})")
        except RuntimeError as e:
            print(f"  [SKIP] CuTe: {e}")
    print()


def run_benchmark(B=4096, N=2048):
    print("=" * 60)
    print(f"Benchmark  ({B}x{N} float32)")
    print("=" * 60)
    print(get_gpu_info())
    print()

    X = torch.randn(B, N, device="cuda")
    # softmax: 读 2N，写 N 个 float（exp 两次 + write）
    bytes_accessed = B * N * 3 * 4

    res = benchmark_func(softmax_pytorch, X)
    bw = compute_bandwidth(bytes_accessed, res["mean_ms"])
    print(f"PyTorch    : {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s")
    baseline = res["mean_ms"]

    res = benchmark_func(softmax_triton, X)
    bw = compute_bandwidth(bytes_accessed, res["mean_ms"])
    print(f"Triton     : {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s  {baseline/res['mean_ms']:.2f}x")

    lib = load_cuda_lib()
    if lib:
        for v in ["softmax_cuda_v1", "softmax_cuda_v2", "softmax_cuda_v3"]:
            res = benchmark_func(run_cuda, lib, v, X)
            bw = compute_bandwidth(bytes_accessed, res["mean_ms"])
            print(f"CUDA {v.split('_')[-1]:4s}   : {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s  {baseline/res['mean_ms']:.2f}x")

    try:
        from operators.softmax.cutlass.wrapper import softmax_cutlass_v1, softmax_cutlass_v2
        for label, fn in [("CuTe v1", softmax_cutlass_v1), ("CuTe v2", softmax_cutlass_v2)]:
            res = benchmark_func(fn, X)
            bw = compute_bandwidth(bytes_accessed, res["mean_ms"])
            print(f"{label:10s}: {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s  {baseline/res['mean_ms']:.2f}x")
    except RuntimeError as e:
        print(f"[SKIP] CuTe: {e}")
    print()


if __name__ == "__main__":
    test_correctness()
    run_benchmark()
