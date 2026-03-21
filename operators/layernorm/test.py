"""
LayerNorm 正确性测试 + Benchmark

运行方式：
    python test.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

import torch
import ctypes

from common.check import check_correctness
from common.utils import benchmark_func, compute_bandwidth, get_gpu_info
from operators.layernorm.pytorch.baseline import layernorm_pytorch
from operators.layernorm.triton.kernel import layernorm_triton


def load_cuda_lib():
    lib_path = os.path.join(os.path.dirname(__file__), "cuda/layernorm.so")
    if not os.path.exists(lib_path):
        print(f"[SKIP] CUDA lib not found: {lib_path}  (run: cd cuda && bash build.sh)")
        return None
    lib = ctypes.CDLL(lib_path)
    for fn in ["layernorm_cuda_v1", "layernorm_cuda_v2", "layernorm_cuda_v3"]:
        getattr(lib, fn).argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float
        ]
    return lib


def run_cuda(lib, fn, X, weight, bias):
    B = X.shape[0] if X.dim() == 2 else X.reshape(-1, X.shape[-1]).shape[0]
    N = X.shape[-1]
    X2 = X.reshape(B, N)
    Y = torch.empty_like(X2)
    w_ptr = weight.data_ptr() if weight is not None else 0
    b_ptr = bias.data_ptr() if bias is not None else 0
    getattr(lib, fn)(
        X2.data_ptr(), ctypes.c_void_p(w_ptr), ctypes.c_void_p(b_ptr),
        Y.data_ptr(), ctypes.c_int(B), ctypes.c_int(N), ctypes.c_float(1e-5)
    )
    return Y.reshape(X.shape)


def test_correctness():
    print("=" * 60)
    print("Correctness Test")
    print("=" * 60)
    for B, N in [(64, 512), (128, 768), (256, 1024)]:
        print(f"\n  Shape: ({B}, {N})")
        X = torch.randn(B, N, device="cuda")
        W = torch.ones(N, device="cuda")
        b = torch.zeros(N, device="cuda")
        ref = layernorm_pytorch(X, W, b)

        out = layernorm_triton(X, W, b)
        check_correctness(out, ref, name=f"Triton ({B}x{N})")

        lib = load_cuda_lib()
        if lib:
            for v in ["layernorm_cuda_v1", "layernorm_cuda_v2", "layernorm_cuda_v3"]:
                out = run_cuda(lib, v, X, W, b)
                check_correctness(out, ref, name=f"CUDA {v.split('_')[-1]} ({B}x{N})")

        try:
            from operators.layernorm.cutlass.wrapper import layernorm_cutlass_v1, layernorm_cutlass_v2, layernorm_cutlass_v3
            check_correctness(layernorm_cutlass_v1(X, W, b), ref, name=f"CuTe v1 ({B}x{N})")
            check_correctness(layernorm_cutlass_v2(X, W, b), ref, name=f"CuTe v2 ({B}x{N})")
            check_correctness(layernorm_cutlass_v3(X, W, b), ref, name=f"CuTe v3 ({B}x{N})")
        except RuntimeError as e:
            print(f"  [SKIP] CuTe: {e}")
    print()


def run_benchmark(B=4096, N=1024):
    print("=" * 60)
    print(f"Benchmark  ({B}x{N} float32)")
    print("=" * 60)
    print(get_gpu_info())
    print()

    X = torch.randn(B, N, device="cuda")
    W = torch.ones(N, device="cuda")
    b = torch.zeros(N, device="cuda")
    # 读 X, W, B，写 Y → 约 4 * 3 * B * N bytes
    bytes_accessed = B * N * 4 * 4

    res = benchmark_func(layernorm_pytorch, X, W, b)
    bw = compute_bandwidth(bytes_accessed, res["mean_ms"])
    print(f"PyTorch    : {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s")
    baseline = res["mean_ms"]

    res = benchmark_func(layernorm_triton, X, W, b)
    bw = compute_bandwidth(bytes_accessed, res["mean_ms"])
    print(f"Triton     : {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s  {baseline/res['mean_ms']:.2f}x")

    lib = load_cuda_lib()
    if lib:
        for v in ["layernorm_cuda_v1", "layernorm_cuda_v2", "layernorm_cuda_v3"]:
            res = benchmark_func(run_cuda, lib, v, X, W, b)
            bw = compute_bandwidth(bytes_accessed, res["mean_ms"])
            print(f"CUDA {v.split('_')[-1]:4s}  : {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s  {baseline/res['mean_ms']:.2f}x")

    try:
        from operators.layernorm.cutlass.wrapper import layernorm_cutlass_v1, layernorm_cutlass_v2, layernorm_cutlass_v3
        for label, fn in [("CuTe v1", layernorm_cutlass_v1), ("CuTe v2", layernorm_cutlass_v2), ("CuTe v3", layernorm_cutlass_v3)]:
            res = benchmark_func(fn, X, W, b)
            bw = compute_bandwidth(bytes_accessed, res["mean_ms"])
            print(f"{label:10s}: {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s  {baseline/res['mean_ms']:.2f}x")
    except RuntimeError as e:
        print(f"[SKIP] CuTe: {e}")
    print()


if __name__ == "__main__":
    test_correctness()
    run_benchmark()
