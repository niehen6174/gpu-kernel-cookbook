"""
Matrix Transpose 正确性测试 + Benchmark

运行方式：
    python test.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

import torch
import ctypes

from common.check import check_correctness
from common.utils import benchmark_func, compute_bandwidth, get_gpu_info
from operators.transpose.pytorch.baseline import transpose_pytorch
from operators.transpose.triton.kernel import transpose_triton


def load_cuda_lib():
    lib_path = os.path.join(os.path.dirname(__file__), "cuda/transpose.so")
    if not os.path.exists(lib_path):
        print(f"[SKIP] CUDA lib not found: {lib_path}  (run: cd cuda && bash build.sh)")
        return None
    lib = ctypes.CDLL(lib_path)
    for fn in ["transpose_cuda_v1", "transpose_cuda_v2"]:
        getattr(lib, fn).argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int
        ]
    return lib


def run_cuda(lib, fn_name, A):
    M, N = A.shape
    B = torch.empty(N, M, dtype=A.dtype, device=A.device)
    getattr(lib, fn_name)(A.data_ptr(), B.data_ptr(), ctypes.c_int(M), ctypes.c_int(N))
    return B


def test_correctness():
    print("=" * 60)
    print("Correctness Test")
    print("=" * 60)
    for M, N in [(512, 512), (1024, 2048), (3000, 7000)]:
        print(f"\n  Shape: ({M}, {N})")
        A = torch.randn(M, N, device="cuda")
        ref = transpose_pytorch(A)

        out = transpose_triton(A)
        check_correctness(out, ref, name=f"Triton ({M}x{N})")

        lib = load_cuda_lib()
        if lib:
            out = run_cuda(lib, "transpose_cuda_v1", A)
            check_correctness(out, ref, name=f"CUDA v1 ({M}x{N})")
            out = run_cuda(lib, "transpose_cuda_v2", A)
            check_correctness(out, ref, name=f"CUDA v2 ({M}x{N})")

        try:
            from operators.transpose.cutlass.wrapper import transpose_cutlass_v1, transpose_cutlass_v2
            check_correctness(transpose_cutlass_v1(A), ref, name=f"CuTe v1 ({M}x{N})")
            check_correctness(transpose_cutlass_v2(A), ref, name=f"CuTe v2 ({M}x{N})")
        except RuntimeError as e:
            print(f"  [SKIP] CuTe: {e}")

        try:
            from operators.transpose.cute.kernel import transpose_cutedsl_v1, transpose_cutedsl_v2
            check_correctness(transpose_cutedsl_v1(A), ref, name=f"CuteDSL v1 ({M}x{N})")
            check_correctness(transpose_cutedsl_v2(A), ref, name=f"CuteDSL v2 ({M}x{N})")
        except ImportError:
            print("  [SKIP] CuteDSL (cutlass Python package not installed)")
    print()


def run_benchmark(M=4096, N=4096):
    print("=" * 60)
    print(f"Benchmark  ({M}x{N} float32)")
    print("=" * 60)
    print(get_gpu_info())
    print()

    A = torch.randn(M, N, device="cuda")
    # 读 M*N 个 float，写 M*N 个 float
    bytes_accessed = M * N * 4 * 2

    res = benchmark_func(transpose_pytorch, A)
    bw = compute_bandwidth(bytes_accessed, res["mean_ms"])
    print(f"PyTorch    : {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s")
    baseline = res["mean_ms"]

    res = benchmark_func(transpose_triton, A)
    bw = compute_bandwidth(bytes_accessed, res["mean_ms"])
    print(f"Triton     : {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s  {baseline/res['mean_ms']:.2f}x")

    lib = load_cuda_lib()
    if lib:
        for v in ["v1", "v2"]:
            res = benchmark_func(run_cuda, lib, f"transpose_cuda_{v}", A)
            bw = compute_bandwidth(bytes_accessed, res["mean_ms"])
            print(f"CUDA {v}    : {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s  {baseline/res['mean_ms']:.2f}x")

    try:
        from operators.transpose.cutlass.wrapper import transpose_cutlass_v1, transpose_cutlass_v2
        for label, fn in [("CuTe v1", transpose_cutlass_v1), ("CuTe v2", transpose_cutlass_v2)]:
            res = benchmark_func(fn, A)
            bw = compute_bandwidth(bytes_accessed, res["mean_ms"])
            print(f"{label:10s}: {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s  {baseline/res['mean_ms']:.2f}x")
    except RuntimeError as e:
        print(f"[SKIP] CuTe: {e}")

    try:
        from operators.transpose.cute.kernel import transpose_cutedsl_v1, transpose_cutedsl_v2
        for label, fn in [("CuteDSL v1", transpose_cutedsl_v1), ("CuteDSL v2", transpose_cutedsl_v2)]:
            res = benchmark_func(fn, A)
            bw = compute_bandwidth(bytes_accessed, res["mean_ms"])
            print(f"{label:10s}: {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s  {baseline/res['mean_ms']:.2f}x")
    except ImportError:
        print("[SKIP] CuteDSL (cutlass Python package not installed)")
    print()


if __name__ == "__main__":
    test_correctness()
    run_benchmark()
