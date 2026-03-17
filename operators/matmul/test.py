"""
Matmul 正确性测试 + Benchmark

运行方式：
    python test.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

import torch
import ctypes

from common.check import check_correctness
from common.utils import benchmark_func, compute_tflops, get_gpu_info
from operators.matmul.pytorch.baseline import matmul_pytorch
from operators.matmul.triton.kernel import matmul_triton


def load_cuda_lib():
    lib_path = os.path.join(os.path.dirname(__file__), "cuda/matmul.so")
    if not os.path.exists(lib_path):
        print(f"[SKIP] CUDA lib not found: {lib_path}  (run: cd cuda && bash build.sh)")
        return None
    lib = ctypes.CDLL(lib_path)
    for fn in ["matmul_cuda_v1", "matmul_cuda_v2", "matmul_cuda_v3"]:
        getattr(lib, fn).argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
    return lib


def run_cuda(lib, fn, A, B):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, dtype=A.dtype, device=A.device)
    getattr(lib, fn)(
        A.data_ptr(), B.data_ptr(), C.data_ptr(),
        ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(N)
    )
    return C


def test_correctness():
    print("=" * 60)
    print("Correctness Test")
    print("=" * 60)
    for M, K, N in [(128, 256, 128), (512, 512, 512), (1024, 1024, 1024)]:
        print(f"\n  Shape: A({M},{K}) x B({K},{N})")
        A = torch.randn(M, K, device="cuda")
        B = torch.randn(K, N, device="cuda")
        ref = matmul_pytorch(A, B)

        out = matmul_triton(A, B)
        check_correctness(out, ref, name=f"Triton ({M}x{K}x{N})", atol=1.0, rtol=1e-3)

        lib = load_cuda_lib()
        if lib:
            for v in ["matmul_cuda_v1", "matmul_cuda_v2", "matmul_cuda_v3"]:
                out = run_cuda(lib, v, A, B)
                check_correctness(out, ref, name=f"CUDA {v.split('_')[-1]}", atol=1e-3, rtol=1e-3)

        try:
            from operators.matmul.cutlass.wrapper import matmul_cutlass_v1, matmul_cutlass_v2
            check_correctness(matmul_cutlass_v1(A, B), ref, name=f"CuTe v1 ({M}x{K}x{N})", atol=1e-3)
            check_correctness(matmul_cutlass_v2(A, B), ref, name=f"CuTe v2 ({M}x{K}x{N})", atol=1e-3)
        except RuntimeError as e:
            print(f"  [SKIP] CuTe: {e}")

        try:
            from operators.matmul.cutlass.wrapper_highlevel import matmul_cutlass_hl_v1, matmul_cutlass_hl_v2
            # CUTLASS 高层 API 使用 TF32 Tensor Core，精度约 1e-2（与 cuBLAS 同级别）
            check_correctness(matmul_cutlass_hl_v1(A, B), ref, name=f"CUTLASS HL v1 ({M}x{K}x{N})", atol=1.0, rtol=1e-2)
            check_correctness(matmul_cutlass_hl_v2(A, B), ref, name=f"CUTLASS HL v2 ({M}x{K}x{N})", atol=1.0, rtol=1e-2)
        except RuntimeError as e:
            print(f"  [SKIP] CUTLASS HL: {e}")
    print()


def run_benchmark(M=4096, K=4096, N=4096):
    print("=" * 60)
    print(f"Benchmark  ({M}x{K}) x ({K}x{N}) float32")
    print("=" * 60)
    print(get_gpu_info())
    print()

    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    flops = 2 * M * N * K

    res = benchmark_func(matmul_pytorch, A, B)
    tflops = compute_tflops(flops, res["mean_ms"])
    print(f"PyTorch (cuBLAS): {res['mean_ms']:.4f} ms  {tflops:.2f} TFLOPS")
    baseline = res["mean_ms"]

    res = benchmark_func(matmul_triton, A, B)
    tflops = compute_tflops(flops, res["mean_ms"])
    print(f"Triton           : {res['mean_ms']:.4f} ms  {tflops:.2f} TFLOPS  {baseline/res['mean_ms']:.2f}x")

    lib = load_cuda_lib()
    if lib:
        for v in ["matmul_cuda_v1", "matmul_cuda_v2", "matmul_cuda_v3"]:
            res = benchmark_func(run_cuda, lib, v, A, B)
            tflops = compute_tflops(flops, res["mean_ms"])
            print(f"CUDA {v.split('_')[-1]:4s}        : {res['mean_ms']:.4f} ms  {tflops:.2f} TFLOPS  {baseline/res['mean_ms']:.2f}x")

    try:
        from operators.matmul.cutlass.wrapper import matmul_cutlass_v1, matmul_cutlass_v2
        for label, fn in [("CuTe v1", matmul_cutlass_v1), ("CuTe v2", matmul_cutlass_v2)]:
            res = benchmark_func(fn, A, B)
            tflops = compute_tflops(flops, res["mean_ms"])
            print(f"{label:10s}      : {res['mean_ms']:.4f} ms  {tflops:.2f} TFLOPS  {baseline/res['mean_ms']:.2f}x")
    except RuntimeError as e:
        print(f"[SKIP] CuTe: {e}")

    try:
        from operators.matmul.cutlass.wrapper_highlevel import matmul_cutlass_hl_v1, matmul_cutlass_hl_v2
        for label, fn in [("CUTLASS HL v1", matmul_cutlass_hl_v1), ("CUTLASS HL v2", matmul_cutlass_hl_v2)]:
            res = benchmark_func(fn, A, B)
            tflops = compute_tflops(flops, res["mean_ms"])
            print(f"{label:14s}  : {res['mean_ms']:.4f} ms  {tflops:.2f} TFLOPS  {baseline/res['mean_ms']:.2f}x")
    except RuntimeError as e:
        print(f"[SKIP] CUTLASS HL: {e}")
    print()


if __name__ == "__main__":
    test_correctness()
    run_benchmark()
