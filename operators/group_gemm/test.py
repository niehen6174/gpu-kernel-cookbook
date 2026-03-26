"""
Group GEMM 正确性测试 + Benchmark

运行方式：
    python test.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

import torch
import ctypes

from common.check import check_correctness
from common.utils import benchmark_func, compute_tflops, get_gpu_info
from operators.group_gemm.pytorch.baseline import group_gemm_pytorch_fixed
from operators.group_gemm.triton.kernel import group_gemm_triton_fixed


def load_cuda_lib():
    lib_path = os.path.join(os.path.dirname(__file__), "cuda/group_gemm.so")
    if not os.path.exists(lib_path):
        print(f"[SKIP] CUDA lib not found: {lib_path}  (run: cd cuda && bash build.sh)")
        return None
    lib = ctypes.CDLL(lib_path)
    for fn in ["group_gemm_cuda_v1", "group_gemm_cuda_v3"]:
        getattr(lib, fn).argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
    return lib


def run_cuda_fixed(lib, fn, A, B):
    """A: (G, M, K), B: (G, K, N) → C: (G, M, N)"""
    G, M, K = A.shape
    _, K2, N = B.shape
    C = torch.empty(G, M, N, dtype=A.dtype, device=A.device)
    getattr(lib, fn)(
        A.data_ptr(), B.data_ptr(), C.data_ptr(),
        ctypes.c_int(G), ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(N),
    )
    return C


def test_correctness():
    print("=" * 60)
    print("Correctness Test")
    print("=" * 60)

    for G, M, K, N in [(4, 128, 256, 128), (8, 256, 256, 256), (16, 512, 512, 512)]:
        print(f"\n  Shape: G={G}, A({M},{K}) x B({K},{N})")
        A = torch.randn(G, M, K, device="cuda")
        B = torch.randn(G, K, N, device="cuda")
        ref = group_gemm_pytorch_fixed(A, B)

        out = group_gemm_triton_fixed(A, B)
        check_correctness(out, ref, name=f"Triton ({G}x{M}x{K}x{N})", atol=1.0, rtol=1e-3)

        lib = load_cuda_lib()
        if lib:
            for v_fn in ["group_gemm_cuda_v1", "group_gemm_cuda_v3"]:
                out = run_cuda_fixed(lib, v_fn, A, B)
                tag = v_fn.split("_")[-1]
                check_correctness(out, ref, name=f"CUDA {tag} ({G}x{M}x{K}x{N})", atol=1e-3, rtol=1e-3)

        try:
            from operators.group_gemm.cutlass.wrapper import group_gemm_cutlass_v1, group_gemm_cutlass_v2
            check_correctness(group_gemm_cutlass_v1(A, B), ref, name=f"CuTe v1 ({G}x{M}x{K}x{N})", atol=1e-3)
            check_correctness(group_gemm_cutlass_v2(A, B), ref, name=f"CuTe v2 ({G}x{M}x{K}x{N})", atol=1e-3)
        except RuntimeError as e:
            print(f"  [SKIP] CuTe: {e}")

        try:
            from operators.group_gemm.cutlass.wrapper_grouped import group_gemm_cutlass_hl_v1, group_gemm_cutlass_hl_v2
            # CUTLASS 3.x uses TF32 accumulation on Hopper → ~1e-2 vs FP32 ref
            check_correctness(group_gemm_cutlass_hl_v1(A, B), ref, name=f"CUTLASS HL v1 ({G}x{M}x{K}x{N})", atol=0.1)
            check_correctness(group_gemm_cutlass_hl_v2(A, B), ref, name=f"CUTLASS HL v2 ({G}x{M}x{K}x{N})", atol=0.1)
        except RuntimeError as e:
            print(f"  [SKIP] CUTLASS HL: {e}")

        try:
            from operators.group_gemm.cute.kernel import group_gemm_cutedsl_v1, group_gemm_cutedsl_v2
            check_correctness(group_gemm_cutedsl_v1(A, B), ref, name=f"CuteDSL v1 ({G}x{M}x{K}x{N})", atol=1e-3)
            check_correctness(group_gemm_cutedsl_v2(A, B), ref, name=f"CuteDSL v2 ({G}x{M}x{K}x{N})", atol=1e-3)
        except ImportError:
            print("  [SKIP] CuteDSL (cutlass Python package not installed)")
    print()


def run_benchmark(G=16, M=1024, K=1024, N=1024):
    print("=" * 60)
    print(f"Benchmark  G={G}, ({M}x{K}) x ({K}x{N}) float32")
    print("=" * 60)
    print(get_gpu_info())
    print()

    A = torch.randn(G, M, K, device="cuda")
    B = torch.randn(G, K, N, device="cuda")
    # 每个 group 2*M*N*K FLOP，共 G 个 group
    flops = 2 * G * M * N * K

    res = benchmark_func(group_gemm_pytorch_fixed, A, B)
    tflops = compute_tflops(flops, res["mean_ms"])
    print(f"PyTorch (bmm)    : {res['mean_ms']:.4f} ms  {tflops:.2f} TFLOPS")
    baseline = res["mean_ms"]

    res = benchmark_func(group_gemm_triton_fixed, A, B)
    tflops = compute_tflops(flops, res["mean_ms"])
    print(f"Triton           : {res['mean_ms']:.4f} ms  {tflops:.2f} TFLOPS  {baseline/res['mean_ms']:.2f}x")

    lib = load_cuda_lib()
    if lib:
        for v_fn, label in [("group_gemm_cuda_v1", "CUDA v1"), ("group_gemm_cuda_v3", "CUDA v3")]:
            res = benchmark_func(run_cuda_fixed, lib, v_fn, A, B)
            tflops = compute_tflops(flops, res["mean_ms"])
            print(f"{label:16s} : {res['mean_ms']:.4f} ms  {tflops:.2f} TFLOPS  {baseline/res['mean_ms']:.2f}x")

    try:
        from operators.group_gemm.cutlass.wrapper import group_gemm_cutlass_v1, group_gemm_cutlass_v2
        for label, fn in [("CuTe v1", group_gemm_cutlass_v1), ("CuTe v2", group_gemm_cutlass_v2)]:
            res = benchmark_func(fn, A, B)
            tflops = compute_tflops(flops, res["mean_ms"])
            print(f"{label:16s} : {res['mean_ms']:.4f} ms  {tflops:.2f} TFLOPS  {baseline/res['mean_ms']:.2f}x")
    except RuntimeError as e:
        print(f"[SKIP] CuTe: {e}")

    try:
        from operators.group_gemm.cute.kernel import group_gemm_cutedsl_v1, group_gemm_cutedsl_v2
        for label, fn in [("CuteDSL v1", group_gemm_cutedsl_v1), ("CuteDSL v2", group_gemm_cutedsl_v2)]:
            res = benchmark_func(fn, A, B)
            tflops = compute_tflops(flops, res["mean_ms"])
            print(f"{label:16s} : {res['mean_ms']:.4f} ms  {tflops:.2f} TFLOPS  {baseline/res['mean_ms']:.2f}x")
    except ImportError:
        print("[SKIP] CuteDSL (cutlass Python package not installed)")

    try:
        from operators.group_gemm.cutlass.wrapper_grouped import group_gemm_cutlass_hl_v1, group_gemm_cutlass_hl_v2
        for label, fn in [("CUTLASS HL v1", group_gemm_cutlass_hl_v1), ("CUTLASS HL v2", group_gemm_cutlass_hl_v2)]:
            res = benchmark_func(fn, A, B)
            tflops = compute_tflops(flops, res["mean_ms"])
            print(f"{label:16s} : {res['mean_ms']:.4f} ms  {tflops:.2f} TFLOPS  {baseline/res['mean_ms']:.2f}x")
    except RuntimeError as e:
        print(f"[SKIP] CUTLASS HL: {e}")
    print()


if __name__ == "__main__":
    test_correctness()
    run_benchmark()
