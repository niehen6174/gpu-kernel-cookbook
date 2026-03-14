"""
Vector Add 正确性测试 + Benchmark

运行方式：
    python test.py

测试内容：
  1. CUDA kernel v1 (naive)
  2. CUDA kernel v2 (vectorized float4)
  3. Triton kernel
  4. CuTe kernel (需要 cutlass 安装)
  5. PyTorch baseline

每个实现与 PyTorch baseline 做误差比较。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

import torch
import ctypes
import numpy as np

from common.check import check_correctness
from common.utils import benchmark_func, compute_bandwidth, print_benchmark_result, get_gpu_info
from operators.vector_add.pytorch.baseline import vector_add_pytorch
from operators.vector_add.triton.kernel import vector_add_triton


# -------------------------------------------------------------------------
# 加载 CUDA shared library（需要先编译 kernel.cu）
# -------------------------------------------------------------------------
def load_cuda_lib():
    lib_path = os.path.join(os.path.dirname(__file__), "cuda/vector_add.so")
    if not os.path.exists(lib_path):
        print(f"[SKIP] CUDA lib not found: {lib_path}")
        print("       请先运行: cd cuda && bash build.sh")
        return None
    lib = ctypes.CDLL(lib_path)
    lib.vector_add_cuda_v1.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
    ]
    lib.vector_add_cuda_v2.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
    ]
    return lib


def run_cuda_v1(lib, A, B):
    C = torch.empty_like(A)
    lib.vector_add_cuda_v1(
        A.data_ptr(), B.data_ptr(), C.data_ptr(), ctypes.c_int(A.numel())
    )
    return C


def run_cuda_v2(lib, A, B):
    C = torch.empty_like(A)
    lib.vector_add_cuda_v2(
        A.data_ptr(), B.data_ptr(), C.data_ptr(), ctypes.c_int(A.numel())
    )
    return C


# -------------------------------------------------------------------------
# 正确性测试
# -------------------------------------------------------------------------
def test_correctness(N=1024 * 1024):
    print("=" * 60)
    print("Correctness Test  (N = {:,})".format(N))
    print("=" * 60)

    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, device="cuda", dtype=torch.float32)
    ref = vector_add_pytorch(A, B)

    # Triton
    out_triton = vector_add_triton(A, B)
    check_correctness(out_triton, ref, name="Triton")

    # CUDA
    lib = load_cuda_lib()
    if lib is not None:
        out_v1 = run_cuda_v1(lib, A, B)
        check_correctness(out_v1, ref, name="CUDA v1 (naive)")

        out_v2 = run_cuda_v2(lib, A, B)
        check_correctness(out_v2, ref, name="CUDA v2 (float4)")

    # CuTe（可选）
    try:
        from operators.vector_add.cute.kernel import run_vector_add_cute
        out_cute = run_vector_add_cute(A, B)
        check_correctness(out_cute, ref, name="CuTe DSL")
    except ImportError:
        print("[SKIP] CuTe DSL (cutlass not installed)")

    print()


# -------------------------------------------------------------------------
# Benchmark
# -------------------------------------------------------------------------
def run_benchmark(N=1024 * 1024 * 16):
    print("=" * 60)
    print("Benchmark  (N = {:,}, dtype=float32)".format(N))
    print("=" * 60)
    print(get_gpu_info())
    print()

    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, device="cuda", dtype=torch.float32)

    # 每个元素读 2 个 float，写 1 个 float，共 3 * 4 bytes
    bytes_accessed = N * 3 * 4

    results = {}

    # PyTorch baseline
    res = benchmark_func(vector_add_pytorch, A, B)
    results["PyTorch"] = res
    bw = compute_bandwidth(bytes_accessed, res["mean_ms"])
    print(f"PyTorch     : {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s")

    # Triton
    res = benchmark_func(vector_add_triton, A, B)
    results["Triton"] = res
    bw = compute_bandwidth(bytes_accessed, res["mean_ms"])
    speedup = results["PyTorch"]["mean_ms"] / res["mean_ms"]
    print(f"Triton      : {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s  speedup={speedup:.2f}x")

    # CUDA
    lib = load_cuda_lib()
    if lib is not None:
        res = benchmark_func(run_cuda_v1, lib, A, B)
        results["CUDA v1"] = res
        bw = compute_bandwidth(bytes_accessed, res["mean_ms"])
        speedup = results["PyTorch"]["mean_ms"] / res["mean_ms"]
        print(f"CUDA v1     : {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s  speedup={speedup:.2f}x")

        res = benchmark_func(run_cuda_v2, lib, A, B)
        results["CUDA v2"] = res
        bw = compute_bandwidth(bytes_accessed, res["mean_ms"])
        speedup = results["PyTorch"]["mean_ms"] / res["mean_ms"]
        print(f"CUDA v2 f4  : {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s  speedup={speedup:.2f}x")

    print()


if __name__ == "__main__":
    test_correctness()
    run_benchmark()
