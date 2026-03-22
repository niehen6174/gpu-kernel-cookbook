"""
RMSNorm 正确性测试 + Benchmark

运行方式：
    python -m operators.rms_norm.test
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

import torch
import ctypes

from common.check import check_correctness
from common.utils import benchmark_func, compute_bandwidth, get_gpu_info
from operators.rms_norm.pytorch.baseline import rms_norm_pytorch, fused_add_rms_norm_pytorch
from operators.rms_norm.triton.kernel import rms_norm_triton, fused_add_rms_norm_triton


# -------------------------------------------------------------------------
# CUDA .so loader
# -------------------------------------------------------------------------
def load_cuda_lib():
    lib_path = os.path.join(os.path.dirname(__file__), "cuda/rms_norm.so")
    if not os.path.exists(lib_path):
        print(f"[SKIP] CUDA lib not found: {lib_path}  "
              "(run: CUDA_ARCH=sm_90 bash operators/rms_norm/cuda/build.sh)")
        return None
    lib = ctypes.CDLL(lib_path)
    _rn_args = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_float]
    _fused_args = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_int, ctypes.c_int, ctypes.c_float]
    for fn in ["rms_norm_cuda_v1", "rms_norm_cuda_v2", "rms_norm_cuda_v3"]:
        getattr(lib, fn).argtypes = _rn_args
        getattr(lib, fn).restype = None
    for fn in ["fused_add_rms_norm_cuda", "fused_add_rms_norm_cuda_v3"]:
        getattr(lib, fn).argtypes = _fused_args
        getattr(lib, fn).restype = None
    return lib


def run_cuda_rms_norm(lib, fn, x, weight, eps=1e-6):
    orig = x.shape
    N = x.shape[-1]
    x2 = x.reshape(-1, N).contiguous()
    B = x2.shape[0]
    y = torch.empty_like(x2)
    getattr(lib, fn)(
        x2.data_ptr(), weight.data_ptr(), y.data_ptr(),
        ctypes.c_int(B), ctypes.c_int(N), ctypes.c_float(eps)
    )
    return y.reshape(orig)


def run_cuda_fused(lib, fn, x, residual, weight, eps=1e-6):
    orig = x.shape
    N = x.shape[-1]
    x2 = x.reshape(-1, N).contiguous()
    r2 = residual.reshape(-1, N).clone()
    B = x2.shape[0]
    y = torch.empty_like(x2)
    getattr(lib, fn)(
        x2.data_ptr(), r2.data_ptr(), weight.data_ptr(), y.data_ptr(),
        ctypes.c_int(B), ctypes.c_int(N), ctypes.c_float(eps)
    )
    return y.reshape(orig), r2.reshape(orig)


# -------------------------------------------------------------------------
# Correctness
# -------------------------------------------------------------------------
def test_correctness():
    print("=" * 60)
    print("RMSNorm Correctness Test")
    print("=" * 60)

    for B, N in [(1024, 512), (2048, 768), (4096, 4096)]:
        print(f"\n  Shape: ({B}, {N})")
        x = torch.randn(B, N, device="cuda")
        w = torch.randn(N, device="cuda")
        eps = 1e-6

        ref = rms_norm_pytorch(x, w, eps)

        out = rms_norm_triton(x, w, eps)
        check_correctness(out, ref, name=f"Triton v1 ({B}x{N})", atol=1e-5)

        lib = load_cuda_lib()
        if lib:
            for v in ["rms_norm_cuda_v1", "rms_norm_cuda_v2", "rms_norm_cuda_v3"]:
                out = run_cuda_rms_norm(lib, v, x, w, eps)
                check_correctness(out, ref, name=f"CUDA {v.split('_')[-1]} ({B}x{N})", atol=1e-5)

        try:
            from operators.rms_norm.cutlass.wrapper import rms_norm_cutlass_v1, rms_norm_cutlass_v2, rms_norm_cutlass_v3
            check_correctness(rms_norm_cutlass_v1(x, w, eps), ref, name=f"CuTe v1 ({B}x{N})", atol=1e-5)
            check_correctness(rms_norm_cutlass_v2(x, w, eps), ref, name=f"CuTe v2 ({B}x{N})", atol=1e-5)
            check_correctness(rms_norm_cutlass_v3(x, w, eps), ref, name=f"CuTe v3 ({B}x{N})", atol=1e-5)
        except RuntimeError as e:
            print(f"  [SKIP] CuTe: {e}")

    print()
    print("  Fused Add + RMSNorm")
    print()

    for B, N in [(1024, 512), (2048, 768), (4096, 4096)]:
        print(f"\n  Fused Shape: ({B}, {N})")
        x = torch.randn(B, N, device="cuda")
        residual = torch.randn(B, N, device="cuda")
        w = torch.randn(N, device="cuda")
        eps = 1e-6

        ref_out, ref_res = fused_add_rms_norm_pytorch(x, residual, w, eps)

        t_out, t_res = fused_add_rms_norm_triton(x, residual, w, eps)
        check_correctness(t_out, ref_out, name=f"Triton fused output ({B}x{N})", atol=1e-5)
        check_correctness(t_res, ref_res, name=f"Triton fused residual ({B}x{N})", atol=1e-5)

        lib = load_cuda_lib()
        if lib:
            for fn in ["fused_add_rms_norm_cuda", "fused_add_rms_norm_cuda_v3"]:
                c_out, c_res = run_cuda_fused(lib, fn, x, residual, w, eps)
                lbl = fn.replace("fused_add_rms_norm_", "CUDA fused ")
                check_correctness(c_out, ref_out, name=f"{lbl} output ({B}x{N})", atol=1e-5)
                check_correctness(c_res, ref_res, name=f"{lbl} residual ({B}x{N})", atol=1e-5)

        try:
            from operators.rms_norm.cutlass.wrapper import fused_add_rms_norm_cutlass_v1, fused_add_rms_norm_cutlass_v3
            for fn, lbl in [(fused_add_rms_norm_cutlass_v1, "CuTe fused v1"),
                            (fused_add_rms_norm_cutlass_v3, "CuTe fused v3")]:
                c_out, c_res = fn(x, residual.clone(), w, eps)
                check_correctness(c_out, ref_out, name=f"{lbl} output ({B}x{N})", atol=1e-5)
                check_correctness(c_res, ref_res, name=f"{lbl} residual ({B}x{N})", atol=1e-5)
        except RuntimeError as e:
            print(f"  [SKIP] CuTe fused: {e}")
    print()


# -------------------------------------------------------------------------
# Benchmark
# -------------------------------------------------------------------------
def run_benchmark(B=4096, N=4096):
    print("=" * 60)
    print(f"RMSNorm Benchmark  ({B}x{N} float32)")
    print("=" * 60)
    print(get_gpu_info())
    print()

    x = torch.randn(B, N, device="cuda")
    w = torch.randn(N, device="cuda")
    eps = 1e-6
    # 2 reads (x, w) + 1 write (y) → 3 * B * N * 4 bytes
    bytes_rn = B * N * 4 * 3

    res = benchmark_func(rms_norm_pytorch, x, w, eps)
    bw = compute_bandwidth(bytes_rn, res["mean_ms"])
    print(f"PyTorch    : {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s")
    baseline = res["mean_ms"]

    res = benchmark_func(rms_norm_triton, x, w, eps)
    bw = compute_bandwidth(bytes_rn, res["mean_ms"])
    print(f"Triton     : {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s  {baseline/res['mean_ms']:.2f}x")

    lib = load_cuda_lib()
    if lib:
        for v in ["rms_norm_cuda_v1", "rms_norm_cuda_v2", "rms_norm_cuda_v3"]:
            res = benchmark_func(run_cuda_rms_norm, lib, v, x, w, eps)
            bw = compute_bandwidth(bytes_rn, res["mean_ms"])
            print(f"CUDA {v.split('_')[-1]:4s}  : {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s  {baseline/res['mean_ms']:.2f}x")

    try:
        from operators.rms_norm.cutlass.wrapper import rms_norm_cutlass_v1, rms_norm_cutlass_v2, rms_norm_cutlass_v3
        for label, fn in [("CuTe v1", rms_norm_cutlass_v1), ("CuTe v2", rms_norm_cutlass_v2), ("CuTe v3", rms_norm_cutlass_v3)]:
            res = benchmark_func(fn, x, w, eps)
            bw = compute_bandwidth(bytes_rn, res["mean_ms"])
            print(f"{label:10s}: {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s  {baseline/res['mean_ms']:.2f}x")
    except RuntimeError as e:
        print(f"[SKIP] CuTe: {e}")
    print()

    # ---- fused benchmark ----
    print("=" * 60)
    print(f"Fused Add+RMSNorm Benchmark  ({B}x{N} float32)")
    print("=" * 60)
    # x(1R) + residual(1R+1W) + w(1R) + y(1W) = 5 passes
    bytes_fused = B * N * 4 * 5

    r = torch.randn(B, N, device="cuda")

    def pt_fused(x_, r_, w_):
        return fused_add_rms_norm_pytorch(x_, r_.clone(), w_, eps)

    res = benchmark_func(pt_fused, x, r, w)
    bw = compute_bandwidth(bytes_fused, res["mean_ms"])
    print(f"PyTorch (add+RN) : {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s")
    fused_baseline = res["mean_ms"]

    if lib:
        for fn in ["fused_add_rms_norm_cuda", "fused_add_rms_norm_cuda_v3"]:
            lbl = fn.replace("fused_add_rms_norm_", "CUDA fused ")
            res = benchmark_func(run_cuda_fused, lib, fn, x, r.clone(), w, eps)
            bw = compute_bandwidth(bytes_fused, res["mean_ms"])
            print(f"{lbl:16s} : {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s  {fused_baseline/res['mean_ms']:.2f}x")

    try:
        from operators.rms_norm.cutlass.wrapper import fused_add_rms_norm_cutlass_v1, fused_add_rms_norm_cutlass_v3
        for fn, lbl in [(fused_add_rms_norm_cutlass_v1, "CuTe fused v1"),
                        (fused_add_rms_norm_cutlass_v3, "CuTe fused v3")]:
            res = benchmark_func(fn, x, r.clone(), w, eps)
            bw = compute_bandwidth(bytes_fused, res["mean_ms"])
            print(f"{lbl:16s} : {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s  {fused_baseline/res['mean_ms']:.2f}x")
    except RuntimeError as e:
        print(f"[SKIP] CuTe fused: {e}")
    print()


if __name__ == "__main__":
    test_correctness()
    run_benchmark()
