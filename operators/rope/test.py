"""
RoPE 正确性测试 + Benchmark

运行方式：
    python -m operators.rope.test
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

import torch
import ctypes

from common.check import check_correctness
from common.utils import benchmark_func, compute_bandwidth, get_gpu_info
from operators.rope.pytorch.baseline import build_cos_sin_cache, apply_rope_pytorch
from operators.rope.triton.kernel import rope_triton, rope_interleaved_triton


# -------------------------------------------------------------------------
# CUDA .so loader
# -------------------------------------------------------------------------
def load_cuda_lib():
    lib_path = os.path.join(os.path.dirname(__file__), "cuda/rope.so")
    if not os.path.exists(lib_path):
        print(f"[SKIP] CUDA lib not found: {lib_path}  "
              "(run: CUDA_ARCH=sm_90 bash operators/rope/cuda/build.sh)")
        return None
    lib = ctypes.CDLL(lib_path)
    _args = [ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_void_p,
             ctypes.c_int, ctypes.c_int, ctypes.c_int]
    for fn in ["rope_cuda_v1", "rope_cuda_v2", "rope_cuda_v3"]:
        getattr(lib, fn).argtypes = _args
        getattr(lib, fn).restype = None
    return lib


def run_cuda_rope(lib, fn, q, k, cos_cache, sin_cache, positions):
    q_out = q.clone()
    k_out = k.clone()
    seq_len, num_heads, head_dim = q.shape
    pos32 = positions.to(torch.int32).contiguous()
    getattr(lib, fn)(
        q_out.data_ptr(), k_out.data_ptr(),
        cos_cache.data_ptr(), sin_cache.data_ptr(),
        pos32.data_ptr(),
        ctypes.c_int(seq_len), ctypes.c_int(num_heads), ctypes.c_int(head_dim)
    )
    return q_out, k_out


# -------------------------------------------------------------------------
# Correctness
# -------------------------------------------------------------------------
def test_correctness():
    print("=" * 60)
    print("RoPE Correctness Test")
    print("=" * 60)

    for seq_len, num_heads, head_dim in [(128, 8, 64), (512, 16, 64), (1024, 32, 64)]:
        print(f"\n  Shape: seq={seq_len}, heads={num_heads}, dim={head_dim}")
        max_seq = seq_len + 64
        cos_cache, sin_cache = build_cos_sin_cache(max_seq, head_dim, device="cuda")

        q = torch.randn(seq_len, num_heads, head_dim, device="cuda")
        k = torch.randn(seq_len, num_heads, head_dim, device="cuda")
        positions = torch.arange(seq_len, device="cuda")

        ref_q, ref_k = apply_rope_pytorch(q, k, cos_cache, sin_cache, positions)

        # Triton V1
        tq, tk = rope_triton(q, k, cos_cache, sin_cache, positions)
        check_correctness(tq, ref_q, name=f"Triton v1 Q ({seq_len},{num_heads},{head_dim})", atol=1e-5)
        check_correctness(tk, ref_k, name=f"Triton v1 K ({seq_len},{num_heads},{head_dim})", atol=1e-5)

        # CUDA
        lib = load_cuda_lib()
        if lib:
            for v in ["rope_cuda_v1", "rope_cuda_v2", "rope_cuda_v3"]:
                cq, ck = run_cuda_rope(lib, v, q, k, cos_cache, sin_cache, positions)
                check_correctness(cq, ref_q, name=f"CUDA {v.split('_')[-1]} Q ({seq_len})", atol=1e-5)
                check_correctness(ck, ref_k, name=f"CUDA {v.split('_')[-1]} K ({seq_len})", atol=1e-5)

        # CuTe
        try:
            from operators.rope.cutlass.wrapper import rope_cutlass_v1, rope_cutlass_v2, rope_cutlass_v3
            cq, ck = rope_cutlass_v1(q, k, cos_cache, sin_cache, positions)
            check_correctness(cq, ref_q, name=f"CuTe v1 Q ({seq_len})", atol=1e-5)
            check_correctness(ck, ref_k, name=f"CuTe v1 K ({seq_len})", atol=1e-5)
            cq, ck = rope_cutlass_v2(q, k, cos_cache, sin_cache, positions)
            check_correctness(cq, ref_q, name=f"CuTe v2 Q ({seq_len})", atol=1e-5)
            check_correctness(ck, ref_k, name=f"CuTe v2 K ({seq_len})", atol=1e-5)
            cq, ck = rope_cutlass_v3(q, k, cos_cache, sin_cache, positions)
            check_correctness(cq, ref_q, name=f"CuTe v3 Q ({seq_len})", atol=1e-5)
            check_correctness(ck, ref_k, name=f"CuTe v3 K ({seq_len})", atol=1e-5)
        except RuntimeError as e:
            print(f"  [SKIP] CuTe: {e}")

    print()


# -------------------------------------------------------------------------
# Benchmark
# -------------------------------------------------------------------------
def run_benchmark(seq_len=4096, num_heads=32, head_dim=64):
    print("=" * 60)
    print(f"RoPE Benchmark  (seq={seq_len}, heads={num_heads}, dim={head_dim})")
    print("=" * 60)
    print(get_gpu_info())
    print()

    cos_cache, sin_cache = build_cos_sin_cache(seq_len + 64, head_dim, device="cuda")
    q = torch.randn(seq_len, num_heads, head_dim, device="cuda")
    k = torch.randn(seq_len, num_heads, head_dim, device="cuda")
    positions = torch.arange(seq_len, device="cuda")

    # Q + K: 2 reads + 2 writes = 4 * seq_len * num_heads * head_dim * 4 bytes
    # cos/sin: 2 reads = 2 * seq_len * half_dim * 4 bytes (negligible)
    bytes_accessed = 4 * seq_len * num_heads * head_dim * 4

    res = benchmark_func(apply_rope_pytorch, q, k, cos_cache, sin_cache, positions)
    bw = compute_bandwidth(bytes_accessed, res["mean_ms"])
    print(f"PyTorch    : {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s")
    baseline = res["mean_ms"]

    res = benchmark_func(rope_triton, q, k, cos_cache, sin_cache, positions)
    bw = compute_bandwidth(bytes_accessed, res["mean_ms"])
    print(f"Triton v1  : {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s  {baseline/res['mean_ms']:.2f}x")

    lib = load_cuda_lib()
    if lib:
        for v in ["rope_cuda_v1", "rope_cuda_v2", "rope_cuda_v3"]:
            res = benchmark_func(run_cuda_rope, lib, v, q, k, cos_cache, sin_cache, positions)
            bw = compute_bandwidth(bytes_accessed, res["mean_ms"])
            print(f"CUDA {v.split('_')[-1]:4s}  : {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s  {baseline/res['mean_ms']:.2f}x")

    try:
        from operators.rope.cutlass.wrapper import rope_cutlass_v1, rope_cutlass_v2, rope_cutlass_v3
        for label, fn in [("CuTe v1", rope_cutlass_v1), ("CuTe v2", rope_cutlass_v2), ("CuTe v3", rope_cutlass_v3)]:
            res = benchmark_func(fn, q, k, cos_cache, sin_cache, positions)
            bw = compute_bandwidth(bytes_accessed, res["mean_ms"])
            print(f"{label:10s}: {res['mean_ms']:.4f} ms  BW={bw:.1f} GB/s  {baseline/res['mean_ms']:.2f}x")
    except RuntimeError as e:
        print(f"[SKIP] CuTe: {e}")
    print()


if __name__ == "__main__":
    test_correctness()
    run_benchmark()
