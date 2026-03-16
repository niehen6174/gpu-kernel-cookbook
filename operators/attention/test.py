"""
Attention 正确性测试 + Benchmark

运行方式：
    python test.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

import torch

from common.check import check_correctness
from common.utils import benchmark_func, compute_tflops, get_gpu_info
from operators.attention.pytorch.baseline import attention_pytorch, attention_pytorch_sdpa
from operators.attention.triton.kernel import flash_attention_triton


def test_correctness():
    print("=" * 60)
    print("Correctness Test")
    print("=" * 60)
    for B, H, N, D in [(2, 4, 128, 64), (1, 8, 512, 64), (2, 4, 256, 64)]:
        print(f"\n  Shape: B={B} H={H} N={N} D={D}")
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)

        ref = attention_pytorch(Q, K, V)

        # Triton Flash Attention
        out = flash_attention_triton(Q, K, V)
        check_correctness(out, ref, name=f"Triton Flash Attn ({N}x{D})", atol=5e-3, rtol=1e-3)

        # PyTorch SDPA (如果支持)
        try:
            out_sdpa = attention_pytorch_sdpa(Q, K, V)
            check_correctness(out_sdpa, ref, name=f"PyTorch SDPA ({N}x{D})", atol=1e-3, rtol=1e-3)
        except Exception as e:
            print(f"  [SKIP] PyTorch SDPA: {e}")

        # CuTe Flash Attention
        try:
            from operators.attention.cutlass.wrapper import flash_attention_cutlass_v1, flash_attention_cutlass_v2
            check_correctness(flash_attention_cutlass_v1(Q, K, V), ref, name=f"CuTe v1 ({N}x{D})", atol=5e-3)
            check_correctness(flash_attention_cutlass_v2(Q, K, V), ref, name=f"CuTe v2 ({N}x{D})", atol=5e-3)
        except RuntimeError as e:
            print(f"  [SKIP] CuTe: {e}")

    # Causal attention test
    print("\n  Causal attention test:")
    Q = torch.randn(2, 4, 256, 64, device="cuda")
    K = torch.randn(2, 4, 256, 64, device="cuda")
    V = torch.randn(2, 4, 256, 64, device="cuda")
    ref_causal = attention_pytorch(Q, K, V, causal=True)
    out_causal = flash_attention_triton(Q, K, V, causal=True)
    check_correctness(out_causal, ref_causal, name="Triton Flash Attn Causal", atol=5e-3)

    try:
        from operators.attention.cutlass.wrapper import flash_attention_cutlass_v2
        check_correctness(flash_attention_cutlass_v2(Q, K, V, causal=True), ref_causal,
                          name="CuTe v2 Causal", atol=5e-3)
    except RuntimeError as e:
        print(f"  [SKIP] CuTe: {e}")
    print()


def run_benchmark(B=4, H=8, N=1024, D=64):
    print("=" * 60)
    print(f"Benchmark  B={B} H={H} N={N} D={D} float32")
    print("=" * 60)
    print(get_gpu_info())
    print()

    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)

    # FLOPs: 2 * B * H * N * N * D (QK^T) + 2 * B * H * N * N * D (PV)
    flops = 4 * B * H * N * N * D

    res = benchmark_func(attention_pytorch, Q, K, V)
    tflops = compute_tflops(flops, res["mean_ms"])
    print(f"PyTorch (naive) : {res['mean_ms']:.4f} ms  {tflops:.2f} TFLOPS")
    baseline = res["mean_ms"]

    try:
        res = benchmark_func(attention_pytorch_sdpa, Q, K, V)
        tflops = compute_tflops(flops, res["mean_ms"])
        print(f"PyTorch SDPA    : {res['mean_ms']:.4f} ms  {tflops:.2f} TFLOPS  {baseline/res['mean_ms']:.2f}x")
    except Exception as e:
        print(f"PyTorch SDPA    : [SKIP] {e}")

    res = benchmark_func(flash_attention_triton, Q, K, V)
    tflops = compute_tflops(flops, res["mean_ms"])
    print(f"Triton Flash    : {res['mean_ms']:.4f} ms  {tflops:.2f} TFLOPS  {baseline/res['mean_ms']:.2f}x")

    try:
        from operators.attention.cutlass.wrapper import flash_attention_cutlass_v1, flash_attention_cutlass_v2
        for label, fn in [("CuTe v1", flash_attention_cutlass_v1),
                          ("CuTe v2", flash_attention_cutlass_v2)]:
            res = benchmark_func(fn, Q, K, V)
            tflops = compute_tflops(flops, res["mean_ms"])
            print(f"{label:<16}: {res['mean_ms']:.4f} ms  {tflops:.2f} TFLOPS  {baseline/res['mean_ms']:.2f}x")
    except RuntimeError as e:
        print(f"[SKIP] CuTe: {e}")

    print()

    # 测试不同序列长度
    print("Scaling with N:")
    for n in [256, 512, 1024, 2048, 4096]:
        Q2 = torch.randn(1, 8, n, D, device="cuda")
        K2 = torch.randn(1, 8, n, D, device="cuda")
        V2 = torch.randn(1, 8, n, D, device="cuda")
        flops2 = 4 * 1 * 8 * n * n * D

        r1 = benchmark_func(attention_pytorch, Q2, K2, V2)
        r2 = benchmark_func(flash_attention_triton, Q2, K2, V2)
        tf1 = compute_tflops(flops2, r1["mean_ms"])
        tf2 = compute_tflops(flops2, r2["mean_ms"])
        print(f"  N={n:4d}: PyTorch={r1['mean_ms']:.3f}ms ({tf1:.2f}T)  "
              f"Flash={r2['mean_ms']:.3f}ms ({tf2:.2f}T)  "
              f"speedup={r1['mean_ms']/r2['mean_ms']:.2f}x")
    print()


if __name__ == "__main__":
    test_correctness()
    run_benchmark()
