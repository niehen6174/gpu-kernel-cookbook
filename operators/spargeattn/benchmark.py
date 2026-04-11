"""
SpargeAttention comprehensive benchmark.

Measures:
  1. Correctness:  Cosine similarity vs PyTorch SDPA at various topk
  2. Sparsity:     Block sparsity ratio at various topk and sequence lengths
  3. Kernel perf:  SpargeAttn vs SDPA vs FlashAttention (if available)
  4. Scaling:      How speedup changes with sequence length
  5. Topk sweep:   Accuracy vs speed tradeoff at different topk

Usage:
    cd operators
    python spargeattn/benchmark.py
    python spargeattn/benchmark.py --quick           # fast mode
    python spargeattn/benchmark.py --video-gen       # video-gen params
"""

import sys
import os
import time
import argparse
import torch
import torch.nn.functional as F

# ─── Add parent directory to path ─────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


# ─── Configuration ─────────────────────────────────────────────────────────────
DEVICE = "cuda"
WARMUP = 10
REPEAT = 30


def parse_args():
    parser = argparse.ArgumentParser(description="SpargeAttention Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer configs")
    parser.add_argument("--video-gen", action="store_true", help="Use video-gen model parameters")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--repeat", type=int, default=REPEAT)
    return parser.parse_args()


# ─── Helpers ───────────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    """Cosine similarity between two tensors."""
    a_flat, b_flat = a.float().reshape(-1), b.float().reshape(-1)
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def l1_relative(a, b):
    """Relative L1 error."""
    a_flat, b_flat = a.float(), b.float()
    return ((a_flat - b_flat).abs().sum() / b_flat.abs().sum()).item()


def benchmark_fn(fn, warmup=WARMUP, repeat=REPEAT):
    """Benchmark, return median time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    times.sort()
    return times[len(times) // 2]


def compute_flops(B, H, N, D):
    """Standard attention FLOPs: 2*(QK^T) + 2*(PV) = 4*B*H*N*N*D."""
    return 4 * B * H * N * N * D


def print_header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


# ─── Test 1: Correctness across topk values ───────────────────────────────────

def test_correctness(B, H, N, D, topk_values, warmup, repeat):
    from spargeattn import spas_sage2_attn_meansim_topk_cuda

    print_header(f"CORRECTNESS TEST — B={B}, H={H}, N={N}, D={D}")
    print(f"  {'topk':>6s}  {'Sparsity':>10s}  {'Cossim':>10s}  {'L1_rel':>10s}  {'RMSE':>10s}")
    print(f"  {'-'*52}")

    q = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)

    ref = F.scaled_dot_product_attention(q, k, v)

    for topk in topk_values:
        out, sparsity = spas_sage2_attn_meansim_topk_cuda(
            q, k, v, topk=topk, return_sparsity=True
        )
        cs = cosine_sim(out, ref)
        l1 = l1_relative(out, ref)
        rmse = torch.sqrt(torch.mean((out.float() - ref.float()) ** 2)).item()
        print(f"  {topk:6.2f}  {sparsity:10.4f}  {cs:10.6f}  {l1:10.6f}  {rmse:10.6f}")


# ─── Test 2: Sparsity analysis ────────────────────────────────────────────────

def test_sparsity(configs, topk_values, warmup, repeat):
    from spargeattn import spas_sage2_attn_meansim_topk_cuda

    print_header("SPARSITY ANALYSIS")

    header = f"  {'B':>3s} {'H':>3s} {'N':>6s} {'D':>4s}"
    for topk in topk_values:
        header += f"  {'k='+str(topk):>8s}"
    print(header)
    print(f"  {'-'*(20 + 10*len(topk_values))}")

    for B, H, N, D in configs:
        q = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)
        k = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)
        v = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)

        line = f"  {B:3d} {H:3d} {N:6d} {D:4d}"
        for topk in topk_values:
            _, sparsity = spas_sage2_attn_meansim_topk_cuda(
                q, k, v, topk=topk, return_sparsity=True
            )
            line += f"  {sparsity:8.2%}"
        print(line)


# ─── Test 3: Performance benchmark ────────────────────────────────────────────

def test_performance(configs, topk, warmup, repeat):
    from spargeattn import spas_sage2_attn_meansim_topk_cuda

    print_header(f"PERFORMANCE BENCHMARK — topk={topk}")
    print(f"  {'B':>3s} {'H':>3s} {'N':>6s} {'D':>4s}  "
          f"{'SDPA(ms)':>10s}  {'Sparge(ms)':>11s}  {'Speedup':>8s}  "
          f"{'Sparsity':>10s}  {'SDPA TFLOPS':>12s}  {'Sparge TFLOPS':>14s}")
    print(f"  {'-'*95}")

    for B, H, N, D in configs:
        q = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)
        k = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)
        v = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)

        # SDPA
        sdpa_ms = benchmark_fn(
            lambda: F.scaled_dot_product_attention(q, k, v),
            warmup=warmup, repeat=repeat
        )

        # SpargeAttn
        sparge_out = [None]
        sparsity_val = [0.0]
        def run_sparge():
            o, s = spas_sage2_attn_meansim_topk_cuda(
                q, k, v, topk=topk, return_sparsity=True
            )
            sparge_out[0] = o
            sparsity_val[0] = s

        sparge_ms = benchmark_fn(run_sparge, warmup=warmup, repeat=repeat)

        flops = compute_flops(B, H, N, D)
        sdpa_tflops = flops / (sdpa_ms / 1000) / 1e12
        sparge_tflops = flops / (sparge_ms / 1000) / 1e12
        speedup = sdpa_ms / sparge_ms

        print(f"  {B:3d} {H:3d} {N:6d} {D:4d}  "
              f"{sdpa_ms:10.3f}  {sparge_ms:11.3f}  {speedup:7.2f}x  "
              f"{sparsity_val[0]:10.2%}  {sdpa_tflops:12.2f}  {sparge_tflops:14.2f}")


# ─── Test 4: Topk sweep — accuracy vs speed tradeoff ──────────────────────────

def test_topk_sweep(B, H, N, D, topk_values, warmup, repeat):
    from spargeattn import spas_sage2_attn_meansim_topk_cuda

    print_header(f"TOPK SWEEP — B={B}, H={H}, N={N}, D={D}")
    print(f"  {'topk':>6s}  {'Sparsity':>10s}  {'Cossim':>10s}  "
          f"{'Time(ms)':>10s}  {'Speedup':>8s}  {'TFLOPS':>8s}")
    print(f"  {'-'*62}")

    q = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)

    ref = F.scaled_dot_product_attention(q, k, v)

    # SDPA baseline
    sdpa_ms = benchmark_fn(
        lambda: F.scaled_dot_product_attention(q, k, v),
        warmup=warmup, repeat=repeat
    )
    flops = compute_flops(B, H, N, D)

    for topk in topk_values:
        out, sparsity = spas_sage2_attn_meansim_topk_cuda(
            q, k, v, topk=topk, return_sparsity=True
        )
        cs = cosine_sim(out, ref)

        ms = benchmark_fn(
            lambda tk=topk: spas_sage2_attn_meansim_topk_cuda(q, k, v, topk=tk),
            warmup=warmup, repeat=repeat
        )
        speedup = sdpa_ms / ms
        tflops = flops / (ms / 1000) / 1e12

        print(f"  {topk:6.2f}  {sparsity:10.2%}  {cs:10.6f}  "
              f"{ms:10.3f}  {speedup:7.2f}x  {tflops:8.2f}")

    print(f"  {'SDPA':>6s}  {'0.00%':>10s}  {'1.000000':>10s}  "
          f"{sdpa_ms:10.3f}  {'1.00x':>8s}  {flops/(sdpa_ms/1000)/1e12:8.2f}")


# ─── Test 5: Causal attention benchmark ───────────────────────────────────────

def test_causal(configs, topk, warmup, repeat):
    from spargeattn import spas_sage2_attn_meansim_topk_cuda

    print_header(f"CAUSAL ATTENTION — topk={topk}")
    print(f"  {'B':>3s} {'H':>3s} {'N':>6s} {'D':>4s}  "
          f"{'SDPA(ms)':>10s}  {'Sparge(ms)':>11s}  {'Speedup':>8s}  "
          f"{'Cossim':>10s}  {'Sparsity':>10s}")
    print(f"  {'-'*78}")

    for B, H, N, D in configs:
        q = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)
        k = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)
        v = torch.randn(B, H, N, D, device=DEVICE, dtype=torch.float16)

        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        sdpa_ms = benchmark_fn(
            lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
            warmup=warmup, repeat=repeat
        )

        out, sparsity = spas_sage2_attn_meansim_topk_cuda(
            q, k, v, topk=topk, is_causal=True, return_sparsity=True
        )
        cs = cosine_sim(out, ref)

        sparge_ms = benchmark_fn(
            lambda: spas_sage2_attn_meansim_topk_cuda(q, k, v, topk=topk, is_causal=True),
            warmup=warmup, repeat=repeat
        )
        speedup = sdpa_ms / sparge_ms

        print(f"  {B:3d} {H:3d} {N:6d} {D:4d}  "
              f"{sdpa_ms:10.3f}  {sparge_ms:11.3f}  {speedup:7.2f}x  "
              f"{cs:10.6f}  {sparsity:10.2%}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 80)
    print("  SpargeAttention Comprehensive Benchmark")
    print("=" * 80)

    # Device info
    dev = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(f"  GPU:     {dev.name}")
    print(f"  Arch:    sm{major}{minor}")
    print(f"  CUDA:    {torch.version.cuda}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Mode:    {'quick' if args.quick else 'video-gen' if args.video_gen else 'standard'}")

    warmup = args.warmup
    repeat = args.repeat

    if args.video_gen:
        # Video generation model parameters (e.g., CogVideoX / Wan)
        configs = [
            (1, 30, 1024, 128),
            (1, 30, 4096, 128),
            (1, 30, 8192, 128),
            (1, 30, 16384, 128),
            (1, 30, 24576, 128),
        ]
        topk_values = [1.0, 0.75, 0.5, 0.3]
        sweep_cfg = (1, 30, 8192, 128)
    elif args.quick:
        configs = [
            (2, 8, 1024, 128),
            (2, 8, 4096, 128),
            (2, 8, 8192, 128),
        ]
        topk_values = [1.0, 0.75, 0.5]
        sweep_cfg = (2, 8, 4096, 128)
    else:
        # Standard benchmark configs
        configs = [
            (2, 8, 256, 128),
            (2, 8, 512, 128),
            (2, 8, 1024, 128),
            (2, 8, 2048, 128),
            (2, 8, 4096, 128),
            (2, 8, 8192, 128),
            (2, 8, 16384, 128),
            # Different head dims
            (2, 16, 4096, 64),
            (2, 16, 4096, 128),
            # LLM-like configs
            (1, 32, 4096, 128),
            (1, 32, 8192, 128),
        ]
        topk_values = [1.0, 0.9, 0.75, 0.5, 0.3, 0.2]
        sweep_cfg = (2, 8, 8192, 128)

    # ── Run benchmarks ──

    # 1. Correctness
    test_correctness(2, 8, 1024, 128, topk_values, warmup, repeat)
    test_correctness(2, 8, 1024, 64, topk_values, warmup, repeat)

    # 2. Sparsity analysis
    test_sparsity(configs, [0.75, 0.5, 0.3], warmup, repeat)

    # 3. Performance benchmark
    test_performance(configs, topk=0.5, warmup=warmup, repeat=repeat)

    # 4. Topk sweep
    B, H, N, D = sweep_cfg
    test_topk_sweep(B, H, N, D, topk_values, warmup, repeat)

    # 5. Causal attention
    causal_configs = [(B, H, N, D) for B, H, N, D in configs if N >= 512 and N <= 8192]
    if causal_configs:
        test_causal(causal_configs, topk=0.5, warmup=warmup, repeat=repeat)

    print(f"\n{'='*80}")
    print("  BENCHMARK COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
