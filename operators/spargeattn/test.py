"""
SpargeAttention test and benchmark script.

Tests correctness against PyTorch SDPA and benchmarks performance.
"""

import torch
import torch.nn.functional as F
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def precision_metric(quant_o, fa2_o, verbose=True, round_num=4):
    """Compute precision metrics between quantized output and reference."""
    if quant_o.shape[-2] > 200000:
        quant_o, fa2_o = quant_o.cpu(), fa2_o.cpu()
    x, xx = quant_o.float(), fa2_o.float()
    sim = F.cosine_similarity(x.reshape(1, -1), xx.reshape(1, -1)).item()
    l1 = ((x - xx).abs().sum() / xx.abs().sum()).item()
    rmse = torch.sqrt(torch.mean((x - xx) ** 2)).item()
    sim = round(sim, round_num)
    l1 = round(l1, round_num)
    rmse = round(rmse, round_num)
    if verbose:
        print(f'  Cossim: {sim:.6f}, L1: {l1:.6f}, RMSE:{rmse:.6f}')
    return {"Cossim": sim, "L1": l1, "RMSE": rmse}


def test_correctness(B=2, H=8, N=1024, D=128, topk_values=[1.0, 0.75, 0.5]):
    """Test SpargeAttention correctness against PyTorch SDPA."""
    from spargeattn import spas_sage2_attn_meansim_topk_cuda

    print(f"\n{'='*60}")
    print(f"Correctness Test: B={B}, H={H}, N={N}, D={D}")
    print(f"{'='*60}")

    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    # Reference: PyTorch SDPA
    ref_out = F.scaled_dot_product_attention(q, k, v)

    for topk in topk_values:
        print(f"\n  topk={topk}:")
        out, sparsity = spas_sage2_attn_meansim_topk_cuda(
            q, k, v, topk=topk, is_causal=False, return_sparsity=True
        )
        print(f"  Sparsity: {sparsity:.4f}")
        precision_metric(out, ref_out)


def test_causal(B=2, H=8, N=1024, D=128, topk=0.5):
    """Test causal attention mode."""
    from spargeattn import spas_sage2_attn_meansim_topk_cuda

    print(f"\n{'='*60}")
    print(f"Causal Test: B={B}, H={H}, N={N}, D={D}, topk={topk}")
    print(f"{'='*60}")

    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    ref_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    out, sparsity = spas_sage2_attn_meansim_topk_cuda(
        q, k, v, topk=topk, is_causal=True, return_sparsity=True
    )
    print(f"  Sparsity: {sparsity:.4f}")
    precision_metric(out, ref_out)


def test_block_sparse_custom_mask(B=2, H=8, N=1024, D=128):
    """Test block-sparse attention with custom mask."""
    from spargeattn import block_sparse_sage2_attn_cuda

    print(f"\n{'='*60}")
    print(f"Custom Mask Test: B={B}, H={H}, N={N}, D={D}")
    print(f"{'='*60}")

    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    # Detect arch for block sizes
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm{major}{minor}"
    if arch == "sm90":
        BLKQ, BLKK = 64, 128
    else:
        BLKQ, BLKK = 128, 64

    num_q_blocks = (N + BLKQ - 1) // BLKQ
    num_k_blocks = (N + BLKK - 1) // BLKK

    # All-ones mask = dense attention
    mask_id = torch.ones(B, H, num_q_blocks, num_k_blocks, dtype=torch.bool, device='cuda')

    ref_out = F.scaled_dot_product_attention(q, k, v)

    out = block_sparse_sage2_attn_cuda(q, k, v, mask_id=mask_id)
    print("  Dense mask (all blocks selected):")
    precision_metric(out, ref_out)


def benchmark(B=2, H=8, N=4096, D=128, topk=0.5, warmup=10, repeat=50):
    """Benchmark SpargeAttention vs PyTorch SDPA."""
    from spargeattn import spas_sage2_attn_meansim_topk_cuda

    print(f"\n{'='*60}")
    print(f"Benchmark: B={B}, H={H}, N={N}, D={D}, topk={topk}")
    print(f"{'='*60}")

    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    # Warmup
    for _ in range(warmup):
        _ = F.scaled_dot_product_attention(q, k, v)
        _ = spas_sage2_attn_meansim_topk_cuda(q, k, v, topk=topk)
    torch.cuda.synchronize()

    # Benchmark SDPA
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        _ = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    sdpa_time = (time.perf_counter() - start) / repeat * 1000
    print(f"  SDPA:        {sdpa_time:.3f} ms")

    # Benchmark SpargeAttn
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        _ = spas_sage2_attn_meansim_topk_cuda(q, k, v, topk=topk)
    torch.cuda.synchronize()
    sparge_time = (time.perf_counter() - start) / repeat * 1000
    print(f"  SpargeAttn:  {sparge_time:.3f} ms")
    print(f"  Speedup:     {sdpa_time/sparge_time:.2f}x")


if __name__ == "__main__":
    print("SpargeAttention Test Suite")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    major, minor = torch.cuda.get_device_capability(0)
    print(f"Compute capability: sm{major}{minor}")

    # Correctness tests
    test_correctness(B=2, H=8, N=1024, D=128)
    test_correctness(B=2, H=8, N=1024, D=64)

    # Causal test
    test_causal(B=2, H=8, N=1024, D=128)

    # Custom mask test
    test_block_sparse_custom_mask(B=2, H=8, N=1024, D=128)

    # Benchmark
    benchmark(B=2, H=8, N=4096, D=128, topk=0.5)
    benchmark(B=2, H=16, N=8192, D=128, topk=0.5)

    print("\nAll tests complete!")
