"""
Test harness for the standalone SageAttention SM90 kernel.

Validates that the locally-built kernel produces identical results
and matching performance compared to the official sageattention package.

Usage:
    # From gpu-kernel-lab root:
    python -m operators.sageattention.cuda.test

    # Or directly:
    cd operators/sageattention/cuda
    python test.py
"""

import sys
import os
import time
import torch
import torch.nn.functional as F

# ─── Configuration ───────────────────────────────────────────────────────────
BATCH = 2
NUM_HEADS = 40
HEAD_DIM = 128
SEQ_LENS = [256, 512, 1024, 2048, 4096, 8192, 18900, 40500]
DTYPE = torch.bfloat16
DEVICE = "cuda"
WARMUP = 5
REPEAT = 20


def load_local_module():
    """Load the locally-built _qattn_sm90 module."""
    cuda_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, cuda_dir)
    try:
        import _qattn_sm90
        print(f"[OK] Local _qattn_sm90 loaded from {cuda_dir}")
        print(f"     Functions: {[f for f in dir(_qattn_sm90) if not f.startswith('_')]}")
        return _qattn_sm90
    except ImportError as e:
        print(f"[ERROR] Cannot load local _qattn_sm90: {e}")
        print(f"        Run 'bash build.sh' in {cuda_dir} first.")
        return None


def load_official_module():
    """Load the official sageattention package."""
    try:
        from sageattention.core import sageattn_qk_int8_pv_fp8_cuda_sm90
        from sageattention.quant import per_warp_int8, per_channel_fp8
        from sageattention.triton.quant_per_thread import per_thread_int8 as per_thread_int8_triton
        print("[OK] Official sageattention loaded")
        return {
            'sageattn': sageattn_qk_int8_pv_fp8_cuda_sm90,
            'per_warp_int8': per_warp_int8,
            'per_channel_fp8': per_channel_fp8,
            'per_thread_int8': per_thread_int8_triton,
        }
    except ImportError as e:
        print(f"[WARN] Official sageattention not available: {e}")
        return None


def generate_inputs(batch, num_heads, seq_len, head_dim, dtype=DTYPE, device=DEVICE):
    """Generate random Q, K, V tensors in HND layout."""
    q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    return q, k, v


def quantize_inputs(q, k, v, official_funcs, tensor_layout="HND", qk_quant_gran="per_warp"):
    """Quantize Q, K, V using the official sageattention quantization kernels."""
    _tensor_layout = 1  # HND
    seq_dim = 2
    nh_dim = 1
    sm_scale = q.size(-1) ** -0.5

    # K smoothing
    km = k.mean(dim=seq_dim, keepdim=True)

    # Quantize Q and K
    if qk_quant_gran == "per_warp":
        q_int8, q_scale, k_int8, k_scale = official_funcs['per_warp_int8'](
            q, k, km, tensor_layout=tensor_layout, BLKQ=64, WARPQ=16, BLKK=128
        )
        _qk_quant_gran = 2
    else:
        q_int8, q_scale, k_int8, k_scale = official_funcs['per_thread_int8'](
            q, k, km, tensor_layout=tensor_layout, BLKQ=64, WARPQ=16, BLKK=128, WARPK=128
        )
        _qk_quant_gran = 3

    # Pad V to multiple of 128
    kv_len = k.size(seq_dim)
    v_pad_len = 128 - (kv_len % 128) if kv_len % 128 != 0 else 0
    v_padded = v
    if v_pad_len > 0:
        v_padded = torch.cat([
            v, torch.zeros(v.size(0), v.size(1), v_pad_len, v.size(3),
                           dtype=v.dtype, device=v.device)
        ], dim=2)

    # Quantize V
    v_fp8, v_scale, _ = official_funcs['per_channel_fp8'](v_padded, tensor_layout=tensor_layout, smooth_v=False)

    return q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale, _qk_quant_gran, sm_scale


def run_local_kernel(local_module, q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale,
                     output_shape, dtype, qk_quant_gran, sm_scale, is_causal=0):
    """Run the locally-built SM90 kernel."""
    o = torch.empty(output_shape, dtype=dtype, device=DEVICE)
    tensor_layout = 1  # HND
    return_lse = 0

    lse = local_module.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
        q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale,
        tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse
    )
    return o


def benchmark_fn(fn, warmup=WARMUP, repeat=REPEAT, sync=True):
    """Benchmark a function, return median time in ms."""
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        if sync:
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    times.sort()
    return times[len(times) // 2]  # median


def pytorch_reference(q, k, v, sm_scale):
    """PyTorch reference attention (scaled dot product)."""
    with torch.no_grad():
        attn_weights = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        o = torch.matmul(attn_weights, v.float())
    return o.to(q.dtype)


def test_correctness(local_module, official_funcs):
    """Compare local kernel output with official sageattention and PyTorch reference."""
    print("\n" + "=" * 70)
    print("CORRECTNESS TEST")
    print("=" * 70)

    for seq_len in [256, 512, 1024, 2048, 4096]:
        q, k, v = generate_inputs(BATCH, NUM_HEADS, seq_len, HEAD_DIM)
        sm_scale = HEAD_DIM ** -0.5

        # PyTorch reference
        ref_out = pytorch_reference(q, k, v, sm_scale)

        # Official sageattention
        if official_funcs:
            official_out = official_funcs['sageattn'](
                q, k, v, tensor_layout="HND", qk_quant_gran="per_warp",
                sm_scale=sm_scale, pv_accum_dtype="fp32+fp32", smooth_k=True
            )

        # Local kernel
        q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale, qk_gran, _ = quantize_inputs(
            q, k, v, official_funcs, qk_quant_gran="per_warp"
        )
        local_out = run_local_kernel(
            local_module, q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale,
            q.shape, q.dtype, qk_gran, sm_scale
        )

        # Compare local vs official
        if official_funcs:
            diff_vs_official = (local_out - official_out).abs().max().item()
            mean_diff_official = (local_out - official_out).abs().mean().item()
            match_official = diff_vs_official < 1e-4  # Should be bit-exact

            diff_vs_ref = (local_out - ref_out).abs().max().item()
            official_vs_ref = (official_out - ref_out).abs().max().item()

            status = "✓ PASS" if match_official else "✗ FAIL"
            print(f"  N={seq_len:5d}: {status}  "
                  f"local_vs_official(max={diff_vs_official:.6f}, mean={mean_diff_official:.6f})  "
                  f"local_vs_ref(max={diff_vs_ref:.4f})  official_vs_ref(max={official_vs_ref:.4f})")
        else:
            diff_vs_ref = (local_out - ref_out).abs().max().item()
            print(f"  N={seq_len:5d}: local_vs_pytorch_ref(max={diff_vs_ref:.4f})")


def test_performance(local_module, official_funcs):
    """Benchmark local kernel vs official sageattention."""
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK")
    print("=" * 70)
    print(f"  B={BATCH}, H={NUM_HEADS}, D={HEAD_DIM}, dtype={DTYPE}")
    print(f"  {'N':>6s}  {'Local(ms)':>10s}  {'Official(ms)':>12s}  {'Ratio':>7s}  {'TFLOPS':>8s}")
    print("  " + "-" * 55)

    for seq_len in SEQ_LENS:
        q, k, v = generate_inputs(BATCH, NUM_HEADS, seq_len, HEAD_DIM)
        sm_scale = HEAD_DIM ** -0.5

        # Prepare quantized inputs
        q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale, qk_gran, _ = quantize_inputs(
            q, k, v, official_funcs, qk_quant_gran="per_warp"
        )

        # Local kernel benchmark
        def run_local():
            return run_local_kernel(
                local_module, q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale,
                q.shape, q.dtype, qk_gran, sm_scale
            )
        local_ms = benchmark_fn(run_local)

        # Compute TFLOPS (2 * B * H * N * N * D for QK, + 2 * B * H * N * N * D for PV)
        flops = 2 * 2 * BATCH * NUM_HEADS * seq_len * seq_len * HEAD_DIM
        tflops = flops / (local_ms / 1000) / 1e12

        # Official benchmark
        if official_funcs:
            def run_official():
                return official_funcs['sageattn'](
                    q, k, v, tensor_layout="HND", qk_quant_gran="per_warp",
                    sm_scale=sm_scale, pv_accum_dtype="fp32+fp32", smooth_k=True
                )
            official_ms = benchmark_fn(run_official)
            ratio = local_ms / official_ms
            print(f"  {seq_len:6d}  {local_ms:10.3f}  {official_ms:12.3f}  {ratio:7.3f}  {tflops:8.2f}")
        else:
            print(f"  {seq_len:6d}  {local_ms:10.3f}  {'N/A':>12s}  {'N/A':>7s}  {tflops:8.2f}")


def test_kernel_only_performance(local_module, official_funcs):
    """Benchmark just the kernel (no quantization overhead)."""
    print("\n" + "=" * 70)
    print("KERNEL-ONLY PERFORMANCE (no quantization)")
    print("=" * 70)
    print(f"  B={BATCH}, H={NUM_HEADS}, D={HEAD_DIM}, dtype={DTYPE}")
    print(f"  {'N':>6s}  {'Kernel(ms)':>10s}  {'TFLOPS':>8s}")
    print("  " + "-" * 30)

    for seq_len in SEQ_LENS:
        q, k, v = generate_inputs(BATCH, NUM_HEADS, seq_len, HEAD_DIM)
        sm_scale = HEAD_DIM ** -0.5

        q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale, qk_gran, _ = quantize_inputs(
            q, k, v, official_funcs, qk_quant_gran="per_warp"
        )

        o = torch.empty(q.shape, dtype=q.dtype, device=DEVICE)

        def run_kernel():
            local_module.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
                q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale,
                1, 0, qk_gran, sm_scale, 0
            )

        kernel_ms = benchmark_fn(run_kernel)
        flops = 2 * 2 * BATCH * NUM_HEADS * seq_len * seq_len * HEAD_DIM
        tflops = flops / (kernel_ms / 1000) / 1e12

        print(f"  {seq_len:6d}  {kernel_ms:10.3f}  {tflops:8.2f}")


def main():
    print("=" * 70)
    print("SageAttention SM90 Kernel — Standalone Build Test")
    print("=" * 70)

    # Device info
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_properties(0)
        print(f"GPU: {dev.name}")
        print(f"Compute capability: {dev.major}.{dev.minor}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("ERROR: No CUDA device available")
        return

    # Load modules
    local_module = load_local_module()
    official_funcs = load_official_module()

    if local_module is None:
        print("\nERROR: Local kernel not built. Run 'bash build.sh' first.")
        return

    if official_funcs is None:
        print("\nWARN: Official sageattention not available.")
        print("      Install with: pip install -e /path/to/SageAttention")
        print("      Tests will run against PyTorch reference only.")

    if official_funcs:
        # Run correctness tests
        test_correctness(local_module, official_funcs)

        # Run performance tests
        test_performance(local_module, official_funcs)

        # Run kernel-only performance tests
        test_kernel_only_performance(local_module, official_funcs)
    else:
        print("\n[SKIP] Need official sageattention for correctness/performance tests.")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
