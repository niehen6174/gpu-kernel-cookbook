"""
Benchmark SageAttention SM90 tile-skip on real DiT attention inputs.

Loads dumped Q, K, V tensors from a video generation model and runs
the locally-built kernel with various skip_threshold values.

Reports:
  - Per-block kernel latency at each threshold
  - Speedup vs baseline (threshold=0)
  - Output difference (max / mean) vs baseline
  - Attention sparsity analysis (how "skippable" the attention is)

Usage:
    cd operators/sageattention/cuda
    python test_real_data.py [--data-dir /path/to/run000] [--blocks 0,1,2]
"""

import sys
import os
import argparse
import time
import glob
import torch
import torch.nn.functional as F

# ─── Defaults ─────────────────────────────────────────────────────────────────
DATA_DIR = "/home/leo/data/dit_attn_inputs/run000"
DEVICE = "cuda"
CFG = {"warmup": 5, "repeat": 20}
THRESHOLDS = [0.0, 5.0, 8.0, 10.0, 14.0, 16.0, 20.0]


def load_local_module():
    cuda_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, cuda_dir)
    import _qattn_sm90
    return _qattn_sm90


def load_official_funcs():
    from sageattention.quant import per_warp_int8, per_channel_fp8
    return {"per_warp_int8": per_warp_int8, "per_channel_fp8": per_channel_fp8}


def discover_blocks(data_dir):
    """Return sorted list of (block_id, q_path, k_path, v_path)."""
    q_files = sorted(glob.glob(os.path.join(data_dir, "step00_block*.q.pt")))
    blocks = []
    for qf in q_files:
        base = qf.replace(".q.pt", "")
        bid = int(base.split("block")[-1])
        kf = base + ".k.pt"
        vf = base + ".v.pt"
        if os.path.exists(kf) and os.path.exists(vf):
            blocks.append((bid, qf, kf, vf))
    return blocks


def load_block(q_path, k_path, v_path, device=DEVICE):
    """Load Q, K, V tensors and move to device.

    Input layout: [B, N, H, D] (NHD) → convert to [B, H, N, D] (HND) for kernel.
    """
    q = torch.load(q_path, map_location="cpu", weights_only=True)
    k = torch.load(k_path, map_location="cpu", weights_only=True)
    v = torch.load(v_path, map_location="cpu", weights_only=True)

    # Detect layout: if dim1 >> dim2, it's [B, N, H, D]; otherwise [B, H, N, D]
    if q.dim() == 4 and q.size(1) > q.size(2):
        # [B, N, H, D] → [B, H, N, D]
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

    q = q.to(device)
    k = k.to(device)
    v = v.to(device)
    return q, k, v


def quantize_for_kernel(q, k, v, official_funcs):
    """Quantize QKV using official sageattention functions (HND layout)."""
    seq_dim = 2  # HND: [B, H, N, D]
    sm_scale = q.size(-1) ** -0.5

    # K smoothing
    km = k.mean(dim=seq_dim, keepdim=True)

    # Quantize QK (per_warp)
    q_int8, q_scale, k_int8, k_scale = official_funcs["per_warp_int8"](
        q, k, km, tensor_layout="HND", BLKQ=64, WARPQ=16, BLKK=128
    )
    qk_quant_gran = 2  # per_warp

    # Pad V to multiple of 128
    kv_len = k.size(seq_dim)
    v_pad_len = 128 - (kv_len % 128) if kv_len % 128 != 0 else 0
    v_padded = v
    if v_pad_len > 0:
        v_padded = torch.cat([
            v,
            torch.zeros(v.size(0), v.size(1), v_pad_len, v.size(3),
                         dtype=v.dtype, device=v.device)
        ], dim=2)

    # Quantize V (per_channel fp8)
    v_fp8, v_scale, _ = official_funcs["per_channel_fp8"](
        v_padded, tensor_layout="HND", smooth_v=False
    )

    return q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale, qk_quant_gran, sm_scale


def benchmark_fn(fn, warmup=None, repeat=None):
    if warmup is None:
        warmup = CFG["warmup"]
    if repeat is None:
        repeat = CFG["repeat"]
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


def analyze_attention_sparsity(q, k, sm_scale, block_id):
    """Analyze how sparse the attention pattern is for a single head.

    Computes full attention logits for head 0 and reports statistics
    on how many KV tiles could be skipped at various thresholds.

    Only analyzes a single head to avoid OOM on large sequences.
    """
    B, H, N, D = q.shape
    CTA_K = 128
    num_kv_tiles = (N + CTA_K - 1) // CTA_K

    # Use head 0 only, batch 0
    q_h = q[0, 0].float()  # [N, D]
    k_h = k[0, 0].float()  # [N, D]

    # Don't compute full N×N attention for huge N — sample query tiles
    CTA_Q = 64
    num_q_tiles = (N + CTA_Q - 1) // CTA_Q
    # sample at most 16 query tiles to keep memory manageable
    max_q_tiles = min(num_q_tiles, 16)
    sample_indices = torch.linspace(0, num_q_tiles - 1, max_q_tiles).long()

    skip_counts = {t: 0 for t in THRESHOLDS if t > 0}
    total_pairs = 0

    for qi_idx in sample_indices:
        qi = qi_idx.item()
        q_start = qi * CTA_Q
        q_end = min(q_start + CTA_Q, N)
        q_tile = q_h[q_start:q_end]  # [CTA_Q, D]

        # Compute logits for all KV tiles
        # logits: [CTA_Q, N]
        logits = (q_tile @ k_h.T) * sm_scale  # [CTA_Q, N]

        # Running max simulation: sweep KV tiles left-to-right
        running_max = torch.full((q_tile.size(0),), -1e9, device=q.device)
        for ki in range(num_kv_tiles):
            k_start = ki * CTA_K
            k_end = min(k_start + CTA_K, N)
            tile_logits = logits[:, k_start:k_end]  # [CTA_Q, tile_len]
            tile_max = tile_logits.max(dim=-1).values  # [CTA_Q]

            # Check skip condition for each threshold
            for t in skip_counts:
                # A tile is "skippable" if ALL query rows agree
                can_skip = (tile_max < running_max - t).all().item()
                if can_skip:
                    skip_counts[t] += 1

            # Update running max
            running_max = torch.max(running_max, tile_max)
            total_pairs += 1

    print(f"\n  [Sparsity] Block {block_id:02d}, head=0, N={N}, "
          f"sampled {max_q_tiles}/{num_q_tiles} Q-tiles × {num_kv_tiles} K-tiles = {total_pairs} tile-pairs")
    print(f"  {'Threshold':>10s}  {'Skip Rate':>10s}  {'Skipped':>8s} / {'Total':>6s}")
    for t in sorted(skip_counts.keys()):
        rate = skip_counts[t] / total_pairs * 100 if total_pairs > 0 else 0
        print(f"  {t:10.1f}  {rate:9.1f}%  {skip_counts[t]:8d} / {total_pairs:6d}")


def run_benchmark_on_block(local_module, q_int8, k_int8, v_fp8, q_scale, k_scale,
                            v_scale, output_shape, dtype, qk_quant_gran, sm_scale,
                            block_id):
    """Benchmark one block at all thresholds, report latency + accuracy."""
    B, H, N, D = output_shape
    flops = 2 * 2 * B * H * N * N * D

    results = {}
    baseline_out = None

    for threshold in THRESHOLDS:
        o = torch.empty(output_shape, dtype=dtype, device=DEVICE)

        def run_fn(t=threshold, out=o):
            local_module.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
                q_int8, k_int8, v_fp8, out, q_scale, k_scale, v_scale,
                1, 0, qk_quant_gran, sm_scale, 0, t
            )

        ms = benchmark_fn(run_fn)
        tflops = flops / (ms / 1000) / 1e12

        # Capture output for accuracy comparison
        run_fn()
        torch.cuda.synchronize()

        if threshold == 0.0:
            baseline_out = o.clone()
            max_diff = 0.0
            mean_diff = 0.0
        else:
            diff = (o - baseline_out).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

        results[threshold] = {
            "ms": ms,
            "tflops": tflops,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark tile-skip on real DiT data")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Path to run directory")
    parser.add_argument("--blocks", default=None, help="Comma-separated block IDs (default: all)")
    parser.add_argument("--skip-sparsity", action="store_true", help="Skip sparsity analysis")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    CFG["warmup"] = args.warmup
    CFG["repeat"] = args.repeat

    print("=" * 90)
    print("SageAttention SM90 — Real DiT Data Tile-Skip Benchmark")
    print("=" * 90)

    # Device info
    dev = torch.cuda.get_device_properties(0)
    print(f"GPU: {dev.name}, CUDA: {torch.version.cuda}")

    # Load modules
    local_module = load_local_module()
    official_funcs = load_official_funcs()
    print("[OK] Modules loaded")

    # Discover blocks
    all_blocks = discover_blocks(args.data_dir)
    print(f"[OK] Found {len(all_blocks)} blocks in {args.data_dir}")

    if args.blocks is not None:
        selected = set(int(x) for x in args.blocks.split(","))
        all_blocks = [(bid, q, k, v) for (bid, q, k, v) in all_blocks if bid in selected]
        print(f"     Selected blocks: {[b[0] for b in all_blocks]}")

    if not all_blocks:
        print("ERROR: No blocks found.")
        return

    # Peek at first block to show data shape
    bid0, qp0, kp0, vp0 = all_blocks[0]
    q_peek = torch.load(qp0, map_location="cpu", weights_only=True)
    print(f"     Tensor shape (on disk): {q_peek.shape}, dtype={q_peek.dtype}")
    del q_peek

    # ─── Per-block results ────────────────────────────────────────────────────
    all_results = {}
    thresholds_header = THRESHOLDS

    print("\n" + "=" * 90)
    print("PER-BLOCK BENCHMARK")
    print("=" * 90)

    for bid, q_path, k_path, v_path in all_blocks:
        print(f"\n{'─' * 90}")
        print(f"Block {bid:02d}")
        print(f"{'─' * 90}")

        # Load
        q, k, v = load_block(q_path, k_path, v_path)
        B, H, N, D = q.shape
        print(f"  Shape: B={B}, H={H}, N={N}, D={D}")

        # Sparsity analysis
        if not args.skip_sparsity:
            analyze_attention_sparsity(q, k, q.size(-1) ** -0.5, bid)

        # Quantize
        t0 = time.perf_counter()
        q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale, qk_gran, sm_scale = \
            quantize_for_kernel(q, k, v, official_funcs)
        quant_ms = (time.perf_counter() - t0) * 1000
        print(f"  Quantization: {quant_ms:.1f} ms")

        # Free original tensors to save GPU memory
        del q, k, v
        torch.cuda.empty_cache()

        # Benchmark
        results = run_benchmark_on_block(
            local_module, q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale,
            (B, H, N, D), torch.bfloat16, qk_gran, sm_scale, bid
        )
        all_results[bid] = results

        # Print table
        print(f"\n  {'Threshold':>10s}  {'Time(ms)':>10s}  {'TFLOPS':>8s}  {'Speedup':>8s}  "
              f"{'MaxDiff':>10s}  {'MeanDiff':>10s}")
        print(f"  {'─' * 70}")
        base_ms = results[0.0]["ms"]
        for t in THRESHOLDS:
            r = results[t]
            speedup = base_ms / r["ms"]
            print(f"  {t:10.1f}  {r['ms']:10.3f}  {r['tflops']:8.2f}  {speedup:7.3f}x  "
                  f"{r['max_diff']:10.6f}  {r['mean_diff']:10.6f}")

        # Free quantized tensors
        del q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale
        torch.cuda.empty_cache()

    # ─── Summary ──────────────────────────────────────────────────────────────
    if len(all_results) > 1:
        print("\n" + "=" * 90)
        print("SUMMARY (averaged across all blocks)")
        print("=" * 90)

        print(f"\n  {'Threshold':>10s}  {'Avg Time(ms)':>13s}  {'Avg TFLOPS':>11s}  "
              f"{'Avg Speedup':>12s}  {'Max MaxDiff':>12s}")
        print(f"  {'─' * 72}")

        for t in THRESHOLDS:
            times = [all_results[bid][t]["ms"] for bid in all_results]
            tflops_list = [all_results[bid][t]["tflops"] for bid in all_results]
            speedups = [all_results[bid][0.0]["ms"] / all_results[bid][t]["ms"] for bid in all_results]
            max_diffs = [all_results[bid][t]["max_diff"] for bid in all_results]

            avg_ms = sum(times) / len(times)
            avg_tflops = sum(tflops_list) / len(tflops_list)
            avg_speedup = sum(speedups) / len(speedups)
            worst_diff = max(max_diffs)

            print(f"  {t:10.1f}  {avg_ms:13.3f}  {avg_tflops:11.2f}  "
                  f"{avg_speedup:11.3f}x  {worst_diff:12.6f}")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()
