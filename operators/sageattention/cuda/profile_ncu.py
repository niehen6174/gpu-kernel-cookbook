"""
Profiling script for NCU (Nsight Compute).
Runs the SM90 kernel once with a representative workload for profiling.

Usage:
    # Full profile:
    ncu --set full -o sage_profile python profile_ncu.py

    # Quick key metrics:
    ncu --kernel-name qk_int8_sv_f8_attn_kernel --launch-skip 3 --launch-count 1 \
        --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_active,\
smsp__warps_active.avg.per_cycle_active,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__inst_executed.avg.per_cycle_elapsed \
        python profile_ncu.py

    # Just run (no ncu) to verify it works:
    python profile_ncu.py
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _qattn_sm90

from sageattention.quant import per_warp_int8, per_channel_fp8

# ─── Config (use smaller N for ncu, larger N for standalone) ───
BATCH = 2
NUM_HEADS = 40
HEAD_DIM = 128
SEQ_LEN = int(os.environ.get("PROFILE_SEQ_LEN", "4096"))
DTYPE = torch.bfloat16
DEVICE = "cuda"


def main():
    print(f"Profiling config: B={BATCH}, H={NUM_HEADS}, D={HEAD_DIM}, N={SEQ_LEN}, dtype={DTYPE}")

    q = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    k = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    v = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    sm_scale = HEAD_DIM ** -0.5

    # Quantize
    km = k.mean(dim=2, keepdim=True)
    q_int8, q_scale, k_int8, k_scale = per_warp_int8(
        q, k, km, tensor_layout="HND", BLKQ=64, WARPQ=16, BLKK=128
    )

    kv_len = k.size(2)
    v_pad_len = 128 - (kv_len % 128) if kv_len % 128 != 0 else 0
    if v_pad_len > 0:
        v = torch.cat([v, torch.zeros(v.size(0), v.size(1), v_pad_len, v.size(3),
                                      dtype=v.dtype, device=v.device)], dim=2)
    v_fp8, v_scale, _ = per_channel_fp8(v, tensor_layout="HND", smooth_v=False)

    o = torch.empty(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=DTYPE, device=DEVICE)

    # Warmup (ncu will skip these with --launch-skip)
    for _ in range(3):
        _qattn_sm90.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
            q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale,
            1, 0, 2, sm_scale, 0
        )
    torch.cuda.synchronize()

    # Profiling run (ncu captures this one with --launch-count 1)
    print("Running kernel for profiling...")
    _qattn_sm90.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
        q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale,
        1, 0, 2, sm_scale, 0
    )
    torch.cuda.synchronize()
    print("Done.")


if __name__ == "__main__":
    main()
