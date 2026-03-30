"""
FP8 量化算子 Benchmark

比较各版本实现的性能和精度：
  - FP16 baseline（cuBLAS）
  - PyTorch per-tensor / per-block
  - Triton per-tensor / per-block
  - CuTe per-tensor
  - CUTLASS V1 per-tensor
  - CUTLASS V2 per-block

运行方式：
    python operators/fp8_quant/benchmark.py
    python operators/fp8_quant/benchmark.py --warmup 5 --repeat 100
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

import torch

from common.utils import benchmark_func, compute_tflops, get_gpu_info
from operators.fp8_quant.pytorch.fp8_torch import (
    fp8_per_tensor_quant,
    fp8_per_tensor_dequant,
    fp8_per_tensor_gemm,
    fp8_per_block_act_quant,
    fp8_per_block_weight_quant,
    fp8_per_block_gemm,
    compute_quant_error,
)

FP8_MAX = 448.0


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def make_tensors(M, K, N, dtype=torch.float16, device="cuda", seed=42):
    """生成标准测试张量"""
    torch.manual_seed(seed)
    x = torch.randn(M, K, dtype=dtype, device=device) * 1.0
    w = torch.randn(N, K, dtype=dtype, device=device) * 0.1
    return x, w


def compute_gemm_flops(M, K, N):
    """计算 GEMM FLOPs"""
    return 2 * M * K * N


def print_header():
    header = (
        f"{'Impl':<30} {'Mean(ms)':<10} {'TFLOPS':<8} {'vs_fp16':<10} "
        f"{'SNR(dB)':<10} {'RMSE':<10}"
    )
    print(header)
    print("-" * len(header))


def print_row(impl, mean_ms, tflops, vs_fp16, snr_db=None, rmse=None):
    snr_str = f"{snr_db:.1f}" if snr_db is not None else "N/A"
    rmse_str = f"{rmse:.2e}" if rmse is not None else "N/A"
    vs_str = f"{vs_fp16:.2f}x" if vs_fp16 is not None else "N/A"
    print(
        f"  {impl:<28} {mean_ms:<10.3f} {tflops:<8.1f} {vs_str:<10} "
        f"{snr_str:<10} {rmse_str:<10}"
    )


# ---------------------------------------------------------------------------
# 精度分析（对比 FP32 reference）
# ---------------------------------------------------------------------------

def analyze_quant_accuracy(x, w, M, K, N):
    """
    计算 per-tensor 和 per-block 量化精度。
    """
    ref_out = x.float() @ w.float().T   # FP32 reference GEMM

    # Per-Tensor 量化
    q_a_t, inv_s_a_t = fp8_per_tensor_quant(x)
    x_dq_t = fp8_per_tensor_dequant(q_a_t, inv_s_a_t, out_dtype=torch.float32)
    err_tensor = compute_quant_error(x.float(), x_dq_t)

    # Per-Block 量化
    q_a_b, inv_s_a_b = fp8_per_block_act_quant(x, group_size=128)
    q_f32 = q_a_b.float().reshape(M, K // 128, 128)
    x_dq_b = (q_f32 * inv_s_a_b.unsqueeze(-1)).reshape(M, K).float()
    err_block = compute_quant_error(x.float(), x_dq_b)

    return err_tensor, err_block


# ---------------------------------------------------------------------------
# GEMM accuracy 计算（对比 FP16 GEMM output）
# ---------------------------------------------------------------------------

def gemm_accuracy(out, ref_fp16):
    """对比 GEMM 输出与 FP16 参考的量化误差"""
    err = compute_quant_error(ref_fp16.float(), out.float())
    return err


# ---------------------------------------------------------------------------
# Benchmark 主函数
# ---------------------------------------------------------------------------

def run_benchmark(configs, warmup=5, repeat=50, device="cuda"):
    print(f"\n{'='*80}")
    print(f"FP8 量化算子 Benchmark")
    print(get_gpu_info())
    print(f"warmup={warmup}, repeat={repeat}")
    print(f"{'='*80}\n")

    for M, K, N in configs:
        print(f"\nConfig: M={M}, K={K}, N={N}")
        print(f"{'='*60}")

        x, w = make_tensors(M, K, N)
        flops = compute_gemm_flops(M, K, N)

        # --- 精度分析 ---
        err_tensor, err_block = analyze_quant_accuracy(x, w, M, K, N)
        print(f"\n量化精度分析（激活量化误差，FP32 reference）:")
        print(f"  Per-Tensor: SNR={err_tensor['snr_db']:.1f}dB  "
              f"RMSE={err_tensor['rmse']:.2e}  "
              f"max_err={err_tensor['max_abs_error']:.2e}  "
              f"cos_sim={err_tensor['cosine_sim']:.6f}")
        print(f"  Per-Block:  SNR={err_block['snr_db']:.1f}dB  "
              f"RMSE={err_block['rmse']:.2e}  "
              f"max_err={err_block['max_abs_error']:.2e}  "
              f"cos_sim={err_block['cosine_sim']:.6f}")
        snr_gain = err_block['snr_db'] - err_tensor['snr_db']
        rmse_red = (1 - err_block['rmse'] / err_tensor['rmse']) * 100
        print(f"  → Per-Block 精度优势: SNR +{snr_gain:.1f}dB，RMSE 减少 {rmse_red:.0f}%")

        print(f"\n{'性能 Benchmark':}")
        print_header()

        # --- FP16 baseline ---
        fp16_ref = None
        try:
            def fp16_gemm():
                return x @ w.T

            stats = benchmark_func(fp16_gemm, warmup=warmup, repeat=repeat)
            mean_ms = stats['mean_ms']
            tflops = compute_tflops(flops, mean_ms)
            fp16_ref = mean_ms

            fp16_out = fp16_gemm()
            print_row("fp16_baseline", mean_ms, tflops, 1.0, snr_db=float('inf'), rmse=0.0)
        except Exception as e:
            print(f"  [fp16_baseline FAIL] {e}")

        # --- PyTorch Per-Tensor ---
        try:
            q_a_t, inv_s_a_t = fp8_per_tensor_quant(x)
            q_b_t, inv_s_b_t = fp8_per_tensor_quant(w)

            def pytorch_per_tensor():
                return fp8_per_tensor_gemm(q_a_t, inv_s_a_t, q_b_t, inv_s_b_t,
                                           out_dtype=torch.bfloat16)

            # warmup 一次
            out_pt = pytorch_per_tensor()
            stats = benchmark_func(pytorch_per_tensor, warmup=warmup, repeat=repeat)
            mean_ms = stats['mean_ms']
            tflops = compute_tflops(flops, mean_ms)
            vs = fp16_ref / mean_ms if fp16_ref else None

            err = gemm_accuracy(out_pt, fp16_out.bfloat16() if fp16_out is not None else out_pt)
            print_row("pytorch_per_tensor", mean_ms, tflops, vs,
                      snr_db=err['snr_db'], rmse=err['rmse'])
        except Exception as e:
            print(f"  [pytorch_per_tensor SKIP] {e}")

        # --- PyTorch Per-Block ---
        try:
            # Per-block 需要 N、K 可被 128 整除
            N_pb = (N // 128) * 128
            K_pb = (K // 128) * 128
            if N_pb == 0 or K_pb == 0:
                raise ValueError(f"N={N} or K={K} too small for per-block")

            x_pb, w_pb = make_tensors(M, K_pb, N_pb)
            q_a_pb, inv_s_a_pb = fp8_per_block_act_quant(x_pb, group_size=128)
            q_b_pb, inv_s_b_pb = fp8_per_block_weight_quant(w_pb, block_size=128)

            def pytorch_per_block():
                return fp8_per_block_gemm(q_a_pb, inv_s_a_pb, q_b_pb, inv_s_b_pb,
                                          out_dtype=torch.bfloat16)

            out_ppb = pytorch_per_block()
            stats = benchmark_func(pytorch_per_block, warmup=warmup, repeat=repeat)
            mean_ms = stats['mean_ms']
            flops_pb = compute_gemm_flops(M, K_pb, N_pb)
            tflops = compute_tflops(flops_pb, mean_ms)
            vs = fp16_ref / mean_ms if fp16_ref else None

            ref_pb = x_pb.float() @ w_pb.float().T
            err = gemm_accuracy(out_ppb, ref_pb.to(torch.bfloat16))
            print_row("pytorch_per_block", mean_ms, tflops, vs,
                      snr_db=err['snr_db'], rmse=err['rmse'])
        except Exception as e:
            print(f"  [pytorch_per_block SKIP] {e}")

        # --- Triton Per-Tensor ---
        try:
            from operators.fp8_quant.triton.kernel import triton_fp8_per_tensor_gemm
            q_a_t, inv_s_a_t = fp8_per_tensor_quant(x)
            q_b_t, inv_s_b_t = fp8_per_tensor_quant(w)

            def triton_pt():
                return triton_fp8_per_tensor_gemm(q_a_t, inv_s_a_t, q_b_t, inv_s_b_t,
                                                  out_dtype=torch.bfloat16)

            out_tpt = triton_pt()
            stats = benchmark_func(triton_pt, warmup=warmup, repeat=repeat)
            mean_ms = stats['mean_ms']
            tflops = compute_tflops(flops, mean_ms)
            vs = fp16_ref / mean_ms if fp16_ref else None
            err = gemm_accuracy(out_tpt, fp16_out.bfloat16() if fp16_out is not None else out_tpt)
            print_row("triton_per_tensor", mean_ms, tflops, vs,
                      snr_db=err['snr_db'], rmse=err['rmse'])
        except Exception as e:
            print(f"  [triton_per_tensor SKIP] {e}")

        # --- Triton Per-Block ---
        try:
            from operators.fp8_quant.triton.kernel import triton_fp8_per_block_gemm
            N_pb2 = (N // 128) * 128
            K_pb2 = (K // 128) * 128
            if N_pb2 == 0 or K_pb2 == 0:
                raise ValueError(f"N or K too small for per-block")

            x_pb2, w_pb2 = make_tensors(M, K_pb2, N_pb2)
            q_a_pb2, inv_s_a_pb2 = fp8_per_block_act_quant(x_pb2, group_size=128)
            q_b_pb2, inv_s_b_pb2 = fp8_per_block_weight_quant(w_pb2, block_size=128)

            def triton_pb():
                return triton_fp8_per_block_gemm(q_a_pb2, inv_s_a_pb2,
                                                  q_b_pb2, inv_s_b_pb2,
                                                  out_dtype=torch.bfloat16)

            out_tpb = triton_pb()
            stats = benchmark_func(triton_pb, warmup=warmup, repeat=repeat)
            mean_ms = stats['mean_ms']
            flops_pb2 = compute_gemm_flops(M, K_pb2, N_pb2)
            tflops = compute_tflops(flops_pb2, mean_ms)
            vs = fp16_ref / mean_ms if fp16_ref else None
            ref_pb2 = x_pb2.float() @ w_pb2.float().T
            err = gemm_accuracy(out_tpb, ref_pb2.to(torch.bfloat16))
            print_row("triton_per_block", mean_ms, tflops, vs,
                      snr_db=err['snr_db'], rmse=err['rmse'])
        except Exception as e:
            print(f"  [triton_per_block SKIP] {e}")

        # --- CuTe Per-Tensor ---
        try:
            from operators.fp8_quant.cute.kernel import cute_fp8_per_tensor_gemm
            q_a_cute, inv_s_a_cute = fp8_per_tensor_quant(x)
            q_b_cute, inv_s_b_cute = fp8_per_tensor_quant(w)

            def cute_pt():
                return cute_fp8_per_tensor_gemm(q_a_cute, inv_s_a_cute,
                                                 q_b_cute, inv_s_b_cute,
                                                 out_dtype=torch.bfloat16)

            out_cute = cute_pt()
            stats = benchmark_func(cute_pt, warmup=warmup, repeat=repeat)
            mean_ms = stats['mean_ms']
            tflops = compute_tflops(flops, mean_ms)
            vs = fp16_ref / mean_ms if fp16_ref else None
            err = gemm_accuracy(out_cute, fp16_out.bfloat16() if fp16_out is not None else out_cute)
            print_row("cute_per_tensor", mean_ms, tflops, vs,
                      snr_db=err['snr_db'], rmse=err['rmse'])
        except Exception as e:
            print(f"  [cute_per_tensor SKIP] {e}")

        # --- CUTLASS V1 Per-Tensor ---
        try:
            import operators.fp8_quant.cutlass_fp8.kernel as _fp8_mod
            _fp8_mod._load_extension_v1()
            if not _fp8_mod.FP8_V1_AVAILABLE:
                raise RuntimeError("FP8 V1 not available")
            from operators.fp8_quant.cutlass_fp8.kernel import fp8_per_tensor_gemm_cutlass

            q_a_v1, inv_s_a_v1 = fp8_per_tensor_quant(x)
            q_b_v1, inv_s_b_v1 = fp8_per_tensor_quant(w)

            def cutlass_v1():
                return fp8_per_tensor_gemm_cutlass(q_a_v1, q_b_v1, inv_s_a_v1, inv_s_b_v1)

            out_v1 = cutlass_v1()
            stats = benchmark_func(cutlass_v1, warmup=warmup, repeat=repeat)
            mean_ms = stats['mean_ms']
            tflops = compute_tflops(flops, mean_ms)
            vs = fp16_ref / mean_ms if fp16_ref else None
            err = gemm_accuracy(out_v1, fp16_out.bfloat16() if fp16_out is not None else out_v1)
            print_row("cutlass_v1_per_tensor", mean_ms, tflops, vs,
                      snr_db=err['snr_db'], rmse=err['rmse'])
        except Exception as e:
            print(f"  [cutlass_v1_per_tensor SKIP] {e}")

        # --- CUTLASS V2 Per-Block ---
        try:
            import operators.fp8_quant.cutlass_fp8.kernel as _fp8_mod2
            _fp8_mod2._load_extension_v2()
            if not _fp8_mod2.FP8_V2_AVAILABLE:
                raise RuntimeError("FP8 V2 not available")
            from operators.fp8_quant.cutlass_fp8.kernel import fp8_per_block_gemm_cutlass

            N_v2 = (N // 128) * 128
            K_v2 = (K // 128) * 128
            if N_v2 == 0 or K_v2 == 0:
                raise ValueError(f"N or K too small")

            x_v2, w_v2 = make_tensors(M, K_v2, N_v2)
            q_a_v2, inv_s_a_v2 = fp8_per_block_act_quant(x_v2, group_size=128)
            q_b_v2, inv_s_b_v2 = fp8_per_block_weight_quant(w_v2, block_size=128)

            def cutlass_v2():
                return fp8_per_block_gemm_cutlass(q_a_v2, inv_s_a_v2, q_b_v2, inv_s_b_v2)

            out_v2 = cutlass_v2()
            stats = benchmark_func(cutlass_v2, warmup=warmup, repeat=repeat)
            mean_ms = stats['mean_ms']
            flops_v2 = compute_gemm_flops(M, K_v2, N_v2)
            tflops = compute_tflops(flops_v2, mean_ms)
            vs = fp16_ref / mean_ms if fp16_ref else None
            ref_v2 = x_v2.float() @ w_v2.float().T
            err = gemm_accuracy(out_v2, ref_v2.to(torch.bfloat16))
            print_row("cutlass_v2_per_block", mean_ms, tflops, vs,
                      snr_db=err['snr_db'], rmse=err['rmse'])
        except Exception as e:
            print(f"  [cutlass_v2_per_block SKIP] {e}")

        print()


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FP8 Quant Benchmark")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=50, help="Benchmark iterations")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available!")
        sys.exit(1)

    if not hasattr(torch, 'float8_e4m3fn'):
        print("ERROR: torch.float8_e4m3fn not available. Need PyTorch >= 2.1")
        sys.exit(1)

    # 测试矩阵尺寸（与计划中一致）
    configs = [
        (64,   512,  512),
        (256,  2048, 2048),
        (1024, 4096, 4096),
        (4096, 4096, 14336),
    ]

    run_benchmark(configs, warmup=args.warmup, repeat=args.repeat)


if __name__ == "__main__":
    main()
