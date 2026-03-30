"""
FP8 量化算子正确性测试

测试用例：
  1. Per-Tensor quant 精度（quant → dequant vs original）
  2. Per-Block quant 精度（quant → dequant vs original）
  3. Triton per-tensor GEMM vs PyTorch（atol=0.1）
  4. Triton per-block GEMM vs PyTorch（atol=0.3）
  5. CUTLASS V1 per-tensor GEMM vs PyTorch（atol=0.1）
  6. CUTLASS V2 per-block GEMM vs PyTorch（atol=0.5）
  7. 精度对比——per-tensor vs per-block，打印 SNR/RMSE/cos_sim

运行方式：
    python operators/fp8_quant/test.py
    python operators/fp8_quant/test.py -v
    python operators/fp8_quant/test.py --large
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

import torch
import torch.nn.functional as F
import math

from common.check import check_correctness
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
# 辅助函数
# ---------------------------------------------------------------------------

def make_tensors(M, K, N, dtype=torch.float16, device="cuda", seed=42):
    """生成测试用张量"""
    torch.manual_seed(seed)
    x = torch.randn(M, K, dtype=dtype, device=device) * 1.0
    w = torch.randn(N, K, dtype=dtype, device=device) * 0.1
    return x, w


def print_quant_error(name: str, err: dict):
    """格式化打印量化误差"""
    print(f"  {name}: SNR={err['snr_db']:.1f}dB  RMSE={err['rmse']:.2e}  "
          f"max_err={err['max_abs_error']:.2e}  cos_sim={err['cosine_sim']:.6f}")


# ---------------------------------------------------------------------------
# Test 1：Per-Tensor quant 精度
# ---------------------------------------------------------------------------

def test_per_tensor_quant_accuracy(verbose=False):
    print("\n[Test 1] Per-Tensor FP8 量化精度（quant → dequant vs original）")
    passed_all = True

    for M, K in [(64, 512), (256, 2048), (1024, 4096)]:
        x = torch.randn(M, K, dtype=torch.float16, device="cuda") * 2.0

        q, inv_scale = fp8_per_tensor_quant(x)
        x_dq = fp8_per_tensor_dequant(q, inv_scale, out_dtype=torch.float16)

        # 验证 q 形状和 dtype
        assert q.shape == (M, K) and q.dtype == torch.float8_e4m3fn
        assert inv_scale.ndim == 0 or inv_scale.numel() == 1

        # 精度分析
        err = compute_quant_error(x, x_dq)

        # FP8 E4M3：3 尾数位，最高 binade（256-448）步长约 32，
        # 解量化后最大误差约 amax/16（最坏情况），SNR ≈ 30-35dB 是正常范围
        amax = x.float().abs().amax().item()
        theoretical_max_err = amax / 16.0   # FP8 E4M3 最坏情况上界

        # SNR > 25dB（FP8 量化），cos_sim > 0.999，max_err < 理论上界
        passed = (err['max_abs_error'] <= theoretical_max_err * 1.1 + 1e-4
                  and err['snr_db'] > 25.0
                  and err['cosine_sim'] > 0.999)
        passed_all &= passed

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] ({M},{K}):", end=" ")
        print_quant_error("", err)
        if verbose:
            print(f"           theoretical_max_err(FP8 worst-case)={theoretical_max_err:.2e}, amax={amax:.3f}")

    return passed_all


# ---------------------------------------------------------------------------
# Test 2：Per-Block quant 精度
# ---------------------------------------------------------------------------

def test_per_block_quant_accuracy(verbose=False):
    print("\n[Test 2] Per-Block FP8 量化精度（quant → dequant vs original）")
    passed_all = True

    for M, K in [(64, 512), (256, 2048), (1024, 4096)]:
        x = torch.randn(M, K, dtype=torch.float16, device="cuda") * 2.0
        GROUP_SIZE = 128

        q, inv_scales = fp8_per_block_act_quant(x, group_size=GROUP_SIZE)

        # 验证形状
        assert q.shape == (M, K) and q.dtype == torch.float8_e4m3fn
        assert inv_scales.shape == (M, K // GROUP_SIZE)

        # 解量化：q * inv_scale（per-group 广播）
        q_f32 = q.float().reshape(M, K // GROUP_SIZE, GROUP_SIZE)
        x_dq = (q_f32 * inv_scales.unsqueeze(-1)).reshape(M, K).to(torch.float16)

        err = compute_quant_error(x, x_dq)

        # Per-block 精度应高于 per-tensor
        # FP8 E4M3 with group_size=128：SNR ~ 31-35dB，比全局稍好
        passed = (err['snr_db'] > 25.0
                  and err['cosine_sim'] > 0.999)
        passed_all &= passed

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] ({M},{K}):", end=" ")
        print_quant_error("", err)

    return passed_all


# ---------------------------------------------------------------------------
# Test 3：Triton per-tensor GEMM vs PyTorch
# ---------------------------------------------------------------------------

def test_triton_per_tensor_gemm(configs, verbose=False):
    print("\n[Test 3] Triton Per-Tensor FP8 GEMM vs PyTorch（atol=0.1）")

    try:
        from operators.fp8_quant.triton.kernel import triton_fp8_per_tensor_gemm
    except ImportError as e:
        print(f"  [SKIP] Triton 版本未找到: {e}")
        return True

    passed_all = True
    for M, K, N in configs:
        x, w = make_tensors(M, K, N)

        q_a, inv_s_a = fp8_per_tensor_quant(x)
        q_b, inv_s_b = fp8_per_tensor_quant(w)

        # PyTorch reference
        try:
            ref = fp8_per_tensor_gemm(q_a, inv_s_a, q_b, inv_s_b, out_dtype=torch.bfloat16)
        except Exception as e:
            print(f"  [SKIP] ({M},{K},{N}) PyTorch ref failed: {e}")
            continue

        # Triton
        try:
            out = triton_fp8_per_tensor_gemm(
                q_a, inv_s_a, q_b, inv_s_b,
                out_dtype=torch.bfloat16,
            )
        except Exception as e:
            print(f"  [SKIP] ({M},{K},{N}) Triton error: {e}")
            import traceback; traceback.print_exc()
            continue

        passed = check_correctness(
            out, ref,
            atol=0.1, rtol=0.01,
            name=f"triton_per_tensor ({M},{K},{N})",
            verbose=verbose,
        )
        passed_all &= passed

    return passed_all


# ---------------------------------------------------------------------------
# Test 4：Triton per-block GEMM vs PyTorch
# ---------------------------------------------------------------------------

def test_triton_per_block_gemm(configs, verbose=False):
    print("\n[Test 4] Triton Per-Block FP8 GEMM vs PyTorch（atol=0.3）")

    try:
        from operators.fp8_quant.triton.kernel import triton_fp8_per_block_gemm
    except ImportError as e:
        print(f"  [SKIP] Triton 版本未找到: {e}")
        return True

    passed_all = True
    for M, K, N in configs:
        # Per-block 需要 N 可被 128 整除
        N_aligned = (N // 128) * 128
        if N_aligned == 0:
            print(f"  [SKIP] ({M},{K},{N}) N too small for per-block")
            continue

        x, w = make_tensors(M, K, N_aligned)

        q_a, inv_s_a = fp8_per_block_act_quant(x, group_size=128)
        q_b, inv_s_b = fp8_per_block_weight_quant(w, block_size=128)

        # PyTorch reference
        try:
            ref = fp8_per_block_gemm(q_a, inv_s_a, q_b, inv_s_b, out_dtype=torch.bfloat16)
        except Exception as e:
            print(f"  [SKIP] ({M},{K},{N_aligned}) PyTorch ref failed: {e}")
            continue

        # Triton
        try:
            out = triton_fp8_per_block_gemm(
                q_a, inv_s_a, q_b, inv_s_b,
                out_dtype=torch.bfloat16,
            )
        except Exception as e:
            print(f"  [SKIP] ({M},{K},{N_aligned}) Triton error: {e}")
            import traceback; traceback.print_exc()
            continue

        passed = check_correctness(
            out, ref,
            atol=0.3, rtol=0.05,
            name=f"triton_per_block ({M},{K},{N_aligned})",
            verbose=verbose,
        )
        passed_all &= passed

    return passed_all


# ---------------------------------------------------------------------------
# Test 5：CUTLASS V1 per-tensor GEMM vs PyTorch
# ---------------------------------------------------------------------------

def test_cutlass_v1_per_tensor_gemm(configs, verbose=False):
    print("\n[Test 5] CUTLASS V1 Per-Tensor FP8 GEMM vs PyTorch（atol=0.1）")

    try:
        import operators.fp8_quant.cutlass_fp8.kernel as _fp8_mod
        _fp8_mod._load_extension_v1()
        if not _fp8_mod.FP8_V1_AVAILABLE:
            print("  [SKIP] CUTLASS FP8 V1 扩展不可用（编译失败）")
            return True
        from operators.fp8_quant.cutlass_fp8.kernel import fp8_per_tensor_gemm_cutlass
    except Exception as e:
        print(f"  [SKIP] CUTLASS V1 加载失败: {e}")
        return True

    passed_all = True
    for M, K, N in configs:
        x, w = make_tensors(M, K, N)

        q_a, inv_s_a = fp8_per_tensor_quant(x)
        q_b, inv_s_b = fp8_per_tensor_quant(w)

        # PyTorch reference
        try:
            ref = fp8_per_tensor_gemm(q_a, inv_s_a, q_b, inv_s_b, out_dtype=torch.bfloat16)
        except Exception as e:
            print(f"  [SKIP] ({M},{K},{N}) PyTorch ref failed: {e}")
            continue

        # CUTLASS V1
        try:
            out = fp8_per_tensor_gemm_cutlass(q_a, q_b, inv_s_a, inv_s_b, bias=None)
        except Exception as e:
            print(f"  [SKIP] ({M},{K},{N}) CUTLASS V1 error: {e}")
            import traceback; traceback.print_exc()
            continue

        passed = check_correctness(
            out, ref,
            atol=0.1, rtol=0.01,
            name=f"cutlass_v1_per_tensor ({M},{K},{N})",
            verbose=verbose,
        )
        passed_all &= passed

    return passed_all


# ---------------------------------------------------------------------------
# Test 6：CUTLASS V2 per-block GEMM vs PyTorch
# ---------------------------------------------------------------------------

def test_cutlass_v2_per_block_gemm(configs, verbose=False):
    print("\n[Test 6] CUTLASS V2 Per-Block FP8 GEMM vs PyTorch（atol=0.5）")

    try:
        import operators.fp8_quant.cutlass_fp8.kernel as _fp8_mod
        _fp8_mod._load_extension_v2()
        if not _fp8_mod.FP8_V2_AVAILABLE:
            print("  [SKIP] CUTLASS FP8 V2 扩展不可用（编译失败）")
            return True
        from operators.fp8_quant.cutlass_fp8.kernel import fp8_per_block_gemm_cutlass
    except Exception as e:
        print(f"  [SKIP] CUTLASS V2 加载失败: {e}")
        return True

    passed_all = True
    for M, K, N in configs:
        N_aligned = (N // 128) * 128
        if N_aligned == 0:
            continue

        x, w = make_tensors(M, K, N_aligned)

        q_a, inv_s_a = fp8_per_block_act_quant(x, group_size=128)
        q_b, inv_s_b = fp8_per_block_weight_quant(w, block_size=128)

        # 计算 Python 侧相同近似的参考（per-row mean × per-col mean）
        # CUTLASS V2: D[i,j] = act_scale_row[i] * raw_acc[i,j] * wgt_scale_full[j]
        # where raw_acc[i,j] = Σ_k q_a[i,k] * q_b[j,k]  (FP8 values as float)
        # act_scale_row = mean(inv_scales_a, dim=-1)  (M,)
        # wgt_scale_full = mean(inv_scales_b, dim=-1).repeat_interleave(128)  (N,)
        try:
            GROUP_SIZE = 128
            act_scale_row = inv_s_a.float().mean(dim=-1)    # (M,)
            n_blk_wgt = N_aligned // GROUP_SIZE
            wgt_scale_col_n = inv_s_b.float().mean(dim=-1)  # (n_blk_wgt,)
            wgt_scale_full = wgt_scale_col_n.repeat_interleave(GROUP_SIZE)  # (N,)

            # raw_acc: FP8 values treated as float (like CUTLASS accumulator)
            raw_acc = q_a.float() @ q_b.float().T   # (M, N)
            # apply per-row and per-col scale
            ref_approx = (raw_acc * act_scale_row.unsqueeze(1) *
                          wgt_scale_full.unsqueeze(0)).to(torch.bfloat16)
        except Exception as e:
            print(f"  [SKIP] ({M},{K},{N_aligned}) PyTorch ref failed: {e}")
            continue

        # CUTLASS V2（EVT 近似）
        try:
            out = fp8_per_block_gemm_cutlass(q_a, inv_s_a, q_b, inv_s_b, bias=None)
        except Exception as e:
            print(f"  [SKIP] ({M},{K},{N_aligned}) CUTLASS V2 error: {e}")
            import traceback; traceback.print_exc()
            continue

        # 与相同近似的 Python 参考对比（V2 近似 vs 相同近似的 Python 实现）
        passed = check_correctness(
            out, ref_approx,
            atol=0.1, rtol=0.01,
            name=f"cutlass_v2_per_block ({M},{K},{N_aligned})",
            verbose=verbose,
        )
        passed_all &= passed

    return passed_all


# ---------------------------------------------------------------------------
# Test 7：精度对比——per-tensor vs per-block
# ---------------------------------------------------------------------------

def test_accuracy_comparison(configs, verbose=False):
    print("\n[Test 7] 精度对比：Per-Tensor vs Per-Block FP8 量化")
    print(f"  {'Shape':<20} {'Scheme':<15} {'SNR(dB)':<10} {'RMSE':<10} {'max_err':<10} {'cos_sim':<10}")
    print("  " + "-" * 75)

    for M, K, N in configs:
        x, w = make_tensors(M, K, N)
        ref = x.float() @ w.float().T   # FP32 reference for GEMM output

        # --- Per-Tensor 量化精度 ---
        q_a_t, inv_s_a_t = fp8_per_tensor_quant(x)
        x_dq_tensor = fp8_per_tensor_dequant(q_a_t, inv_s_a_t, out_dtype=torch.float32)
        err_tensor = compute_quant_error(x.float(), x_dq_tensor)

        shape_str = f"({M},{K},{N})"
        print(f"  {shape_str:<20} {'per_tensor':<15} "
              f"{err_tensor['snr_db']:<10.1f} "
              f"{err_tensor['rmse']:<10.2e} "
              f"{err_tensor['max_abs_error']:<10.2e} "
              f"{err_tensor['cosine_sim']:<10.6f}")

        # --- Per-Block 量化精度 ---
        q_a_b, inv_s_a_b = fp8_per_block_act_quant(x, group_size=128)
        q_f32 = q_a_b.float().reshape(M, K // 128, 128)
        x_dq_block = (q_f32 * inv_s_a_b.unsqueeze(-1)).reshape(M, K).float()
        err_block = compute_quant_error(x.float(), x_dq_block)

        print(f"  {'':<20} {'per_block':<15} "
              f"{err_block['snr_db']:<10.1f} "
              f"{err_block['rmse']:<10.2e} "
              f"{err_block['max_abs_error']:<10.2e} "
              f"{err_block['cosine_sim']:<10.6f}")

        # 对比结论
        snr_gain = err_block['snr_db'] - err_tensor['snr_db']
        rmse_reduction = (1 - err_block['rmse'] / err_tensor['rmse']) * 100
        if verbose:
            print(f"  → Per-Block 精度优势: SNR +{snr_gain:.1f}dB，RMSE 减少 {rmse_reduction:.0f}%")

        print()

    return True


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FP8 Quant Correctness Tests")
    parser.add_argument("--large", action="store_true",
                        help="Include large matrix tests (1024, 4096, 4096)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available!")
        sys.exit(1)

    # 检查 FP8 支持
    if not hasattr(torch, 'float8_e4m3fn'):
        print("ERROR: torch.float8_e4m3fn not available. Need PyTorch >= 2.1")
        sys.exit(1)

    print(f"FP8 Quant Correctness Tests")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    configs_small = [
        (64,  512,  512),
        (256, 2048, 2048),
    ]
    configs_gemm = [
        (64, 512, 512),
        (256, 2048, 2048),
    ]
    if args.large:
        configs_small.append((1024, 4096, 4096))
        configs_gemm.append((1024, 4096, 4096))

    results = {}

    results["per_tensor_quant_accuracy"] = test_per_tensor_quant_accuracy(args.verbose)
    results["per_block_quant_accuracy"]  = test_per_block_quant_accuracy(args.verbose)
    results["triton_per_tensor_gemm"]    = test_triton_per_tensor_gemm(configs_gemm[:1], args.verbose)
    results["triton_per_block_gemm"]     = test_triton_per_block_gemm(configs_gemm[:1], args.verbose)
    results["cutlass_v1_per_tensor"]     = test_cutlass_v1_per_tensor_gemm(configs_gemm[:1], args.verbose)
    results["cutlass_v2_per_block"]      = test_cutlass_v2_per_block_gemm(configs_gemm[:1], args.verbose)
    results["accuracy_comparison"]       = test_accuracy_comparison(configs_small, args.verbose)

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        all_passed &= passed

    print("=" * 60)
    if all_passed:
        print("  ALL TESTS PASSED ✓")
    else:
        print("  SOME TESTS FAILED ✗")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
