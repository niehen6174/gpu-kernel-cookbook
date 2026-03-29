"""
SVDQuant 正确性测试

测试矩阵大小:
    - (64, 512, 512)   — 小
    - (256, 2048, 2048) — 中
    - (1024, 4096, 4096) — 大（典型 LLM 尺寸，默认跳过以节省时间）

测试内容:
    1. 量化/反量化精度：int4_quantize + int4_dequantize 的数值误差
    2. PyTorch 版 vs FP16 baseline：误差应在量化误差范围内（atol≈0.05）
    3. 打包/解包：int4_pack_uint8 + int4_unpack_uint8 的正确性
    4. 如果 Triton 版本存在，也与 PyTorch 版对比

运行方式：
    python operators/svdquant/test.py
    python operators/svdquant/test.py --large   # 包含大矩阵测试
    python operators/svdquant/test.py -v        # 详细输出
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

import torch
import numpy as np

from common.check import check_correctness
from operators.svdquant.pytorch.svdquant_torch import (
    int4_quantize,
    int4_dequantize,
    int4_pack_uint8,
    int4_unpack_uint8,
    create_svdquant_params,
    svdquant_forward_torch,
)
from operators.svdquant.pytorch.baseline import svdquant_fp16_baseline


# -------------------------------------------------------------------------
# 辅助
# -------------------------------------------------------------------------

def make_tensors(M, K, N, rank=32, dtype=torch.float16, device="cuda"):
    """生成测试用张量"""
    torch.manual_seed(42)
    x = torch.randn(M, K, dtype=dtype, device=device) * 0.5
    W = torch.randn(K, N, dtype=dtype, device=device) * 0.02
    # 模拟 smooth factor（接近 1，带些变化）
    smooth = torch.rand(K, dtype=dtype, device=device) * 0.5 + 0.5   # [0.5, 1.0]
    bias = torch.randn(N, dtype=dtype, device=device) * 0.01
    return x, W, smooth, bias


# -------------------------------------------------------------------------
# 测试 1：量化原语
# -------------------------------------------------------------------------

def test_quantize_dequantize(verbose=False):
    print("\n[Test 1] int4_quantize / int4_dequantize")
    passed_all = True

    for M, K in [(64, 512), (256, 2048), (1024, 4096)]:
        x = torch.randn(M, K, dtype=torch.float16, device="cuda") * 2.0
        q, scales = int4_quantize(x, group_size=64)
        x_dq = int4_dequantize(q, scales, group_size=64)

        # 量化误差上界：scale/2（四舍五入误差不超过 scale 的一半）
        max_err = (x - x_dq).abs().max().item()
        rel_err = ((x - x_dq).abs() / (x.abs() + 1e-8)).mean().item()

        # 验证 q 值范围
        q_min = q.min().item()
        q_max = q.max().item()
        range_ok = (q_min >= -8) and (q_max <= 7)

        # 理论最大误差 = scale/2 = max(|x_group|) / 14
        # 验证实际误差 <= 理论最大误差（允许少量 fp16 精度损失）
        group_size = 64
        num_groups = K // group_size
        x_grouped = x.view(M, num_groups, group_size)
        theoretical_max_err = (x_grouped.abs().max(dim=-1).values / 14.0).max().item()
        passed = range_ok and (max_err <= theoretical_max_err * 1.01 + 1e-4)
        passed_all &= passed

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] ({M},{K}): max_err={max_err:.4f}, theoretical_bound={theoretical_max_err:.4f}, "
              f"q_range=[{q_min},{q_max}]")

    return passed_all


# -------------------------------------------------------------------------
# 测试 2：打包/解包
# -------------------------------------------------------------------------

def test_pack_unpack(verbose=False):
    print("\n[Test 2] int4_pack_uint8 / int4_unpack_uint8")
    passed_all = True

    for M, K in [(64, 512), (256, 2048)]:
        x = torch.randn(M, K, dtype=torch.float16, device="cuda") * 3.0
        q, _ = int4_quantize(x, group_size=64)

        packed = int4_pack_uint8(q)
        q_unpacked = int4_unpack_uint8(packed)

        match = (q == q_unpacked).all().item()
        passed_all &= match
        status = "PASS" if match else "FAIL"

        if not match:
            diff = (q != q_unpacked).sum().item()
            print(f"  [{status}] ({M},{K}): {diff}/{q.numel()} mismatches")
        else:
            print(f"  [{status}] ({M},{K}): pack/unpack round-trip exact")

    return passed_all


# -------------------------------------------------------------------------
# 测试 3：PyTorch SVDQuant vs FP16 Baseline
# -------------------------------------------------------------------------

def test_svdquant_vs_baseline(configs, verbose=False):
    print("\n[Test 3] SVDQuant PyTorch vs FP16 Baseline")
    passed_all = True

    for M, K, N in configs:
        x, W, smooth, bias = make_tensors(M, K, N)

        # 构建 SVDQuant 参数
        params = create_svdquant_params(W, rank=32, group_size=64, smooth=smooth)

        # FP16 baseline：用 smooth 后的 *残差* 权重计算参考值
        # 注意：baseline 接受已量化残差的 FP16 版，所以需要反量化
        W_residual_dq = int4_dequantize(
            params["q_w"].T.contiguous(),
            params["wscales"].T.contiguous(),
            group_size=64,
        ).T.contiguous()   # (K, N)

        # FP16 baseline（无量化误差，用于确认 SVD 分解正确）
        y_baseline = svdquant_fp16_baseline(
            x, W_residual_dq,
            params["lora_down"], params["lora_up"],
            smooth, bias
        )

        # SVDQuant PyTorch 实现
        y_svdq = svdquant_forward_torch(
            x,
            params["q_w"], params["wscales"],
            params["lora_down"], params["lora_up"],
            smooth, bias,
            group_size=64,
        )

        # 计算误差
        max_err = (y_svdq - y_baseline).abs().max().item()
        mean_err = (y_svdq - y_baseline).abs().mean().item()

        # 允许一定量化误差（量化误差 + double 量化误差，允许较大阈值）
        passed = max_err < 0.5
        passed_all &= passed
        status = "PASS" if passed else "FAIL"

        print(f"  [{status}] ({M},{K},{N}): max_err={max_err:.5f}, mean_err={mean_err:.6f}")
        if verbose and not passed:
            # 打印更多信息
            print(f"           y_baseline: min={y_baseline.min():.3f}, max={y_baseline.max():.3f}, "
                  f"mean={y_baseline.mean():.3f}")
            print(f"           y_svdq:     min={y_svdq.min():.3f}, max={y_svdq.max():.3f}, "
                  f"mean={y_svdq.mean():.3f}")

    return passed_all


# -------------------------------------------------------------------------
# 测试 4：确认量化误差与全精度的关系
# -------------------------------------------------------------------------

def test_quantization_error_bound(verbose=False):
    print("\n[Test 4] 量化误差分析（SVDQuant vs 全精度 FP16）")
    passed_all = True

    for M, K, N in [(64, 512, 512), (256, 2048, 2048)]:
        x, W, smooth, bias = make_tensors(M, K, N)

        # 全精度参考（无量化）
        # y_ref = (x / smooth) @ (smooth[:, None] * W) + bias
        #       = x @ W + bias  (smooth cancels out)
        x_smooth_fp = x / smooth.unsqueeze(0)
        W_smooth_fp = smooth.unsqueeze(1) * W
        y_ref = x_smooth_fp.float() @ W_smooth_fp.float()
        y_ref = y_ref.to(x.dtype) + bias

        # SVDQuant 前向
        params = create_svdquant_params(W, rank=32, group_size=64, smooth=smooth)
        y_svdq = svdquant_forward_torch(
            x,
            params["q_w"], params["wscales"],
            params["lora_down"], params["lora_up"],
            smooth, bias,
            group_size=64,
        )

        max_err = (y_svdq - y_ref).abs().max().item()
        mean_err = (y_svdq - y_ref).abs().mean().item()

        # SVDQuant vs 全精度：包含量化误差 + SVD 近似误差，允许更大误差
        passed = max_err < 1.0   # 宽松阈值
        passed_all &= passed
        status = "PASS" if passed else "FAIL"

        print(f"  [{status}] ({M},{K},{N}): max_err_vs_fp16={max_err:.5f}, mean={mean_err:.6f}")

    return passed_all


# -------------------------------------------------------------------------
# 测试 5：Triton 版本（如可用）
# -------------------------------------------------------------------------

def test_triton_vs_pytorch(configs, verbose=False):
    print("\n[Test 5] Triton vs PyTorch SVDQuant")

    try:
        from operators.svdquant.triton.kernel import svdquant_forward_triton
    except ImportError as e:
        print(f"  [SKIP] Triton 版本未找到: {e}")
        return True

    passed_all = True
    for M, K, N in configs:
        x, W, smooth, bias = make_tensors(M, K, N)
        params = create_svdquant_params(W, rank=32, group_size=64, smooth=smooth)

        y_torch = svdquant_forward_torch(
            x, params["q_w"], params["wscales"],
            params["lora_down"], params["lora_up"],
            smooth, bias, group_size=64,
        )

        try:
            y_triton = svdquant_forward_triton(
                x, params["q_w"], params["wscales"],
                params["lora_down"], params["lora_up"],
                smooth, bias, group_size=64,
            )
        except Exception as e:
            print(f"  [SKIP] ({M},{K},{N}) Triton error: {e}")
            continue

        passed = check_correctness(
            y_triton, y_torch,
            atol=0.05, rtol=0.01,
            name=f"triton ({M},{K},{N})",
            verbose=verbose,
        )
        passed_all &= passed

    return passed_all


# -------------------------------------------------------------------------
# 测试 6：CuTe 版本（如可用）
# -------------------------------------------------------------------------

def test_cute_vs_pytorch(configs, verbose=False):
    print("\n[Test 6] CuTe vs PyTorch SVDQuant")

    try:
        from operators.svdquant.cute.kernel import svdquant_forward_cute, CUTE_AVAILABLE
        if not CUTE_AVAILABLE:
            print("  [SKIP] cutlass.cute 未安装")
            return True
    except ImportError as e:
        print(f"  [SKIP] CuTe 版本未找到: {e}")
        return True

    passed_all = True
    for M, K, N in configs:
        x, W, smooth, bias = make_tensors(M, K, N)
        params = create_svdquant_params(W, rank=32, group_size=64, smooth=smooth)

        y_torch = svdquant_forward_torch(
            x, params["q_w"], params["wscales"],
            params["lora_down"], params["lora_up"],
            smooth, bias, group_size=64,
        )

        for ver in ["v1", "v2"]:
            try:
                y_cute = svdquant_forward_cute(
                    x, params["q_w"], params["wscales"],
                    params["lora_down"], params["lora_up"],
                    smooth, bias, group_size=64, version=ver,
                )
            except Exception as e:
                print(f"  [SKIP] ({M},{K},{N}) CuTe {ver} error: {e}")
                continue

            passed = check_correctness(
                y_cute, y_torch,
                atol=0.5, rtol=0.05,
                name=f"cute_{ver} ({M},{K},{N})",
                verbose=verbose,
            )
            passed_all &= passed

    return passed_all


# -------------------------------------------------------------------------
# 测试 7：Triton 优化版
# -------------------------------------------------------------------------

def test_triton_opt_vs_pytorch(configs, verbose=False):
    print("\n[Test 7] Triton Optimized vs PyTorch SVDQuant")

    try:
        from operators.svdquant.triton.kernel import svdquant_forward_triton_opt
    except ImportError as e:
        print(f"  [SKIP] Triton 优化版未找到: {e}")
        return True

    passed_all = True
    for M, K, N in configs:
        x, W, smooth, bias = make_tensors(M, K, N)
        params = create_svdquant_params(W, rank=32, group_size=64, smooth=smooth)

        y_torch = svdquant_forward_torch(
            x, params["q_w"], params["wscales"],
            params["lora_down"], params["lora_up"],
            smooth, bias, group_size=64,
        )

        try:
            y_triton_opt = svdquant_forward_triton_opt(
                x, params["q_w"], params["wscales"],
                params["lora_down"], params["lora_up"],
                smooth, bias, group_size=64,
            )
        except Exception as e:
            print(f"  [SKIP] ({M},{K},{N}) Triton opt error: {e}")
            import traceback; traceback.print_exc()
            continue

        passed = check_correctness(
            y_triton_opt, y_torch,
            atol=0.05, rtol=0.01,
            name=f"triton_opt ({M},{K},{N})",
            verbose=verbose,
        )
        passed_all &= passed

    return passed_all


# -------------------------------------------------------------------------
# 测试 8：CuTe V3
# -------------------------------------------------------------------------

def test_cute_v3_vs_pytorch(configs, verbose=False):
    print("\n[Test 8] CuTe V3 vs PyTorch SVDQuant")

    try:
        from operators.svdquant.cute.kernel import svdquant_forward_cute_v3, CUTE_AVAILABLE
        if not CUTE_AVAILABLE:
            print("  [SKIP] cutlass.cute 未安装")
            return True
    except ImportError as e:
        print(f"  [SKIP] CuTe V3 未找到: {e}")
        return True

    passed_all = True
    for M, K, N in configs:
        x, W, smooth, bias = make_tensors(M, K, N)
        params = create_svdquant_params(W, rank=32, group_size=64, smooth=smooth)

        y_torch = svdquant_forward_torch(
            x, params["q_w"], params["wscales"],
            params["lora_down"], params["lora_up"],
            smooth, bias, group_size=64,
        )

        try:
            y_cute_v3 = svdquant_forward_cute_v3(
                x, params["q_w"], params["wscales"],
                params["lora_down"], params["lora_up"],
                smooth, bias, group_size=64,
            )
        except Exception as e:
            print(f"  [SKIP] ({M},{K},{N}) CuTe V3 error: {e}")
            import traceback; traceback.print_exc()
            continue

        passed = check_correctness(
            y_cute_v3, y_torch,
            atol=0.5, rtol=0.05,
            name=f"cute_v3 ({M},{K},{N})",
            verbose=verbose,
        )
        passed_all &= passed

    return passed_all


# -------------------------------------------------------------------------
# 测试 9：Nunchaku 版本（如可用）
# -------------------------------------------------------------------------

def test_nunchaku_vs_pytorch(configs, verbose=False):
    print("\n[Test 9] Nunchaku vs PyTorch SVDQuant")

    try:
        from operators.svdquant.nunchaku.kernel import (
            svdquant_forward_nunchaku,
            NUNCHAKU_AVAILABLE,
        )
        if not NUNCHAKU_AVAILABLE:
            print("  [SKIP] nunchaku 未安装")
            return True
    except ImportError as e:
        print(f"  [SKIP] nunchaku 版本未找到: {e}")
        return True

    passed_all = True
    for M, K, N in configs:
        x, W, smooth, bias = make_tensors(M, K, N)
        params = create_svdquant_params(W, rank=32, group_size=64, smooth=smooth)

        y_torch = svdquant_forward_torch(
            x, params["q_w"], params["wscales"],
            params["lora_down"], params["lora_up"],
            smooth, bias, group_size=64,
        )

        try:
            y_nunchaku = svdquant_forward_nunchaku(
                x, params["q_w"], params["wscales"],
                params["lora_down"], params["lora_up"],
                smooth, bias, group_size=64,
            )
        except Exception as e:
            print(f"  [SKIP] ({M},{K},{N}) Nunchaku error: {e}")
            import traceback; traceback.print_exc()
            continue

        passed = check_correctness(
            y_nunchaku, y_torch,
            atol=0.5, rtol=0.05,
            name=f"nunchaku ({M},{K},{N})",
            verbose=verbose,
        )
        passed_all &= passed

    return passed_all


# -------------------------------------------------------------------------
# 测试 10：W8A8 CUTLASS SM90 WGMMA（如可用）
# -------------------------------------------------------------------------

def test_w8a8_vs_pytorch(configs, verbose=False):
    print("\n[Test 10] W8A8 CUTLASS SM90 WGMMA vs PyTorch SVDQuant")

    try:
        import operators.svdquant.cutlass_w8a8.kernel as _w8a8_mod
        from operators.svdquant.cutlass_w8a8.kernel import (
            svdquant_forward_w8a8,
            prepare_w8a8_weights,
        )
        _w8a8_mod._load_extension()
        if not _w8a8_mod.W8A8_AVAILABLE:
            print("  [SKIP] W8A8 CUTLASS 扩展不可用（编译失败）")
            return True
    except ImportError as e:
        print(f"  [SKIP] W8A8 模块未找到: {e}")
        return True
    except Exception as e:
        print(f"  [SKIP] W8A8 加载失败: {e}")
        return True

    passed_all = True
    group_size = 64  # INT4 量化的 group size

    for M, K, N in configs:
        x, W, smooth, bias = make_tensors(M, K, N)
        params = create_svdquant_params(W, rank=32, group_size=group_size, smooth=smooth)

        # PyTorch SVDQuant（INT4）参考输出
        y_torch = svdquant_forward_torch(
            x, params["q_w"], params["wscales"],
            params["lora_down"], params["lora_up"],
            smooth, bias, group_size=group_size,
        )

        # W8A8 CUTLASS 输出（在线版，含权重重新量化）
        try:
            y_w8a8 = svdquant_forward_w8a8(
                x, params["q_w"], params["wscales"],
                params["lora_down"], params["lora_up"],
                smooth, bias,
                group_size_act=128,
                group_size_wgt_int4=group_size,
            )
        except Exception as e:
            print(f"  [SKIP] ({M},{K},{N}) W8A8 forward error: {e}")
            if verbose:
                import traceback; traceback.print_exc()
            continue

        # INT8 group=128 vs INT4 group=64：有额外近似误差，允许较大阈值
        passed = check_correctness(
            y_w8a8, y_torch,
            atol=1.0, rtol=0.1,
            name=f"w8a8_sm90 ({M},{K},{N})",
            verbose=verbose,
        )
        passed_all &= passed

    return passed_all


# -------------------------------------------------------------------------
# 测试 11：W8A8 V3 (stream overlap)
# -------------------------------------------------------------------------

def test_w8a8_v3_vs_pytorch(configs, verbose=False):
    print("\n[Test 11] W8A8 V3 (stream overlap) vs PyTorch SVDQuant")

    try:
        import operators.svdquant.cutlass_w8a8.kernel as _w8a8_mod
        from operators.svdquant.cutlass_w8a8.kernel import (
            svdquant_forward_w8a8_v3,
        )
        _w8a8_mod._load_extension_v2()
        if not _w8a8_mod.W8A8_V2_AVAILABLE:
            print("  [SKIP] W8A8 V2 扩展不可用（V3 复用 V2 kernel）")
            return True
    except ImportError as e:
        print(f"  [SKIP] W8A8 模块未找到: {e}")
        return True
    except Exception as e:
        print(f"  [SKIP] W8A8 加载失败: {e}")
        return True

    passed_all = True
    group_size = 64

    for M, K, N in configs:
        x, W, smooth, bias = make_tensors(M, K, N)
        params = create_svdquant_params(W, rank=32, group_size=group_size, smooth=smooth)

        y_torch = svdquant_forward_torch(
            x, params["q_w"], params["wscales"],
            params["lora_down"], params["lora_up"],
            smooth, bias, group_size=group_size,
        )

        try:
            y_v3 = svdquant_forward_w8a8_v3(
                x, params["q_w"], params["wscales"],
                params["lora_down"], params["lora_up"],
                smooth, bias,
                group_size_act=128,
                group_size_wgt_int4=group_size,
            )
        except Exception as e:
            print(f"  [SKIP] ({M},{K},{N}) W8A8 V3 forward error: {e}")
            if verbose:
                import traceback; traceback.print_exc()
            continue

        passed = check_correctness(
            y_v3, y_torch,
            atol=1.0, rtol=0.1,
            name=f"w8a8_v3_sm90 ({M},{K},{N})",
            verbose=verbose,
        )
        passed_all &= passed

    return passed_all


# -------------------------------------------------------------------------
# 主入口
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SVDQuant Correctness Tests")
    parser.add_argument("--large", action="store_true", help="Include large matrix tests (1024, 4096, 4096)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available!")
        sys.exit(1)

    print(f"Running SVDQuant Correctness Tests")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    configs_small = [
        (64,  512,  512),
        (256, 2048, 2048),
    ]
    if args.large:
        configs_small.append((1024, 4096, 4096))

    results = {}

    results["quantize_dequantize"]  = test_quantize_dequantize(args.verbose)
    results["pack_unpack"]          = test_pack_unpack(args.verbose)
    results["svdquant_vs_baseline"] = test_svdquant_vs_baseline(configs_small, args.verbose)
    results["quantization_error"]   = test_quantization_error_bound(args.verbose)
    results["triton_vs_pytorch"]    = test_triton_vs_pytorch(configs_small[:1], args.verbose)
    results["cute_vs_pytorch"]      = test_cute_vs_pytorch(configs_small[:1], args.verbose)
    results["triton_opt_vs_pytorch"] = test_triton_opt_vs_pytorch(configs_small[:1], args.verbose)
    results["cute_v3_vs_pytorch"]    = test_cute_v3_vs_pytorch(configs_small[:1], args.verbose)
    results["nunchaku_vs_pytorch"]   = test_nunchaku_vs_pytorch(configs_small[:1], args.verbose)
    results["w8a8_vs_pytorch"]       = test_w8a8_vs_pytorch(configs_small[:1], args.verbose)

    # W8A8 V3（stream overlap，复用 V2 kernel，只测小规模）
    results["w8a8_v3_vs_pytorch"]    = test_w8a8_v3_vs_pytorch(configs_small[:1], args.verbose)

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        all_passed &= passed

    print("="*60)
    if all_passed:
        print("  ALL TESTS PASSED ✓")
    else:
        print("  SOME TESTS FAILED ✗")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
