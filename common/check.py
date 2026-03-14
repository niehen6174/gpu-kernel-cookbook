"""
正确性检验工具：比较 kernel 输出与 PyTorch baseline
"""

import torch
import numpy as np
from typing import Optional


def check_correctness(
    out_kernel: torch.Tensor,
    out_ref: torch.Tensor,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    name: str = "kernel",
    verbose: bool = True,
) -> bool:
    """
    比较 kernel 输出与参考实现（PyTorch）的误差。

    Args:
        out_kernel: kernel 输出张量
        out_ref:    参考输出张量（PyTorch baseline）
        atol:       绝对误差容忍度
        rtol:       相对误差容忍度
        name:       kernel 名称（用于打印）
        verbose:    是否打印详细信息

    Returns:
        True 表示通过测试
    """
    out_kernel = out_kernel.float().cpu()
    out_ref = out_ref.float().cpu()

    max_abs_err = (out_kernel - out_ref).abs().max().item()
    mean_abs_err = (out_kernel - out_ref).abs().mean().item()
    max_rel_err = ((out_kernel - out_ref).abs() / (out_ref.abs() + 1e-8)).max().item()

    passed = torch.allclose(out_kernel, out_ref, atol=atol, rtol=rtol)

    if verbose:
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"[{name}] {status}")
        print(f"  max_abs_error : {max_abs_err:.2e}")
        print(f"  mean_abs_error: {mean_abs_err:.2e}")
        print(f"  max_rel_error : {max_rel_err:.2e}")
        if not passed:
            # 打印不匹配位置示例
            diff = (out_kernel - out_ref).abs()
            idx = diff.argmax()
            print(f"  worst mismatch @ idx={idx.item()}: "
                  f"kernel={out_kernel.flatten()[idx].item():.6f}, "
                  f"ref={out_ref.flatten()[idx].item():.6f}")

    return passed


def allclose_fp16(
    out_kernel: torch.Tensor,
    out_ref: torch.Tensor,
    name: str = "kernel",
) -> bool:
    """FP16 精度下的宽松对比（误差容忍更大）"""
    return check_correctness(out_kernel, out_ref, atol=1e-2, rtol=1e-2, name=name)
