"""
通用工具函数：计时、性能指标计算
"""

import time
import torch
import numpy as np
from typing import Callable, Dict, Any, Optional


def benchmark_func(
    func: Callable,
    *args,
    warmup: int = 10,
    repeat: int = 100,
    **kwargs
) -> Dict[str, float]:
    """
    对函数进行 GPU benchmark，返回延迟统计数据。

    Args:
        func: 待测试的函数
        warmup: 预热次数（排除 JIT 编译等影响）
        repeat: 测量次数
        *args, **kwargs: 传给 func 的参数

    Returns:
        包含 mean_ms, min_ms, max_ms, std_ms 的字典
    """
    # 预热
    for _ in range(warmup):
        func(*args, **kwargs)
    torch.cuda.synchronize()

    # 正式测量
    latencies = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))  # ms

    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "std_ms": float(np.std(latencies)),
        "median_ms": float(np.median(latencies)),
    }


def compute_bandwidth(bytes_accessed: int, latency_ms: float) -> float:
    """
    计算内存带宽（GB/s）。

    Args:
        bytes_accessed: 访问的字节数（读 + 写）
        latency_ms: kernel 延迟（毫秒）

    Returns:
        带宽（GB/s）
    """
    return bytes_accessed / (latency_ms * 1e-3) / 1e9


def compute_tflops(flop_count: int, latency_ms: float) -> float:
    """
    计算计算效率（TFLOPS）。

    Args:
        flop_count: 浮点运算次数
        latency_ms: kernel 延迟（毫秒）

    Returns:
        TFLOPS
    """
    return flop_count / (latency_ms * 1e-3) / 1e12


def print_benchmark_result(
    operator: str,
    impl: str,
    result: Dict[str, float],
    bandwidth_gb: Optional[float] = None,
    tflops: Optional[float] = None,
    baseline_ms: Optional[float] = None,
):
    """打印格式化的 benchmark 结果"""
    speedup = f"{baseline_ms / result['mean_ms']:.2f}x" if baseline_ms else "N/A"
    bw_str = f"{bandwidth_gb:.1f} GB/s" if bandwidth_gb else "N/A"
    tflops_str = f"{tflops:.3f}" if tflops else "N/A"
    print(
        f"{'operator':<12} {'impl':<12} {'mean(ms)':<10} {'min(ms)':<10} "
        f"{'BW(GB/s)':<12} {'TFLOPS':<10} {'speedup':<10}"
    )
    print("-" * 80)
    print(
        f"{operator:<12} {impl:<12} {result['mean_ms']:<10.4f} {result['min_ms']:<10.4f} "
        f"{bw_str:<12} {tflops_str:<10} {speedup:<10}"
    )


def get_gpu_info() -> str:
    """获取当前 GPU 信息"""
    if not torch.cuda.is_available():
        return "No GPU available"
    props = torch.cuda.get_device_properties(0)
    lines = [
        f"GPU: {props.name}",
        f"  SM count: {props.multi_processor_count}",
        f"  Memory: {props.total_memory/1e9:.1f} GB",
    ]
    # 部分属性在较新版本 PyTorch 中可能不存在
    for attr, label, scale, unit in [
        ("clock_rate", "Clock", 1e6, "GHz"),
        ("memory_clock_rate", "Memory clock", 1e6, "GHz"),
        ("memory_bus_width", "Memory bus", 1, "bit"),
    ]:
        if hasattr(props, attr):
            val = getattr(props, attr)
            if scale != 1:
                lines.append(f"  {label}: {val/scale:.2f} {unit}")
            else:
                lines.append(f"  {label}: {val} {unit}")
    return "\n".join(lines)
