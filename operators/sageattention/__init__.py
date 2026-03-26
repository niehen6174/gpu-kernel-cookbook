"""
SageAttention 算子包

包含 SageAttention v1 和 v2 的 Triton 实现：
  - v1: per-block INT8 Q/K + FP16 V + K Smoothing
  - v2: per-warp INT8 Q + per-block INT8 K + FP8 V + K/V Smoothing
"""
from .triton.kernel_v1 import sageattn_v1
from .triton.kernel_v2 import sageattn_v2

__all__ = ["sageattn_v1", "sageattn_v2"]
