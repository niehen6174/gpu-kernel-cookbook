"""
Vector Add — CuTe C++ 实现的 Python 封装

通过 ctypes 调用编译好的 vector_add_cutlass.so
"""

import ctypes
import os
import torch


def _load_lib():
    lib_path = os.path.join(os.path.dirname(__file__), "vector_add_cutlass.so")
    if not os.path.exists(lib_path):
        return None
    lib = ctypes.CDLL(lib_path)
    for fn in ["vector_add_cutlass_v1", "vector_add_cutlass_v2"]:
        f = getattr(lib, fn)
        f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        f.restype = None
    return lib


_lib = None


def _get_lib():
    global _lib
    if _lib is None:
        _lib = _load_lib()
    return _lib


def vector_add_cutlass_v1(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """CuTe v1：直接索引（展示 Tensor / local_tile / local_partition）"""
    lib = _get_lib()
    if lib is None:
        raise RuntimeError(
            "vector_add_cutlass.so not found. Run: "
            "CUDA_ARCH=sm_90 bash operators/vector_add/cutlass/build.sh"
        )
    C = torch.empty_like(A)
    lib.vector_add_cutlass_v1(
        A.data_ptr(), B.data_ptr(), C.data_ptr(), ctypes.c_int(A.numel())
    )
    return C


def vector_add_cutlass_v2(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """CuTe v2：128-bit 向量化（展示 Copy Atom / uint128_t）"""
    lib = _get_lib()
    if lib is None:
        raise RuntimeError(
            "vector_add_cutlass.so not found. Run: "
            "CUDA_ARCH=sm_90 bash operators/vector_add/cutlass/build.sh"
        )
    C = torch.empty_like(A)
    lib.vector_add_cutlass_v2(
        A.data_ptr(), B.data_ptr(), C.data_ptr(), ctypes.c_int(A.numel())
    )
    return C
