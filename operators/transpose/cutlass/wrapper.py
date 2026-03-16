import ctypes, os, torch

def _load():
    p = os.path.join(os.path.dirname(__file__), "transpose_cutlass.so")
    if not os.path.exists(p):
        return None
    lib = ctypes.CDLL(p)
    for fn in ["transpose_cutlass_v1", "transpose_cutlass_v2"]:
        f = getattr(lib, fn)
        f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        f.restype = None
    return lib

_lib = None
def _get():
    global _lib
    if _lib is None: _lib = _load()
    if _lib is None: raise RuntimeError("transpose_cutlass.so not found. Run: CUDA_ARCH=sm_90 bash operators/transpose/cutlass/build.sh")
    return _lib

def transpose_cutlass_v1(A: torch.Tensor) -> torch.Tensor:
    """CuTe v1: 2D Tensor 直接索引（non-coalesced 写，教学用）"""
    M, N = A.shape
    B = torch.empty(N, M, dtype=A.dtype, device=A.device)
    _get().transpose_cutlass_v1(A.data_ptr(), B.data_ptr(), ctypes.c_int(M), ctypes.c_int(N))
    return B

def transpose_cutlass_v2(A: torch.Tensor) -> torch.Tensor:
    """CuTe v2: Shared Memory Tiling + bank conflict padding"""
    M, N = A.shape
    B = torch.empty(N, M, dtype=A.dtype, device=A.device)
    _get().transpose_cutlass_v2(A.data_ptr(), B.data_ptr(), ctypes.c_int(M), ctypes.c_int(N))
    return B
