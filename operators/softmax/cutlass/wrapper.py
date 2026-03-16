import ctypes, os, torch

def _load():
    p = os.path.join(os.path.dirname(__file__), "softmax_cutlass.so")
    if not os.path.exists(p):
        return None
    lib = ctypes.CDLL(p)
    for fn in ["softmax_cutlass_v1", "softmax_cutlass_v2"]:
        f = getattr(lib, fn)
        f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        f.restype = None
    return lib

_lib = None
def _get():
    global _lib
    if _lib is None: _lib = _load()
    if _lib is None: raise RuntimeError("softmax_cutlass.so not found. Run: CUDA_ARCH=sm_90 bash operators/softmax/cutlass/build.sh")
    return _lib

def softmax_cutlass_v1(X: torch.Tensor) -> torch.Tensor:
    """CuTe v1: 两趟扫描 + CuTe Tensor 包装"""
    B, N = X.shape
    Y = torch.empty_like(X)
    _get().softmax_cutlass_v1(X.data_ptr(), Y.data_ptr(), ctypes.c_int(B), ctypes.c_int(N))
    return Y

def softmax_cutlass_v2(X: torch.Tensor) -> torch.Tensor:
    """CuTe v2: Online softmax + Warp shuffle"""
    B, N = X.shape
    Y = torch.empty_like(X)
    _get().softmax_cutlass_v2(X.data_ptr(), Y.data_ptr(), ctypes.c_int(B), ctypes.c_int(N))
    return Y
