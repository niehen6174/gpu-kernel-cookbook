import ctypes, os, torch

def _load():
    p = os.path.join(os.path.dirname(__file__), "matmul_cutlass.so")
    if not os.path.exists(p):
        return None
    lib = ctypes.CDLL(p)
    for fn in ["matmul_cutlass_v1", "matmul_cutlass_v2"]:
        f = getattr(lib, fn)
        f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                      ctypes.c_int, ctypes.c_int, ctypes.c_int]
        f.restype = None
    return lib

_lib = None
def _get():
    global _lib
    if _lib is None: _lib = _load()
    if _lib is None: raise RuntimeError("matmul_cutlass.so not found. Run: CUDA_ARCH=sm_90 bash operators/matmul/cutlass/build.sh")
    return _lib

def _call(fn, A, B):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, dtype=A.dtype, device=A.device)
    getattr(_get(), fn)(
        A.data_ptr(), B.data_ptr(), C.data_ptr(),
        ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(N)
    )
    return C

def matmul_cutlass_v1(A, B):
    """CuTe v1: Naive，2D Tensor 直接索引"""
    return _call("matmul_cutlass_v1", A, B)

def matmul_cutlass_v2(A, B):
    """CuTe v2: Shared Memory Tiling + cute::copy"""
    return _call("matmul_cutlass_v2", A, B)
