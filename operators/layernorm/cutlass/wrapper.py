import ctypes, os, torch

def _load():
    p = os.path.join(os.path.dirname(__file__), "layernorm_cutlass.so")
    if not os.path.exists(p):
        return None
    lib = ctypes.CDLL(p)
    for fn in ["layernorm_cutlass_v1", "layernorm_cutlass_v2", "layernorm_cutlass_v3"]:
        f = getattr(lib, fn)
        f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                      ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float]
        f.restype = None
    return lib

_lib = None
def _get():
    global _lib
    if _lib is None: _lib = _load()
    if _lib is None: raise RuntimeError("layernorm_cutlass.so not found. Run: CUDA_ARCH=sm_90 bash operators/layernorm/cutlass/build.sh")
    return _lib

def _call(fn, X, weight, bias):
    B, N = X.shape
    Y = torch.empty_like(X)
    w = weight.data_ptr() if weight is not None else 0
    b = bias.data_ptr()   if bias   is not None else 0
    getattr(_get(), fn)(
        X.data_ptr(), ctypes.c_void_p(w), ctypes.c_void_p(b),
        Y.data_ptr(), ctypes.c_int(B), ctypes.c_int(N), ctypes.c_float(1e-5)
    )
    return Y

def layernorm_cutlass_v1(X, weight=None, bias=None):
    """CuTe v1: Two-pass + CuTe Tensor 包装"""
    return _call("layernorm_cutlass_v1", X, weight, bias)

def layernorm_cutlass_v2(X, weight=None, bias=None):
    """CuTe v2: Welford + Warp Reduction + CuTe Tensor"""
    return _call("layernorm_cutlass_v2", X, weight, bias)

def layernorm_cutlass_v3(X, weight=None, bias=None):
    """CuTe v3: float4/LDG.128 + 寄存器缓存 x/w/b + 两路独立 reduce"""
    return _call("layernorm_cutlass_v3", X, weight, bias)
