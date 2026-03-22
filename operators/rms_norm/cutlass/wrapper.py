import ctypes, os, torch

def _load():
    p = os.path.join(os.path.dirname(__file__), "rms_norm_cutlass.so")
    if not os.path.exists(p):
        return None
    lib = ctypes.CDLL(p)
    _rn_args    = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_int, ctypes.c_int, ctypes.c_float]
    _fused_args = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_int, ctypes.c_int, ctypes.c_float]
    for fn in ["rms_norm_cutlass_v1", "rms_norm_cutlass_v2", "rms_norm_cutlass_v3"]:
        f = getattr(lib, fn)
        f.argtypes = _rn_args
        f.restype = None
    for fn in ["fused_add_rms_norm_cutlass_v1", "fused_add_rms_norm_cutlass_v3"]:
        f = getattr(lib, fn)
        f.argtypes = _fused_args
        f.restype = None
    return lib

_lib = None
def _get():
    global _lib
    if _lib is None: _lib = _load()
    if _lib is None:
        raise RuntimeError(
            "rms_norm_cutlass.so not found. Run: "
            "CUDA_ARCH=sm_90 bash operators/rms_norm/cutlass/build.sh"
        )
    return _lib

def _call(fn, x, weight, eps=1e-6):
    orig_shape = x.shape
    N = x.shape[-1]
    x2 = x.reshape(-1, N)
    B = x2.shape[0]
    y = torch.empty_like(x2)
    getattr(_get(), fn)(
        x2.data_ptr(), weight.data_ptr(), y.data_ptr(),
        ctypes.c_int(B), ctypes.c_int(N), ctypes.c_float(eps)
    )
    return y.reshape(orig_shape)

def _call_fused(fn, x, residual, weight, eps=1e-6):
    orig_shape = x.shape
    N = x.shape[-1]
    x2 = x.reshape(-1, N).contiguous()
    r2 = residual.reshape(-1, N)   # modified inplace
    B = x2.shape[0]
    y = torch.empty_like(x2)
    getattr(_get(), fn)(
        x2.data_ptr(), r2.data_ptr(), weight.data_ptr(), y.data_ptr(),
        ctypes.c_int(B), ctypes.c_int(N), ctypes.c_float(eps)
    )
    return y.reshape(orig_shape), r2.reshape(orig_shape)

def rms_norm_cutlass_v1(x, weight, eps=1e-6):
    """CuTe v1: Two-pass + CuTe Tensor 包装"""
    return _call("rms_norm_cutlass_v1", x, weight, eps)

def rms_norm_cutlass_v2(x, weight, eps=1e-6):
    """CuTe v2: float4 向量化 + CuTe Tensor"""
    return _call("rms_norm_cutlass_v2", x, weight, eps)

def rms_norm_cutlass_v3(x, weight, eps=1e-6):
    """CuTe v3: LDG.128 + 寄存器缓存 x/w（借鉴 sglang norm_fusion）"""
    return _call("rms_norm_cutlass_v3", x, weight, eps)

def fused_add_rms_norm_cutlass_v1(x, residual, weight, eps=1e-6):
    """CuTe fused v1: residual add + RMSNorm, scalar two-pass"""
    return _call_fused("fused_add_rms_norm_cutlass_v1", x, residual, weight, eps)

def fused_add_rms_norm_cutlass_v3(x, residual, weight, eps=1e-6):
    """CuTe fused v3: float4 + register cache updated residual"""
    return _call_fused("fused_add_rms_norm_cutlass_v3", x, residual, weight, eps)
