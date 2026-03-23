import ctypes, os, torch

def _load():
    p = os.path.join(os.path.dirname(__file__), "rope_cutlass.so")
    if not os.path.exists(p):
        return None
    lib = ctypes.CDLL(p)
    # void fn(float* q, float* k, float* cos, float* sin, int* pos,
    #         int seq_len, int num_heads, int head_dim)
    _args = [ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_void_p,
             ctypes.c_int, ctypes.c_int, ctypes.c_int]
    for fn in ["rope_cutlass_v1", "rope_cutlass_v2", "rope_cutlass_v3"]:
        f = getattr(lib, fn)
        f.argtypes = _args
        f.restype = None
    return lib

_lib = None
def _get():
    global _lib
    if _lib is None: _lib = _load()
    if _lib is None:
        raise RuntimeError(
            "rope_cutlass.so not found. Run: "
            "CUDA_ARCH=sm_90 bash operators/rope/cutlass/build.sh"
        )
    return _lib

def _call(fn, q, k, cos_cache, sin_cache, positions):
    seq_len, num_heads, head_dim = q.shape
    q_out = q.clone()
    k_out = k.clone()
    pos32 = positions.to(torch.int32).contiguous()
    getattr(_get(), fn)(
        q_out.data_ptr(), k_out.data_ptr(),
        cos_cache.data_ptr(), sin_cache.data_ptr(),
        pos32.data_ptr(),
        ctypes.c_int(seq_len), ctypes.c_int(num_heads), ctypes.c_int(head_dim)
    )
    return q_out, k_out

def rope_cutlass_v1(q, k, cos_cache, sin_cache, positions):
    """CuTe v1: 3D Tensor + local_tile"""
    return _call("rope_cutlass_v1", q, k, cos_cache, sin_cache, positions)

def rope_cutlass_v2(q, k, cos_cache, sin_cache, positions):
    """CuTe v2: float2 向量化"""
    return _call("rope_cutlass_v2", q, k, cos_cache, sin_cache, positions)

def rope_cutlass_v3(q, k, cos_cache, sin_cache, positions):
    """CuTe v3: shared cos/sin + all-heads-per-block (1 block/token)"""
    return _call("rope_cutlass_v3", q, k, cos_cache, sin_cache, positions)
