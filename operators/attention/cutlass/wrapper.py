import ctypes, os, torch

def _load():
    p = os.path.join(os.path.dirname(__file__), "attention_cutlass.so")
    if not os.path.exists(p):
        return None
    lib = ctypes.CDLL(p)

    # V1: flash_attention_cutlass_v1(Q, K, V, O, B, H, N, d)
    f = lib.flash_attention_cutlass_v1
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                  ctypes.c_void_p,
                  ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    f.restype = None

    # V2: flash_attention_cutlass_v2(Q, K, V, O, B, H, N, d, causal)
    f = lib.flash_attention_cutlass_v2
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                  ctypes.c_void_p,
                  ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                  ctypes.c_int]
    f.restype = None
    return lib

_lib = None
def _get():
    global _lib
    if _lib is None: _lib = _load()
    if _lib is None: raise RuntimeError("attention_cutlass.so not found. Run: CUDA_ARCH=sm_90 bash operators/attention/cutlass/build.sh")
    return _lib


def flash_attention_cutlass_v1(Q, K, V):
    """CuTe V1: 单头串行 Flash Attention，清晰展示 CuTe local_tile"""
    B, H, N, D = Q.shape
    O = torch.empty_like(Q)
    _get().flash_attention_cutlass_v1(
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(),
        ctypes.c_int(B), ctypes.c_int(H), ctypes.c_int(N), ctypes.c_int(D)
    )
    return O


def flash_attention_cutlass_v2(Q, K, V, causal=False):
    """CuTe V2: 多头并行 Flash Attention + causal mask 支持"""
    B, H, N, D = Q.shape
    O = torch.empty_like(Q)
    _get().flash_attention_cutlass_v2(
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(),
        ctypes.c_int(B), ctypes.c_int(H), ctypes.c_int(N), ctypes.c_int(D),
        ctypes.c_int(1 if causal else 0)
    )
    return O
