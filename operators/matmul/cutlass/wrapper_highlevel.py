import ctypes, os, torch

def _load():
    p = os.path.join(os.path.dirname(__file__), "matmul_cutlass_hl.so")
    if not os.path.exists(p):
        return None
    lib = ctypes.CDLL(p)
    _args = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_int, ctypes.c_int, ctypes.c_int]
    for fn in ["matmul_cutlass_hl_v1", "matmul_cutlass_hl_v2"]:
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
            "matmul_cutlass_hl.so not found. Run: "
            "CUDA_ARCH=sm_90 bash operators/matmul/cutlass/build_highlevel.sh"
        )
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

def matmul_cutlass_hl_v1(A, B):
    """CUTLASS 高层 API v1: TileShape 128×128×32, TmaWarpSpecializedCooperative"""
    return _call("matmul_cutlass_hl_v1", A, B)

def matmul_cutlass_hl_v2(A, B):
    """CUTLASS 高层 API v2: TileShape 64×128×32, TmaWarpSpecializedPingpong"""
    return _call("matmul_cutlass_hl_v2", A, B)
