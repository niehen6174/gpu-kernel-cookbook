import ctypes, os, torch


def _load():
    p = os.path.join(os.path.dirname(__file__), "group_gemm_cutlass.so")
    if not os.path.exists(p):
        return None
    lib = ctypes.CDLL(p)
    for fn in ["group_gemm_cutlass_v1", "group_gemm_cutlass_v2"]:
        f = getattr(lib, fn)
        f.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        f.restype = None
    return lib


_lib = None


def _get():
    global _lib
    if _lib is None:
        _lib = _load()
    if _lib is None:
        raise RuntimeError(
            "group_gemm_cutlass.so not found. "
            "Run: CUDA_ARCH=sm_90 bash operators/group_gemm/cutlass/build.sh"
        )
    return _lib


def _call(fn, A, B):
    G, M, K = A.shape
    G2, K2, N = B.shape
    assert G == G2 and K == K2
    C = torch.empty(G, M, N, dtype=A.dtype, device=A.device)
    getattr(_get(), fn)(
        A.data_ptr(), B.data_ptr(), C.data_ptr(),
        ctypes.c_int(G), ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(N),
    )
    return C


def group_gemm_cutlass_v1(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """CuTe v1: 3D Tensor 直接索引"""
    return _call("group_gemm_cutlass_v1", A, B)


def group_gemm_cutlass_v2(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """CuTe v2: Shared memory tiling"""
    return _call("group_gemm_cutlass_v2", A, B)
