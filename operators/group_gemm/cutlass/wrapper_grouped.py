"""
CUTLASS 3.x 高层 API Group GEMM wrapper（sm90a / Hopper）

注意：CUTLASS Grouped GEMM 约定 B 维度为 (N, K)（ColMajor* 表示 K 最快）。
Python 侧需传入 B(G, K, N)，本模块自动做 transpose → B^T(G, N, K) 再调 C 函数。
"""
import ctypes, os
import torch


def _load():
    p = os.path.join(os.path.dirname(__file__), "group_gemm_cutlass_hl.so")
    if not os.path.exists(p):
        return None
    lib = ctypes.CDLL(p)
    for fn in ["group_gemm_cutlass_hl_v1", "group_gemm_cutlass_hl_v2"]:
        f = getattr(lib, fn)
        f.argtypes = [
            ctypes.c_void_p,   # A   (G, M, K)
            ctypes.c_void_p,   # BT  (G, N, K)  = B.transpose(1,2).contiguous()
            ctypes.c_void_p,   # C   (G, M, N)
            ctypes.c_int,      # G
            ctypes.c_int,      # M
            ctypes.c_int,      # K
            ctypes.c_int,      # N
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
            "group_gemm_cutlass_hl.so not found. "
            "Run: CUDA_ARCH=sm_90 bash operators/group_gemm/cutlass/build_grouped.sh"
        )
    return _lib


def _call(fn: str, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    A: (G, M, K) float32 contiguous
    B: (G, K, N) float32 — transposed internally to BT(G, N, K)
    返回 C: (G, M, N) float32
    """
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only float32 supported"
    assert A.is_cuda and B.is_cuda
    G, M, K = A.shape
    G2, K2, N = B.shape
    assert G == G2 and K == K2, f"Shape mismatch: A({G},{M},{K}) B({G2},{K2},{N})"

    A  = A.contiguous()
    BT = B.transpose(1, 2).contiguous()   # (G, N, K) — K-major for CUTLASS ColumnMajor*
    C  = torch.empty(G, M, N, dtype=torch.float32, device=A.device)

    getattr(_get(), fn)(
        A.data_ptr(), BT.data_ptr(), C.data_ptr(),
        ctypes.c_int(G), ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(N),
    )
    return C


def group_gemm_cutlass_hl_v1(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """CUTLASS 3.x Grouped GEMM V1: KernelPtrArrayTmaWarpSpecializedCooperative (TileShape 128×128×32)"""
    return _call("group_gemm_cutlass_hl_v1", A, B)


def group_gemm_cutlass_hl_v2(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """CUTLASS 3.x Grouped GEMM V2: KernelPtrArrayTmaWarpSpecializedPingpong (TileShape 64×128×32)"""
    return _call("group_gemm_cutlass_hl_v2", A, B)
