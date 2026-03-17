"""
统一 Benchmark 入口

运行方式：
    python benchmarks/benchmark.py                     # 运行所有算子
    python benchmarks/benchmark.py --op matmul         # 只运行指定算子
    python benchmarks/benchmark.py --save              # 保存结果到 results/
"""

import sys
import os
import argparse
import json
import datetime
import ctypes

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))

import torch
from common.utils import benchmark_func, compute_bandwidth, compute_tflops, get_gpu_info

ROOT = os.path.join(os.path.dirname(__file__), "../")


# -------------------------------------------------------------------------
# 辅助：加载各算子的 CUDA .so
# -------------------------------------------------------------------------
def _load_so(rel_path, fn_specs):
    """加载 .so 并配置函数签名。fn_specs: {fn_name: (argtypes, restype)}"""
    path = os.path.join(ROOT, rel_path)
    if not os.path.exists(path):
        return None
    lib = ctypes.CDLL(path)
    for fn_name, (argtypes, restype) in fn_specs.items():
        f = getattr(lib, fn_name)
        f.argtypes = argtypes
        f.restype = restype
    return lib


# -------------------------------------------------------------------------
# vector_add
# -------------------------------------------------------------------------
def benchmark_vector_add():
    from operators.vector_add.pytorch.baseline import vector_add_pytorch
    from operators.vector_add.triton.kernel import vector_add_triton

    _ptr3_int = ([ctypes.c_void_p] * 3 + [ctypes.c_int], None)
    cuda_lib = _load_so("operators/vector_add/cuda/vector_add.so", {
        "vector_add_cuda_v1": _ptr3_int,
        "vector_add_cuda_v2": _ptr3_int,
    })
    cutlass_lib = _load_so("operators/vector_add/cutlass/vector_add_cutlass.so", {
        "vector_add_cutlass_v1": _ptr3_int,
        "vector_add_cutlass_v2": _ptr3_int,
    })

    def run_cuda(lib, fn, A, B):
        C = torch.empty_like(A)
        getattr(lib, fn)(A.data_ptr(), B.data_ptr(), C.data_ptr(), ctypes.c_int(A.numel()))
        return C

    results = {}
    for N in [1024*1024, 1024*1024*16, 1024*1024*64]:
        A = torch.randn(N, device="cuda")
        B = torch.randn(N, device="cuda")
        bytes_accessed = N * 3 * 4
        key = f"N={N//1024//1024}M"
        results[key] = {}

        impls = [
            ("pytorch",        lambda: vector_add_pytorch(A, B)),
            ("triton",         lambda: vector_add_triton(A, B)),
        ]
        if cuda_lib:
            impls += [
                ("cuda_v1",    lambda: run_cuda(cuda_lib, "vector_add_cuda_v1", A, B)),
                ("cuda_v2_f4", lambda: run_cuda(cuda_lib, "vector_add_cuda_v2", A, B)),
            ]
        if cutlass_lib:
            impls += [
                ("cute_v1",    lambda: run_cuda(cutlass_lib, "vector_add_cutlass_v1", A, B)),
                ("cute_v2",    lambda: run_cuda(cutlass_lib, "vector_add_cutlass_v2", A, B)),
            ]

        for name, fn in impls:
            r = benchmark_func(fn)
            results[key][name] = {
                "mean_ms": r["mean_ms"],
                "bw_gbs": compute_bandwidth(bytes_accessed, r["mean_ms"]),
            }
    return results


# -------------------------------------------------------------------------
# transpose
# -------------------------------------------------------------------------
def benchmark_transpose():
    from operators.transpose.pytorch.baseline import transpose_pytorch
    from operators.transpose.triton.kernel import transpose_triton

    _ptr2_int2 = ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int], None)
    cuda_lib = _load_so("operators/transpose/cuda/transpose.so", {
        "transpose_cuda_v1": _ptr2_int2,
        "transpose_cuda_v2": _ptr2_int2,
    })

    cutlass_lib = _load_so("operators/transpose/cutlass/transpose_cutlass.so", {
        "transpose_cutlass_v1": _ptr2_int2,
        "transpose_cutlass_v2": _ptr2_int2,
    })

    def run_cuda(lib, fn, A):
        M, N = A.shape
        B = torch.empty(N, M, dtype=A.dtype, device=A.device)
        getattr(lib, fn)(A.data_ptr(), B.data_ptr(), ctypes.c_int(M), ctypes.c_int(N))
        return B

    results = {}
    for M, N in [(1024, 1024), (4096, 4096), (8192, 8192)]:
        A = torch.randn(M, N, device="cuda")
        bytes_accessed = M * N * 4 * 2
        key = f"{M}x{N}"
        results[key] = {}

        impls = [
            ("pytorch", lambda: transpose_pytorch(A)),
            ("triton",  lambda: transpose_triton(A)),
        ]
        if cuda_lib:
            impls += [
                ("cuda_v1", lambda: run_cuda(cuda_lib, "transpose_cuda_v1", A)),
                ("cuda_v2", lambda: run_cuda(cuda_lib, "transpose_cuda_v2", A)),
            ]
        if cutlass_lib:
            impls += [
                ("cute_v1", lambda: run_cuda(cutlass_lib, "transpose_cutlass_v1", A)),
                ("cute_v2", lambda: run_cuda(cutlass_lib, "transpose_cutlass_v2", A)),
            ]

        for name, fn in impls:
            r = benchmark_func(fn)
            results[key][name] = {
                "mean_ms": r["mean_ms"],
                "bw_gbs": compute_bandwidth(bytes_accessed, r["mean_ms"]),
            }
    return results


# -------------------------------------------------------------------------
# softmax
# -------------------------------------------------------------------------
def benchmark_softmax():
    from operators.softmax.pytorch.baseline import softmax_pytorch
    from operators.softmax.triton.kernel import softmax_triton

    _ptr2_int2 = ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int], None)
    cuda_lib = _load_so("operators/softmax/cuda/softmax.so", {
        "softmax_cuda_v1": _ptr2_int2,
        "softmax_cuda_v2": _ptr2_int2,
        "softmax_cuda_v3": _ptr2_int2,
    })

    cutlass_lib = _load_so("operators/softmax/cutlass/softmax_cutlass.so", {
        "softmax_cutlass_v1": _ptr2_int2,
        "softmax_cutlass_v2": _ptr2_int2,
    })

    def run_cuda(lib, fn, X):
        B, N = X.shape
        Y = torch.empty_like(X)
        getattr(lib, fn)(X.data_ptr(), Y.data_ptr(), ctypes.c_int(B), ctypes.c_int(N))
        return Y

    results = {}
    for B, N in [(1024, 512), (4096, 2048), (4096, 8192)]:
        X = torch.randn(B, N, device="cuda")
        bytes_accessed = B * N * 3 * 4
        key = f"B={B},N={N}"
        results[key] = {}

        impls = [
            ("pytorch", lambda: softmax_pytorch(X)),
            ("triton",  lambda: softmax_triton(X)),
        ]
        if cuda_lib:
            impls += [
                ("cuda_v1", lambda: run_cuda(cuda_lib, "softmax_cuda_v1", X)),
                ("cuda_v2", lambda: run_cuda(cuda_lib, "softmax_cuda_v2", X)),
                ("cuda_v3", lambda: run_cuda(cuda_lib, "softmax_cuda_v3", X)),
            ]
        if cutlass_lib:
            impls += [
                ("cute_v1", lambda: run_cuda(cutlass_lib, "softmax_cutlass_v1", X)),
                ("cute_v2", lambda: run_cuda(cutlass_lib, "softmax_cutlass_v2", X)),
            ]

        for name, fn in impls:
            r = benchmark_func(fn)
            results[key][name] = {
                "mean_ms": r["mean_ms"],
                "bw_gbs": compute_bandwidth(bytes_accessed, r["mean_ms"]),
            }
    return results


# -------------------------------------------------------------------------
# layernorm
# -------------------------------------------------------------------------
def benchmark_layernorm():
    from operators.layernorm.pytorch.baseline import layernorm_pytorch
    from operators.layernorm.triton.kernel import layernorm_triton

    _ln_args = ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float], None)
    cuda_lib = _load_so("operators/layernorm/cuda/layernorm.so", {
        "layernorm_cuda_v1": _ln_args,
        "layernorm_cuda_v2": _ln_args,
    })

    cutlass_lib = _load_so("operators/layernorm/cutlass/layernorm_cutlass.so", {
        "layernorm_cutlass_v1": _ln_args,
        "layernorm_cutlass_v2": _ln_args,
    })

    def run_cuda(lib, fn, X, W, b):
        B, N = X.shape
        Y = torch.empty_like(X)
        w_ptr = W.data_ptr() if W is not None else 0
        b_ptr = b.data_ptr() if b is not None else 0
        getattr(lib, fn)(
            X.data_ptr(), ctypes.c_void_p(w_ptr), ctypes.c_void_p(b_ptr),
            Y.data_ptr(), ctypes.c_int(B), ctypes.c_int(N), ctypes.c_float(1e-5)
        )
        return Y

    results = {}
    for B, N in [(4096, 512), (4096, 1024), (4096, 4096)]:
        X = torch.randn(B, N, device="cuda")
        W = torch.ones(N, device="cuda")
        b = torch.zeros(N, device="cuda")
        bytes_accessed = B * N * 4 * 4
        key = f"B={B},N={N}"
        results[key] = {}

        impls = [
            ("pytorch", lambda: layernorm_pytorch(X, W, b)),
            ("triton",  lambda: layernorm_triton(X, W, b)),
        ]
        if cuda_lib:
            impls += [
                ("cuda_v1", lambda: run_cuda(cuda_lib, "layernorm_cuda_v1", X, W, b)),
                ("cuda_v2", lambda: run_cuda(cuda_lib, "layernorm_cuda_v2", X, W, b)),
            ]
        if cutlass_lib:
            impls += [
                ("cute_v1", lambda: run_cuda(cutlass_lib, "layernorm_cutlass_v1", X, W, b)),
                ("cute_v2", lambda: run_cuda(cutlass_lib, "layernorm_cutlass_v2", X, W, b)),
            ]

        for name, fn in impls:
            r = benchmark_func(fn)
            results[key][name] = {
                "mean_ms": r["mean_ms"],
                "bw_gbs": compute_bandwidth(bytes_accessed, r["mean_ms"]),
            }
    return results


# -------------------------------------------------------------------------
# matmul
# -------------------------------------------------------------------------
def benchmark_matmul():
    from operators.matmul.pytorch.baseline import matmul_pytorch
    from operators.matmul.triton.kernel import matmul_triton

    _mm_args = ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_int, ctypes.c_int, ctypes.c_int], None)
    cuda_lib = _load_so("operators/matmul/cuda/matmul.so", {
        "matmul_cuda_v1": _mm_args,
        "matmul_cuda_v2": _mm_args,
        "matmul_cuda_v3": _mm_args,
    })

    cutlass_lib = _load_so("operators/matmul/cutlass/matmul_cutlass.so", {
        "matmul_cutlass_v1": _mm_args,
        "matmul_cutlass_v2": _mm_args,
    })
    cutlass_hl_lib = _load_so("operators/matmul/cutlass/matmul_cutlass_hl.so", {
        "matmul_cutlass_hl_v1": _mm_args,
        "matmul_cutlass_hl_v2": _mm_args,
    })

    def run_cuda(lib, fn, A, B):
        M, K = A.shape
        _, N = B.shape
        C = torch.empty(M, N, dtype=A.dtype, device=A.device)
        getattr(lib, fn)(
            A.data_ptr(), B.data_ptr(), C.data_ptr(),
            ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(N)
        )
        return C

    results = {}
    for M, K, N in [(512, 512, 512), (2048, 2048, 2048), (4096, 4096, 4096)]:
        A = torch.randn(M, K, device="cuda")
        B = torch.randn(K, N, device="cuda")
        flops = 2 * M * N * K
        key = f"{M}x{K}x{N}"
        results[key] = {}

        impls = [
            ("pytorch (cuBLAS)", lambda: matmul_pytorch(A, B)),
            ("triton",           lambda: matmul_triton(A, B)),
        ]
        if cuda_lib:
            impls += [
                ("cuda_v1", lambda: run_cuda(cuda_lib, "matmul_cuda_v1", A, B)),
                ("cuda_v2", lambda: run_cuda(cuda_lib, "matmul_cuda_v2", A, B)),
                ("cuda_v3", lambda: run_cuda(cuda_lib, "matmul_cuda_v3", A, B)),
            ]
        if cutlass_lib:
            impls += [
                ("cute_v1", lambda: run_cuda(cutlass_lib, "matmul_cutlass_v1", A, B)),
                ("cute_v2", lambda: run_cuda(cutlass_lib, "matmul_cutlass_v2", A, B)),
            ]
        if cutlass_hl_lib:
            impls += [
                ("cutlass_hl_v1", lambda: run_cuda(cutlass_hl_lib, "matmul_cutlass_hl_v1", A, B)),
                ("cutlass_hl_v2", lambda: run_cuda(cutlass_hl_lib, "matmul_cutlass_hl_v2", A, B)),
            ]

        for name, fn in impls:
            r = benchmark_func(fn)
            results[key][name] = {
                "mean_ms": r["mean_ms"],
                "tflops": compute_tflops(flops, r["mean_ms"]),
            }
    return results


# -------------------------------------------------------------------------
# attention
# -------------------------------------------------------------------------
def benchmark_attention():
    from operators.attention.pytorch.baseline import attention_pytorch
    from operators.attention.triton.kernel import flash_attention_triton

    cutlass_lib_available = False
    try:
        from operators.attention.cutlass.wrapper import flash_attention_cutlass_v1, flash_attention_cutlass_v2
        cutlass_lib_available = True
    except RuntimeError:
        pass

    results = {}
    for B, H, N, D in [(4, 8, 512, 64), (4, 8, 1024, 64), (1, 8, 4096, 64)]:
        Q = torch.randn(B, H, N, D, device="cuda")
        K = torch.randn(B, H, N, D, device="cuda")
        V = torch.randn(B, H, N, D, device="cuda")
        flops = 4 * B * H * N * N * D
        key = f"B={B},H={H},N={N},D={D}"
        results[key] = {}

        impls = [
            ("pytorch (naive)", lambda: attention_pytorch(Q, K, V)),
            ("triton flash",    lambda: flash_attention_triton(Q, K, V)),
        ]
        if cutlass_lib_available:
            impls += [
                ("cute_v1", lambda: flash_attention_cutlass_v1(Q, K, V)),
                ("cute_v2", lambda: flash_attention_cutlass_v2(Q, K, V)),
            ]

        for name, fn in impls:
            r = benchmark_func(fn)
            results[key][name] = {
                "mean_ms": r["mean_ms"],
                "tflops": compute_tflops(flops, r["mean_ms"]),
            }
    return results


# -------------------------------------------------------------------------
# rms_norm
# -------------------------------------------------------------------------
def benchmark_rms_norm():
    from operators.rms_norm.pytorch.baseline import rms_norm_pytorch
    from operators.rms_norm.triton.kernel import rms_norm_triton

    _rn_args = ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                 ctypes.c_int, ctypes.c_int, ctypes.c_float], None)
    cuda_lib = _load_so("operators/rms_norm/cuda/rms_norm.so", {
        "rms_norm_cuda_v1": _rn_args,
        "rms_norm_cuda_v2": _rn_args,
    })
    cutlass_lib = _load_so("operators/rms_norm/cutlass/rms_norm_cutlass.so", {
        "rms_norm_cutlass_v1": _rn_args,
        "rms_norm_cutlass_v2": _rn_args,
    })

    def run_cuda(lib, fn, x, w):
        B, N = x.shape
        y = torch.empty_like(x)
        getattr(lib, fn)(
            x.data_ptr(), w.data_ptr(), y.data_ptr(),
            ctypes.c_int(B), ctypes.c_int(N), ctypes.c_float(1e-6)
        )
        return y

    results = {}
    for B, N in [(1024, 512), (4096, 4096), (4096, 8192)]:
        x = torch.randn(B, N, device="cuda")
        w = torch.randn(N, device="cuda")
        # 2 reads (x, w) + 1 write (y) = 3 * B * N * 4 bytes
        bytes_accessed = B * N * 3 * 4
        key = f"B={B},N={N}"
        results[key] = {}

        impls = [
            ("pytorch",    lambda: rms_norm_pytorch(x, w, 1e-6)),
            ("triton",     lambda: rms_norm_triton(x, w, 1e-6)),
        ]
        if cuda_lib:
            impls += [
                ("cuda_v1", lambda: run_cuda(cuda_lib, "rms_norm_cuda_v1", x, w)),
                ("cuda_v2", lambda: run_cuda(cuda_lib, "rms_norm_cuda_v2", x, w)),
            ]
        if cutlass_lib:
            impls += [
                ("cute_v1", lambda: run_cuda(cutlass_lib, "rms_norm_cutlass_v1", x, w)),
                ("cute_v2", lambda: run_cuda(cutlass_lib, "rms_norm_cutlass_v2", x, w)),
            ]

        for name, fn in impls:
            r = benchmark_func(fn)
            results[key][name] = {
                "mean_ms": r["mean_ms"],
                "bw_gbs": compute_bandwidth(bytes_accessed, r["mean_ms"]),
            }
    return results


# -------------------------------------------------------------------------
# rope
# -------------------------------------------------------------------------
def benchmark_rope():
    from operators.rope.pytorch.baseline import build_cos_sin_cache, apply_rope_pytorch
    from operators.rope.triton.kernel import rope_triton

    _rope_args = ([ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_void_p,
                   ctypes.c_int, ctypes.c_int, ctypes.c_int], None)
    cuda_lib = _load_so("operators/rope/cuda/rope.so", {
        "rope_cuda_v1": _rope_args,
        "rope_cuda_v2": _rope_args,
    })
    cutlass_lib = _load_so("operators/rope/cutlass/rope_cutlass.so", {
        "rope_cutlass_v1": _rope_args,
        "rope_cutlass_v2": _rope_args,
    })

    def run_cuda(lib, fn, q, k, cos_cache, sin_cache, positions):
        q_out = q.clone()
        k_out = k.clone()
        seq_len, num_heads, head_dim = q.shape
        pos32 = positions.to(torch.int32).contiguous()
        getattr(lib, fn)(
            q_out.data_ptr(), k_out.data_ptr(),
            cos_cache.data_ptr(), sin_cache.data_ptr(),
            pos32.data_ptr(),
            ctypes.c_int(seq_len), ctypes.c_int(num_heads), ctypes.c_int(head_dim)
        )
        return q_out, k_out

    results = {}
    for seq_len, num_heads, head_dim in [(512, 8, 64), (2048, 32, 64), (4096, 32, 64)]:
        cos_cache, sin_cache = build_cos_sin_cache(seq_len + 64, head_dim, device="cuda")
        q = torch.randn(seq_len, num_heads, head_dim, device="cuda")
        k = torch.randn(seq_len, num_heads, head_dim, device="cuda")
        positions = torch.arange(seq_len, device="cuda")
        # Q + K: 2 reads + 2 writes = 4 * seq_len * num_heads * head_dim * 4 bytes
        bytes_accessed = 4 * seq_len * num_heads * head_dim * 4
        key = f"seq={seq_len},h={num_heads},d={head_dim}"
        results[key] = {}

        impls = [
            ("pytorch",    lambda: apply_rope_pytorch(q, k, cos_cache, sin_cache, positions)),
            ("triton_v1",  lambda: rope_triton(q, k, cos_cache, sin_cache, positions)),
        ]
        if cuda_lib:
            impls += [
                ("cuda_v1", lambda: run_cuda(cuda_lib, "rope_cuda_v1", q, k, cos_cache, sin_cache, positions)),
                ("cuda_v2", lambda: run_cuda(cuda_lib, "rope_cuda_v2", q, k, cos_cache, sin_cache, positions)),
            ]
        if cutlass_lib:
            impls += [
                ("cute_v1", lambda: run_cuda(cutlass_lib, "rope_cutlass_v1", q, k, cos_cache, sin_cache, positions)),
                ("cute_v2", lambda: run_cuda(cutlass_lib, "rope_cutlass_v2", q, k, cos_cache, sin_cache, positions)),
            ]

        for name, fn in impls:
            r = benchmark_func(fn)
            results[key][name] = {
                "mean_ms": r["mean_ms"],
                "bw_gbs": compute_bandwidth(bytes_accessed, r["mean_ms"]),
            }
    return results


# -------------------------------------------------------------------------
# 打印 & 主入口
# -------------------------------------------------------------------------
BENCHMARKS = {
    "vector_add": benchmark_vector_add,
    "transpose":  benchmark_transpose,
    "softmax":    benchmark_softmax,
    "layernorm":  benchmark_layernorm,
    "matmul":     benchmark_matmul,
    "attention":  benchmark_attention,
    "rms_norm":   benchmark_rms_norm,
    "rope":       benchmark_rope,
}


def print_results(op_name, results):
    print(f"\n{'='*60}")
    print(f"  {op_name.upper()}")
    print(f"{'='*60}")
    for config, impls in results.items():
        print(f"\n  Config: {config}")
        baseline_ms = None
        for impl, metrics in impls.items():
            ms = metrics["mean_ms"]
            if baseline_ms is None:
                baseline_ms = ms
            speedup = baseline_ms / ms
            extra = ""
            if "bw_gbs" in metrics:
                extra = f"  BW={metrics['bw_gbs']:.1f} GB/s"
            elif "tflops" in metrics:
                extra = f"  {metrics['tflops']:.2f} TFLOPS"
            print(f"    {impl:<22} {ms:.4f} ms  speedup={speedup:.2f}x{extra}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=str, default="all",
                        help="Operator to benchmark (all, vector_add, transpose, ...)")
    parser.add_argument("--save", action="store_true",
                        help="Save results to results/ directory")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA GPU available!")
        return

    print(get_gpu_info())

    ops_to_run = list(BENCHMARKS.keys()) if args.op == "all" else [args.op]
    all_results = {}

    for op in ops_to_run:
        if op not in BENCHMARKS:
            print(f"Unknown operator: {op}")
            continue
        print(f"\nBenchmarking {op}...")
        try:
            results = BENCHMARKS[op]()
            all_results[op] = results
            print_results(op, results)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    if args.save:
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"results/benchmark_{timestamp}.json"
        with open(fname, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {fname}")


if __name__ == "__main__":
    main()
