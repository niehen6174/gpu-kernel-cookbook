"""
统一 NCU Profile Driver
=======================

ncu 会 capture 所有 CUDA kernel，但我们只关心目标算子的几次调用。
通过 cudaProfilerStart/Stop 精确框定范围，配合 ncu --target-processes all
即可只看目标 kernel，屏蔽 warmup 和无关调用。

用法（由 scripts/profile.sh 驱动，也可直接调用）：

  # 只 profile 某个算子的某个 kernel
  python profiling/profile_driver.py --op rms_norm --kernel cuda_v2

  # profile 一个算子的所有 kernel（逐个跑，分别 start/stop）
  python profiling/profile_driver.py --op rms_norm

  # 列出所有可用算子和 kernel
  python profiling/profile_driver.py --list

然后包在 ncu 里：

  ncu --target-processes all \\
      --section SpeedOfLight --section MemoryWorkloadAnalysis \\
      --kernel-name <kernel_name> \\
      python profiling/profile_driver.py --op rms_norm --kernel cuda_v2
"""

import sys
import os
import argparse
import ctypes
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.join(os.path.dirname(__file__), "..")


# =========================================================================
# cudaProfilerStart / Stop 封装
# =========================================================================
def profiler_start():
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()

def profiler_stop():
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


# =========================================================================
# 辅助：加载 .so
# =========================================================================
def _load_so(rel_path, fn_specs):
    path = os.path.join(ROOT, rel_path)
    if not os.path.exists(path):
        return None
    lib = ctypes.CDLL(path)
    for fn_name, (argtypes, restype) in fn_specs.items():
        f = getattr(lib, fn_name)
        f.argtypes = argtypes
        f.restype = restype
    return lib


# =========================================================================
# Kernel Registry
#
# 每个 op 是一个函数，接受 warmup=True/False，返回 {kernel_name: callable}。
# callable() 触发一次 kernel 调用（不含 synchronize）。
#
# 约定：
#   - warmup=True  时调用但不做 profiler start/stop
#   - warmup=False 时由框架在调用前后插入 start/stop
# =========================================================================

def _kernels_rms_norm():
    """rms_norm: B=4096, N=4096"""
    B, N = 4096, 4096
    x = torch.randn(B, N, device="cuda").contiguous()
    w = torch.randn(N, device="cuda").contiguous()
    y = torch.empty(B, N, device="cuda")

    _args = [ctypes.c_void_p] * 3 + [ctypes.c_int, ctypes.c_int, ctypes.c_float]
    cuda_lib = _load_so("operators/rms_norm/cuda/rms_norm.so", {
        "rms_norm_cuda_v1": (_args, None),
        "rms_norm_cuda_v2": (_args, None),
        "rms_norm_cuda_v3": (_args, None),
    })
    cute_lib = _load_so("operators/rms_norm/cutlass/rms_norm_cutlass.so", {
        "rms_norm_cutlass_v1": (_args, None),
        "rms_norm_cutlass_v2": (_args, None),
        "rms_norm_cutlass_v3": (_args, None),
    })

    def _call(lib, fn):
        return lambda: getattr(lib, fn)(
            x.data_ptr(), w.data_ptr(), y.data_ptr(),
            ctypes.c_int(B), ctypes.c_int(N), ctypes.c_float(1e-6)
        )

    kernels = {}
    if cuda_lib:
        kernels["cuda_v1"] = _call(cuda_lib, "rms_norm_cuda_v1")
        kernels["cuda_v2"] = _call(cuda_lib, "rms_norm_cuda_v2")
        kernels["cuda_v3"] = _call(cuda_lib, "rms_norm_cuda_v3")
    if cute_lib:
        kernels["cute_v1"] = _call(cute_lib, "rms_norm_cutlass_v1")
        kernels["cute_v2"] = _call(cute_lib, "rms_norm_cutlass_v2")
        kernels["cute_v3"] = _call(cute_lib, "rms_norm_cutlass_v3")
    return kernels


def _kernels_rope():
    """rope: seq=4096, heads=32, head_dim=64"""
    from operators.rope.pytorch.baseline import build_cos_sin_cache
    seq_len, num_heads, head_dim = 4096, 32, 64
    cos_cache, sin_cache = build_cos_sin_cache(seq_len + 64, head_dim, device="cuda")
    q = torch.randn(seq_len, num_heads, head_dim, device="cuda").contiguous()
    k = torch.randn(seq_len, num_heads, head_dim, device="cuda").contiguous()
    positions = torch.arange(seq_len, device="cuda", dtype=torch.int32).contiguous()

    _args = [ctypes.c_void_p] * 4 + [ctypes.c_void_p] + [ctypes.c_int] * 3
    cuda_lib = _load_so("operators/rope/cuda/rope.so", {
        "rope_cuda_v1": (_args, None),
        "rope_cuda_v2": (_args, None),
    })
    cute_lib = _load_so("operators/rope/cutlass/rope_cutlass.so", {
        "rope_cutlass_v1": (_args, None),
        "rope_cutlass_v2": (_args, None),
    })

    def _call(lib, fn):
        return lambda: getattr(lib, fn)(
            q.data_ptr(), k.data_ptr(),
            cos_cache.data_ptr(), sin_cache.data_ptr(),
            positions.data_ptr(),
            ctypes.c_int(seq_len), ctypes.c_int(num_heads), ctypes.c_int(head_dim)
        )

    kernels = {}
    if cuda_lib:
        kernels["cuda_v1"] = _call(cuda_lib, "rope_cuda_v1")
        kernels["cuda_v2"] = _call(cuda_lib, "rope_cuda_v2")
    if cute_lib:
        kernels["cute_v1"] = _call(cute_lib, "rope_cutlass_v1")
        kernels["cute_v2"] = _call(cute_lib, "rope_cutlass_v2")
    return kernels


def _kernels_softmax():
    """softmax: B=4096, N=4096"""
    B, N = 4096, 4096
    x = torch.randn(B, N, device="cuda").contiguous()
    y = torch.empty(B, N, device="cuda")

    _args = [ctypes.c_void_p] * 2 + [ctypes.c_int, ctypes.c_int]
    cuda_lib = _load_so("operators/softmax/cuda/softmax.so", {
        "softmax_cuda_v1": (_args, None),
        "softmax_cuda_v2": (_args, None),
        "softmax_cuda_v3": (_args, None),
    })
    cute_lib = _load_so("operators/softmax/cutlass/softmax_cutlass.so", {
        "softmax_cutlass_v1": (_args, None),
        "softmax_cutlass_v2": (_args, None),
    })

    def _call(lib, fn):
        return lambda: getattr(lib, fn)(
            x.data_ptr(), y.data_ptr(), ctypes.c_int(B), ctypes.c_int(N)
        )

    kernels = {}
    if cuda_lib:
        for v in ["v1", "v2", "v3"]:
            kernels[f"cuda_{v}"] = _call(cuda_lib, f"softmax_cuda_{v}")
    if cute_lib:
        for v in ["v1", "v2"]:
            kernels[f"cute_{v}"] = _call(cute_lib, f"softmax_cutlass_{v}")
    return kernels


def _kernels_layernorm():
    """layernorm: B=4096, N=4096"""
    B, N = 4096, 4096
    x = torch.randn(B, N, device="cuda").contiguous()
    w = torch.ones(N, device="cuda").contiguous()
    b = torch.zeros(N, device="cuda").contiguous()
    y = torch.empty(B, N, device="cuda")

    _args = [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_int, ctypes.c_float]
    cuda_lib = _load_so("operators/layernorm/cuda/layernorm.so", {
        "layernorm_cuda_v1": (_args, None),
        "layernorm_cuda_v2": (_args, None),
        "layernorm_cuda_v3": (_args, None),
    })
    cute_lib = _load_so("operators/layernorm/cutlass/layernorm_cutlass.so", {
        "layernorm_cutlass_v1": (_args, None),
        "layernorm_cutlass_v2": (_args, None),
        "layernorm_cutlass_v3": (_args, None),
    })

    def _call(lib, fn):
        return lambda: getattr(lib, fn)(
            x.data_ptr(), w.data_ptr(), b.data_ptr(), y.data_ptr(),
            ctypes.c_int(B), ctypes.c_int(N), ctypes.c_float(1e-5)
        )

    kernels = {}
    if cuda_lib:
        kernels["cuda_v1"] = _call(cuda_lib, "layernorm_cuda_v1")
        kernels["cuda_v2"] = _call(cuda_lib, "layernorm_cuda_v2")
        kernels["cuda_v3"] = _call(cuda_lib, "layernorm_cuda_v3")
    if cute_lib:
        kernels["cute_v1"] = _call(cute_lib, "layernorm_cutlass_v1")
        kernels["cute_v2"] = _call(cute_lib, "layernorm_cutlass_v2")
        kernels["cute_v3"] = _call(cute_lib, "layernorm_cutlass_v3")
    return kernels


def _kernels_matmul():
    """matmul: 4096x4096x4096"""
    M, K, N = 4096, 4096, 4096
    A = torch.randn(M, K, device="cuda").contiguous()
    B = torch.randn(K, N, device="cuda").contiguous()
    C = torch.empty(M, N, device="cuda")

    _args = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3
    cuda_lib = _load_so("operators/matmul/cuda/matmul.so", {
        "matmul_cuda_v1": (_args, None),
        "matmul_cuda_v2": (_args, None),
        "matmul_cuda_v3": (_args, None),
    })
    cute_lib = _load_so("operators/matmul/cutlass/matmul_cutlass.so", {
        "matmul_cutlass_v1": (_args, None),
        "matmul_cutlass_v2": (_args, None),
    })
    hl_lib = _load_so("operators/matmul/cutlass/matmul_cutlass_hl.so", {
        "matmul_cutlass_hl_v1": (_args, None),
        "matmul_cutlass_hl_v2": (_args, None),
    })

    def _call(lib, fn):
        return lambda: getattr(lib, fn)(
            A.data_ptr(), B.data_ptr(), C.data_ptr(),
            ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(N)
        )

    kernels = {}
    if cuda_lib:
        for v in ["v1", "v2", "v3"]:
            kernels[f"cuda_{v}"] = _call(cuda_lib, f"matmul_cuda_{v}")
    if cute_lib:
        kernels["cute_v1"] = _call(cute_lib, "matmul_cutlass_v1")
        kernels["cute_v2"] = _call(cute_lib, "matmul_cutlass_v2")
    if hl_lib:
        kernels["cutlass_hl_v1"] = _call(hl_lib, "matmul_cutlass_hl_v1")
        kernels["cutlass_hl_v2"] = _call(hl_lib, "matmul_cutlass_hl_v2")
    return kernels


def _kernels_transpose():
    """transpose: 4096x4096"""
    M, N = 4096, 4096
    A = torch.randn(M, N, device="cuda").contiguous()
    B = torch.empty(N, M, device="cuda")

    _args = [ctypes.c_void_p] * 2 + [ctypes.c_int, ctypes.c_int]
    cuda_lib = _load_so("operators/transpose/cuda/transpose.so", {
        "transpose_cuda_v1": (_args, None),
        "transpose_cuda_v2": (_args, None),
    })
    cute_lib = _load_so("operators/transpose/cutlass/transpose_cutlass.so", {
        "transpose_cutlass_v1": (_args, None),
        "transpose_cutlass_v2": (_args, None),
    })

    def _call(lib, fn):
        return lambda: getattr(lib, fn)(
            A.data_ptr(), B.data_ptr(), ctypes.c_int(M), ctypes.c_int(N)
        )

    kernels = {}
    if cuda_lib:
        kernels["cuda_v1"] = _call(cuda_lib, "transpose_cuda_v1")
        kernels["cuda_v2"] = _call(cuda_lib, "transpose_cuda_v2")
    if cute_lib:
        kernels["cute_v1"] = _call(cute_lib, "transpose_cutlass_v1")
        kernels["cute_v2"] = _call(cute_lib, "transpose_cutlass_v2")
    return kernels


def _kernels_vector_add():
    """vector_add: N=64M"""
    N = 1024 * 1024 * 64
    A = torch.randn(N, device="cuda").contiguous()
    B = torch.randn(N, device="cuda").contiguous()
    C = torch.empty(N, device="cuda")

    _args = [ctypes.c_void_p] * 3 + [ctypes.c_int]
    cuda_lib = _load_so("operators/vector_add/cuda/vector_add.so", {
        "vector_add_cuda_v1": (_args, None),
        "vector_add_cuda_v2": (_args, None),
    })
    cute_lib = _load_so("operators/vector_add/cutlass/vector_add_cutlass.so", {
        "vector_add_cutlass_v1": (_args, None),
        "vector_add_cutlass_v2": (_args, None),
    })

    def _call(lib, fn):
        return lambda: getattr(lib, fn)(
            A.data_ptr(), B.data_ptr(), C.data_ptr(), ctypes.c_int(N)
        )

    kernels = {}
    if cuda_lib:
        kernels["cuda_v1"] = _call(cuda_lib, "vector_add_cuda_v1")
        kernels["cuda_v2"] = _call(cuda_lib, "vector_add_cuda_v2")
    if cute_lib:
        kernels["cute_v1"] = _call(cute_lib, "vector_add_cutlass_v1")
        kernels["cute_v2"] = _call(cute_lib, "vector_add_cutlass_v2")
    return kernels


# =========================================================================
# 全局 Registry
# =========================================================================
REGISTRY = {
    "rms_norm":   _kernels_rms_norm,
    "rope":       _kernels_rope,
    "softmax":    _kernels_softmax,
    "layernorm":  _kernels_layernorm,
    "matmul":     _kernels_matmul,
    "transpose":  _kernels_transpose,
    "vector_add": _kernels_vector_add,
}


# =========================================================================
# 主逻辑
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Profile driver: 用 cudaProfilerStart/Stop 框定目标 kernel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 直接运行（不经过 ncu，用于调试/验证）
  python profiling/profile_driver.py --op rms_norm --kernel cuda_v2

  # 配合 ncu
  ncu --target-processes all --set full \\
      python profiling/profile_driver.py --op rms_norm --kernel cuda_v2

  # 列出所有可用 kernel
  python profiling/profile_driver.py --list
""")
    parser.add_argument("--op",     type=str, help="算子名，如 rms_norm")
    parser.add_argument("--kernel", type=str, help="kernel 名，如 cuda_v2（不指定则跑所有）")
    parser.add_argument("--warmup", type=int, default=3, help="warmup 次数（默认 3）")
    parser.add_argument("--iters",  type=int, default=1, help="profile 次数（默认 1）")
    parser.add_argument("--list",   action="store_true", help="列出所有可用 op/kernel")
    args = parser.parse_args()

    if args.list:
        print("可用 op 和 kernel：")
        for op_name, builder in REGISTRY.items():
            kernels = builder()
            print(f"  {op_name}:")
            for k in kernels:
                print(f"    {k}")
        return

    if not args.op:
        parser.error("请指定 --op")

    if args.op not in REGISTRY:
        print(f"[ERROR] 未知 op: {args.op}，可用: {list(REGISTRY.keys())}")
        sys.exit(1)

    print(f"[profile_driver] Building kernels for op={args.op} ...")
    all_kernels = REGISTRY[args.op]()

    # 筛选目标 kernel
    if args.kernel:
        if args.kernel not in all_kernels:
            print(f"[ERROR] 未知 kernel: {args.kernel}，可用: {list(all_kernels.keys())}")
            sys.exit(1)
        target_kernels = {args.kernel: all_kernels[args.kernel]}
    else:
        target_kernels = all_kernels

    print(f"[profile_driver] 目标 kernel: {list(target_kernels.keys())}")

    # Warmup（profiler 关闭状态）
    print(f"[profile_driver] Warmup x{args.warmup} ...")
    for fn in target_kernels.values():
        for _ in range(args.warmup):
            fn()
    torch.cuda.synchronize()

    # Profile（用 cudaProfilerStart/Stop 框定）
    # ncu 默认只 capture profiler active 期间的 kernel
    print(f"[profile_driver] Profiling x{args.iters} ...")
    for name, fn in target_kernels.items():
        print(f"  -> {name}")
        for _ in range(args.iters):
            profiler_start()
            fn()
            profiler_stop()

    torch.cuda.synchronize()
    print("[profile_driver] Done.")


if __name__ == "__main__":
    main()
