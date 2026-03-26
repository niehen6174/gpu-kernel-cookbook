"""
Group GEMM - Triton 实现

计算 G 组独立的矩阵乘法，所有 group 的 (M, K, N) 相同：
    C[g, i, j] = sum_k A[g, i, k] * B[g, k, j]

实现：
V1: 3D program grid — (pid_m, pid_n, pid_g)，每个 program 负责 group g 中的
    一个 BLOCK_M × BLOCK_N 输出 tile，K 维度分块迭代。

V2: 变大小 group GEMM (jagged) — A/B 按行 concat 为 1D 数组，
    group_offsets 记录每组在 flat 数组中的偏移。
    每个 program 由 (group_id, tile_m, tile_n) 三者确定，
    通过 group_id 查找偏移跳转到正确位置。
"""

import triton
import triton.language as tl
import torch
from typing import List


# -----------------------------------------------------------------------
# V1: Fixed-size group GEMM
# -----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def group_gemm_v1_kernel(
    a_ptr, b_ptr, c_ptr,
    G, M, N, K,
    stride_ag, stride_am, stride_ak,
    stride_bg, stride_bk, stride_bn,
    stride_cg, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    3D grid: axis0 = tile_m * tile_n (2D flattened), axis1 = group_id
    """
    pid      = tl.program_id(axis=0)   # M×N 方向上的 tile 编号
    group_id = tl.program_id(axis=1)   # group 编号

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # 当前 group 的基地址偏移
    a_base = a_ptr + group_id * stride_ag
    b_base = b_ptr + group_id * stride_bg
    c_base = c_ptr + group_id * stride_cg

    # M/N 方向的行列偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_base + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem  = K - k * BLOCK_K
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_rem)
        b_mask = (offs_k[:, None] < k_rem) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs  = c_base + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask  = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def group_gemm_triton_fixed(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Fixed-size group GEMM.
    A: (G, M, K), B: (G, K, N) → C: (G, M, N)
    """
    assert A.is_cuda and B.is_cuda
    G, M, K = A.shape
    G2, K2, N = B.shape
    assert G == G2 and K == K2

    C = torch.empty(G, M, N, dtype=A.dtype, device=A.device)

    num_pid_m = triton.cdiv(M, 64)   # rough estimate for grid
    num_pid_n = triton.cdiv(N, 64)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        G,
    )

    group_gemm_v1_kernel[grid](
        A, B, C,
        G, M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
    )
    return C


# -----------------------------------------------------------------------
# V2: Variable-size group GEMM (jagged layout)
#
# A_flat: (total_a_rows, K) 所有 group A 矩阵行拼接
# B_flat: (total_b_rows, N) 所有 group B 矩阵行拼接（K_g × N_g）
# a_offsets[g]: group g 的 A 矩阵在 A_flat 的起始行
# b_offsets[g]: group g 的 B 矩阵在 B_flat 的起始行
# m_sizes[g]:   group g 的 M_g
# k_sizes[g]:   group g 的 K_g
# n_sizes[g]:   group g 的 N_g
# c_offsets[g]: group g 的 C 矩阵在 C_flat 的起始行
# -----------------------------------------------------------------------
@triton.jit
def group_gemm_v2_kernel(
    a_ptr, b_ptr, c_ptr,
    a_offsets_ptr, b_offsets_ptr, c_offsets_ptr,
    m_sizes_ptr, k_sizes_ptr, n_sizes_ptr,
    G,
    stride_ak, stride_bk, stride_bn, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Jagged group GEMM.
    程序以 (tile_id, group_id) 的方式启动：axis0=tile，axis1=group
    但因为各组 tile 数不同，这里用 axis0 做 flat tile 编号。
    为简单起见，axis0 = group_id * MAX_TILES_PER_GROUP + tile_in_group
    实际上用单独 tile_starts 数组来处理不等大小。

    本实现：
      - axis1 = group_id (G groups)
      - axis0 = (tile_m_in_group * ceil(N/BN))  —— 但N各不同，故改为：
      - axis0 每个 program 查询自己的 group_id 和 tile 坐标（通过 tile_starts 数组）
    实际简化为: 所有 group 平铺 tile，每个 program 由 flat_tile_id 确定所属 group。
    """
    flat_tile_id = tl.program_id(axis=0)
    group_id     = tl.program_id(axis=1)

    # 读取 group 信息
    M_g = tl.load(m_sizes_ptr + group_id)
    K_g = tl.load(k_sizes_ptr + group_id)
    N_g = tl.load(n_sizes_ptr + group_id)

    num_pid_n = tl.cdiv(N_g, BLOCK_N)
    pid_m     = flat_tile_id // num_pid_n
    pid_n     = flat_tile_id % num_pid_n

    # 越界 guard（由 max_tiles 决定）
    if pid_m * BLOCK_M >= M_g:
        return

    a_row_off = tl.load(a_offsets_ptr + group_id)
    b_row_off = tl.load(b_offsets_ptr + group_id)
    c_row_off = tl.load(c_offsets_ptr + group_id)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # A: (total_rows, K_g) 其中当前 group 从 a_row_off 开始，stride_ak = K (full width)
    a_ptrs = a_ptr + (a_row_off + offs_m[:, None]) * stride_ak + offs_k[None, :]
    b_ptrs = b_ptr + (b_row_off + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K_g, BLOCK_K)):
        k_rem  = K_g - k * BLOCK_K
        a_mask = (offs_m[:, None] < M_g) & (offs_k[None, :] < k_rem)
        b_mask = (offs_k[:, None] < k_rem) & (offs_n[None, :] < N_g)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + (c_row_off + offs_m[:, None]) * stride_cn + offs_n[None, :]
    c_mask = (offs_m[:, None] < M_g) & (offs_n[None, :] < N_g)
    tl.store(c_ptrs, acc, mask=c_mask)


def group_gemm_triton_var(
    A_list: List[torch.Tensor],
    B_list: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Variable-size group GEMM via jagged flat layout.
    A_list: G 个 (M_g, K_g) tensors
    B_list: G 个 (K_g, N_g) tensors
    返回 G 个 C_g = A_g @ B_g 矩阵列表
    """
    G = len(A_list)
    assert G == len(B_list)

    m_sizes = torch.tensor([A.shape[0] for A in A_list], dtype=torch.int32, device="cuda")
    k_sizes = torch.tensor([A.shape[1] for A in A_list], dtype=torch.int32, device="cuda")
    n_sizes = torch.tensor([B.shape[1] for B in B_list], dtype=torch.int32, device="cuda")

    # Build flat A (each row of A_g has K_g elements; all K_g must equal for flat layout)
    # For simplicity in this implementation, we use a common K dimension
    # (Variable K requires separate kernel launches or workspace management)
    # Check if all K are same
    k_vals = [A.shape[1] for A in A_list]
    if len(set(k_vals)) > 1:
        # Fallback to sequential for variable K
        return [torch.matmul(A, B) for A, B in zip(A_list, B_list)]

    K = k_vals[0]
    # Flat concat
    A_flat = torch.cat(A_list, dim=0)   # (sum_M, K)
    B_flat = torch.cat(B_list, dim=0)   # (sum_K, N_g) — here K repeated G times but offsets track it

    n_vals = [B.shape[1] for B in B_list]
    N_max  = max(n_vals)
    # Build output list and flat output tensor (with same N for flat storage)
    # For variable N, we need separate outputs per group
    C_list = [torch.empty(A.shape[0], B.shape[1], dtype=A.dtype, device=A.device)
              for A, B in zip(A_list, B_list)]

    a_offsets = torch.zeros(G, dtype=torch.int32, device="cuda")
    b_offsets = torch.zeros(G, dtype=torch.int32, device="cuda")
    c_offsets = torch.zeros(G, dtype=torch.int32, device="cuda")

    a_off = b_off = c_off = 0
    for g in range(G):
        a_offsets[g] = a_off
        b_offsets[g] = b_off
        c_offsets[g] = c_off
        a_off += A_list[g].shape[0]
        b_off += B_list[g].shape[0]
        c_off += C_list[g].shape[0]

    C_flat = torch.cat(C_list, dim=0)  # preallocate; strides may differ per group N

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    max_tiles_per_group = max(
        triton.cdiv(A_list[g].shape[0], BLOCK_M) * triton.cdiv(B_list[g].shape[1], BLOCK_N)
        for g in range(G)
    )

    grid = (max_tiles_per_group, G)

    group_gemm_v2_kernel[grid](
        A_flat, B_flat, C_flat,
        a_offsets, b_offsets, c_offsets,
        m_sizes, k_sizes, n_sizes,
        G,
        A_flat.stride(0),   # stride_ak (columns in A_flat)
        B_flat.stride(0),   # stride_bk (columns in B_flat)
        B_flat.stride(1),   # stride_bn
        C_flat.stride(0),   # stride_cn (columns in C_flat — note variable N is tricky)
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    torch.cuda.synchronize()

    # Split C_flat back to per-group tensors
    offset = 0
    results = []
    for g in range(G):
        M_g = A_list[g].shape[0]
        results.append(C_flat[offset: offset + M_g])
        offset += M_g
    return results
