"""
SVDQuant Triton 实现

两个融合 kernel：
1. quantize_fuse_lora_kernel：
   - 输入: x (M, K) FP16
   - 融合: smooth 应用 + INT4 量化 + LoRA down projection
   - 输出: q_x (M, K//2) uint8, ascales (M, K//G), lora_act (M, rank)

2. w4a4_gemm_dequant_lora_kernel：
   - 输入: q_x (M, K//2) uint8, q_w (N, K//2) uint8, ascales, wscales
           lora_act (M, rank), lora_up (N, rank)
   - 融合: INT4 解量化 + GEMM + 低秩加法 + bias
   - 使用 tl.dot() 利用 Tensor Core
   - Block tiling: BLOCK_M=128, BLOCK_N=256, BLOCK_K=64

设计说明：
- INT4 值存储为 int8（为了让 Triton 正确处理符号），不做 nibble packing
  （nibble 解包在 Triton 中需要位运算，后续版本可以优化）
- W4A4 GEMM 通过先反量化到 FP16 再做 FP16 tl.dot 实现
  （真正的 INT4 MMA 需要 inline PTX，作为后续优化）
- 第一版聚焦于 correctness 和结构清晰
"""

import torch
import triton
import triton.language as tl


# -------------------------------------------------------------------------
# Kernel 1: 量化激活 + LoRA down 投影
# -------------------------------------------------------------------------

@triton.jit
def quantize_fuse_lora_kernel(
    # 输入
    X_ptr,          # (M, K) FP16
    smooth_ptr,     # (K,)   FP16  平滑因子
    lora_down_ptr,  # (K, R) FP16  LoRA down 权重
    # 输出
    Q_X_ptr,        # (M, K) int8  量化激活（暂不 packing，便于 debug）
    ascales_ptr,    # (M, G) FP16  激活 scales，G = K // group_size
    lora_act_ptr,   # (M, R) FP32  LoRA 激活输出
    # 维度
    M: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    GROUP_SIZE: tl.constexpr,   # = 64
    BLOCK_M: tl.constexpr,      # = 64
    BLOCK_K: tl.constexpr,      # = K (每个 thread block 处理整行)
    BLOCK_R: tl.constexpr,      # = R (LoRA rank，通常小)
):
    """
    每个 thread block 处理 BLOCK_M 行激活：
    1. 对每行 x[i, :] 应用 smooth，量化到 INT4，输出 scale
    2. 计算 lora_act[i, :] = x[i, :] @ lora_down  (FP16 matmul)

    这里简化：每个 program 处理一行（pid = row index）
    """
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # 加载一行 x: (K,)
    k_offs = tl.arange(0, BLOCK_K)
    x_row = tl.load(X_ptr + pid_m * K + k_offs, mask=k_offs < K, other=0.0).to(tl.float32)

    # 加载 smooth: (K,)
    smooth = tl.load(smooth_ptr + k_offs, mask=k_offs < K, other=1.0).to(tl.float32)

    # 应用平滑
    x_smooth = x_row / smooth   # (K,)

    # 量化：per-group INT4
    num_groups = K // GROUP_SIZE
    # 计算每组 scale 并量化
    for g in tl.static_range(0, K // GROUP_SIZE):
        g_start = g * GROUP_SIZE
        g_offs  = g_start + tl.arange(0, GROUP_SIZE)
        x_g = tl.load(X_ptr + pid_m * K + g_offs,
                       mask=g_offs < K, other=0.0).to(tl.float32)
        sm_g = tl.load(smooth_ptr + g_offs, mask=g_offs < K, other=1.0).to(tl.float32)
        x_g_smooth = x_g / sm_g

        # scale = max(|x|) / 7
        abs_max = tl.max(tl.abs(x_g_smooth), axis=0)
        scale = abs_max / 7.0
        scale = tl.where(scale < 1e-8, 1e-8, scale)

        # 量化
        q_g = tl.extra.cuda.libdevice.llrint(x_g_smooth / scale)
        q_g = tl.clamp(q_g, -8, 7).to(tl.int8)

        # 存储
        tl.store(Q_X_ptr + pid_m * K + g_offs, q_g, mask=g_offs < K)
        tl.store(ascales_ptr + pid_m * (K // GROUP_SIZE) + g,
                 scale.to(tl.float16))

    # LoRA down projection: lora_act[pid_m, :] = x[pid_m, :] @ lora_down
    # lora_down: (K, R)
    r_offs = tl.arange(0, BLOCK_R)
    acc = tl.zeros([BLOCK_R], dtype=tl.float32)
    for k in range(K):
        x_k = tl.load(X_ptr + pid_m * K + k).to(tl.float32)
        w_k = tl.load(lora_down_ptr + k * R + r_offs,
                      mask=r_offs < R, other=0.0).to(tl.float32)
        acc += x_k * w_k

    tl.store(lora_act_ptr + pid_m * R + r_offs, acc, mask=r_offs < R)


# -------------------------------------------------------------------------
# Kernel 2: W4A4 GEMM + 低秩加法（使用 tl.dot）
# -------------------------------------------------------------------------

@triton.jit
def w4a4_gemm_dequant_lora_kernel(
    # 量化激活
    Q_X_ptr,        # (M, K) int8
    ascales_ptr,    # (M, K//G) FP16
    # 量化权重
    Q_W_ptr,        # (N, K) int8  （转置布局，row-major）
    wscales_ptr,    # (N, K//G) FP16
    # LoRA
    lora_act_ptr,   # (M, R) FP32
    lora_up_ptr,    # (N, R) FP16
    # 偏置 + 输出
    bias_ptr,       # (N,) FP16，可为 NULL (0 = no bias)
    Y_ptr,          # (M, N) FP16 输出
    # 维度
    M, N, K: tl.constexpr,
    R: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    计算 Y = dequant(Q_X) @ dequant(Q_W)^T + lora_act @ lora_up^T + bias

    策略：
    - 标准 tiled GEMM，每个 block 计算 (BLOCK_M, BLOCK_N) 的输出块
    - 在 K 维 iterate，每次加载 (BLOCK_M, BLOCK_K) 激活块和 (BLOCK_N, BLOCK_K) 权重块
    - 在 tile 内对每个 group 做 dequant，转换为 FP16 后用 tl.dot
    - LoRA 部分在主 GEMM 外额外计算（小矩阵，rank 通常 ≤ 64）
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offs = m_start + tl.arange(0, BLOCK_M)
    n_offs = n_start + tl.arange(0, BLOCK_N)

    # 累加器 (BLOCK_M, BLOCK_N)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # K 方向 tiling
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for k_tile in range(num_k_tiles):
        k_start = k_tile * BLOCK_K
        k_offs = k_start + tl.arange(0, BLOCK_K)

        # 加载量化激活块 Q_X[m_offs, k_offs]: (BLOCK_M, BLOCK_K) int8
        qx_ptrs = Q_X_ptr + m_offs[:, None] * K + k_offs[None, :]
        qx = tl.load(qx_ptrs,
                     mask=(m_offs[:, None] < M) & (k_offs[None, :] < K),
                     other=0).to(tl.int8)

        # 加载激活 scales: 需要知道每个 k 对应的 group
        # ascales shape: (M, K//G), group = k // GROUP_SIZE
        # 我们需要对每个 (m, k) 组合找到对应 scale
        # 简化：对 BLOCK_K 内的每个 group 加载一次 scale，然后广播
        # 假设 BLOCK_K = GROUP_SIZE 的整数倍
        num_k_groups = BLOCK_K // GROUP_SIZE
        qx_f = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
        for g_local in tl.static_range(0, BLOCK_K // GROUP_SIZE):
            g_global = k_tile * (BLOCK_K // GROUP_SIZE) + g_local
            g_offs_local = g_local * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

            # 加载激活 scales: (BLOCK_M,)
            as_ptrs = ascales_ptr + m_offs * (K // GROUP_SIZE) + g_global
            a_scale = tl.load(as_ptrs, mask=m_offs < M, other=1.0).to(tl.float32)

            # 加载对应的量化值
            k_local_offs = k_start + g_local * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
            qx_g_ptrs = Q_X_ptr + m_offs[:, None] * K + k_local_offs[None, :]
            qx_g = tl.load(qx_g_ptrs,
                           mask=(m_offs[:, None] < M) & (k_local_offs[None, :] < K),
                           other=0).to(tl.int8).to(tl.float32)
            # dequant: q * scale
            qx_g_dq = qx_g * a_scale[:, None]  # (BLOCK_M, GROUP_SIZE)
            # 写入对应位置
            # （Triton 不支持动态切片赋值，改用累加方式）
            # 这里用 mask 技巧
            for gs in tl.static_range(0, GROUP_SIZE):
                qx_f = tl.where(
                    (tl.arange(0, BLOCK_K)[None, :] == (g_local * GROUP_SIZE + gs)),
                    qx_g_dq[:, gs:gs+1],
                    qx_f
                )

        # 类似地，加载并反量化权重块 Q_W[n_offs, k_offs]: (BLOCK_N, BLOCK_K)
        qw_f = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)
        for g_local in tl.static_range(0, BLOCK_K // GROUP_SIZE):
            g_global = k_tile * (BLOCK_K // GROUP_SIZE) + g_local
            k_local_offs = k_start + g_local * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

            ws_ptrs = wscales_ptr + n_offs * (K // GROUP_SIZE) + g_global
            w_scale = tl.load(ws_ptrs, mask=n_offs < N, other=1.0).to(tl.float32)

            qw_g_ptrs = Q_W_ptr + n_offs[:, None] * K + k_local_offs[None, :]
            qw_g = tl.load(qw_g_ptrs,
                           mask=(n_offs[:, None] < N) & (k_local_offs[None, :] < K),
                           other=0).to(tl.int8).to(tl.float32)
            qw_g_dq = qw_g * w_scale[:, None]

            for gs in tl.static_range(0, GROUP_SIZE):
                qw_f = tl.where(
                    (tl.arange(0, BLOCK_K)[None, :] == (g_local * GROUP_SIZE + gs)),
                    qw_g_dq[:, gs:gs+1],
                    qw_f
                )

        # dot product: (BLOCK_M, BLOCK_K) x (BLOCK_N, BLOCK_K)^T -> (BLOCK_M, BLOCK_N)
        acc += tl.dot(qx_f.to(tl.float16), tl.trans(qw_f.to(tl.float16)))

    # LoRA 部分: acc += lora_act @ lora_up^T
    # lora_act: (M, R), lora_up: (N, R)
    # 简化版本：逐 rank 累加（rank 通常小）
    for r in range(R):
        la_ptrs = lora_act_ptr + m_offs * R + r
        la = tl.load(la_ptrs, mask=m_offs < M, other=0.0).to(tl.float32)

        lu_ptrs = lora_up_ptr + n_offs * R + r
        lu = tl.load(lu_ptrs, mask=n_offs < N, other=0.0).to(tl.float32)

        acc += la[:, None] * lu[None, :]

    # 偏置
    if HAS_BIAS:
        bias = tl.load(bias_ptr + n_offs, mask=n_offs < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    # 存储输出
    y_ptrs = Y_ptr + m_offs[:, None] * N + n_offs[None, :]
    tl.store(y_ptrs, acc.to(tl.float16),
             mask=(m_offs[:, None] < M) & (n_offs[None, :] < N))


# -------------------------------------------------------------------------
# 简化版 Triton kernel（单 kernel 实现，便于调试）
# -------------------------------------------------------------------------

@triton.jit
def svdquant_simple_kernel(
    # 输入
    X_ptr,          # (M, K) FP16 输入激活
    smooth_ptr,     # (K,)   FP16 平滑因子
    Q_W_ptr,        # (N, K) int8 量化权重（已转置，N×K 布局）
    wscales_ptr,    # (N, K//G) FP16 权重 scales
    lora_down_ptr,  # (K, R) FP16
    lora_up_ptr,    # (N, R) FP16
    bias_ptr,       # (N,) FP16，可为 0
    # 输出
    Y_ptr,          # (M, N) FP16
    # 维度
    M, N, K,
    R: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    融合 kernel：smooth + 量化 + W4A4 GEMM + LoRA

    每个 block 负责输出矩阵的 (BLOCK_M, BLOCK_N) 块。
    在 K 维迭代：对每个 K tile，反量化激活和权重后做 FP16 tl.dot。
    LoRA 部分在 K 循环外做（小矩阵）。

    注意：这是 correctness-first 的参考实现，性能不是主要目标。
    BLOCK_K 必须等于 GROUP_SIZE（简化 group 索引）。
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offs = m_start + tl.arange(0, BLOCK_M)
    n_offs = n_start + tl.arange(0, BLOCK_N)
    k_range = tl.arange(0, BLOCK_K)   # BLOCK_K == GROUP_SIZE

    # 主 GEMM 累加器
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for k_tile in range(num_k_tiles):
        k_offs = k_tile * BLOCK_K + k_range
        g_idx  = k_tile   # group index（因为 BLOCK_K == GROUP_SIZE）

        # ------- 激活块 -------
        # 加载 x[m_offs, k_offs]: (BLOCK_M, BLOCK_K) FP16
        x_ptrs = X_ptr + m_offs[:, None] * K + k_offs[None, :]
        x_blk  = tl.load(x_ptrs,
                         mask=(m_offs[:, None] < M) & (k_offs[None, :] < K),
                         other=0.0)   # FP16

        # 加载 smooth[k_offs]
        sm_blk = tl.load(smooth_ptr + k_offs,
                         mask=k_offs < K, other=1.0)  # FP16

        # 应用平滑并量化
        x_sm = x_blk.to(tl.float32) / sm_blk.to(tl.float32)  # (BM, BK) FP32

        # per-row scale（对这个 K tile，每行一个 scale）
        a_scale = tl.max(tl.abs(x_sm), axis=1) / 7.0  # (BLOCK_M,)
        a_scale = tl.where(a_scale < 1e-8, 1e-8, a_scale)

        # 量化 → 反量化（模拟 INT4 精度）
        # 使用 floor + 0.5 实现 round，避免 libdevice 依赖
        x_scaled = x_sm / a_scale[:, None]
        x_q = tl.floor(x_scaled + 0.5)          # round-to-nearest
        x_q = tl.minimum(tl.maximum(x_q, -8.0), 7.0)   # clamp to [-8, 7]
        x_dq = (x_q * a_scale[:, None]).to(tl.float16)  # 反量化

        # ------- 权重块 -------
        # 加载 Q_W[n_offs, k_offs]: (BLOCK_N, BLOCK_K) int8
        qw_ptrs = Q_W_ptr + n_offs[:, None] * K + k_offs[None, :]
        q_w_blk = tl.load(qw_ptrs,
                          mask=(n_offs[:, None] < N) & (k_offs[None, :] < K),
                          other=0).to(tl.int8)

        # 加载权重 scale: wscales[n_offs, g_idx] -> (BLOCK_N,)
        ws_ptrs = wscales_ptr + n_offs * (K // GROUP_SIZE) + g_idx
        w_scale = tl.load(ws_ptrs, mask=n_offs < N, other=1.0)   # FP16

        # 反量化权重
        w_dq = (q_w_blk.to(tl.float32) * w_scale.to(tl.float32)[:, None]).to(tl.float16)

        # tl.dot: (BLOCK_M, BLOCK_K) x (BLOCK_N, BLOCK_K)^T -> (BLOCK_M, BLOCK_N)
        acc += tl.dot(x_dq, tl.trans(w_dq))

    # ------- LoRA 部分 -------
    # lora_act = x[m_offs, :] @ lora_down → (BLOCK_M, R)
    # 串行计算（rank 小，K 较大时性能差，但逻辑清晰）
    lora_acc = tl.zeros([BLOCK_M, R], dtype=tl.float32)
    for k_tile in range(num_k_tiles):
        k_offs = k_tile * BLOCK_K + k_range
        x_ptrs = X_ptr + m_offs[:, None] * K + k_offs[None, :]
        x_blk  = tl.load(x_ptrs,
                         mask=(m_offs[:, None] < M) & (k_offs[None, :] < K),
                         other=0.0).to(tl.float16)

        ld_ptrs = lora_down_ptr + k_offs[:, None] * R + tl.arange(0, R)[None, :]
        ld_blk  = tl.load(ld_ptrs,
                          mask=(k_offs[:, None] < K) & (tl.arange(0, R)[None, :] < R),
                          other=0.0).to(tl.float16)

        lora_acc += tl.dot(x_blk, ld_blk)

    # lora_out = lora_acc @ lora_up^T: (BLOCK_M, R) x (BLOCK_N, R)^T -> (BLOCK_M, BLOCK_N)
    lu_ptrs = lora_up_ptr + n_offs[:, None] * R + tl.arange(0, R)[None, :]
    lu_blk  = tl.load(lu_ptrs,
                      mask=(n_offs[:, None] < N) & (tl.arange(0, R)[None, :] < R),
                      other=0.0).to(tl.float16)

    acc += tl.dot(lora_acc.to(tl.float16), tl.trans(lu_blk))

    # 偏置
    if HAS_BIAS:
        bias = tl.load(bias_ptr + n_offs, mask=n_offs < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    # 写出
    y_ptrs = Y_ptr + m_offs[:, None] * N + n_offs[None, :]
    tl.store(y_ptrs, acc.to(tl.float16),
             mask=(m_offs[:, None] < M) & (n_offs[None, :] < N))


# -------------------------------------------------------------------------
# 优化版 Kernel：单 K-loop 融合 GEMM + LoRA + autotune
# -------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'num_warps': 8,  'num_stages': 3}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'num_warps': 4,  'num_stages': 3}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'num_warps': 4,  'num_stages': 3}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'num_warps': 4,  'num_stages': 2}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'num_warps': 8,  'num_stages': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'num_warps': 8,  'num_stages': 3}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def svdquant_optimized_kernel(
    # 输入
    X_ptr,          # (M, K) FP16 输入激活
    smooth_ptr,     # (K,)   FP16 平滑因子
    Q_W_ptr,        # (N, K) int8 量化权重（N×K 布局）
    wscales_ptr,    # (N, K//G) FP16 权重 scales
    lora_down_ptr,  # (K, R) FP16  LoRA down
    lora_up_ptr,    # (N, R) FP16  LoRA up（N×R 布局）
    bias_ptr,       # (N,) FP16，可为 0
    # 输出
    Y_ptr,          # (M, N) FP16
    # 维度
    M, N, K,
    R: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    优化版融合 kernel：单 K-loop 同时完成 GEMM 和 LoRA down，消除 x 的二次 HBM 读取。

    优化点：
    1. 单 K-loop：x 只从 HBM 读一次，同时用于 GEMM 和 LoRA down
    2. 更大 tile（128×128），通过 autotune 自动搜索最优配置
    3. BLOCK_K == GROUP_SIZE，每 tile 对应一个量化 group，避免分支
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offs = m_start + tl.arange(0, BLOCK_M)
    n_offs = n_start + tl.arange(0, BLOCK_N)
    k_range = tl.arange(0, BLOCK_K)       # BLOCK_K == GROUP_SIZE
    r_range = tl.arange(0, R)

    # 主 GEMM 累加器
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    # LoRA down 累加器 (M × R)
    lora_acc = tl.zeros([BLOCK_M, R], dtype=tl.float32)

    num_k_tiles = tl.cdiv(K, BLOCK_K)

    # -----------------------------------------------------------------------
    # 单 K-loop：x 只读一次，同时参与 GEMM 和 LoRA down
    # -----------------------------------------------------------------------
    for k_tile in range(num_k_tiles):
        k_offs = k_tile * BLOCK_K + k_range
        g_idx  = k_tile   # group index（BLOCK_K == GROUP_SIZE）

        # 加载 x[m_offs, k_offs]: (BLOCK_M, BLOCK_K) FP16
        x_ptrs = X_ptr + m_offs[:, None] * K + k_offs[None, :]
        x_blk  = tl.load(x_ptrs,
                         mask=(m_offs[:, None] < M) & (k_offs[None, :] < K),
                         other=0.0)   # FP16

        # ------- GEMM part：量化 x + 反量化 weight -------
        # 应用 smooth 并模拟 INT4 量化
        sm_blk = tl.load(smooth_ptr + k_offs, mask=k_offs < K, other=1.0)  # FP16
        x_sm = x_blk.to(tl.float32) / sm_blk.to(tl.float32)  # (BM, BK) FP32

        # per-row scale（对这个 K tile）
        a_scale = tl.max(tl.abs(x_sm), axis=1) / 7.0  # (BLOCK_M,)
        a_scale = tl.where(a_scale < 1e-8, 1e-8, a_scale)

        # 量化 → 反量化（模拟 INT4 精度）
        x_scaled = x_sm / a_scale[:, None]
        x_q = tl.floor(x_scaled + 0.5)
        x_q = tl.minimum(tl.maximum(x_q, -8.0), 7.0)
        x_dq = (x_q * a_scale[:, None]).to(tl.float16)  # (BLOCK_M, BLOCK_K) FP16

        # 加载权重并反量化
        qw_ptrs = Q_W_ptr + n_offs[:, None] * K + k_offs[None, :]
        q_w_blk = tl.load(qw_ptrs,
                          mask=(n_offs[:, None] < N) & (k_offs[None, :] < K),
                          other=0).to(tl.int8)

        ws_ptrs = wscales_ptr + n_offs * (K // GROUP_SIZE) + g_idx
        w_scale = tl.load(ws_ptrs, mask=n_offs < N, other=1.0)  # FP16
        w_dq = (q_w_blk.to(tl.float32) * w_scale.to(tl.float32)[:, None]).to(tl.float16)
        # (BLOCK_N, BLOCK_K) FP16

        # GEMM tile
        acc += tl.dot(x_dq, tl.trans(w_dq))

        # ------- LoRA down part：复用同一个 x_blk -------
        # lora_down[k_offs, r_range]: (BLOCK_K, R) FP16
        ld_ptrs = lora_down_ptr + k_offs[:, None] * R + r_range[None, :]
        ld_blk  = tl.load(ld_ptrs,
                          mask=(k_offs[:, None] < K) & (r_range[None, :] < R),
                          other=0.0).to(tl.float16)

        # lora_acc += x_blk @ ld_blk: (BLOCK_M, BLOCK_K) x (BLOCK_K, R) -> (BLOCK_M, R)
        lora_acc += tl.dot(x_blk.to(tl.float16), ld_blk)

    # -----------------------------------------------------------------------
    # K-loop 结束：LoRA up projection
    # lora_out = lora_acc @ lora_up^T: (BLOCK_M, R) x (BLOCK_N, R)^T -> (BLOCK_M, BLOCK_N)
    # -----------------------------------------------------------------------
    lu_ptrs = lora_up_ptr + n_offs[:, None] * R + r_range[None, :]
    lu_blk  = tl.load(lu_ptrs,
                      mask=(n_offs[:, None] < N) & (r_range[None, :] < R),
                      other=0.0).to(tl.float16)

    acc += tl.dot(lora_acc.to(tl.float16), tl.trans(lu_blk))

    # 偏置
    if HAS_BIAS:
        bias = tl.load(bias_ptr + n_offs, mask=n_offs < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    # 写出
    y_ptrs = Y_ptr + m_offs[:, None] * N + n_offs[None, :]
    tl.store(y_ptrs, acc.to(tl.float16),
             mask=(m_offs[:, None] < M) & (n_offs[None, :] < N))


# -------------------------------------------------------------------------
# Python 封装（优化版）
# -------------------------------------------------------------------------

def svdquant_forward_triton_opt(
    x: torch.Tensor,           # (M, K) FP16
    q_w: torch.Tensor,         # (K, N) int8 量化残差权重
    wscales: torch.Tensor,     # (K//group_size, N) FP16 weight scales
    lora_down: torch.Tensor,   # (K, rank) FP16
    lora_up: torch.Tensor,     # (rank, N) FP16
    smooth: torch.Tensor,      # (K,) FP16
    bias: torch.Tensor | None = None,
    group_size: int = 64,
) -> torch.Tensor:
    """
    SVDQuant 前向传播 Triton 优化实现。

    使用 svdquant_optimized_kernel：
    - 单 K-loop 融合：GEMM + LoRA down，消除 x 二次 HBM 读取
    - 更大 tile（128×128）+ autotune 自动搜索最优配置
    - BLOCK_K == group_size，每 tile 一个量化 group

    Args:
        x:         (M, K) FP16 输入激活
        q_w:       (K, N) int8 量化残差权重
        wscales:   (K//group_size, N) FP16 权重 scales
        lora_down: (K, rank) FP16
        lora_up:   (rank, N) FP16
        smooth:    (K,) FP16 平滑因子
        bias:      (N,) FP16 偏置（可选）
        group_size: INT4 量化 group size（默认 64，kernel constexpr）

    Returns:
        y: (M, N) FP16
    """
    M, K = x.shape
    K2, N = q_w.shape
    rank = lora_down.shape[1]

    assert K == K2, f"K mismatch: x.K={K}, q_w.K={K2}"
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    assert lora_down.shape == (K, rank)
    assert lora_up.shape == (rank, N)

    # 权重转置：q_w (K, N) -> (N, K)
    q_w_T = q_w.T.contiguous()          # (N, K) int8
    wscales_T = wscales.T.contiguous()   # (N, K//group_size) FP16

    x = x.contiguous()
    lora_down = lora_down.contiguous()   # (K, rank) row-major

    y = torch.empty(M, N, dtype=torch.float16, device=x.device)

    has_bias = bias is not None
    if not has_bias:
        bias_ptr = torch.zeros(N, dtype=torch.float16, device=x.device)
    else:
        bias_ptr = bias.to(torch.float16)

    # 确保 rank 是 2 的幂（tl.dot 要求 constexpr 维度为 2 的幂）
    rank_pow2 = triton.next_power_of_2(rank)

    if rank_pow2 != rank:
        lora_down_pad = torch.zeros(K, rank_pow2, dtype=lora_down.dtype, device=lora_down.device)
        lora_down_pad[:, :rank] = lora_down
        lora_up_pad = torch.zeros(rank_pow2, N, dtype=lora_up.dtype, device=lora_up.device)
        lora_up_pad[:rank, :] = lora_up
    else:
        lora_down_pad = lora_down
        lora_up_pad = lora_up

    lora_up_T = lora_up_pad.T.contiguous()   # (N, rank_pow2)

    BLOCK_K = group_size

    # autotune 根据 BLOCK_M/BLOCK_N 确定 grid
    # 使用 lambda 在 autotune 选好配置后计算实际 grid
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    svdquant_optimized_kernel[grid](
        x, smooth,
        q_w_T, wscales_T,
        lora_down_pad, lora_up_T,
        bias_ptr, y,
        M, N, K,
        R=rank_pow2,
        GROUP_SIZE=group_size,
        BLOCK_K=BLOCK_K,
        HAS_BIAS=has_bias,
    )

    return y


# -------------------------------------------------------------------------
# Python 封装（原始版）
# -------------------------------------------------------------------------

def svdquant_forward_triton(
    x: torch.Tensor,           # (M, K) FP16
    q_w: torch.Tensor,         # (K, N) int8 量化残差权重
    wscales: torch.Tensor,     # (K//group_size, N) FP16 weight scales
    lora_down: torch.Tensor,   # (K, rank) FP16
    lora_up: torch.Tensor,     # (rank, N) FP16
    smooth: torch.Tensor,      # (K,) FP16
    bias: torch.Tensor | None = None,
    group_size: int = 64,
) -> torch.Tensor:
    """
    SVDQuant 前向传播 Triton 实现。

    使用 svdquant_simple_kernel：融合 smooth + INT4 量化 + W4A4 GEMM + LoRA。
    BLOCK_K 固定等于 group_size。

    Args:
        x:         (M, K) FP16 输入激活
        q_w:       (K, N) int8 量化残差权重
        wscales:   (K//group_size, N) FP16 权重 scales
        lora_down: (K, rank) FP16
        lora_up:   (rank, N) FP16
        smooth:    (K,) FP16 平滑因子
        bias:      (N,) FP16 偏置（可选）
        group_size: INT4 量化 group size（默认 64，kernel constexpr）

    Returns:
        y: (M, N) FP16
    """
    M, K = x.shape
    K2, N = q_w.shape
    rank = lora_down.shape[1]

    assert K == K2, f"K mismatch: x.K={K}, q_w.K={K2}"
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    assert lora_down.shape == (K, rank)
    assert lora_up.shape == (rank, N)

    # 权重需要转置：q_w (K, N) -> q_w_T (N, K)，per-row 量化对应 N 维
    q_w_T = q_w.T.contiguous()          # (N, K) int8
    wscales_T = wscales.T.contiguous()   # (N, K//group_size) FP16

    # 确保 lora 张量是连续的（row-major），kernel 假设 row-major 布局
    x = x.contiguous()
    lora_down = lora_down.contiguous()   # (K, rank) row-major: stride=(rank, 1)

    # 输出
    y = torch.empty(M, N, dtype=torch.float16, device=x.device)

    # 准备偏置
    has_bias = bias is not None
    if not has_bias:
        bias_ptr = torch.zeros(N, dtype=torch.float16, device=x.device)
    else:
        bias_ptr = bias.to(torch.float16)

    # 确保 rank 是 2 的幂（Triton constexpr 要求）
    rank_pow2 = triton.next_power_of_2(rank)

    # Pad lora_down 和 lora_up 到 rank_pow2
    if rank_pow2 != rank:
        lora_down_pad = torch.zeros(K, rank_pow2, dtype=lora_down.dtype, device=lora_down.device)
        lora_down_pad[:, :rank] = lora_down
        lora_up_pad = torch.zeros(rank_pow2, N, dtype=lora_up.dtype, device=lora_up.device)
        lora_up_pad[:rank, :] = lora_up
    else:
        lora_down_pad = lora_down
        lora_up_pad = lora_up

    # lora_up 转置为 (N, rank) 便于 kernel 访问
    lora_up_T = lora_up_pad.T.contiguous()   # (N, rank_pow2)

    # Block size：BLOCK_K = group_size
    BLOCK_M = min(64, triton.next_power_of_2(M))
    BLOCK_N = min(64, triton.next_power_of_2(N))
    BLOCK_K = group_size   # 每个 tile 恰好是一个 group

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    svdquant_simple_kernel[grid](
        x, smooth,
        q_w_T, wscales_T,
        lora_down_pad, lora_up_T,
        bias_ptr, y,
        M, N, K,
        rank_pow2, group_size,
        BLOCK_M, BLOCK_N, BLOCK_K,
        has_bias,
    )

    return y
