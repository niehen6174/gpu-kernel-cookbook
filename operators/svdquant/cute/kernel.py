"""
SVDQuant CuTe Python DSL 实现

使用 NVIDIA CuTe Python DSL (cutlass.cute) 实现 SVDQuant 前向传播。

V1: Naive — 每个 thread 计算一个输出元素，串行 K 循环，在线反量化
V2: 共享内存分块 — BLOCK_M×BLOCK_N tile，sX + sW smem，BLOCK_K=64
V3: cuBLAS 主 GEMM + CuTe LoRA epilogue（最高性能方案）

Launch 语法（CuTe DSL）：
    kernel(args).launch(grid=(...), block=(...))   ← 注意：不是 kernel[grid, block](args)

共享内存 API：
    smem = SmemAllocator()
    s = smem.allocate_tensor(dtype, layout, alignment)
    cute.arch.barrier()   ← 非 syncthreads()
"""

import torch

try:
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cutlass.utils import SmemAllocator
    CUTE_AVAILABLE = True
except ImportError:
    CUTE_AVAILABLE = False


if CUTE_AVAILABLE:
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    # V2 使用更大的 BLOCK_K（64）以减少 barrier 次数并提升 smem 带宽利用率
    # 注意：BLOCK_K_V2 = BLOCK_N_V2（每个 thread 负责加载一列 K tile）
    BLOCK_K_V2 = 64

    # V3 LoRA epilogue block size
    BLOCK_M_V3 = 32
    BLOCK_N_V3 = 32

    # -----------------------------------------------------------------------
    # V1: Naive — 1 thread per output element, 串行 K，在线反量化
    # -----------------------------------------------------------------------
    @cute.kernel
    def svdquant_cute_v1_kernel(
        X: cute.Tensor,         # (M, K) FP16，已应用 smooth
        Q_W: cute.Tensor,       # (N, K) FP16（int8 已转 fp16 传入）
        W_scales: cute.Tensor,  # (N, G) FP16，G = K // group_size
        Lora_act: cute.Tensor,  # (M, R) FP32，已预计算
        Lora_up: cute.Tensor,   # (N, R) FP16
        Y: cute.Tensor,         # (M, N) FP16 输出
        M: cute.Int32,
        N: cute.Int32,
        K: cute.Int32,
        R: cute.Int32,
        GROUP_SIZE: cute.Int32,
    ):
        tidx, tidy, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        row = bidy * BLOCK_M + tidy
        col = bidx * BLOCK_N + tidx

        if row < M and col < N:
            # W4A4 GEMM：在线反量化 + 累加
            acc = 0.0
            for k in range(K):
                g     = k // GROUP_SIZE
                x_val = X[row, k]
                qw    = Q_W[col, k]
                ws    = W_scales[col, g]
                acc   = acc + x_val * (qw * ws)

            # LoRA：lora_act @ lora_up^T
            lora_sum = 0.0
            for r in range(R):
                lora_sum = lora_sum + Lora_act[row, r] * Lora_up[col, r]

            Y[row, col] = cute.Float16(acc + lora_sum)

    @cute.jit
    def _launch_v1(
        X: cute.Tensor,
        Q_W: cute.Tensor,
        W_scales: cute.Tensor,
        Lora_act: cute.Tensor,
        Lora_up: cute.Tensor,
        Y: cute.Tensor,
        M: cute.Int32,
        N: cute.Int32,
        K: cute.Int32,
        R: cute.Int32,
        GROUP_SIZE: cute.Int32,
    ):
        svdquant_cute_v1_kernel(
            X, Q_W, W_scales, Lora_act, Lora_up, Y, M, N, K, R, GROUP_SIZE
        ).launch(
            grid=((N + BLOCK_N - 1) // BLOCK_N,
                  (M + BLOCK_M - 1) // BLOCK_M,
                  1),
            block=(BLOCK_N, BLOCK_M, 1),
        )

    # -----------------------------------------------------------------------
    # V2: 共享内存分块 — sX + sW tile，BLOCK_K=64，减少 barrier 次数
    # -----------------------------------------------------------------------
    @cute.kernel
    def svdquant_cute_v2_kernel(
        X: cute.Tensor,         # (M, K) FP16
        Q_W: cute.Tensor,       # (N, K) FP16
        W_scales: cute.Tensor,  # (N, G) FP16
        Lora_act: cute.Tensor,  # (M, R) FP32
        Lora_up: cute.Tensor,   # (N, R) FP16
        Y: cute.Tensor,         # (M, N) FP16
        M: cute.Int32,
        N: cute.Int32,
        K: cute.Int32,
        R: cute.Int32,
        GROUP_SIZE: cute.Int32,
    ):
        tidx, tidy, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        # 共享内存：每个 block 申请 X tile 和 W tile，BLOCK_K_V2=64
        smem = SmemAllocator()
        sX = smem.allocate_tensor(
            cute.Float16,
            cute.make_layout((BLOCK_M, BLOCK_K_V2), stride=(BLOCK_K_V2, 1)),
            16,
        )
        sW = smem.allocate_tensor(
            cute.Float16,
            cute.make_layout((BLOCK_N, BLOCK_K_V2), stride=(BLOCK_K_V2, 1)),
            16,
        )

        row = bidy * BLOCK_M + tidy
        col = bidx * BLOCK_N + tidx
        acc = 0.0

        for k_start in range(0, K, BLOCK_K_V2):
            # 每个 thread 负责加载 2 个 K 元素（因为 BLOCK_K_V2=64，BLOCK_N=32）
            for kk in range(BLOCK_K_V2 // BLOCK_N):
                k_local = kk * BLOCK_N + tidx
                if row < M and (k_start + k_local) < K:
                    sX[tidy, k_local] = X[row, k_start + k_local]
                else:
                    sX[tidy, k_local] = cute.Float16(0.0)

                col_w = bidx * BLOCK_N + tidy
                if col_w < N and (k_start + k_local) < K:
                    k_abs = k_start + k_local
                    g     = k_abs // GROUP_SIZE
                    qw    = Q_W[col_w, k_abs]
                    ws    = W_scales[col_w, g]
                    sW[tidy, k_local] = qw * ws
                else:
                    sW[tidy, k_local] = cute.Float16(0.0)

            cute.arch.barrier()

            # 内核乘加（串行 K 维，覆盖更多 K）
            if row < M and col < N:
                k_lim = min(BLOCK_K_V2, K - k_start)
                for k in range(k_lim):
                    acc = acc + sX[tidy, k] * sW[tidx, k]

            cute.arch.barrier()

        # LoRA 部分
        if row < M and col < N:
            lora_sum = 0.0
            for r in range(R):
                lora_sum = lora_sum + Lora_act[row, r] * Lora_up[col, r]
            Y[row, col] = cute.Float16(acc + lora_sum)

    @cute.jit
    def _launch_v2(
        X: cute.Tensor,
        Q_W: cute.Tensor,
        W_scales: cute.Tensor,
        Lora_act: cute.Tensor,
        Lora_up: cute.Tensor,
        Y: cute.Tensor,
        M: cute.Int32,
        N: cute.Int32,
        K: cute.Int32,
        R: cute.Int32,
        GROUP_SIZE: cute.Int32,
    ):
        svdquant_cute_v2_kernel(
            X, Q_W, W_scales, Lora_act, Lora_up, Y, M, N, K, R, GROUP_SIZE
        ).launch(
            grid=((N + BLOCK_N - 1) // BLOCK_N,
                  (M + BLOCK_M - 1) // BLOCK_M,
                  1),
            block=(BLOCK_N, BLOCK_M, 1),
        )


# -----------------------------------------------------------------------
# AOT 编译缓存
# -----------------------------------------------------------------------
_cache_v1: dict = {}
_cache_v2: dict = {}

if CUTE_AVAILABLE:
    # -----------------------------------------------------------------------
    # V3: CuTe LoRA Epilogue Kernel
    # 主 GEMM 由 Python 侧 torch.matmul (cuBLAS) 完成，
    # CuTe kernel 只负责 epilogue：Y += lora_act @ lora_up^T + bias
    # -----------------------------------------------------------------------
    @cute.kernel
    def svdquant_cute_v3_lora_epilog_kernel(
        Lora_act: cute.Tensor,   # (M, R) FP32
        Lora_up: cute.Tensor,    # (N, R) FP16
        Y: cute.Tensor,          # (M, N) FP16  (in-place update)
        Bias: cute.Tensor,       # (N,) FP16，若 HAS_BIAS=False 传零张量
        M: cute.Int32,
        N: cute.Int32,
        R: cute.Int32,
        HAS_BIAS: cute.Int32,
    ):
        """
        每个 thread 负责一个输出元素 Y[row, col]：
          Y[row, col] += sum_r(Lora_act[row, r] * Lora_up[col, r]) + Bias[col]
        rank R 通常很小（≤64），串行 r 循环开销小。
        """
        tidx, tidy, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        row = bidy * BLOCK_M_V3 + tidy
        col = bidx * BLOCK_N_V3 + tidx

        if row < M and col < N:
            lora_sum = 0.0
            for r in range(R):
                lora_sum = lora_sum + Lora_act[row, r] * Lora_up[col, r]
            val = Y[row, col] + cute.Float16(lora_sum)
            if HAS_BIAS:
                val = val + Bias[col]
            Y[row, col] = val

    @cute.jit
    def _launch_v3_epilog(
        Lora_act: cute.Tensor,
        Lora_up: cute.Tensor,
        Y: cute.Tensor,
        Bias: cute.Tensor,
        M: cute.Int32,
        N: cute.Int32,
        R: cute.Int32,
        HAS_BIAS: cute.Int32,
    ):
        svdquant_cute_v3_lora_epilog_kernel(
            Lora_act, Lora_up, Y, Bias, M, N, R, HAS_BIAS
        ).launch(
            grid=((N + BLOCK_N_V3 - 1) // BLOCK_N_V3,
                  (M + BLOCK_M_V3 - 1) // BLOCK_M_V3,
                  1),
            block=(BLOCK_N_V3, BLOCK_M_V3, 1),
        )

_cache_v3: dict = {}


def svdquant_forward_cute_v3(
    x: torch.Tensor,           # (M, K) FP16
    q_w: torch.Tensor,         # (K, N) int8
    wscales: torch.Tensor,     # (K//group_size, N) FP16
    lora_down: torch.Tensor,   # (K, rank) FP16
    lora_up: torch.Tensor,     # (rank, N) FP16
    smooth: torch.Tensor,      # (K,) FP16
    bias: torch.Tensor | None = None,
    group_size: int = 64,
) -> torch.Tensor:
    """
    SVDQuant V3：cuBLAS 主 GEMM + CuTe LoRA Epilogue。

    策略：
    1. Python 侧反量化权重（离线或在线）
    2. torch.matmul 调用 cuBLAS Tensor Core 做主 GEMM（最快路径）
    3. CuTe kernel 做 epilogue：y += lora_act @ lora_up^T + bias

    性能特性：
    - 主 GEMM 走 cuBLAS，性能接近 FP16 理论上限
    - CuTe epilogue 只处理小型低秩计算（rank ≤ 64）
    """
    if not CUTE_AVAILABLE:
        raise ImportError("cutlass.cute 不可用，请执行: pip install nvidia-cutlass")

    from operators.svdquant.pytorch.svdquant_torch import int4_dequantize

    M, K = x.shape
    _, N = q_w.shape
    rank = lora_down.shape[1]

    # Step 1: 应用 smooth
    x_smooth = (x / smooth.unsqueeze(0)).contiguous()    # (M, K) FP16

    # Step 2: 反量化权重（在线），得到 FP16 权重
    # q_w: (K, N) int8, wscales: (K//G, N) FP16
    W_dq = int4_dequantize(
        q_w.T.contiguous(),
        wscales.T.contiguous(),
        group_size=group_size,
    ).T.contiguous()    # (K, N) FP16

    # Step 3: 主 GEMM via cuBLAS
    y = torch.matmul(x_smooth, W_dq)    # (M, N) FP16

    # Step 4: LoRA down projection（FP16）
    lora_act = (x.float() @ lora_down.contiguous().float()).half()   # (M, rank) FP16

    # Step 5: CuTe epilogue：y += lora_act @ lora_up^T + bias
    lora_up_T = lora_up.T.contiguous()   # (N, rank) FP16
    has_bias = bias is not None
    if not has_bias:
        bias_tensor = torch.zeros(N, dtype=torch.float16, device=x.device)
    else:
        bias_tensor = bias.to(torch.float16).contiguous()

    y_ct       = from_dlpack(y)
    la_ct      = from_dlpack(lora_act)
    lu_ct      = from_dlpack(lora_up_T)
    bias_ct    = from_dlpack(bias_tensor)

    key = (M, K, N, rank, group_size)
    if key not in _cache_v3:
        _cache_v3[key] = cute.compile(
            _launch_v3_epilog,
            la_ct, lu_ct, y_ct, bias_ct,
            M, N, rank, int(has_bias),
        )
    _cache_v3[key](la_ct, lu_ct, y_ct, bias_ct, M, N, rank, int(has_bias))

    return y


def svdquant_forward_cute(
    x: torch.Tensor,           # (M, K) FP16
    q_w: torch.Tensor,         # (K, N) int8
    wscales: torch.Tensor,     # (K//group_size, N) FP16
    lora_down: torch.Tensor,   # (K, rank) FP16
    lora_up: torch.Tensor,     # (rank, N) FP16
    smooth: torch.Tensor,      # (K,) FP16
    bias: torch.Tensor | None = None,
    group_size: int = 64,
    version: str = "v1",
) -> torch.Tensor:
    """
    SVDQuant 前向传播 CuTe 实现。

    预处理（Python 侧）：
      - smooth 应用：x_smooth = x / smooth
      - LoRA down 预计算：lora_act = x @ lora_down（FP32）
      - 权重转置：q_w (K,N) → (N,K)，int8 → fp16（DSL 目前不支持 int8）

    CuTe kernel 负责：
      - 在线反量化权重：qw * wscale
      - 主 GEMM：x_smooth @ W_dequant^T
      - LoRA 加法：lora_act @ lora_up^T

    Args:
        version: "v1"（naive）或 "v2"（共享内存分块）
    """
    if not CUTE_AVAILABLE:
        raise ImportError("cutlass.cute 不可用，请执行: pip install nvidia-cutlass")

    M, K = x.shape
    _, N = q_w.shape
    rank = lora_down.shape[1]

    # 预处理
    x_smooth  = (x / smooth.unsqueeze(0)).contiguous()          # (M, K) FP16
    lora_act  = (x.float() @ lora_down.contiguous().float())     # (M, rank) FP32
    q_w_T     = q_w.T.contiguous().to(torch.float16)             # (N, K) FP16
    wscales_T = wscales.T.contiguous()                           # (N, K//G) FP16
    lora_up_T = lora_up.T.contiguous()                           # (N, rank) FP16
    y         = torch.empty(M, N, dtype=torch.float16, device=x.device)

    # CuTe Tensor
    x_ct  = from_dlpack(x_smooth)
    qw_ct = from_dlpack(q_w_T)
    ws_ct = from_dlpack(wscales_T)
    la_ct = from_dlpack(lora_act.contiguous())
    lu_ct = from_dlpack(lora_up_T)
    y_ct  = from_dlpack(y)

    key = (M, K, N, rank, group_size)

    if version == "v1":
        if key not in _cache_v1:
            _cache_v1[key] = cute.compile(
                _launch_v1, x_ct, qw_ct, ws_ct, la_ct, lu_ct, y_ct,
                M, N, K, rank, group_size,
            )
        _cache_v1[key](x_ct, qw_ct, ws_ct, la_ct, lu_ct, y_ct, M, N, K, rank, group_size)
    else:
        if key not in _cache_v2:
            _cache_v2[key] = cute.compile(
                _launch_v2, x_ct, qw_ct, ws_ct, la_ct, lu_ct, y_ct,
                M, N, K, rank, group_size,
            )
        _cache_v2[key](x_ct, qw_ct, ws_ct, la_ct, lu_ct, y_ct, M, N, K, rank, group_size)

    if bias is not None:
        y = y + bias.to(torch.float16)
    return y
