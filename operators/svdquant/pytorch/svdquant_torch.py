"""
SVDQuant PyTorch 参考实现

实现 SVDQuant W4A4 量化的完整前向传播：

    y = dequant(Q_x) @ dequant(Q_residual) + (x @ lora_down) @ lora_up + bias

数学原理：
    1. 平滑迁移：x̂ = x / smooth_factor（将激活 outlier 迁移到权重）
    2. 激活量化：Q_x = Quantize(x̂, group_size=64)，对称 INT4
    3. SVD 分解（离线）：W_smooth = U@S@V^T，提取低秩部分
    4. 残差量化：Q_residual = Quantize(residual, group_size=64)
    5. 前向：y = dequant(Q_x) @ dequant(Q_residual) + (x @ lora_down) @ lora_up + bias

量化格式（对称 INT4, per-group）：
    - 范围: [-8, 7]
    - Scale: s = max(|x_group|) / 7
    - Quant: q = round(x / s).clamp(-8, 7)
    - Dequant: x̂ = q * s
"""

import torch
import torch.nn.functional as F
import math


# -------------------------------------------------------------------------
# 基本量化原语
# -------------------------------------------------------------------------

def int4_quantize(
    x: torch.Tensor,   # (..., K) FP16/BF16
    group_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对称 INT4 per-group 量化。

    Args:
        x:          (..., K) 输入张量，最后一维按 group_size 分组
        group_size: 每个量化组的大小，K 必须整除 group_size

    Returns:
        q:      (..., K) int8，量化值（范围 [-8, 7]）
        scales: (..., K // group_size) FP16/BF16，每组的缩放因子
    """
    orig_shape = x.shape
    K = x.shape[-1]
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    num_groups = K // group_size

    # reshape: (..., num_groups, group_size)
    x_grouped = x.view(*x.shape[:-1], num_groups, group_size)

    # 计算每组的 scale: max(|x|) / 7
    # scales shape: (..., num_groups)
    scales = x_grouped.abs().max(dim=-1).values / 7.0
    scales = scales.clamp(min=1e-8)   # 防止除零

    # 量化
    # x_grouped / scales[..., None]: (..., num_groups, group_size)
    q_grouped = torch.round(x_grouped / scales.unsqueeze(-1)).clamp(-8, 7).to(torch.int8)

    # 恢复形状
    q = q_grouped.view(orig_shape)
    return q, scales


def int4_dequantize(
    q: torch.Tensor,      # (..., K) int8
    scales: torch.Tensor, # (..., K // group_size) FP16/BF16
    group_size: int = 64,
) -> torch.Tensor:
    """
    INT4 per-group 反量化。

    Returns:
        x: (..., K) FP16/BF16，反量化结果
    """
    K = q.shape[-1]
    num_groups = K // group_size

    q_grouped = q.view(*q.shape[:-1], num_groups, group_size).float()
    # scales: (..., num_groups) -> (..., num_groups, 1)
    x_grouped = q_grouped * scales.float().unsqueeze(-1)
    return x_grouped.view(q.shape).to(scales.dtype)


def int4_pack_uint8(q: torch.Tensor) -> torch.Tensor:
    """
    将 int8 张量（实际范围 [-8, 7]，使用 4 bit）打包成 uint8。
    每两个 INT4 值打包成一个 uint8。

    输入:  (..., K)  int8，K 为偶数
    输出:  (..., K//2) uint8

    低 nibble = 偶数索引元素，高 nibble = 奇数索引元素
    """
    assert q.shape[-1] % 2 == 0, "K must be even for INT4 packing"
    # 将值映射到 [0, 15]：+8 使 [-8,7] -> [0,15]
    q_uint = (q.to(torch.int32) + 8).to(torch.uint8)  # (..., K)
    q_even = q_uint[..., 0::2]  # (..., K//2)
    q_odd  = q_uint[..., 1::2]  # (..., K//2)
    packed = q_even | (q_odd << 4)
    return packed


def int4_unpack_uint8(packed: torch.Tensor) -> torch.Tensor:
    """
    从 uint8 中解包 INT4 值。

    输入:  (..., K//2) uint8
    输出:  (..., K)    int8，值范围 [-8, 7]
    """
    q_even = (packed & 0x0F).to(torch.int8) - 8   # 低 nibble
    q_odd  = (packed >> 4).to(torch.int8) - 8      # 高 nibble
    # 交织合并：(..., K//2, 2) -> (..., K)
    out = torch.stack([q_even, q_odd], dim=-1).view(*packed.shape[:-1], packed.shape[-1] * 2)
    return out


# -------------------------------------------------------------------------
# 离线参数构建
# -------------------------------------------------------------------------

def create_svdquant_params(
    W: torch.Tensor,            # (K, N) FP16/BF16 原始权重
    rank: int = 32,
    group_size: int = 64,
    smooth: torch.Tensor | None = None,  # (K,) 平滑因子，None 则用 1
) -> dict:
    """
    离线构建 SVDQuant 所需参数：
    1. 应用平滑因子：W_smooth = diag(smooth) @ W
    2. SVD 低秩分解：W_smooth ≈ lora_down @ lora_up
    3. 量化残差：W_residual = W_smooth - lora_down @ lora_up

    Args:
        W:          (K, N) 原始权重（PyTorch 惯例 y = xW）
        rank:       SVD 截断秩
        group_size: INT4 量化 group size
        smooth:     (K,) 平滑因子，None 则不做平滑

    Returns:
        dict with keys:
            lora_down:  (K, rank) FP16
            lora_up:    (rank, N) FP16
            q_w:        (K//2, N) uint8 packed 量化残差（或 (K, N//2) 视布局）
            wscales:    (K // group_size, N) FP16 残差权重 scales
            smooth:     (K,) FP16 平滑因子
    """
    K, N = W.shape
    dtype = W.dtype

    # 平滑因子
    if smooth is None:
        smooth = torch.ones(K, dtype=dtype, device=W.device)

    # Step 1: 应用平滑 W_smooth = diag(smooth) @ W
    W_smooth = smooth.unsqueeze(1) * W   # (K, N)

    # Step 2: SVD 低秩分解
    # torch.linalg.svd 返回 U(K,K), S(min(K,N)), Vh(N,N)
    # 使用 full_matrices=False 获取截断 SVD
    W_f32 = W_smooth.float()   # SVD 用 FP32 提高精度
    try:
        U, S, Vh = torch.linalg.svd(W_f32, full_matrices=False)
        # U: (K, min(K,N)), S: (min(K,N),), Vh: (min(K,N), N)
    except Exception as e:
        print(f"SVD failed: {e}, falling back to random LoRA")
        lora_down = torch.randn(K, rank, device=W.device, dtype=dtype) * 0.01
        lora_up   = torch.randn(rank, N, device=W.device, dtype=dtype) * 0.01
        W_residual = W_smooth
    else:
        r = min(rank, S.shape[0])
        S_sqrt = S[:r].sqrt()
        lora_down = (U[:, :r] * S_sqrt.unsqueeze(0)).to(dtype)   # (K, r)
        lora_up   = (Vh[:r, :] * S_sqrt.unsqueeze(1)).to(dtype)  # (r, N)
        # Step 3: 残差
        W_residual = W_smooth - lora_down.float() @ lora_up.float()
        W_residual = W_residual.to(dtype)

    # Step 4: 量化残差权重
    # W_residual shape: (K, N)，沿 K 维度 per-group 量化（按列分组不太合理，
    # 实际中按行（K 维）分组更常见，这里保持与激活量化一致的 group 方向）
    # 我们对权重按 N 维度转置后量化，即每 N 个一组
    # 常见方式：per-output-channel group，沿 K 维 per-group
    # 这里选择：W.T shape (N, K)，对 K 维按 group_size 量化
    W_T = W_residual.T.contiguous()  # (N, K)
    q_w_T, wscales_T = int4_quantize(W_T, group_size=group_size)
    # q_w_T: (N, K) int8, wscales_T: (N, K//group_size)
    # 转置回来，便于 GEMM
    q_w = q_w_T.T.contiguous()     # (K, N) int8
    wscales = wscales_T.T.contiguous()  # (K//group_size, N)

    return {
        "lora_down":  lora_down,        # (K, rank) FP16
        "lora_up":    lora_up,          # (rank, N) FP16
        "q_w":        q_w,              # (K, N) int8 量化残差权重
        "wscales":    wscales,          # (K//group_size, N) FP16 weight scales
        "smooth":     smooth,           # (K,) FP16
        "group_size": group_size,
        "rank":       rank,
    }


# -------------------------------------------------------------------------
# 完整前向传播
# -------------------------------------------------------------------------

def svdquant_forward_torch(
    x: torch.Tensor,           # (M, K) FP16/BF16 输入激活
    q_w: torch.Tensor,         # (K, N) int8 量化残差权重
    wscales: torch.Tensor,     # (K//group_size, N) FP16 weight scales
    lora_down: torch.Tensor,   # (K, rank) FP16
    lora_up: torch.Tensor,     # (rank, N) FP16
    smooth: torch.Tensor,      # (K,) FP16 平滑因子
    bias: torch.Tensor | None = None,  # (N,) FP16
    group_size: int = 64,
) -> torch.Tensor:
    """
    SVDQuant W4A4 完整前向传播（PyTorch 参考实现）。

    计算流程：
        x̂ = x / smooth                           # 平滑激活（将 outlier 转移到权重）
        Q_x, ascales = int4_quantize(x̂)          # 量化激活
        x̂_dequant = int4_dequantize(Q_x, ascales) # 反量化激活
        W_dequant = int4_dequantize(q_w, wscales) # 反量化权重
        y_main = x̂_dequant @ W_dequant            # INT4 GEMM（模拟）
        lora_act = x @ lora_down                  # FP16 LoRA down
        y_lora = lora_act @ lora_up               # FP16 LoRA up
        y = y_main + y_lora + bias

    注意：
    - 这是数值参考实现，性能不是目标
    - 实际 INT4 GEMM 需要专用 hardware kernel（Triton/CUDA 版本）

    Returns:
        y: (M, N) FP16/BF16
    """
    M, K = x.shape
    dtype = x.dtype

    # Step 1: 平滑激活
    x_smooth = x / smooth.unsqueeze(0).to(dtype)     # (M, K)

    # Step 2: 量化激活
    q_x, ascales = int4_quantize(x_smooth, group_size=group_size)
    # q_x: (M, K) int8, ascales: (M, K//group_size)

    # Step 3: 反量化激活和权重，模拟 INT4 GEMM
    x_dq = int4_dequantize(q_x, ascales, group_size=group_size)   # (M, K)
    W_dq = int4_dequantize(q_w.T.contiguous(), wscales.T.contiguous(), group_size=group_size)
    # W_dq: (N, K), 转置得 (K, N)
    W_dq = W_dq.T.contiguous()

    # Step 4: 主 GEMM
    y_main = torch.matmul(x_dq.float(), W_dq.float()).to(dtype)   # (M, N)

    # Step 5: LoRA 部分（用原始精度 x）
    lora_act = torch.matmul(x.float(), lora_down.float()).to(dtype)  # (M, rank)
    y_lora   = torch.matmul(lora_act.float(), lora_up.float()).to(dtype)  # (M, N)

    # Step 6: 合并
    y = y_main + y_lora
    if bias is not None:
        y = y + bias.to(dtype)

    return y


# -------------------------------------------------------------------------
# 便捷封装
# -------------------------------------------------------------------------

class SVDQuantLinear(torch.nn.Module):
    """
    SVDQuant 量化线性层（PyTorch 参考实现）。

    使用方式：
        # 从 FP16 权重创建
        layer = SVDQuantLinear.from_fp16(W, rank=32)
        y = layer(x)
    """

    def __init__(
        self,
        q_w: torch.Tensor,
        wscales: torch.Tensor,
        lora_down: torch.Tensor,
        lora_up: torch.Tensor,
        smooth: torch.Tensor,
        bias: torch.Tensor | None = None,
        group_size: int = 64,
    ):
        super().__init__()
        self.register_buffer("q_w", q_w)
        self.register_buffer("wscales", wscales)
        self.register_buffer("lora_down", lora_down)
        self.register_buffer("lora_up", lora_up)
        self.register_buffer("smooth", smooth)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        self.group_size = group_size

    @classmethod
    def from_fp16(
        cls,
        W: torch.Tensor,
        rank: int = 32,
        group_size: int = 64,
        smooth: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> "SVDQuantLinear":
        """从 FP16 权重构建量化层（包含离线 SVD 分解和量化）。"""
        params = create_svdquant_params(W, rank=rank, group_size=group_size, smooth=smooth)
        return cls(
            q_w=params["q_w"],
            wscales=params["wscales"],
            lora_down=params["lora_down"],
            lora_up=params["lora_up"],
            smooth=params["smooth"],
            bias=bias,
            group_size=group_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return svdquant_forward_torch(
            x, self.q_w, self.wscales,
            self.lora_down, self.lora_up,
            self.smooth, self.bias,
            self.group_size,
        )
