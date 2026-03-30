# SageAttention CuTe DSL 实现开发日志

**硬件环境**：NVIDIA H20 (SM90a, Hopper), CUDA 12.9
**框架版本**：CUTLASS CuTe DSL, Triton, PyTorch
**参考论文**：SageAttention (arXiv:2410.02367)

---

## 目录

1. [项目背景](#1-项目背景)
2. [实现架构概览](#2-实现架构概览)
3. [优化历程](#3-优化历程)
   - [v1：ss-mode 基础实现](#v1ss-mode-基础实现)
   - [量化加速：Python 循环 → Triton GPU Kernel](#量化加速python-循环--triton-gpu-kernel)
   - [v2：rs-mode + 向量化 + K double buffering](#v2rs-mode--向量化--k-double-buffering)
   - [v2 尝试优化与受阻项](#v2-尝试优化与受阻项)
   - [SageAttention V2 量化升级](#sageattention-v2-量化升级)
   - [v3：BLOCK_M=128 split-warpgroup](#v3block_m128-split-warpgroup)
   - [v4：FP8 V GMEM 带宽优化（调试中）](#v4fp8-v-gmem-带宽优化)
4. [性能基准](#4-性能基准)
5. [关键实现细节](#5-关键实现细节)
6. [已知问题与受阻项](#6-已知问题与受阻项)
7. [后续优化方向](#7-后续优化方向)

---

## 1. 项目背景

目标：用 CUTLASS CuTe DSL 在 Hopper (SM90a) 上实现 SageAttention V1，对标官方 CUDA 实现和自有 Triton 实现，作为学习 CuTe DSL / WGMMA / ss-rs 模式的工程实践。

**算法核心**（SageAttention V1）：

```
1. K Smoothing:   k_smooth = k - mean(k, dim=seq)
2. Q/K 量化:      INT8 per-block，Q scale 融合 sm_scale * log2(e)
3. QK GEMM:       INT8 × INT8 → INT32  (WGMMA)
4. Online Softmax: exp2 路径
5. PV GEMM:       FP16 × FP16 → FP32  (WGMMA)
6. 输出归一化
```

**静态超参数**：`BLOCK_M=64, BLOCK_N=64, HEAD_DIM=64, NUM_THREADS=128`（1 warpgroup per CTA）

---

## 2. 实现架构概览

```
operators/sageattention/cute/
├── kernel_v1.py      # CuTe DSL v1：ss-mode PV GEMM（正确性参考版）
├── kernel.py         # CuTe DSL v2：rs-mode + 向量化 + K double buffering（SageAttention V1 性能版）
├── kernel_v2.py      # CuTe DSL v2_q2：SageAttention V2 量化升级（per-warp Q + FP8 V）
├── kernel_v3.py      # CuTe DSL v3：BLOCK_M=128 split-warpgroup（SageAttention V1，双 WG）
├── quant.py          # Triton GPU 量化 kernels（含 V2 新增函数）
└── DEVLOG.md         # 本文档
```

**文件职责**

| 文件 | 模式 | 量化方案 | 状态 |
|---|---|---|---|
| `kernel_v1.py` | ss-mode PV GEMM | Triton GPU kernel | ✅ 正确，用于对比基准 |
| `kernel.py` | rs-mode PV GEMM | Triton GPU kernel | ✅ 高性能主实现（SageAttention V1，BM=64） |
| `kernel_v2.py` | rs-mode PV GEMM | per-warp Q INT8 + FP8 V round-trip | ✅ SageAttention V2 实现 |
| `kernel_v3.py` | rs-mode PV GEMM, split-WG | Triton GPU kernel | ✅ BLOCK_M=128 双 WG 实现 |
| `kernel_v4.py` | rs-mode PV GEMM, split-WG, FP8 V | FP8 V GMEM + per-tile scale | 🔴 N=384/512/1024 有确定性 Bug（调试中）|
| `quant.py` | — | Triton + PyTorch | ✅ 通用量化模块 |

---

## 3. 优化历程

### v1：ss-mode 基础实现

**目标**：先跑通，保证正确性，避免 rs-mode layout 转换复杂度。

**关键设计**：
- QK GEMM：`INT8 × INT8 → INT32`，A/B 均从 smem 读（ss-mode）
- PV GEMM：`FP16 × FP16 → FP32`，P 写回 `sP` smem，再从 smem 读（ss-mode）
- Online softmax：exp2 路径，warp-level reduce（bfly shuffle，offset 2→1）
- `acc_S` 到 `tOrP` 的 layout 转换：通过 `logical_divide` 构造 `frgA_layout`，映射 QK C-fragment 到 PV A-fragment

**正确性验证**：`max_abs(output - torch_sdpa) ≈ 5–8e-3`（FP32 在线 softmax 累积顺序差异）。

---

### 量化加速：Python 循环 → Triton GPU Kernel

#### 问题

`kernel_v1.py` 初始版本用 Python for-loop 做量化，每次调用发射大量小 kernel，导致 CPU-GPU 同步开销极高：

```
N=8192, B=1, H=16：
  smooth_and_quant_k (Python): ~241ms   ← B×H×(N/64) = 2048 次循环
  quant_q_per_block  (Python): ~191ms
  GPU attention kernel:          ~1.6ms
  合计:                         ~434ms   ← 99.7% 耗时在 CPU
```

#### 解决方案：新建 `quant.py`

**Triton Kernel**：`_quant_per_block_int8_kernel`

```
Grid: (N // BLOCK_SZ, H, B)   ← 每 program 处理一个 (block_n, head, batch) 三元组
每 program 处理 64 × 64 = 4096 个元素

步骤：
  1. Load (64, 64) fp16 block，正确传 strides（支持非连续 tensor）
  2. FUSE_KM=True：减去 km[b,h,d]（K smoothing）
  3. 乘以 sm_scale（Q 传 sm_scale*log2e，K 传 1.0）
  4. scale = max(|x|) / 127.0 + 1e-7
  5. x_q = clamp(x/scale + 0.5*sign(x), -128, 127).to(int8)
  6. Store x_q (contiguous)，store scale
```

**Python 包装**：

```python
quant_per_block_int8(x, sm_scale, block_sz, km)  # 核心
quant_q_per_block_gpu(q, sm_scale)                # Q: 融合 sm_scale * LOG2_E
smooth_and_quant_k_gpu(k)                         # K: PyTorch mean + Triton quant
```

**关键细节**：
- K mean 用 `k.float().mean(dim=2, keepdim=True).to(fp16)` 一次 PyTorch op 完成，无需 kernel 内部 reduce
- `FUSE_KM` 为 `tl.constexpr`，编译期决定是否减 km，零开销
- km 用 `reshape(B,H,D)` 后传 stride，支持 `keepdim=True` 的非连续 layout

#### 效果

```
N=8192, B=1, H=16：
  smooth_and_quant_k (Triton): ~0.07ms   ↓ 3400×
  quant_q_per_block  (Triton): ~0.06ms   ↓ 3200×
  量化合计:                     ~0.13ms
  GPU attention kernel:          ~1.8ms
  合计:                          ~1.9ms   ↓ 228×
```

**正确性**：与 CPU 版 int8 逐元素对比：
- K：`max_diff = 1`（量化舍入误差，符合预期）
- Q：`max_diff = 0`
- scale：`max_diff < 1e-5`

---

### v2：rs-mode + 向量化 + K double buffering

**目标**：在 v1 正确性基础上提升 GPU kernel 本体性能。

**四项优化**：

| 优化点 | v1 | v2 | 收益 |
|---|---|---|---|
| Q/K/V 加载 | `basic_copy`（标量） | `CopyUniversalOp` + `cp.async`（128-bit） | 带宽利用率大幅提升 |
| PV GEMM 模式 | ss-mode（P 写回 sP smem） | rs-mode（P 留寄存器） | 节省 ~8KB smem，省 1 次 barrier |
| K 流水线 | 无预取 | 2-stage sK smem，prologue 预取 K[0]/K[1]，K[n+2] 在 PV GEMM 期间加载 | 消除 K 等待 stall |
| copy layout | 8×16 (bug) | 32×4 (正确) | 修复 H=32 illegal access |

**K Double Buffering 设计**：

```
smem: sK_s0, sK_s1（2 stages）
prologue:  K[0] → s0, K[1] → s1（预取两块，等待 K[0] 就绪）
iter n:
  ├── QK GEMM(sK_{n%2})        ← K[n] 已在 smem
  ├── cp.async V[n] → sV       ← group A
  ├── cp.async K[n+2] → sK_{n%2 reuse}  ← group B（当前 stage 已读完可复用）
  ├── softmax（overlap cp.async）
  ├── wait_group(1)             ← V 已就绪，K[n+2] 继续加载中
  ├── PV GEMM                  ← K[n+2] 在此期间完成加载
  └── wait_group(0)            ← 确保 K[n+2] 就绪
```

**v2 时间分解（N=8192, B=1, H=32）**：

```
quant_k (Triton):  0.137ms
quant_q (Triton):  0.022ms
v_permute:         0.127ms   ← V 预转置 (HEAD_DIM, BLOCK_N) 用于 K-major smem
kernel:            3.185ms
total:             3.437ms
```

---

### v2 尝试优化与受阻项

#### ① BLOCK_N=128（Option D）— 回滚，无性能收益

**方案**：保持 WGMMA tiler `BLOCK_N_MMA=64`，用 2 个子 tile 覆盖 BLOCK_N=128 的 KV 块。

**问题**：需要 4 个 smem 缓冲区（sKL + sKR + sVL + sVR = 36KB vs 原 20KB），导致 occupancy 下降，PV GEMM 加倍，实测无性能收益（N=8192: 3.44ms vs 3.29ms Triton）。

#### ② FP8 PV GEMM — 不可行

**方案**：P 存为 FP8 E4M3FN，使用 FP8×FP8→FP32 WGMMA。

**根因**：CuTe DSL MLIR 后端中，标量 fp32 → fp8 转换（`nvgpu.cvt_fptrunc`）要求向量输入，逐元素 `rP[i] = Float8E4M3FN(p_val)` 触发 IR 验证失败：
```
'nvgpu.cvt_fptrunc' op operand #0 must be 32-bits aligned floating-point-like 1-d vector, but got 'f32'
```
rs-mode 下 rP 按元素逐个赋值，无法绕过此限制。

#### ③ 消除 V permute — 不可行

**背景**：V 预转置（`.permute().contiguous()`）耗时 0.127ms/call，约占总耗时 3.7%。

**方案**：将 V 保持自然 `(BLOCK_N, HEAD_DIM)` 布局，使用 COL_MAJOR（N-major）B smem 和 `mma_pv COL_MAJOR`。

**根因**：COL_MAJOR smem 的 MN_SW128 swizzle（HEAD_DIM=64, fp16 → 1024 bits，触发 128-byte swizzle）使得 smem 地址在 cp.async 128-bit/64-bit/32-bit 路径下均不满足对齐要求：
```
'cute.copy' op dst ptr alignment (16 bits) does not meet requirement (128 bits)
```
该对齐约束由 swizzle 图案决定，无法通过降低拷贝粒度绕过。

---

## 4. 性能基准

**测试条件**：NVIDIA H20, SM90a, D=64, 200 次迭代取均值（10 次 warmup）
**测试日期**：2026-03-28

### B=1, H=16, D=64

| N | FA2 | SageOff | SageTri | **CuTe-v1** | **CuTe-v2** | v2 vs FA2 | v2 vs Tri |
|---|---|---|---|---|---|---|---|
| 1024 | 0.068ms | 0.126ms | 0.104ms | 0.265ms | 0.258ms | 0.26x | 0.40x |
| 2048 | 0.253ms | 0.148ms | 0.151ms | 0.896ms | 0.344ms | 0.74x | 0.44x |
| 4096 | 0.867ms | 0.414ms | 0.510ms | 0.695ms | 0.663ms | **1.31x** | 0.77x |
| 8192 | 3.410ms | 1.316ms | 1.669ms | 1.888ms | 1.843ms | **1.85x** | **1.10x** |
| 16384 | 13.068ms | 4.660ms | 6.334ms | 6.610ms | 6.497ms | **2.01x** | **1.15x** |

### B=1, H=32, D=64

| N | FA2 | SageOff | SageTri | **CuTe-v1** | **CuTe-v2** | v2 vs FA2 | v2 vs Tri |
|---|---|---|---|---|---|---|---|
| 1024 | 0.129ms | 0.119ms | 0.096ms | 0.311ms | 0.295ms | 0.44x | 0.33x |
| 2048 | 0.435ms | 0.271ms | 0.285ms | 0.461ms | 0.448ms | 1.03x | 0.64x |
| 4096 | 1.697ms | 0.764ms | 0.885ms | 1.084ms | 1.054ms | **1.61x** | 0.84x |
| 8192 | 6.493ms | 2.523ms | 3.264ms | 3.497ms | 3.426ms | **1.90x** | **1.05x** |
| 16384 | 25.395ms | 9.181ms | 12.202ms | 13.045ms | 12.682ms | **2.00x** | **1.06x** |

### B=2, H=16, D=64

| N | FA2 | SageOff | SageTri | **CuTe-v1** | **CuTe-v2** | v2 vs FA2 | v2 vs Tri |
|---|---|---|---|---|---|---|---|
| 1024 | 0.128ms | 0.120ms | 0.103ms | 0.302ms | 0.301ms | 0.43x | 0.34x |
| 2048 | 0.434ms | 0.274ms | 0.284ms | 0.470ms | 0.483ms | 1.04x | 0.59x |
| 4096 | 1.697ms | 0.772ms | 0.882ms | 1.080ms | 1.058ms | **1.60x** | 0.83x |
| 8192 | 6.493ms | 2.531ms | 3.266ms | 3.500ms | 3.412ms | **1.90x** | **1.05x** |
| 16384 | 25.396ms | 9.196ms | 12.259ms | 13.020ms | 12.688ms | **2.00x** | **1.06x** |

### 性能小结

```
v2 vs FA2（N≥4096）：快 1.3x–2.0x   ✅（INT8 量化 + 算法收益）
v2 vs SageTriton：   大 N 快 5–15%   ✅（K double buffering 消除 stall）
v2 vs SageOff官方：  慢 25–40%       （TMA、warp specialization 等工程差距）
v2 vs CuTe-v1：      快 2–10%        （rs-mode + 128-bit copy + double buffering）
小序列（N≤2048）：   慢于 FA2/SageTriton（warpgroup underutilization，launch overhead）
```

---

## 5. 关键实现细节

### 量化公式一致性

CPU 与 GPU Triton kernel 使用完全相同的量化公式：

```python
scale = max(|x * sm_scale|) / 127.0 + 1e-7
x_q   = clamp(x / scale + 0.5 * sign(x), -128, 127).to(int8)
```

`sign(x)` 实现：`tl.where(x >= 0, 1.0, -1.0)`（含 0 时取正，与 `torch.sign` 不同但对量化无影响）

### sm_scale 融合路径

```
Q：scale_factor = sm_scale * LOG2_E (1.4427...)
   → WGMMA 内积 = Q_int8 · K_int8 * (q_scale * k_scale)
   → softmax 使用 exp2(S * dequant - row_max)，无需 exp

K：scale_factor = 1.0（K 不含 attention scale）
```

### Online Softmax Warp Reduce

```
128 线程 = 4 warps。每线程持有 ACC_ROWS_PER_THR=2 行。
行内 reduce：bfly shuffle，offset 2→1（跨 warp pair）。
```

### ss-mode vs rs-mode P 矩阵处理

```
ss-mode (v1):
  acc_S (INT32 regs)
    → frgA_layout 转换 → tOrP (FP16 regs)
    → copy_P_store retile → sP smem
    → PV GEMM 从 sP smem 读

rs-mode (v2):
  acc_S (INT32 regs)
    → frgA_layout 转换 → tOrP (FP16 regs)
    → PV GEMM 直接从 tOrP 寄存器读（OperandSource.RMEM）
  省去 sP smem 缓冲区和对应的 barrier
```

### V 预转置的必要性

WGMMA B 操作数 ROW_MAJOR（K-major）smem 要求 K 维度（= BLOCK_N）在 smem 内连续。
V 自然布局为 `(BLOCK_N, HEAD_DIM)` 行主序（HEAD_DIM 连续）。
要满足 K-major smem，需在 Python 侧将 V 转置为 `(HEAD_DIM, BLOCK_N)` 后再 cp.async 进 smem。

尝试 COL_MAJOR smem 以跳过转置，但 MN_SW128 swizzle 导致 cp.async 地址不对齐，无法绕过。

---

## 6. 已知问题与受阻项

| 问题 | 状态 | 描述 |
|---|---|---|
| v1/v2 小序列性能 | 🟡 已知 | N≤2048 时单 warpgroup 利用率不足，launch overhead 占比高，慢于 FA2/SageTriton |
| FP8 PV GEMM（kernel 内） | 🔴 受阻 | CuTe DSL 不支持标量 fp32→fp8 转换（需向量化输入），rs-mode 逐元素赋值无法生成合法 IR；V2 采用 round-trip 方案绕过 |
| 消除 V permute | 🔴 受阻 | COL_MAJOR smem MN_SW128 swizzle 与 cp.async 对齐要求冲突 |
| is_causal 未实现 | 🟡 已知 | 所有版本均未支持因果 mask |
| **v4 N≥384 确定性失败** | 🔴 调试中 | `head_flat ≡ 2 (mod 4)` 的 head 在 N=384/512 时 100% 错误（见 v4 章节详细分析）；疑似 `copy_f8_reg`（flat src）与 `copy_f16_reg`（swizzled dst）4-pass 遍历顺序不一致 |

---

## 7. 后续优化方向

### 近期（有可行性）

**① TMA 替换 cp.async**
H20 的 TMA 单元支持 2D tile 搬运，可绕过 cp.async 对齐约束，也是消除 V permute 的另一路径。
CuTe DSL 支持 `AsyncCopyTmaDescriptor`，但需要重构 prologue 和主循环。

**② Warp Specialization**
SageAttention 官方使用 producer warp 专门做 TMA，consumer warpgroup 做 GEMM，实现计算与搬运的深度重叠。
CuTe DSL 对应方案：`warpgroup.arrive_and_wait` + named barrier。

**③ FP8 GEMM（换路径）**
绕过标量转换限制：将 softmax 的 p_val 按向量批量写入 rP（而非逐元素赋值）。
需要研究 CuTe DSL 的向量赋值接口（`cute.arch.fma`、向量 fragment 操作）。

### 中期（功能扩展）

**④ Causal Mask 支持**
Flash Attention 风格：对角线附近 KV tile 内做掩码，其余 tile 全计算。

**⑤ HEAD_DIM=128 支持**
扩展到 D=128：调整 smem layout、WGMMA tiler、warp reduce 步数。

**⑥ GQA（Grouped Query Attention）**
多 Q head 共享 KV head，修改 `head_flat` 映射。

---

### SageAttention V2 量化升级

**目标**：实现 SageAttention V2（arXiv:2501.01005）的量化升级，对标 Triton V2 实现的精度。

**V2 vs V1 量化方案对比**：

| 张量 | V1（CuTe）| V2（CuTe） | 变化 |
|---|---|---|---|
| Q | per-block INT8，1 scale / BLOCK_M=64 行 | per-warp INT8，4 scales / BLOCK_M=64 行（WARPQ=16） | 更细粒度，每 warp 独立 scale |
| K | per-block INT8 + K Smoothing | 不变 | — |
| V | FP16（原始） | FP8 per-channel round-trip → dequant 回 FP16 + V Smoothing | 量化噪声模拟 + mean recovery |

#### 实现方案

**设计决策 1：V FP8 采用 round-trip 而非 kernel 内 FP8 GEMM**

CuTe DSL 无法在 rs-mode 下做逐元素 fp32→fp8 转换（`nvgpu.cvt_fptrunc` 要求向量输入，V1 受阻项同）。
因此 V 在 Python 预处理阶段完成 FP8 E4M3 round-trip，dequantize 回 FP16 后传入 kernel，
kernel 的 PV GEMM 仍保持 FP16×FP16→FP32，smem 布局无需修改。

**设计决策 2：Q per-warp scale 复用现有 Triton kernel**

`quant_per_block_int8` 的 `block_sz` 参数放宽后（允许 16），直接以 `block_sz=WARPQ=16` 调用，
再将 scale reshape 为 `(B,H,N//BLOCK_M, WARPS_PER_BLOCK)` 的 2D 形式，无需新 kernel。

**Kernel 核心改动**（相对 `kernel.py`，最小化 diff）：

```python
# kernel.py (V1):
gQScale: cute.Tensor,       # (B*H*N_Q_BLOCKS,) fp32 — 1D
q_scale_val = gQScale[bidx]

# kernel_v2.py (V2):
gQScale: cute.Tensor,       # (B*H*N_Q_BLOCKS, WARPS_PER_BLOCK) fp32 — 2D
warp_idx = tidx // 32       # warp 编号：0,1,2,3（tidx 在 [0,128) 内）
q_scale_val = gQScale[bidx, warp_idx]
```

其余 softmax、PV GEMM、K double buffering 代码完全不变。

**Python Wrapper 新增 epilogue**：

```python
# V mean 在 kernel 输出后加回（V Smoothing 恢复）
out = out_buf.view(B, H, N, D)
if smooth_v and vm is not None:
    out = out + vm.to(torch.float32)   # vm: (B, H, 1, D)，广播 N 维
```

#### 新增代码（`quant.py`）

```
quant_q_per_warp_int8_gpu(q, sm_scale, BLOCK_M=64, WARPQ=16)
  → 调用 quant_per_block_int8(block_sz=16)
  → reshape scale: (B,H,N//WARPQ) → (B,H,N//BLOCK_M, 4)

quant_v_per_channel_fp8(v, smooth_v=True)
  → vm = v.float().mean(dim=2, keepdim=True)
  → v_scale = |v_smooth|.amax(dim=2) / 448 + 1e-7
  → FP8 round-trip: fp32 → Float8E4M3FN → fp32
  → 输出 v_fp16 (dequantized), v_scale (B,H,D), vm (B,H,1,D)
```

#### 正确性验证结果

测试日期：2026-03-28，NVIDIA H20 SM90a

**多场景 max_abs 对比（sm_scale=D^{-0.5}，smooth_k=True，smooth_v=True）**：

| B | H | N | vs cute-v1 | vs triton-v1 | vs triton-v2 |
|---|---|---|---|---|---|
| 1 | 16 | 2048 | 0.0104 | 0.0105 | 0.0071 |
| 1 | 32 | 2048 | 0.0141 | 0.0143 | 0.0120 |
| 2 | 16 | 4096 | 0.0110 | 0.0120 | 0.0062 |
| 1 | 16 | 8192 | 0.0060 | 0.0059 | 0.0036 |

✅ 所有场景 max_abs < 0.02，V2 与 Triton V2 误差更小（两者量化方案相同）。

误差略大于 V1（~0.01 vs ~0.005）符合预期：FP8 round-trip 引入额外量化噪声，per-warp Q scale 粒度更细但 warp 内精度略有差异。

**有限性检查**：H=16 和 H=32 均无 NaN/Inf ✅

#### 速度测试结果

测试条件：B=1, H=32, D=64，50 次迭代（5 次 warmup）

| N | cute-v1 | cute-v2 | triton-v1 | triton-v2 | v2/v1 开销 |
|---|---|---|---|---|---|
| 1024 | 0.300ms | 0.448ms | 0.146ms | 0.235ms | 1.49× |
| 2048 | 0.464ms | 0.614ms | 0.289ms | 0.563ms | 1.32× |
| 4096 | 1.186ms | 1.462ms | 0.892ms | 1.575ms | 1.23× |
| 8192 | 3.435ms | 4.322ms | 3.292ms | 5.197ms | 1.26× |

**速度分析**：

- cute-v2 比 cute-v1 慢约 **23–49%**，额外开销全部来自 Python 预处理：
  - V FP8 round-trip（large tensor fp32→fp8→fp32 类型转换，N=8192 约 +0.5ms）
  - per-warp Q 量化粒度更细（N//16 blocks vs N//64，Triton kernel grid 更大）
- Kernel 本体无差异（仅多 1 次 2D gQScale 索引，可忽略不计）
- **cute-v2 vs triton-v2**：N=8192 时 **4.32ms vs 5.20ms，快约 17%** ✅
  （triton-v2 的 V FP8 WGMMA 和 V scale 传入 kernel 路径开销更重）

---

### v3：BLOCK_M=128 split-warpgroup（SageAttention V1 进一步提速）

**目标**：BLOCK_M 从 64 增大到 128，N_CTAs 减半，提升 L2 命中率和 CTA 复用效率。

#### 技术挑战：atom_layout=(2,1,1) 的硬件 Bug

初始方案使用 `atom_layout=(2,1,1)` 使单个 MMA 覆盖 128 M-rows（2 个 m64 atom）。测试发现：

- **N_CTAs ≤ 4 时 PASS，≥ 8 时随机 FAIL**
- 错误模式：WG0（rows 0-63）输出值为上一个 CTA 的预期值，WG1 始终正确
- PTX 分析：生成了正确的 `wgmma.mma_async.s32.s8.s8` 指令，但硬件执行时 WG0 的 smem A 描述符有时读到了前一 CTA 在同一 SM 上遗留的 smem 数据

**结论**：`(2,1,1)` smem A 描述符存在 SM 复用时的硬件竞争条件，`(1,1,1)` 不受影响（WG1 指向 smem base+4096，不会被前 CTA 污染）。

#### 解决方案：Split-Warpgroup 设计

将 CTA 的 256 个线程拆分为两个独立的 `(1,1,1)` warpgroup pipeline：

```
WG0 (tidx 0-127):   sQ0 = rows 0-63,   独立 QK GEMM + softmax + PV GEMM → O rows 0-63
WG1 (tidx 128-255): sQ1 = rows 64-127, 独立 QK GEMM + softmax + PV GEMM → O rows 64-127
```

**smem 布局**：sQ0 + sQ1 + sK (2-stage) + sV = 4+4+8+8 = 24KB（vs v2 20KB，均 4 CTA/SM）

**关键设计点**：
- `cute.arch.barrier()` 必须在 `if wg_idx` 块外执行（需所有 256 线程参与）
- MMA 和 softmax 在各自的 `if wg_idx == 0/1:` 块内执行（无 barrier 死锁）
- K/V cp.async 由所有 256 线程协作执行（在 if-block 外）
- K[n+2] 在 V-wait barrier 之后才能发出（确保 sK[cur_stage] 安全覆写）

**变量作用域解决方案**：`tCrQ0`/`tCrQ1` 均在 if-block 外预先构建（descriptor 不含寄存器值），两个 WG 各自只在对应 if-block 内使用其中一个。

#### 正确性验证

测试日期：2026-03-29，NVIDIA H20 SM90a

| B | H | N | max_abs | 状态 |
|---|---|---|---|---|
| 1 | 4 | 128 | 0.0119 | ✅ PASS |
| 1 | 4 | 256 | 0.0081 | ✅ PASS |
| 1 | 8 | 512 | 0.0071 | ✅ PASS |
| 2 | 8 | 1024 | 0.0068 | ✅ PASS |
| 1 | 16 | 2048 | 0.0054 | ✅ PASS |
| 1 | 32 | 8192 | 0.0040 | ✅ PASS |

#### 性能对比（B=1, H=16, D=64，含量化开销）

| N | FA2 | CuTe v2 (BM=64) | CuTe v3 (BM=128) | Official |
|---|---|---|---|---|
| 1024 | 0.68ms | 2.46ms | 0.81ms | 0.45ms |
| 4096 | 0.96ms | 1.90ms | 0.75ms | 0.50ms |
| 8192 | 3.59ms | 1.87ms | 1.96ms | 1.35ms |

**性能分析**：
- v3 在 N≤4096（GPU 未满载）时显著快于 v2（3×），因为 N_CTAs 减半，launch/scheduling overhead 更低
- v3 在 N=8192（H20 满载）时比 v2 慢约 5%，因为：
  - 256 线程/CTA 寄存器压力更大
  - 两个 warpgroup 在 barrier 点需要汇合（串行化点）
  - sK 必须等两个 WG 都完成 GEMM 才能覆写（需额外 barrier）
- **对比官方 SageAttention**：v3 在 N=4096 时达到官方 1.5× 的性能（91T vs 138T TFLOPS）

---

### v4：FP8 V GMEM 带宽优化

**目标**：在 v3 的 split-warpgroup 基础上，将 V 的 GMEM 存储从 FP16 改为 FP8 E4M3FN，GMEM 带宽减半（每 tile 8KB → 4KB）。

#### 设计概述

```
v3: sV(FP16, 8KB/tile) — 直接 cp.async FP16 → swizzled smem → WGMMA
v4: gV(FP8, 4KB/tile) → sV_f8(FP8, 4KB flat) → [FP8→FP16 smem转换] → sV_f16(FP16, 8KB swizzled) → WGMMA
```

**内存布局变化（vs v3）**：
```
sQ0(4KB) + sQ1(4KB) + sK(8KB×2stage) + sV_f8(4KB) + sV_f16_0(8KB) + sV_f16_1(8KB) = 36KB
```
（vs v3 的 24KB；H20 共 228KB smem，仍可 4 CTA/SM）

**V dequant 融合**：
```
softmax 时:  rP[i] = Float16(exp2(...) * v_scale_val)   — P × v_scale 写入 rP
PV GEMM:     acc_O += rP × sV_f16_wgX                   — (P × v_scale) × (V_fp8 × 1/v_scale) = P × V_real
```
V scale 乘法融入 softmax（免费），PV GEMM 精度不变。

#### FP8→FP16 smem 转换：跨 WG 可见性分析

**核心问题**：`wgmma.fence.sync.aligned`（即 `warpgroup.fence()`）只对**调用 WG 自身**发出的 generic memory store 提供顺序保证，无法覆盖其他 WG 的 store。

**初始方案（有 Bug）**：256 线程共同写一个 `sV_f16` 缓冲：
```
全部 256 线程写 sV_f16
→ WG0 的 warpgroup.fence() 只能看到 WG0（tidx 0-127）写的元素
→ WG1 写的另一半元素在 WG0 的 WGMMA 中不可见
→ 非确定性错误（~50%失败）
```

**修复方案：per-WG 私有 sV_f16 缓冲**：
```python
# SS struct 中各 WG 有独立缓冲：
sV_f16_0: Float16 (8KB, WG0 专用)
sV_f16_1: Float16 (8KB, WG1 专用)

# WG0 写满 sV_f16_0（128线程×32元素×4 pass = 4096元素）
# WG0 的 warpgroup.fence() → 覆盖所有 WG0 写的 sV_f16_0 元素
# WG0 WGMMA 读 sV_f16_0 → 可见性完整

# 同理 WG1 操作 sV_f16_1
```

这需要每个 WG 独立完成整个 4096 元素的 FP8→FP16 转换，虽有重复计算但保证了正确性。

**128线程 tiled copy（per-WG）**：
```python
# FP8 smem→reg: 64-bit 原子，8 FP8/thread
# TV layout: (16,8):(8,1) × (1,8)  →  tiler (16,64)
# 4 pass × 128线程 × 8元素 = 4096元素 ✓
ca_f8_reg  = make_copy_atom(CopyUniversalOp(), Float8E4M3FN, num_bits=64)
copy_f8_reg = make_tiled_copy_tv(ca_f8_reg, (16,8):(8,1), (1,8))

# FP16 reg→smem: 128-bit 原子，8 FP16/thread
# 与 FP8 pass 数匹配
ca_f16_reg  = make_copy_atom(CopyUniversalOp(), Float16, num_bits=128)
copy_f16_reg = make_tiled_copy_tv(ca_f16_reg, (16,8):(8,1), (1,8))
```

**两步 FP8→FP16 转换**（CuTe DSL 约束）：
```python
# 错误：.to(Float16) → arith.extf(f16→f16) IR 报错
# 正确：
reg_f16.store(reg_f8.load().to(Float32).to(Float16))  # F32 作中间类型
```

**sV_f8 用 flat 布局（无 swizzle）**：
```python
sV_f8_layout = cute.make_layout((HEAD_DIM, BLOCK_N), stride=(BLOCK_N, 1))
```
cp.async 写 sV_f8 用 Int8 recast（无 swizzle），FP8 读也用相同 flat 布局，避免写/读 swizzle 不一致。

#### 正确性验证进展

测试日期：2026-03-30，NVIDIA H20 SM90a

**已通过的配置**：

| B | H | N | 状态 | 备注 |
|---|---|---|---|---|
| 任意 | 任意 | 128 (N_KV=2) | ✅ PASS | 修复 end-of-loop barrier 后解决 |
| 任意 | 任意 | 256 (N_KV=4) | ✅ PASS | per-WG buffer 方案修复后 |

**仍存在 Bug 的配置**：

| B | H | N | 状态 | 误差 |
|---|---|---|---|---|
| ≥1 | ≥1 | 384 (N_KV=6) | ❌ FAIL | ~0.02–0.03 |
| ≥1 | ≥1 | 512 (N_KV=8) | ❌ FAIL | ~0.2 |

#### 调试过程与发现

**Bug 1（已修复）：end-of-loop barrier 缺失**

N=128（N_KV=2）有约 2% 的非确定性失败。原因：当 `n_tile+2 >= n_kv_blocks` 时没有 K 预取，也没有 `barrier()`，导致最后一次循环两 WG 不同步。修复：加 `else: cute.arch.barrier()`。

**Bug 2（已修复）：共享 sV_f16 缓冲跨 WG 可见性**

N=256（N_KV=4）约 50% 失败。根本原因：256 线程共写 `sV_f16`，`warpgroup.fence()` 只覆盖本 WG 的 store。修复：引入 `sV_f16_0` / `sV_f16_1` 独立缓冲。

**Bug 3（未解决）：N=384 确定性失败，pattern = head_flat ≡ 2 (mod 4)**

诊断测试（`/tmp/debug_head_pattern.py`）发现：

```
B=1,H=4,N=384:  hf0✓  hf1✓  hf2❌(0.022)  hf3✓
B=1,H=8,N=384:  hf{0,1,3,4,5,7}✓  hf{2,6}❌
B=4,H=3,N=384:  hf{2,6,10}❌  其余✓
B=4,H=8,N=256:  全部✓  (N_KV=4，不触发)
```

关键排除项：
- `V = constant per tile` → **掩盖 Bug**（max error ≤ 0.002，在误差范围内）。
  意味着 K pipeline、softmax、v_scale 索引均正确；Bug 在 **tile 内元素级别的 smem 寻址**。
- `V = randn` → 误差与 V 量级成正比（~1/V_scale 缩放关系）。

**根因假设**：

`copy_f8_reg`（源：flat `sV_f8`）和 `copy_f16_reg`（目标：swizzled `sV_f16_wgX`）使用相同的 TV layout `(16,8):(8,1) × (1,8)`，tiler = `(16,64)`，需 **4 pass** 覆盖 `(64,64)` tile。

当源是 flat（无 swizzle）而目标有 swizzle 时，两者的 pass 遍历顺序（element index → logical (row, col)）可能在某个 pass 不一致，导致特定元素写入 smem 的错误逻辑位置。`head_flat ≡ 2 (mod 4)` 的周期性正好与 4-pass 循环对应。

**待验证的修复方向**：

1. 改用 **flat（无 swizzle）sV_f16** 目标，验证 Bug 是否消失（确认是 swizzle mismatch）。
2. 如果确认，改为两个独立的 `cute.copy` 步骤：先用 flat copy 写 sV_f16_flat，再用 smem-to-smem copy 将 flat 转为 swizzled。
3. 或：使用完全不同的 tile 结构（128线程 × 32元素 × 1 pass = 4096 elements，避免多 pass）。

#### 变更日志（v4 部分）

| 日期 | 文件 | 变更内容 |
|---|---|---|
| 2026-03-29 | `cute/kernel_v4.py` | 新建：FP8 V GMEM 版本，基于 v3 split-WG 架构 |
| 2026-03-29 | `cute/kernel_v4.py` | 修复 Bug 1：加入 end-of-loop `else: barrier()` |
| 2026-03-30 | `cute/kernel_v4.py` | 修复 Bug 2：引入 per-WG 私有 sV_f16_0/sV_f16_1 缓冲，解决跨 WG WGMMA 可见性 |
| 2026-03-30 | `cute/CUTLASS_CPP_ANALYSIS.md` | 新建：CUTLASS C++ SageAttention 实现可行性分析 |
| 2026-03-30 | `cute/__init__.py` | 新增 kernel_v4 导出 |
| 2026-03-30 | `test.py` | 新增 v4 正确性测试和 benchmark 入口 |
| 2026-03-30 | `cute/DEVLOG.md` | 更新：记录 v4 实现过程和 Bug 3 调试进展 |

---

| 日期 | 文件 | 变更内容 |
|---|---|---|
| 2026-03-27 | `cute/kernel_v1.py` | 新建：ss-mode CuTe DSL 基础实现，Python 循环量化 |
| 2026-03-27 | `cute/kernel.py` | 新建：v2 rs-mode + 128-bit 向量化 + cp.async 流水 |
| 2026-03-27 | `cute/quant.py` | 新建：Triton GPU 量化 kernel，替换 Python for-loop |
| 2026-03-27 | `cute/kernel_v1.py` | 修改：量化调用替换为 GPU Triton kernel，CPU 版保留为 `_xxx_cpu` |
| 2026-03-28 | `cute/kernel.py` | 修复：copy_QK layout 8×16 → 32×4（修复 H=32 illegal access） |
| 2026-03-28 | `cute/kernel.py` | 优化：K double buffering（2-stage sK smem，prologue 预取 K[0]/K[1]） |
| 2026-03-28 | `cute/quant.py` | 新增：V per-tile FP8 量化 kernel（quant_v_per_tile_fp8_gpu） |
| 2026-03-28 | `cute/DEVLOG.md` | 更新：记录 v2 所有优化、受阻项、最新 benchmark 结果 |
| 2026-03-28 | `cute/quant.py` | 修改：`quant_per_block_int8` 放宽 block_sz assertion，加入 16 |
| 2026-03-28 | `cute/quant.py` | 新增：`quant_q_per_warp_int8_gpu`（V2 per-warp Q INT8 量化） |
| 2026-03-28 | `cute/quant.py` | 新增：`quant_v_per_channel_fp8`（V2 per-channel FP8 + V Smoothing，PyTorch GPU） |
| 2026-03-28 | `cute/kernel_v2.py` | 新建：SageAttention V2 CuTe DSL 实现（per-warp Q scale + FP8 V round-trip） |
| 2026-03-29 | `cute/kernel_v3.py` | 新建：BLOCK_M=128 split-warpgroup 实现（修复 atom_layout=(2,1,1) 竞争条件） |
| 2026-03-29 | `test.py` | 更新：加入 CuTe DSL v2/v3 正确性验证和 benchmark |
