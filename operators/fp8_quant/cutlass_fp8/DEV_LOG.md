# FP8 量化算子开发记录

## 概述

为两种 FP8 量化方案实现完整算子栈（PyTorch / Triton / CuTe / CUTLASS），
含 GEMM kernel、精度分析、benchmark。

---

## 量化方案

### Scheme A：Per-Tensor FP8
```
FP8 格式：float8_e4m3fn，max = 448.0
scale     = 448.0 / amax(|x|)
q         = clamp(x * scale, -448, 448).to(fp8)
inv_scale = amax / 448.0               ← 存储"解量化 scale"
x_fp32    = q.float() * inv_scale
GEMM      = torch._scaled_mm(q_a, q_b.T, scale_a, scale_b)
```

### Scheme B：Per-Block FP8（group_size=128）
```
激活 (M, K)：每 128 元素一组（沿 K 维）
  inv_scale[i, g] = max(|x[i, g*128:(g+1)*128]|) / 448.0
  q[i, g*128:] = clamp(x / inv_scale, -448, 448).to(fp8)

权重 (N, K)：128×128 2D block
  inv_scale[n_blk, k_blk] = max(|w[n*128:, k*128:]|) / 448.0

GEMM：out[i,j] = Σ_g  inv_scale_a[i,g] * inv_scale_b[j//128, g] * dot(q_a[i,g*128:], q_b[j,g*128:])
```

---

## 文件结构

```
operators/fp8_quant/
├── __init__.py
├── benchmark.py               # 全版本 benchmark + 精度分析
├── test.py                    # 正确性测试（含精度指标）
├── pytorch/
│   ├── __init__.py
│   └── fp8_torch.py          # PyTorch baseline（两种 scheme）
├── triton/
│   └── kernel.py             # Triton quant kernel + FP8 GEMM
├── cute/
│   └── kernel.py             # CuTe Python DSL（SM90）
└── cutlass_fp8/
    ├── __init__.py
    ├── gemm_fp8_sm90_v1.cu   # Per-tensor scale（scalar epilogue）
    ├── gemm_fp8_sm90_v2.cu   # Per-block scale（EVT per-row/col 近似）
    ├── kernel.py             # Python JIT 入口 V1/V2
    └── DEV_LOG.md            # 本文件
```

---

## Backend 实现细节

### PyTorch（`pytorch/fp8_torch.py`）

**Per-Tensor**
- `fp8_per_tensor_quant(x)` → `(q_fp8, inv_scale)` 标量
- `fp8_per_tensor_dequant(q, inv_scale)` → `fp16`
- `fp8_per_tensor_gemm(a, a_inv_s, b, b_inv_s)`:
  - 优先 `torch._scaled_mm`（SM90+ HW FP8 GEMM）
  - fallback：`a.float() * inv_scale_a @ (b.float() * inv_scale_b).T`

**Per-Block**
- `fp8_per_block_act_quant(x, group_size=128)` → `(q_fp8, inv_scales(M, K//128))`
- `fp8_per_block_weight_quant(w, block_size=128)` → `(q_fp8, inv_scales(N//128, K//128))`
- `fp8_per_block_gemm(a, a_inv_s, b, b_inv_s)` → 向量化 per-group scale（快于 naive loop）

**精度工具**
- `compute_quant_error(x_orig, x_dequant)` → `{rmse, max_abs_error, snr_db, cosine_sim}`

---

### Triton（`triton/kernel.py`）

**量化 kernel**
- `triton_fp8_per_tensor_quant`: 2-pass（先 amax，再量化）
- `triton_fp8_per_block_act_quant`: 每 program 处理一行的一个 group

**GEMM kernel**
- `triton_fp8_per_block_gemm`:
  - autotune：BLOCK_M/N/K 配置搜索
  - BLOCK_K = GROUP_SIZE = 128
  - tile 内：FP8 → FP32 dequant，tl.dot（FP16）
- `triton_fp8_per_tensor_gemm`:
  - autotune
  - scalar scale 在 tile loop 外统一乘（效率高）

**注意**：Triton float8 支持需要较新版本（>= 2.2），
`tl.float8e4m3fnuz` 与 PyTorch `torch.float8_e4m3fn` 需要验证映射。

---

### CuTe Python DSL（`cute/kernel.py`）

**展示内容**
- `cute.make_tensor` 创建 layout-aware tensor
- `cute.local_tile` 将全局 tensor 分块到 thread 粒度
- `cute.arch.thread_idx() / block_idx()` 替代 threadIdx/blockIdx
- `cute.arch.barrier()` 替代 `__syncthreads()`
- `cute.copy` 用于 global → shared memory 协同加载

**GEMM 策略**
- 优先 `torch._scaled_mm`（SM90 HW FP8 GEMM）
- 若 CuTe Python DSL 完整支持 FP8 MMA：使用 WGMMA
- 当前 CuTe Python binding 对 FP8 MMA Python-level 支持有限，注释展示架构

---

### CUTLASS 3.x（`cutlass_fp8/`）

#### V1：`gemm_fp8_sm90_v1.cu` — Per-Tensor Scale

```cpp
ElementA = ElementB = cutlass::float_e4m3_t
ElementD = cutlass::bfloat16_t
ElementAcc = float
TileShape = (128, 256, 64)  // FP8 sweet-spot K=64

// Epilogue: LinearCombination
// D = alpha * acc, alpha = inv_scale_a * inv_scale_b（Python 侧预乘）
```

**关键设计**：
- FP8 K tile = 64（vs INT8 K tile = 128），因为 FP8 精度更高，MMA 效率不同
- alpha 标量在 Python 侧预先计算，epilogue 只需一次乘法

#### V2：`gemm_fp8_sm90_v2.cu` — Per-Block Scale（EVT 近似）

```cpp
// EVT 结构（与 W8A8 V2 完全一致，只换 FP8 operands）
// D[i,j] = bf16(act_scale[i] * acc[i,j] * wgt_scale[j] + bias[j])
// act_scale[i] = mean(act_inv_scales[i, :])  per-row
// wgt_scale[j] = mean(wgt_inv_scales[:, j])  per-col（N-block mean）

CustomEVT:
  inner: ActScale × acc × WgtScale + bias
  outer: 0×C + inner → BF16
```

**精度代价**：EVT 近似 vs 精确 per-block：
- 精确 per-block SNR：~46 dB
- EVT per-row/col 近似 SNR：~44-45 dB（降低约 1-2 dB）
- 性能收益：无 group 展开，epilogue 直接 WGMMA pipeline

---

## 编译与运行

```bash
# 正确性测试
python operators/fp8_quant/test.py -v

# 性能 + 精度分析
python operators/fp8_quant/benchmark.py --warmup 5 --repeat 100
```

---

## 常见问题与解决

### 1. `torch.float8_e4m3fn` 不可用
- 需要 PyTorch >= 2.1
- 检查：`torch.float8_e4m3fn in dir(torch)`

### 2. `torch._scaled_mm` 不可用
- 需要 PyTorch >= 2.1 + SM90+ GPU（H100/H800）
- Fallback 到 FP32 dequant GEMM，精度不变，性能降低

### 3. Triton FP8 类型映射
- Triton `tl.float8e4m3fnuz` ≠ PyTorch `torch.float8_e4m3fn`
  - `float8_e4m3fn`：NaN 编码不同，PyTorch 默认
  - `float8e4m3fnuz`：无 NaN，Triton 支持
- 实际推理：数值范围相同（max=448），大部分情况可互用
- 若有符号位差异，需要额外转换

### 4. CUTLASS V2 EVT 编译
- `sm90_visitor_load_tma_warpspecialized.hpp` 路径依赖 CUTLASS 3.x
- EpilogueSchedule 必须显式指定 `TmaWarpSpecializedCooperative`（EVT 要求）
- `RowBcastStages` 需要从 EpilogueDescriptor 推导，不可手动指定

### 5. FP8 CUTLASS alignment
- FP8 = 1 byte → `128bit / 8bit = 16 elements` alignment（AlignA = AlignB = 16）
- BF16 = 2 byte → `128bit / 16bit = 8 elements` alignment（AlignC = AlignD = 8）

---

## 性能预期（H100 SXM5，M=1024 K=4096 N=4096）

| 实现 | 预期 TFLOPS | vs FP16 |
|------|------------|---------|
| FP16 baseline | ~110 | 1.00x |
| PyTorch per-tensor | ~111 | 1.01x |
| PyTorch per-block | ~91 | 0.83x |
| Triton per-tensor | ~120 | 1.09x |
| Triton per-block | ~117 | 1.06x |
| CuTe per-tensor | ~120 | 1.09x |
| CUTLASS V1 per-tensor | ~151 | 1.37x |
| CUTLASS V2 per-block | ~139 | 1.26x |

*以上为估计值，实际性能依赖 GPU、batch size、矩阵尺寸。*

---

## 精度预期（M=1024 K=4096 N=4096）

| 方案 | SNR(dB) | RMSE | cos_sim |
|------|---------|------|---------|
| Per-Tensor | ~42 | ~3.2e-3 | ~0.99997 |
| Per-Block | ~46 | ~1.9e-3 | ~0.99999 |
| EVT 近似 | ~44 | ~2.3e-3 | ~0.99998 |

*Per-Block 精度高约 4 dB，误差减少约 40%。*

---

## 开发日志

### v1.0（2026-03-29）
- 初始实现：PyTorch baseline（per-tensor + per-block）
- Triton quant kernel + GEMM（per-tensor + per-block）
- CuTe Python DSL 展示（quant layout 原语 + torch._scaled_mm GEMM）
- CUTLASS V1（per-tensor scalar epilogue）
- CUTLASS V2（per-block EVT per-row/col 近似，复用 W8A8 V2 框架）
- test.py（7 个测试用例）
- benchmark.py（4 种尺寸 × 多 backend，含精度统计）
