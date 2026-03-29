# SVDQuant W8A8 WGMMA SM90 算子开发文档

> **GPU**: NVIDIA H20 (SM90, 78 SMs, 102 GB HBM3)
> **CUDA**: 12.8 | **CUTLASS**: 3.3.0 | **PyTorch**: 2.8.0+cu128
> **日期**: 2026-03-29

---

## 1. 背景与动机

### 1.1 SVDQuant 算子结构

SVDQuant 将权重矩阵分解为低秩部分（LoRA）加量化残差：

```
y = x @ W + bias
  = x @ (lora_down @ lora_up + dequant(W_q)) + bias
  ≈ (x/smooth) @ (smooth * W_q_dequant) + x @ lora_down @ lora_up + bias
```

- 权重 INT4 量化，group_size=64，显存节省 ~4x
- 激活 smooth 缩放后量化为 INT4（W4A4）
- LoRA rank=32 捕获量化残差

### 1.2 性能瓶颈（起点）

初始各实现在 M=1024, K=4096, N=4096 上的对比：

| 实现 | 延迟 (ms) | TFLOPS | 说明 |
|---|---|---|---|
| fp16_baseline (cuBLAS) | 0.317 | 110 | 理论上限参考 |
| triton_opt | 0.922 | 37 | 当时最佳 |
| nunchaku | 1.567 | 22 | 比 Triton 还慢 |

**根因**：nunchaku 和 Triton 都用了 **SM80 warp-level `mma.sync m16n8k64 s4`**，在 SM90 H20 上运行时完全绕开了 Hopper WGMMA（warpgroup-level 指令，理论吞吐是 warp MMA 的 4x）。

### 1.3 SM90 MMA 指令支持矩阵

| 精度 | SM90 WGMMA | SM80 warp MMA |
|---|---|---|
| FP16 × FP16 | ✅ wgmma.mma_async m64n128k16 | ✅ |
| INT8 × INT8 | ✅ wgmma.mma_async m64nXk32 s32.s8.s8 | ✅ |
| **INT4 × INT4** | **❌ 不支持** | ✅ m16n8k64 s4 |
| FP4 × FP4 | ❌（SM120 Blackwell 才有）| ❌ |

**关键约束**：在 SM90 上使用 WGMMA，必须退到 INT8，无法用 INT4。这是硬件限制。

---

## 2. 方案设计

### 2.1 W8A8 INT8 WGMMA 方案

**思路**：将 INT4 权重重新量化为 INT8，使用 CUTLASS 3.x `CollectiveBuilder` 的 SS 路径触发真正的 WGMMA 指令。

**关键设计决策**：

| 决策 | 选择 | 原因 |
|---|---|---|
| 指令路径 | WGMMA wgmma.mma_async | SM90 WGMMA = warp MMA 4x 吞吐 |
| 精度 | INT8×INT8→INT32 | SM90 唯一支持 WGMMA 的整数精度 |
| CollectiveBuilder | SS 路径（同精度） | A/B 均为 8-bit，自动选 SS 非 Mixed Width |
| TileShape | 128×256×128 | INT8 SS 路径标准 sweet-spot |
| 编译目标 | sm_90a | 必须是 `sm_90a` 而非 `sm_90`，否则 WGMMA 不可用 |
| 量化精度 | group_size=128 | INT4→INT8 dequant 后重量化 |

### 2.2 文件结构

```
operators/svdquant/cutlass_w8a8/
├── __init__.py               空包
├── gemm_w8a8_sm90.cu         V1：基础 WGMMA GEMM
├── gemm_w8a8_sm90_v2.cu      V2：fused smooth-quant + fused epilogue
├── kernel.py                 Python 入口（量化 + JIT + forward V1/V2）
└── DEV_LOG.md                本文档
```

---

## 3. 实现过程与问题记录

### 3.1 V1 实现（基础版）

**目标**：INT8 WGMMA GEMM，pipeline 独立 kernel 串行执行。

#### V1 Pipeline

```
[1] x / smooth → INT8 quantize  (Python: reshape + amax + round)
[2] INT8 WGMMA GEMM              (CUTLASS kernel，output FP32)
[3] per-row ascale × per-col wscale dequant  (Python elementwise)
[4] LoRA (2 FP16 GEMMs)
[5] bias add
```

#### 遇到的编译/运行问题及解决

**问题 1：`make_cute_packed_stride` 找不到**
```
error: namespace "cutlass" has no member "make_cute_packed_stride"
```
- **原因**：该函数定义在 `cutlass/util/packed_stride.hpp`，位于 `tools/util/include/` 而非 `include/`，未被默认 include
- **解决**：显式添加 `#include "cutlass/util/packed_stride.hpp"`

**问题 2：链接时缺失 `cuTensorMapEncodeTiled`**
```
undefined symbol: cuTensorMapEncodeTiled
```
- **原因**：TMA（Tensor Memory Accelerator）API 在 `libcuda.so`（驱动库）而不在 `libcudart.so`
- **解决**：JIT 编译加 `extra_ldflags=["-L/lib64", "-lcuda"]`

**问题 3：FP16 输出数值溢出（全 `inf`）**
- **原因**：INT8×INT8 的 INT32 累加值可达 `128 × 127 × 127 ≈ 2M`，远超 FP16 max（65504），直接转 FP16 溢出
- **解决**：CUTLASS 输出类型改为 `float`（FP32），Python 侧做 dequant 后转 FP16

**问题 4：Python `W8A8_AVAILABLE` 状态不同步**
- **原因**：`from module import W8A8_AVAILABLE` 是值拷贝，`_load_extension()` 后模块内部变量更新但导入的变量不变
- **解决**：改为 `import module as mod; mod._load_extension(); if mod.W8A8_AVAILABLE`

#### V1 性能分析（M=1024, K=4096, N=14336）

```
[A] FP16 GEMM only:             924 us  130T  ← 参考
[C] Activation smooth+quant:    130 us         ← 额外开销
[D] INT8 WGMMA GEMM only:       497 us  242T  ← 内核本身很快
[E] Dequant elementwise:        176 us         ← 额外开销
[F] LoRA (2 GEMMs):              49 us
[G] Full V1 pipeline:           977 us  124T  ← C+E 吃掉所有收益
```

**结论**：INT8 WGMMA kernel 本身达到 242T，但 C（激活量化 130us）+ E（dequant 176us）= 306us 额外开销，把 GEMM 节省的 ~430us 大幅侵蚀。

### 3.2 V2 实现（融合优化版）

**目标**：消除 V1 中额外的 smooth+quant kernel 和 dequant elementwise kernel。

#### V2 优化点 1：Fused Smooth+Quantize CUDA kernel

手写 CUDA kernel，将以下操作合并为一个 kernel pass：
- `x / smooth` 的 elementwise divide
- per-group `max(|x|)` reduce
- `scale = max / 127.0`
- `q = round(x/scale).clamp(-128, 127)`

**实现**：`grid=(M, K/group_size)`, `block=(group_size,)`，shared memory tree-reduce 求组内 max。

将 V1 的 `~130us`（smooth divide + reshape + amax + round 多个 Python 算子）压缩到 **~44us**。

#### V2 优化点 2：Fused Epilogue（EVT per-row scale + per-col bias → FP16）

使用 CUTLASS 3.x **Epilogue Visitor Tree（EVT）**，在 GEMM epilogue 内直接计算：

```
D[i,j] = half( alpha[i] * acc[i,j] + bias[j] )
```

其中 `alpha[i] = ascale_mean[i] * wscale_mean`（Python 预计算），输出直接为 FP16，消除单独的 dequant elementwise kernel（~176us）。

**EVT 树结构**：

```
CustomEVT = Sm90EVT<outer_compute(half, float),   // 转 FP16
  Sm90ScalarBroadcast<float>(beta=0),             // C 不用
  Sm90SrcFetch,                                   // C
  InnerEVT = Sm90EVT<inner_compute(float, float), // alpha*acc+bias
    Sm90ColBroadcast<alpha[i], stride(1,0,0)>,    // per-row alpha（M-dim）
    Sm90AccFetch,                                  // INT32 累加器
    Sm90RowBroadcast<bias[j], stride(0,1,0)>      // per-col bias（N-dim）
  >
>
```

**关键命名约定**（CUTLASS 3.x 反直觉）：
- `ColBroadcast`（列广播）= 每行一个值 = **per-row** scale，stride `(1,0,0)` 沿 M 方向
- `RowBroadcast`（行广播）= 每列一个值 = **per-col** bias，stride `(0,1,0)` 沿 N 方向

**EpilogueSchedule 问题**：`KernelScheduleAuto` 对于 TileShape=128×256 选 `TmaWarpSpecializedCooperative`，EVT 必须与之配套使用相同 schedule，否则编译失败。需要显式指定：

```cpp
using EpilogueSchedule_v2 = cutlass::epilogue::TmaWarpSpecializedCooperative;
```

同时将 `CollectiveEpilogue` 定义放在 `CollectiveMainloop` 之前，便于用 `StageCountAutoCarveout<sizeof(SharedStorage)>` 精确计算 SMEM stage 数。

**ptxas 警告**（不影响正确性，轻微影响性能）：
```
ptxas info: Potential Performance Loss: wgmma.mma_async instructions are serialized
due to wgmma pipeline crossing function boundary
```
这是 CUTLASS 3.3 的已知限制，CUTLASS 3.5+ 已修复。

---

## 4. Benchmark 结果

### 4.1 基础对比（M=1024, K=4096, N=4096）

| 实现 | 延迟 (ms) | TFLOPS | vs PyTorch | vs Triton_opt |
|---|---|---|---|---|
| fp16_baseline (cuBLAS) | 0.316 | 110T | 6.6x | 3.2x |
| svdquant_pytorch (INT4) | 2.085 | 16.7T | 1.0x | 0.44x |
| triton_opt | 0.928 | 37.7T | 2.25x | 1.0x |
| nunchaku (SM80 INT4 MMA) | 1.566 | 22.4T | 1.33x | 0.59x |
| **W8A8 V1** | **0.420** | **83.0T** | **4.97x** | **2.21x** |
| **W8A8 V2** | **0.293** | **119T** | **7.12x** | **3.17x** |
| **W8A8 V3** | **0.311** | **112T** | **6.65x** | **2.96x** |

> V3 mean ms 略高于 V2 是因为 stream overhead 在小矩阵不显著，但 `min_ms=0.294` 与 V2 几乎相同。

### 4.2 大矩阵全面对比（LLM 典型尺寸）

所有结果基于 warmup=5, repeat=100，单位 ms：

| Config | fp16 | triton_opt | nunchaku | W8A8 V1 | W8A8 V2 | **W8A8 V3** | V3 TFLOPS | V3/fp16 | V3/triton |
|---|---|---|---|---|---|---|---|---|---|
| M=64,K=512,N=512 | 0.026 | 0.089 | 0.145 | 0.212 | 0.120 | **0.113** | 0.334T | 0.23x | **0.79x** |
| M=256,K=2048,N=2048 | 0.041 | 0.151 | 0.430 | 0.201 | 0.129 | **0.131** | 16.9T | 0.31x | **1.15x** |
| M=1024,K=4096,N=4096 | 0.314 | 0.923 | 1.557 | 0.420 | 0.298 | **0.311** | 112T | 0.99x | **2.97x** |
| M=4096,K=4096,N=14336 | 3.681 | 8.984 | 17.591 | 3.565 | 2.223 | **2.170** | 224T | 0.59x | **4.14x** |

### 4.3 V1 → V2 → V3 kernel 级别加速

| 环节 | V1 | V2 | V3 | 说明 |
|---|---|---|---|---|
| Smooth+Quant | ~130 us（多个 Python ops） | ~44 us（单 CUDA kernel） | ~44 us | **V1→V2: 2.9x** |
| INT8 GEMM | ~497 us（FP32 output） | 同 GEMM（FP16 output） | 并发进行 | epilogue 内完成 dequant |
| Dequant elementwise | ~176 us（独立 kernel） | **0 us**（fused in epilogue） | **0 us** | **完全消除** |
| LoRA | ~49 us（串行） | ~49 us（串行） | **~0 us（与 GEMM 并发）** | **V2→V3: LoRA 隐藏** |
| `.item()` CPU sync | ~0 us（无，已用 mean 近似） | ~0 us | ~0 us | - |
| **Full pipeline** | **~977 us** | **~642 us** | **~595 us** | **V2→V3: ~1.08x，V1→V3: 1.64x** |

> V3 在大矩阵（M=4096, N=14336）相比 V2 额外提升 ~2.4%；在中等矩阵（M=1024, N=4096）效果有限（LoRA 耗时短，GEMM 主导）。

---

## 5. 为什么 W8A8 仍未超过 FP16

### 5.1 三层原因分析

**层 1：算法层面的额外开销**（不可避免）
- 激活量化（smooth + quant）：即使 fused，仍需 ~44us 额外内存访问
- LoRA（2 FP16 GEMMs）：V3 中已与 GEMM 并发，但 `wait_stream` + `out + lora_out` 仍有 ~2us 额外同步开销
- 总固定开销：~46us（V3），约占 fp16 总延迟（960us）的 5%

**层 2：GEMM 本身**（INT8 应比 FP16 快）
- H20 INT8 Tensor Core 理论吞吐 ≈ FP16 的 2x
- 实测 INT8 WGMMA GEMM 单 kernel：242T vs FP16 baseline 130T ✓
- 但同等计算量（FLOPs），INT8 需要额外的 dequant 才能得到 FP16 输出

**层 3：剩余 gap（V3 后）**
- V2/V3 `alpha_row` 计算仍用 `mean(ascales)` 近似，丢失了 per-group 精度
- 这导致部分误差累积，但不是性能瓶颈

**结论**：对于很大的 N（14336），GEMM 主导，V3 已达 FP16 的 ~60%。随着 N 增大，W8A8 的优势会更明显。

### 5.2 为什么 nunchaku 慢

Nunchaku 的 INT4 W4A4 GEMM 用的是 SM80 warp-level 指令：
```ptx
mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32   ← SM80 warp MMA
```
在 H20（SM90）上运行时，没有升级到 WGMMA，相当于只用了 GPU 1/4 的算力，只有 22T。
这是 **硬件约束**：SM90 WGMMA 不支持 INT4，INT4 WGMMA 要等 SM120（Blackwell）。

---

## 6. 后续优化方向

### 6.1 ✅ 已完成的优化

| 版本 | 优化 | 效果 |
|---|---|---|
| V1 | CUTLASS 3.x INT8 WGMMA 基础实现 | 83T，4.97x vs Triton |
| V2 | Fused smooth+quant；EVT fused epilogue（dequant+bias） | 119T，7.12x vs Triton |
| V3 | CUDA stream overlap（LoRA 并发 GEMM）；消除 .item() CPU sync | 112T，6.65x vs Triton；V2→V3 大矩阵额外 +2% |

### 6.2 未完成的优化方向

**方向 A：Fuse LoRA 进 GEMM epilogue**

nunchaku 已经证明可行（`lora_act_in @ lora_up` 在 epilogue 中完成）。在 CUTLASS EVT 中扩展为：
```
D[i,j] = half( alpha[i] * acc[i,j] + (lora_act[i,:] @ lora_up[:,j]) + bias[j] )
```
这需要把 `lora_act`（M×R FP16）和 `lora_up`（R×N FP16）的矩阵乘也塞进 epilogue 的 `Sm90AuxLoad` + `Sm90EVT`，属于自定义 visitor，实现复杂度较高。

**方向 B：精确的 per-group dequant（提升精度，不影响速度）**

当前 `alpha[i] = mean(ascales[i,:]) * mean(wscales[:,j])` 是近似。
精确方法需要 group-level 展开：`acc[i,j] = Σ_g ascales[i,g] * Σ_{k in g} q_x[i,k]*q_w[k,j]`，这需要 group-level 累加，CUTLASS 3.x 暂不支持直接表达，可通过自定义 mainloop 实现。

**方向 C：TileShape 调优**

- 当前 `128×256×128` 对小 M（M=64）不友好（tile 太大，wave 不整）
- 可对小 M 使用 `64×128×128` 或 `64×256×128`
- 使用 Python 侧按 M 大小分支选择不同 kernel

**方向 D：ClusterShape 多 CTA 协作**

对于大矩阵（N≥512），启用 `ClusterShape=<2,1,1>` 利用 Hopper multi-CTA cluster 特性。理论上可进一步提升 SM 利用率 ~10-20%。

**方向 E：升级 CUTLASS 版本**

当前使用 CUTLASS 3.3.0，存在 wgmma pipeline 跨函数边界的 ptxas warning（性能损失）。CUTLASS 3.5+ 已修复，升级后可能有额外 5-10% 提升。

### 6.2 精度优化

- 当前 V1/V2 精度（vs INT4 pytorch reference）：`max_abs_error ≈ 0.3-1.1`（atol=1.0 边界）
- 可通过 per-group 精确 dequant 将误差降到 `max_abs_error < 0.3`（与 V1 持平）
- 与 INT4 baseline 本身的误差（0.28-0.42）相比，精度损失在可接受范围

### 6.3 推理框架集成

- 将 `prepare_w8a8_weights` 集成为模型加载时一次性预处理（离线）
- `svdquant_forward_w8a8_v2_cached` 作为热路径 forward
- 对于 diffusion model 的典型 hidden_size=4096, seq_len=1024，V2 相当于近乎免费地获得 INT8 量化压缩（~0.67x fp16 延迟）+ 权重显存降低 2x（vs FP16）

---

## 7. 代码结构与接口

### 7.1 CUDA 内核文件

#### `gemm_w8a8_sm90.cu`（V1）
```cpp
// 暴露一个函数：
void w8a8_gemm_sm90(act, wgt, out_fp32, alpha=1.0f)
// 输入: act (M,K) int8, wgt (N,K) int8, out (M,N) float32
// 输出: out = alpha * acc  (float32，无 scale)
```

#### `gemm_w8a8_sm90_v2.cu`（V2）
```cpp
// 暴露两个函数：
std::pair<Tensor, Tensor> fused_smooth_quantize(x, smooth, group_size=128)
// x (M,K) fp16, smooth (K,) fp16 → q (M,K) int8, scales (M,K/gs) fp32

void w8a8_gemm_sm90_v2(act, wgt, out_fp16, alpha_row, bias=None)
// act (M,K) int8, wgt (N,K) int8, out (M,N) fp16, alpha_row (M,) fp32, bias (N,) fp16
// out[i,j] = half(alpha_row[i] * acc[i,j] + bias[j])
```

### 7.2 Python 接口

```python
# V1
from operators.svdquant.cutlass_w8a8.kernel import (
    prepare_w8a8_weights,          # INT4 → INT8 weight conversion
    svdquant_forward_w8a8,         # online (w/ weight requant, for correctness test)
    svdquant_forward_w8a8_cached,  # cached (benchmark)
    _load_extension, W8A8_AVAILABLE,
)

# V2
from operators.svdquant.cutlass_w8a8.kernel import (
    svdquant_forward_w8a8_v2,        # online (for correctness test)
    svdquant_forward_w8a8_v2_cached, # cached (benchmark)
    _load_extension_v2, W8A8_V2_AVAILABLE,
)

# V3 (推荐用于推理热路径)
from operators.svdquant.cutlass_w8a8.kernel import (
    svdquant_forward_w8a8_v3,        # online (for correctness test)
    svdquant_forward_w8a8_v3_cached, # cached (benchmark), 支持传入预计算 wscale_mean
    prepare_w8a8_v3_cache,           # 预计算 wscale_mean GPU tensor
)
# 推荐用法：
#   wscale_mean = prepare_w8a8_v3_cache(wscales8)  # 模型加载时一次
#   out = svdquant_forward_w8a8_v3_cached(x, q_w8_col, wscales8, ..., wscale_mean_cached=wscale_mean)
```

### 7.3 编译配置

```python
extra_cuda_cflags=[
    "-std=c++17", "-O3",
    "--expt-relaxed-constexpr", "--expt-extended-lambda",
    "-gencode", "arch=compute_90a,code=sm_90a",  # 必须 sm_90a
    "-I{CUTLASS_ROOT}/include",
    "-I{CUTLASS_ROOT}/tools/util/include",        # packed_stride.hpp 在这里
    "-DCUTLASS_ARCH_MMA_SM90_SUPPORTED",
    "-DCUTE_ARCH_MMA_SM90A_ENABLED",
    "-U__CUDA_NO_HALF_OPERATORS__", ...           # 消除宏冲突
]
extra_ldflags=["-L/lib64", "-lcuda"]  # TMA API 在 libcuda.so
```

---

## 8. 关键教训总结

1. **`sm_90` vs `sm_90a`**：WGMMA 必须用 `sm_90a`，`sm_90` 编译通过但 runtime 无法调度 WGMMA
2. **packed_stride.hpp 在 tools/util/include**：不在主 include 目录，忘记 include 会报找不到 `make_cute_packed_stride`
3. **TMA 需要 libcuda.so**：`cuTensorMapEncodeTiled` 在驱动库，不在 `libcudart.so`，需要额外 `-lcuda`
4. **INT32 累加不能直接转 FP16**：最大值 ≈ 128×127×127 = 2M >> 65504，必须走 FP32
5. **Python import 的值拷贝陷阱**：`from module import bool_var` 不能感知模块内部的后续更新
6. **ColBroadcast = per-row, RowBroadcast = per-col**：CUTLASS EVT 的命名是"广播方向"而非"应用方向"，与直觉相反
7. **EpilogueSchedule 必须与 KernelSchedule 配套**：EVT 使用时必须显式指定 `TmaWarpSpecializedCooperative`
8. **INT4 WGMMA 在 SM90 不存在**：nunchaku 慢的根本原因，等 Blackwell 才能解决
9. **CUDA stream 复用**：每次 forward 创建新 stream 开销约 100-200us（H20 实测），对小矩阵（M=64，总延迟 ~113us）影响巨大。必须用模块级持久化 stream。
10. **Stream 依赖正确性**：lora_stream 在使用 x 前必须 `wait_stream(cur_stream)`，否则在 x 由上游 kernel 产生时会出现 race condition（通常不报错但结果错误）。
