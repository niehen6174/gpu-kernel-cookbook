# SageAttention SM90 内核优化空间评估

> 基于官方 SageAttention v2.2.0 SM90 内核 `qk_int_sv_f8_cuda_sm90.cu` 的完整代码分析
>
> Date: 2026-03-31 | Hardware: NVIDIA H20 (SM90a, Hopper), CUDA 12.9

## 1. 当前架构概览

```
CTA_Q=64, CTA_K=128, NUM_THREADS=128 (1 warpgroup)
QK GEMM: wgmma_s8s8s32 (INT8, SS mode, 两个操作数均来自 smem)
PV GEMM: wgmma_f8f8f32 (FP8, RS mode, A 来自寄存器, B 来自 smem)
数据加载: TMA (cp.async.bulk.tensor.4d) + mbarrier phase-flip 双缓冲
Softmax: exp2 approximation + warp shuffle reduce (CUDA core)
输出: 直接 GMEM store (half2/bfloat162, 非 TMA store)
无 warp specialization, 无深度流水线
支持: GQA, causal mask, per-warp/per-thread 量化粒度, return_lse
```

### 关键数据流

```
GMEM Q (INT8) ──TMA──> sQ (smem)
                                    ├── QK GEMM (WGMMA INT8×INT8→I32)
GMEM K (INT8) ──TMA──> sK (smem)  ─┘
                                    ├── softmax (CUDA core: int→float→exp2→FP8)
GMEM V (FP8) ──TMA──> sV (smem)   ├── PV GEMM (WGMMA FP8×FP8→F32)
                                    └── Output → GMEM (direct store)
```

### 主循环结构

```
加载 Q, K[0], V[0]
for iter = 1 to N-1:
    wait K[iter-1]
    QK GEMM (sQ × sK → RS)
    load K[iter]               ← 在 QK GEMM 后发起
    softmax (RS → exp2 → FP8)
    wait V[iter-1]
    PV GEMM (RS_fp8 × sV → RO_temp), RO += RO_temp
    load V[iter]               ← 在 PV GEMM 后发起

最后一次迭代 (含 masking):
    wait K, QK GEMM, mask, softmax, wait V, PV GEMM

normalize, v_scale, output store
```

## 2. 优化空间分析

### 🔴 高收益 (预期 >10% TFLOPS 提升)

#### 2.1 Warp Specialization (生产者-消费者分离)

**现状**: 128 线程全部参与 TMA load + WGMMA compute，串行执行。

```
当前 (无 Warp Spec):
  ┌─────────┐ ┌──────────┐ ┌─────────┐ ┌────────┐ ┌─────────┐ ┌──────────┐
  │ wait_K  │→│ QK_GEMM  │→│ load_K  │→│softmax │→│ wait_V  │→│ PV_GEMM  │→ load_V
  └─────────┘ └──────────┘ └─────────┘ └────────┘ └─────────┘ └──────────┘
  ^^^^^^^^                                         ^^^^^^^^
  阻塞等待                                         阻塞等待
```

**优化**: 256 线程 → 1 个 LoadWG (TMA) + 1-2 个 MmaWG (GEMM)

```
优化后 (Warp Spec):
  LoadWG:  load_K[0] → load_V[0] → load_K[1] → load_V[1] → ...
  MmaWG:              QK[0] → softmax → PV[0] → QK[1] → softmax → PV[1] → ...
                      ^^^^^^^^^^^^^^^^^^^^^^^^
                      与 load_K[1]/load_V[1] 完全重叠
```

| 指标 | 值 |
|------|----|
| 预期收益 | +10-20% TFLOPS |
| 实现难度 | 高 |
| 时间估计 | 2-3 周 |
| 参考 | FlashAttention-3 |

**关键挑战**:
- 需要重构线程分配 (warpgroup ID 分配)
- 需要 named barrier 而非 `__syncthreads()`
- Load WG 和 MMA WG 之间的同步
- 需要增加到 256+ 线程

#### 2.2 深度流水线 (Multi-stage Pipeline)

**现状**: K 和 V 各用 mbarrier phase-flip 双缓冲，实际上只有 2 个阶段。

```c
// 当前: 简单 phase-flip，每次必须等前一次 TMA 完成
wait(&barrier_K, p);       // 等 K[n] 加载完成
QK_GEMM(sQ, sK);          // 使用 K[n]
load_K(sK, n+1);           // 发起 K[n+1] 加载
// ... 无法在 QK_GEMM 期间同时加载 K[n+2]
```

**优化**: 3-5 stage pipeline，允许多个 inflight TMA 请求

```
3-stage 示例:
  Stage 0: sK_0 ← load_K[n]     (TMA inflight)
  Stage 1: sK_1 ← load_K[n+1]   (TMA inflight)
  Stage 2: sK_2 ← compute QK     (WGMMA using sK_2)
```

| 指标 | 值 |
|------|----|
| 预期收益 | +5-15% TFLOPS |
| 实现难度 | 中 |
| 时间估计 | 1-2 周 |
| SMEM 代价 | 每 stage +8KB (CTA_K=128, D=64, INT8) |
| 参考 | CUTLASS 3.x `PipelineAsync<Stages>` |

### 🟡 中等收益 (预期 5-10% TFLOPS 提升)

#### 2.3 CTA_Q=128 (更大的 Q tile)

**现状**: CTA_Q=64，每个 CTA 处理 64 行 Q

**优化**: CTA_Q=128 → QK GEMM 变为 128×128
- 计算密度翻倍 (更好的 arithmetic intensity)
- CTA 数量减半 (更少的 launch overhead)
- Q tile 数据复用率更高

| 指标 | 值 |
|------|----|
| 预期收益 | +5-10% TFLOPS |
| 实现难度 | 中 |
| 时间估计 | 1 周 |
| SMEM 代价 | +4KB (CTA_Q: 64→128, D=64, INT8) |
| 风险 | 寄存器压力增加, 可能降低 occupancy |
| 参考 | FlashAttention-3 使用 CTA_Q=128, CTA_K=128 |

#### 2.4 TMA Store 输出

**现状** (line 517-551):
```c
// 每个线程逐元素写 GMEM — 标量逐对写入
DTypeOut *O_lane_ptr = O + ...;
for (fq) for (fv) {
    ((half2*)(O_lane_ptr + ...))[0] = __float22half2_rn(...);  // 4 bytes per store
    ((half2*)(O_lane_ptr + ...))[0] = __float22half2_rn(...);
}
```

**优化**: 使用 TMA store (`cp.async.bulk.tensor`) 一次性写出整个 tile
- 减少 store 指令数量
- 利用 TMA 硬件自动处理地址计算

| 指标 | 值 |
|------|----|
| 预期收益 | +3-5% TFLOPS |
| 实现难度 | 低-中 |
| 时间估计 | 2-3 天 |
| 前置条件 | 需要额外的 TMA descriptor for output |

#### 2.5 RO_temp 中间累加器消除

**现状** (line 328, 459):
```c
float RO_temp[num_tiles_q][num_tiles_v][8];   // PV GEMM 写入临时 buffer
wgmma::wgmma_f8f8f32<...>(RO_temp[fq], ...); // WGMMA 写入 RO_temp
// 然后累加:
for (...) RO[fq][fv][k] += RO_temp[fq][fv][k];  // 额外 FADD
```

**问题**:
- 寄存器占用翻倍 (RO + RO_temp)
- 每次 PV GEMM 后额外的 `num_tiles_q × num_tiles_v × 8` 次 FADD
- 增加 register pressure → 可能降低 occupancy

**优化**: 直接让 WGMMA 累加到 RO (利用 `scaleD=1` 模式)

| 指标 | 值 |
|------|----|
| 预期收益 | +3-5% |
| 实现难度 | 低 |
| 时间估计 | 1-2 天 |
| 注意 | 需要确认 online softmax rescaling 兼容性 |

**分析**: 当前使用 RO_temp 的原因是 online softmax 在 PV GEMM 之前需要 rescale RO (`RO *= o_scale`)。如果直接累加，需要在每次 PV GEMM 前先 rescale RO，但 WGMMA 的 scaleD 参数固定为 0 或 1，不支持自定义 scale。因此 RO_temp 是正确做法，但可以考虑将 rescale 融合到 WGMMA 的 scaleD=0 (clear) + 后续手动累加的方式。

### 🟢 低收益但易实现 (预期 1-3%)

#### 2.6 `#pragma unrol` 拼写错误修复

**现状** (line 314, 445):
```c
#pragma unrol    // ← 缺少 'l'，应为 #pragma unroll
for (uint32_t fk = 0; fk < num_tiles_k; fk++) { ... }
```

如果编译器不识别 `#pragma unrol`（视为未知 pragma 并忽略），则这些循环可能不被展开。
影响的循环: `d` 累加循环（softmax 分母累加）

| 指标 | 值 |
|------|----|
| 预期收益 | +1-3% |
| 实现难度 | 极低 |
| 时间估计 | 10 分钟 |
| 验证方法 | cuobjdump 查看 SASS 确认循环是否展开 |

#### 2.7 L2 Cache Promotion

**现状**: TMA descriptor 使用 `CU_TENSOR_MAP_L2_PROMOTION_NONE`

```c
// line 46:
cuTensorMapEncodeTiled(..., CU_TENSOR_MAP_L2_PROMOTION_NONE, ...);
```

**优化**: 对 K/V 开启 L2 promotion (`CU_TENSOR_MAP_L2_PROMOTION_L2_128B` 或 `_L2_256B`)
- 多个 CTA (不同 Q tile) 共享相同的 K/V 数据
- L2 promotion 可以提高这些共享数据的命中率
- H20 L2 = 60MB，对于中等 N 的 K/V 足够缓存

| 指标 | 值 |
|------|----|
| 预期收益 | +1-3% |
| 实现难度 | 极低 |
| 时间估计 | 10 分钟 |
| 改动 | `CU_TENSOR_MAP_L2_PROMOTION_NONE` → `CU_TENSOR_MAP_L2_PROMOTION_L2_128B` |

#### 2.8 Persistent Kernel / Tile Scheduler

**现状**: 每个 Q tile 启动一个独立 CTA

```c
dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
kernel<<<grid, NUM_THREADS, sMemSize>>>(...);
```

**优化**: 持久化内核 + 软件 tile scheduler
- 一个 CTA 处理多个 tile
- 减少 kernel launch/teardown 开销
- 更好的负载均衡 (特别是 causal mask 场景)

| 指标 | 值 |
|------|----|
| 预期收益 | +1-5% (主要改善小 N 场景) |
| 实现难度 | 中 |
| 时间估计 | 1 周 |
| 参考 | FlashAttention-3 `TileScheduler` |

#### 2.9 Softmax CUDA Core 优化

`update_mdo()` 中的 online softmax 在标量 CUDA core 上执行。

可能的优化:
- `__expf()` 近似替代 `exp()` (如果精度可接受) — **已使用 `exp2`**
- 预计算 `log2e * sm_scale` 融合 — **已做 (line 153)**
- 向量化 rescale 操作
- 减少 warp shuffle 次数

| 指标 | 值 |
|------|----|
| 预期收益 | +1-2% |
| 实现难度 | 低 |
| 时间估计 | 1-2 天 |

## 3. 综合评估矩阵

| # | 优化方向 | 预期收益 | 实现难度 | 时间估计 | 优先级 |
|---|---------|---------|---------|---------|-------|
| 1 | Warp Specialization | +10-20% | 高 | 2-3 周 | P1 |
| 2 | 深度流水线 (3-5 stage) | +5-15% | 中 | 1-2 周 | P1 |
| 3 | CTA_Q=128 | +5-10% | 中 | 1 周 | P2 |
| 4 | TMA Store 输出 | +3-5% | 低-中 | 2-3 天 | P2 |
| 5 | RO_temp 消除 | +3-5% | 低 | 1-2 天 | P2 |
| 6 | `#pragma unroll` 修复 | +1-3% | 极低 | 10 分钟 | P0 |
| 7 | L2 Promotion | +1-3% | 极低 | 10 分钟 | P0 |
| 8 | Persistent Kernel | +1-5% | 中 | 1 周 | P3 |
| 9 | Softmax 优化 | +1-2% | 低 | 1-2 天 | P3 |

## 4. 收益估算

```
当前官方: ~215 TFLOPS (H100 参考数据，H20 按比例)

快速修复 (P0, < 1天):
  #pragma unroll + L2 promotion → +2-6% → ~221-228 TFLOPS

中等投入 (P2, 1-2周):
  + TMA store + RO_temp优化 + CTA_Q=128 → +11-20% → ~239-258 TFLOPS

大投入 (P1, 3-5周):
  + Warp Spec + 深度流水线 → +15-35% → ~247-290 TFLOPS

理论上限 (全部优化叠加, 非线性):
  ~260-280 TFLOPS (估计实际 +20-30%)
```

## 5. 实施路径建议

### Phase 0: 立即验证 (< 1 天)

1. **修复 `#pragma unroll` 拼写** (line 314, 445)
2. **开启 L2 Promotion** (修改 TMA descriptor 枚举值)
3. **验证**: cuobjdump 检查 SASS 循环展开, benchmark 确认性能变化

### Phase 1: Quick Wins (1 周)

4. **评估 RO_temp 消除可行性** (需要仔细分析 online softmax 兼容性)
5. **TMA Store 输出** (替换标量写入)

### Phase 2: 架构级改进 (2-4 周)

6. **CTA_Q=128** (需要调整 WGMMA tile, 测试 register pressure)
7. **Warp Specialization** (需要重构线程模型)
8. **Deep Pipeline** (需要增加 smem buffers, 修改 mbarrier 管理)

### Phase 3: 长期优化

9. **Persistent Kernel** (tile scheduler)
10. **Causal mask 优化** (减少 causal 场景的无效计算)

## 6. 与 FlashAttention-3 的差距分析

| 特性 | SageAttention SM90 | FlashAttention-3 | 差距影响 |
|------|-------------------|-------------------|---------|
| CTA_Q | 64 | 128 | -5-10% |
| Pipeline Stages | 2 (phase-flip) | 3-5 (deep) | -5-15% |
| Warp Spec | 无 | 1 Load + 2 MMA WG | -10-20% |
| TMA Store | 无 (direct GMEM) | 是 | -3-5% |
| Persistent Kernel | 无 | 是 (TileScheduler) | -1-5% |
| QK Precision | INT8 | FP16 | **优势**: 2× compute throughput |
| PV Precision | FP8 RS mode | FP16 | **优势**: 2× compute throughput |
| Quantization | 需要额外的量化 kernel | 无 | **劣势**: 额外开销 |

**核心结论**: SageAttention 的量化策略 (INT8 QK + FP8 PV) 提供了 2× 的 tensor core 吞吐量优势，但架构级优化 (Warp Spec + Deep Pipeline + CTA_Q) 的缺失抵消了部分收益。通过补齐这些架构差距，SageAttention 的实际性能有望超过 FlashAttention-3。

## 7. 源码结构

```
operators/sageattention/cuda/
├── OPTIMIZATION_ANALYSIS.md    # 本文档
├── build.sh                    # 编译脚本
├── setup.py                    # torch extension 构建配置
├── test.py                     # 正确性/性能测试
└── csrc/                       # 源码 (复制自官方 SageAttention v2.2.0)
    ├── qk_int_sv_f8_cuda_sm90.cu   # 主 kernel (915 行)
    ├── pybind_sm90.cpp              # pybind11 绑定
    ├── attn_cuda_sm90.h             # 函数声明
    ├── attn_utils.cuh               # 工具函数 (TMA, softmax, 类型转换)
    ├── wgmma.cuh                    # WGMMA 指令封装
    ├── math.cuh                     # PTX 数学函数
    ├── mma.cuh                      # Tensor Core MMA 指令
    ├── utils.cuh                    # CHECK 宏
    ├── dispatch_utils.h             # 模板分发宏
    ├── cp_async.cuh                 # cp.async 封装
    ├── permuted_smem.cuh            # 共享内存 swizzle
    └── numeric_conversion.cuh       # FP8/INT8 类型转换
```
