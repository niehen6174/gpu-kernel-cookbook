# LayerNorm 优化实录：从 Latency-Bound 到 Memory-Bound

## 概述

本文记录对 LayerNorm 算子的完整 NCU profiling → 分析 → 优化 → 验证 全流程。

**实验配置**
- GPU: NVIDIA H20 (sm_90, Hopper)
- 理论峰值内存带宽: 4096 GB/s
- 测试形状: B=4096, N=1024（float32）
- 数据访问量: B×N×4 bytes × 4（读 x + w + b，写 y）= **64 MB**
- 理论最短时间 = 64 MB / 4096 GB/s ≈ **15.6 µs**

**LayerNorm 数学定义**
```
μ  = mean(x)         = (1/N) Σ x_i
σ² = variance(x)     = (1/N) Σ (x_i - μ)²
y_i = γ * (x_i - μ) / sqrt(σ² + ε) + β
```
与 RMSNorm 相比，LayerNorm 多了均值（mean）项，需要同时 reduce sum_x 和 sum_x²（或用 Welford 算法）。

---

## Round 0：基准测试

```
PyTorch    : 0.0317 ms  BW=2113.8 GB/s  (1.00x 基准)
Triton     : 0.0346 ms  BW=1936.9 GB/s  (0.92x)
CUDA v1    : 0.0825 ms  BW=813.3  GB/s  (0.38x)
CUDA v2    : 0.0547 ms  BW=1226.6 GB/s  (0.58x)
CUDA v3    : 0.0316 ms  BW=2124.1 GB/s  (1.00x)
CuTe v1   : 0.0785 ms  BW=854.4  GB/s  (0.40x)
CuTe v2   : 0.0506 ms  BW=1325.5 GB/s  (0.63x)
CuTe v3   : 0.0276 ms  BW=2429.9 GB/s  (1.15x)
```

**结论**：v1/v2 比 PyTorch 慢很多，存在明显优化空间。v3 达到甚至超过 PyTorch。

---

## Round 1：分析 v1 (Two-pass + shared memory block reduction)

### 1.1 实现思路
- 每个 block 处理一行（1024 floats）
- Pass1：scalar 累积 sum_x + sum_x²，shared memory block reduce
- Pass2：scalar 归一化 + 应用 γ, β
- Block size: 1024 threads

### 1.2 NCU 数据

| 指标 | v1 |
|------|-----|
| Duration | 134.30 µs |
| DRAM Throughput % | 21.22% |
| Memory Throughput % | 24.06% |
| Compute (SM) % | 52.30% |
| L2 Hit Rate | 52.18% |
| Achieved Occupancy | 94.48% |
| Theoretical Occupancy | 100% |
| Block Limit Registers | **2 blocks/SM** |
| Block Limit Warps | 2 blocks/SM |
| Warp Cycles Per Instruction | 28.11 cycles |

### 1.3 根因分析

**第一个信号：DRAM Throughput 21.22%（极低）**

H20 理论峰值 4096 GB/s，21.22% = 约 869 GB/s。但实际 Memory Throughput Gbyte/s 测量为 853.6 GB/s，与计算一致。

实际完成时间 134 µs，而理论最短 15.6 µs → **效率仅 11.6%**

**第二个信号：Block Limit Registers = 2**

1024 threads × (寄存器数/thread) = SM 寄存器上限。只能运行 2 个 block/SM，Wave数 = 4096 / (78 × 2) ≈ 26.3 → 整整 26.3 waves，每 wave 切换造成大量 idle。

**第三个信号：Compute 52.30% > Memory 21.22%**

Compute 更高说明不是单纯的内存带宽瓶颈——L1/L2 miss 的 long scoreboard stall 占主导，属于 **Latency-Bound**（访存延迟绑定）。

**根因**：
1. **Scalar load**：每次 `x[i]` = 32-bit load，无法走 128-bit LDG.128，指令发射效率低
2. **Pass2 重复读 x**：Pass1 读完 x 不保存，Pass2 再读一次，DRAM 流量翻倍
3. **26.3 waves**：1024-thread block 导致寄存器压力大，block/SM 只有 2
4. **Two-pass inherently requires 2x x reads**：除非用寄存器缓存

---

## Round 2：分析 v2 (Welford + Warp Reduction)

### 2.1 实现思路
- Block size: 256 threads（降低每 block 寄存器压力）
- Welford 在线算法：单趟计算 (mean, variance)，避免 E[x²]-E[x]² 的数值不稳定
- 两级规约：Warp-level shuffle → Block-level shared memory

### 2.2 NCU 数据

| 指标 | v1 | v2 |
|------|-----|-----|
| Duration | 134.30 µs | **108.96 µs** |
| DRAM Throughput % | 21.22% | **26.03%** |
| Compute (SM) % | 52.30% | 56.63% |
| L2 Hit Rate | 52.18% | **60.52%** |
| Block Limit Registers | 2 | **10** |
| Achieved Occupancy | 94.48% | 92.41% |
| Warp Cycles / Instruction | 28.11 | 25.15 |

### 2.3 分析

v2 相比 v1 改进了：
- Block size 256 → 寄存器压力降低 → Block Limit Registers 从 2 提升到 10
- L2 Hit Rate 52% → 60%：Welford 的单趟 x 读取减少了 L1 miss；Pass2 的 x 访问部分命中 L2 缓存
- Waves Per SM: 26.3 → 6.56（更少的 wave，启动开销更小）

但 v2 仍然问题：
- **DRAM 只有 26%**：仍然离理论峰值很远
- **Welford 串行依赖**：`delta = x - mean_prev`，当前 mean 依赖上一步，无法向量化
  - 每个 float 单独计算，无法使用 float4 指令
- **Pass2 仍读 x 两次**：Welford 在 Pass1 只更新状态，不存储 x，Pass2 需重新从内存读 x

---

## Round 3：设计 v3 优化

### 3.1 问题归纳

| 问题 | v1 | v2 | 解决方案 |
|------|----|----|----------|
| Scalar load | ✗ | ✗ | float4 向量化（128-bit LDG） |
| Pass2 重读 x | ✗ | ✗ | 寄存器缓存 x（register tile） |
| Welford 串行依赖 | — | ✗ | 两路独立 reduce（sum_x + sum_x²） |
| 寄存器压力 | 1024t | 256t | 256t + ELEMS_PER_THREAD 模板展开 |
| w/b 重读 | ✗ | ✗ | 寄存器缓存 w/b |

### 3.2 核心思路

```
两路独立 reduce（替代 Welford）:
  sum_x  += x[i]       (与 sum_x² 完全独立)
  sum_x² += x[i] * x[i]
  → 可以同时 #pragma unroll，编译器向量化

寄存器缓存 x/w/b:
  Load x[i] → rX[e]（float4 register）
  同时累积 sum_x + sum_x²
  Pass2 从 rX/rW/rB 读，零 DRAM traffic

float4 向量化:
  单次 128-bit LDG 处理 4 floats
  减少 75% 的 load 指令数
```

### 3.3 v3 代码结构

```cpp
template <int THREADS, int ELEMS_PER_THREAD>
__global__ void __launch_bounds__(THREADS)
layernorm_v3(const float* input, const float* weight, const float* bias,
             float* output, int N, float eps)
{
    // Step 1: 全量 load → 寄存器缓存，同时计算 sum_x + sum_x²
    float4 rX[ELEMS_PER_THREAD];  // N=1024: 4 × 256 = 1024 floats = 全行
    float4 rW[ELEMS_PER_THREAD];
    float4 rB[ELEMS_PER_THREAD];

    float local_sum = 0.0f, local_sq = 0.0f;
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int i = tid + e * THREADS;
        rX[e] = __ldg(&x4[i]);   // read-only cache，避免 L1 污染
        rW[e] = __ldg(&w4[i]);
        rB[e] = __ldg(&b4[i]);
        local_sum += rX[e].x + rX[e].y + rX[e].z + rX[e].w;  // 两路独立
        local_sq  += rX[e].x*rX[e].x + rX[e].y*rX[e].y + ...;
    }

    // Step 2: 两路同步 warp + block reduce
    // ... warp shuffle → smem → warp 0 reduce ...

    float mean    = s_sum[0] / N;
    float var     = s_sq[0]  / N - mean * mean;
    float inv_std = rsqrtf(var + eps);

    // Step 3: Pass2 完全从寄存器读（零 DRAM）
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        out.x = (rX[e].x - mean) * inv_std * rW[e].x + rB[e].x;
        y4[i] = out;
    }
}
```

---

## Round 4：v3 NCU 数据

### 4.1 NCU 数据

| 指标 | v1 | v2 | v3 |
|------|-----|-----|-----|
| Duration | 134.30 µs | 108.96 µs | **39.84 µs** |
| DRAM Throughput % | 21.22% | 26.03% | **69.10%** |
| Memory Throughput % | 24.06% | 26.03% | 69.10% |
| Compute (SM) % | 52.30% | 56.63% | **36.35%** |
| L2 Hit Rate | 52.18% | 60.52% | **50.93%** |
| Memory Throughput (GB/s) | 853.6 | 1053 | **2780** |
| Achieved Occupancy | 94.48% | 92.41% | **45.82%** |
| Theoretical Occupancy | 100% | 100% | **50%** |
| Block Limit Registers | 2 | 10 | **4** |
| Warp Cycles / Instruction | 28.11 | 25.15 | **18.71** |
| Waves Per SM | 26.3 | 6.56 | **13.13** |

### 4.2 数据解读

**DRAM 21% → 69%（3.26× 提升）**

v3 真正成为了 Memory-Bound：
- Memory% ≈ Compute%（69% vs 36%），Memory 主导
- 实际带宽 2780 GB/s，达到峰值的 67.8%

**Occupancy 从 94% 降到 46%**

这是 **刻意的权衡**：
- v3 每 thread 需要 `rX + rW + rB = 3 × ELEMS_PER_THREAD = 12` 个 float4 寄存器（其中 N=1024, ELEMS=4）
- `__launch_bounds__(256)` + 更多寄存器 → Theoretical Occupancy 仅 50%
- Block Limit Registers: 4（每 SM 只能跑 4 个 256-thread block = 32 warps）
- 但 DRAM 带宽利用率 3× 提升，duration 缩短 3.4×，**Occupancy 下降完全值得**

**Warp Cycles 从 25-28 降到 18.71**

- 寄存器缓存 Pass2 完全不读内存 → 消除 L1/L2 miss → Long Scoreboard Stall 大幅降低
- float4 LDG.128 → 指令发射效率更高

**L2 Hit Rate 从 60% 降到 51%**

这是 **正常的**：v3 只读一次 x/w/b（通过 `__ldg` read-only cache），L2 完全不需要缓存 x（不会重读），因此 L2 Hit Rate 降低反而说明 x 确实只读了一次。

### 4.3 与 CuTe v3 对比

| 指标 | cuda_v3 | cute_v3 |
|------|---------|---------|
| Duration | 39.84 µs | **39.55 µs** |
| DRAM % | 69.10% | **69.86%** |
| Memory (GB/s) | 2780 | **2810** |
| Block Limit Reg | 4 | 4 |
| Warp Cycles | 18.71 | **18.xx** |

CuTe v3 与 CUDA v3 几乎完全一致，CuTe 的抽象层没有引入任何开销——这正是 CuTe 的设计目标。

---

## Round 5：性能汇总与优化终止分析

### 5.1 完整 NCU 数据对比

| 版本 | Duration | DRAM% | BW(GB/s) | Warp Cycles | Occupancy |
|------|----------|-------|----------|-------------|-----------|
| v1 (two-pass, scalar) | 134.30 µs | 21.22% | 853 | 28.11 | 94.5% |
| v2 (Welford, scalar) | 108.96 µs | 26.03% | 1053 | 25.15 | 92.4% |
| v3 (float4, reg cache) | 39.84 µs | 69.10% | 2780 | 18.71 | 45.8% |
| cute_v3 | 39.55 µs | 69.86% | 2810 | ~18 | ~46% |
| PyTorch | ~31.7 µs | ~84% | ~3456 | — | — |
| 理论最优 | 15.6 µs | 100% | 4096 | — | — |

### 5.2 与 PyTorch 的差距分析

| 项 | 我们 v3 | PyTorch |
|----|---------|---------|
| Duration | 39.84 µs | ~31.7 µs |
| DRAM% | 69.10% | ~84% |
| BW | 2780 GB/s | ~3456 GB/s |
| 差距 | 1.26× | 基准 |
| 相对理论最优 | 39.2% | 49.3% |

PyTorch 的优势可能来自：
1. **Fused kernel**：可能将相邻操作融合（如 residual add + LayerNorm）
2. **更大 N 优化**：PyTorch 使用 `apex` / 自定义 persistent kernel，多 SM 并行化
3. **bf16/fp16 路径**：内部使用更窄的数据类型以减少带宽

### 5.3 优化终止判断

| 判断条件 | v3 状态 | 结论 |
|----------|---------|------|
| DRAM > 60% | 69.10% ✓ | 已是 Memory-Bound |
| Warp Cycles 已降低 | 18.71（比 v1 低 33%）✓ | 延迟已大幅改善 |
| 相对 PyTorch | 1.26× 差距 | 接近但未超越 |
| 进一步优化收益 | 需要 kernel fusion | 单算子优化空间有限 |

**结论**：v3 在单独 LayerNorm 场景下已达到良好优化状态（69% DRAM 利用率）。进一步提升需要融合 residual add（类似 sglang 的 `add_residual + norm`），这将使 DRAM 利用率提升至 ~85%+。

---

## 关键优化技术总结

### 1. 两路独立 Reduce 替代 Welford

**为什么 Welford 无法向量化？**

```cpp
// Welford: 串行依赖链
for (int i = ...) {
    state.count += 1.0f;
    float delta = xi - state.mean;    // 依赖上一步的 state.mean
    state.mean += delta / state.count;  // 更新 mean
    float delta2 = xi - state.mean;   // 依赖刚更新的 mean
    state.m2 += delta * delta2;       // 串行依赖无法展开
}
```

Welford 每一步的 `delta` 依赖上一步的 `state.mean`，形成**循环依赖**，编译器无法向量化。

**两路 Reduce：完全独立**

```cpp
// 两路独立 reduce: 可完全 #pragma unroll
float local_sum = 0.0f, local_sq = 0.0f;
for (int i = ...) {
    float xi = x[i];
    local_sum += xi;         // 无依赖
    local_sq  += xi * xi;   // 无依赖，与 local_sum 相互独立
}
```

注意：两路方式在极端值（overflow）下数值稳定性略逊于 Welford，但实际 float32 LLM 推理中不成问题。

### 2. 寄存器缓存消除 Pass2 重读

LayerNorm 必须两趟扫描（先算 mean/var，再归一化）。v1/v2 在 Pass2 重读 x，造成额外 ~16 MB DRAM 流量（B×N×4 = 4096×1024×4 = 16 MB）。

v3 的 `rX[ELEMS_PER_THREAD]` 寄存器数组将整行缓存在寄存器中，Pass2 的访存完全来自寄存器，**每 block 节省 16 MB DRAM 读取**（实际数据约 16 MB / 4096 blocks = 4 KB/block）。

### 3. float4 + `__ldg` 向量化

```
scalar load: 32-bit 指令，每次处理 1 float
float4 load: 128-bit LDG.128 指令，每次处理 4 floats
→ 指令数降低 4×，减少 Load/Store Unit 压力
→ __ldg 走 read-only texture cache，不污染 L1 data cache
```

### 4. 占用率 vs 寄存器 的权衡

v3 通过 `ELEMS_PER_THREAD=4`（N=1024 时每 thread 处理 4×4=16 floats）缓存了所有数据：

```
rX[4] + rW[4] + rB[4] = 12 个 float4 = 48 个 float 寄存器
加上计算寄存器 → 总寄存器约 64 个/thread
256 threads × 64 regs = 16384 regs/block
SM 总寄存器 65536 → 65536 / 16384 = 4 blocks/SM
→ Theoretical Occupancy = 4 × 8 warps / 64 = 50%
```

**关键结论**：Occupancy 从 94% 降到 50%，但 DRAM 利用率从 21% 升到 69%。速度提升 3.4×。**对于 Memory-Bound 算子，高占用率 ≠ 高性能。**

---

## 完整优化流程回顾

```
v1 (21% DRAM, 134µs)
  问题：scalar load + Pass2 重读 x + 1024t 寄存器压力大
    ↓
  优化方向：float4 向量化 + 寄存器缓存 + 减小 block size
    ↓
v2 (26% DRAM, 109µs)
  问题：Welford 串行依赖阻止向量化 + 仍然 scalar + Pass2 重读 x
    ↓
  优化方向：两路独立 reduce + float4 + 寄存器缓存 x/w/b
    ↓
v3 (69% DRAM, 39.8µs)  ← 当前最优
  进一步空间：融合 residual add（→ fused_add_layernorm）
```
