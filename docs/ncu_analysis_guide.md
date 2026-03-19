# NCU 性能分析指南：读懂数据、找到瓶颈、判断优化终点

本文以实际 ncu 输出为例，系统介绍如何解读各 section 的指标、如何定位性能瓶颈、以及如何判断一个 kernel 是否已经优化到位。

---

## 目录

1. [整体分析框架](#1-整体分析框架)
2. [Speed of Light：第一眼看什么](#2-speed-of-light第一眼看什么)
3. [Launch Statistics：配置是否合理](#3-launch-statistics配置是否合理)
4. [Occupancy：占用率的真实含义](#4-occupancy占用率的真实含义)
5. [Memory Workload Analysis：内存瓶颈细化](#5-memory-workload-analysis内存瓶颈细化)
6. [Warp State Statistics：时间去哪了](#6-warp-state-statistics时间去哪了)
7. [Compute Workload Analysis：计算流水线](#7-compute-workload-analysis计算流水线)
8. [Roofline 模型：终极性能上界](#8-roofline-模型终极性能上界)
9. [实战：rms_norm cuda_v2 完整分析](#9-实战rms_norm-cuda_v2-完整分析)
10. [优化终止条件：什么时候可以停了](#10-优化终止条件什么时候可以停了)
11. [常见瓶颈模式与对应优化手段](#11-常见瓶颈模式与对应优化手段)
12. [分析流程 SOP](#12-分析流程-sop)

---

## 1. 整体分析框架

GPU kernel 的性能瓶颈本质上只有两类：

```
┌──────────────────────────────────────────────┐
│  Kernel 性能 = min(Compute 上界, Memory 上界) │
│                                              │
│  Compute-bound: 计算单元先跑满               │
│  Memory-bound:  内存带宽先跑满               │
└──────────────────────────────────────────────┘
```

**第一步**：判断 kernel 属于哪一类
**第二步**：在该类别下，找具体瓶颈
**第三步**：优化后验证瓶颈转移了还是消除了

ncu 的各 section 对应不同层次的分析深度：

```
Speed of Light  →  定性：Memory-bound 还是 Compute-bound？
Launch Stats    →  Grid/Block 配置是否合理？
Occupancy       →  SM 利用率是否够高？
Memory Analysis →  带宽哪一层在受限？coalescing 好不好？
Warp States     →  stall 在哪里？占多少比例？
Compute         →  指令流水线饱和吗？
Roofline        →  距理论上界还差多远？
```

---

## 2. Speed of Light：第一眼看什么

`--set basic` 默认获取的核心 section，是所有分析的起点。

### 关键指标

```
Memory Throughput    %     内存子系统整体利用率（取各层最大值）
DRAM Throughput      %     HBM 带宽利用率（相对峰值）
Compute (SM) Throughput %  SM 计算单元利用率
Duration             us    kernel 实际执行时间
```

### 如何判断瓶颈类型

```
Memory Throughput >> Compute Throughput  →  Memory-bound
Compute Throughput >> Memory Throughput  →  Compute-bound
两者接近且都低                            →  Latency-bound（小 kernel，launch overhead）
两者都接近 100%                           →  理想状态（极少见）
```

### 典型读法示例

```
Memory Throughput   63.04%    ← 主要瓶颈
Compute Throughput  24.57%    ← 计算闲置
→ Memory-bound kernel，优化方向：减少内存访问量或提高带宽利用率
```

```
Memory Throughput   15.23%
Compute Throughput  87.43%    ← 主要瓶颈
→ Compute-bound kernel，优化方向：减少计算量或用更快的指令（如 Tensor Core）
```

### 百分比的参考意义

- **> 80%**：接近硬件上限，优化空间有限
- **60-80%**：较好，但仍有提升空间
- **< 60%**：存在明显瓶颈，值得深入分析
- **< 20%**：kernel 严重受限（可能是 latency-bound 或配置问题）

---

## 3. Launch Statistics：配置是否合理

### 关键指标

```
Grid Size              总 block 数
Block Size             每 block 线程数
Registers Per Thread   每线程寄存器数
Dynamic Shared Memory  动态 smem 用量
Waves Per SM           总 block 数 / SM 数 / 每 SM active block 数
```

### Waves Per SM

```
Waves Per SM = ceil(Grid Size / (# SMs × Blocks per SM))
```

- **Waves ≥ 2**：SM 有机会 overlap 多个 wave，隐藏延迟
- **Waves < 1**：Grid 太小，部分 SM 空闲（常见于小 batch）
- **Waves 是小数**：最后一个 wave 的 SM 利用率较低（尾效应）

例：`Waves Per SM = 6.56` → 6 个完整 wave + 最后 0.56 波，GPU 整体利用率高

### Registers Per Thread

- 每 SM 有固定寄存器（H20: 65536）
- 每 SM 最大 block 数 = 65536 / (Block Size × Registers Per Thread)
- **寄存器太多** → Block 数受限 → Occupancy 下降
- 优化：`__launch_bounds__(blockDim, minBlocks)` 限制寄存器上限

```
Registers = 26, Block Size = 256
→ 每 block 寄存器 = 26 × 256 = 6656
→ 每 SM 可放 65536 / 6656 ≈ 9.8 → 限制到 8 blocks/SM（Block Limit Registers）
```

### Block Size 的选择

| Block Size | 特点 |
|-----------|------|
| 32 (1 warp) | 调度灵活，但 smem/寄存器利用低 |
| 128 | 常用中间值 |
| 256 | Reduction kernel 常用 |
| 512/1024 | 需要确认寄存器不超限 |

**经验原则**：Block Size 是 32 的倍数（warp 大小），且使 `Theoretical Occupancy ≈ 100%`

---

## 4. Occupancy：占用率的真实含义

Occupancy = 实际 active warp 数 / 理论最大 warp 数（per SM）

### 关键指标

```
Theoretical Occupancy     %    理论上限（由 block size、寄存器、smem 决定）
Achieved Occupancy        %    实际运行时均值
Block Limit Registers     block  寄存器限制每 SM 最多几个 block
Block Limit Shared Mem    block  smem 限制每 SM 最多几个 block
Block Limit Warps         block  warp 槽限制
```

### Theoretical vs Achieved

- **Theoretical** 由静态配置决定（编译时可预测）
- **Achieved** 是运行时实测，一般 ≤ Theoretical
- 两者差距大 → 动态负载不均衡（load imbalance）

### 占用率高就好吗？

**不一定**。Occupancy 的作用是**隐藏延迟**：当一个 warp 在等内存时，SM 切换到另一个 warp 继续执行。

```
对 Memory-bound kernel：高 Occupancy 很重要，能 overlap 内存延迟
对 Compute-bound kernel：Occupancy 影响较小，瓶颈在计算单元本身
对 Latency-bound（小 kernel）：Occupancy 再高也救不了 launch overhead
```

**经验**：Achieved Occupancy > 50% 通常就够用了；追求从 70% → 100% 的收益往往很小

### 看 Block Limit 找优化点

```
Block Limit Registers  = 8   ← 寄存器是瓶颈
Block Limit Shared Mem = 28  ← smem 不是瓶颈
Block Limit Warps      = 8   ← warp 槽也到了上限
→ 当前 8 blocks/SM，由寄存器和 warp 槽共同限制，无法再提高
```

如果 `Block Limit Registers` 是最小值，考虑：
- 减少中间变量（减少寄存器用量）
- 用 `__launch_bounds__` 提示编译器

---

## 5. Memory Workload Analysis：内存瓶颈细化

需要 `--section MemoryWorkloadAnalysis` 或 `--set full`。

### 内存层次结构

```
Thread → L1/TEX Cache (per SM) → L2 Cache (shared) → DRAM (HBM)
          ~32 KB/SM               ~40 MB               H20: 4.0 TB/s
```

### 关键指标

```
L1 Hit Rate         %    L1 cache 命中率
L2 Hit Rate         %    L2 cache 命中率
L1/L2/DRAM 各自的实际带宽 GB/s
Global Memory Load/Store Efficiency  %    有效字节 / 实际传输字节
```

### Global Load Efficiency（Coalescing）

这是 **最重要** 的内存优化指标：

```
Load Efficiency = Requested Bytes / Actual Bytes Transferred

= 100%  理想：所有加载的字节都被线程使用
< 100%  存在浪费：加载了不需要的字节（非合并访问）
```

**示例**：
```
Block 内 32 个线程，访问地址：
  合并：thread 0 → addr 0, thread 1 → addr 4, ..., thread 31 → addr 124
  → 一次 128B 事务，效率 100%

  非合并：thread 0 → addr 0, thread 1 → addr 512, ...
  → 32 次独立事务，效率 ~3%
```

### 判断是哪一级在受限

```
DRAM Throughput = 95%  → HBM 带宽饱和，无法再快（已到硬件上限）
L2 Throughput  = 95%   → L2 到 SM 的带宽饱和
L1 Throughput  = 95%   → L1 到寄存器的带宽饱和

DRAM = 60%, L2 = 80%   → 部分数据从 L2 cache 命中，减轻了 DRAM 压力
```

---

## 6. Warp State Statistics：时间去哪了

需要 `--section WarpStateStats` 或 `--set full`。

这是定位 **latency 来源** 的核心 section，直接回答"warp 在等什么"。

### Warp 状态分类

```
Eligible     warp 准备好执行，等待调度器分配（理想状态）
Stalled      warp 在等某种资源，无法执行
  ├── Long Scoreboard    等待全局内存（L2/DRAM）返回数据
  ├── Short Scoreboard   等待 L1 cache / smem
  ├── Wait               等待 barrier（__syncthreads）
  ├── MIO Throttle       内存指令队列满
  ├── Math Pipe Throttle 数学流水线满（compute-bound 的标志）
  ├── Not Selected       warp eligible 但调度器选了别人
  └── No Instructions    warp 没有指令（kernel 边界）
```

### 如何读这些数据

```
Long Scoreboard = 70%  → 大部分时间在等 DRAM/L2 返回数据
  → Memory-bound，提高带宽利用率或减少访存量

Math Pipe Throttle = 60% → 数学运算流水线拥堵
  → Compute-bound，考虑 Tensor Core 或减少计算量

Wait (barrier) = 40%   → 大量时间在 __syncthreads() 等待
  → 同步过多，考虑减少 barrier 或用 warp-level 原语替代

MIO Throttle = 30%     → 内存指令发射太密集
  → 考虑用 float4/float2 减少内存指令数量
```

### 优化优先级

优先处理 **占比最高** 的 stall 原因：

```
Long Scoreboard > Long Scoreboard + ...   ← 先解决内存延迟
├── 提高 Occupancy（更多 warp 可以 hide latency）
├── 使用 prefetch（__ldg, async copy）
└── 减少访存量（算法优化、reuse）
```

---

## 7. Compute Workload Analysis：计算流水线

需要 `--section ComputeWorkloadAnalysis` 或 `--set full`。

### 关键指标

```
Pipe FMA Active         %    FMA（乘加）流水线利用率
Pipe Shared Active      %    shared memory 操作流水线
Pipe LSU Active         %    Load/Store Unit 利用率
Executed Instructions         每 warp 执行的指令数
IPC (Instructions Per Cycle)  每周期指令数
```

### 指令混合分析

```
FMA Throughput 高   → 计算密集，接近 compute-bound
LSU Throughput 高   → 访存密集，接近 memory-bound
两者都低            → latency-bound 或 warp 数量不足
```

### SM 效率

```
SM Active Cycles / Total SM Elapsed Cycles
≈ 1.0  理想，SM 全程有效工作
< 0.8  SM 存在空转（尾效应、load imbalance）
```

---

## 8. Roofline 模型：终极性能上界

需要 `--section SpeedOfLight_HierarchicalSingleRooflineChart`。

### 什么是 Roofline

```
                    |
 Peak Compute (TFLOPS)  -------- ← Compute Roof
                    |        /
  Performance       |       /
  (FLOPS/s)         |      /  ← Roofline
                    |     /
                    |    /
                    |   * ← kernel 实际位置
                    |  /
                    | /
                    |/
                    +-----------------
                    Arithmetic Intensity (FLOPS/Byte)
                           ^
                     Ridge Point
```

### 解读 kernel 在 Roofline 上的位置

```
在 ridge point 左侧：Memory-bound
  → 提高 AI（减少访存 / 增加复用）或接近 Memory Roof（提高带宽利用率）

在 ridge point 右侧：Compute-bound
  → 提高 compute 效率（Tensor Core、更少低效指令）

距离 Roof 的比例：
  kernel 点 / Roof = 实际效率百分比
  如 kernel 在 Memory Roof 的 63%，说明还有 37% 可提升空间
```

### Arithmetic Intensity（算术强度）

```
AI = Total FLOPs / Total Bytes Transferred

AI < Ridge Point → Memory-bound
AI > Ridge Point → Compute-bound

H20 Ridge Point（FP32）≈ 44 TFLOPS / 4.0 TB/s = 11 FLOP/Byte
```

常见算子的理论 AI：
- vector_add：1 FLOP/Byte（极低，严重 memory-bound）
- rms_norm：~3-5 FLOP/Byte（memory-bound）
- matmul（大矩阵）：~N FLOP/Byte（可以 compute-bound）

---

## 9. 实战：rms_norm cuda_v2 完整分析

以前面 profile 拿到的数据为例，做完整解读。

### 原始数据

```
Duration:             44.80 µs
Memory Throughput:    63.04%      → DRAM: 63.04%, L2: 69.39%, L1: 48.52%
Compute Throughput:   24.57%
Achieved Occupancy:   90.64%
Registers/Thread:     26
Block Size:           256
Grid Size:            4096
Waves Per SM:         6.56
Block Limit Registers: 8
Block Limit Shared Mem: 28
```

### 分析步骤

**Step 1：定性判断**
```
Memory 63% >> Compute 24%  →  Memory-bound kernel ✓
（符合 RMSNorm 的理论预期：读 x、读 w、写 y，计算少）
```

**Step 2：带宽利用率**
```
H20 峰值 DRAM 带宽 = 4.0 TB/s
实际 DRAM 带宽 = 4.0 × 63.04% = 2.52 TB/s

理论最优带宽需求（B=4096, N=4096, float32）：
  读 x: 4096×4096×4 = 64 MB
  读 w: 4096×4 = 16 KB（被 L2 cache 命中，不算 DRAM）
  写 y: 64 MB
  总 DRAM ≈ 128 MB

理论最优时间 = 128 MB / 4.0 TB/s = 32 µs
实测时间 = 44.80 µs
效率 = 32 / 44.80 = 71.4%
（与 DRAM Throughput 63% 接近，合理）
```

**Step 3：Occupancy 分析**
```
Achieved Occupancy = 90.64% → 很高，不是瓶颈
Block Limit = 8 blocks/SM（寄存器和 warp 槽共同限制）
Waves = 6.56 → GPU 满载，波次充足
```

**Step 4：定位剩余 37% 差距**
```
实际 DRAM 利用率 63%，距 100% 还有 37% 差距。
可能原因（需要 WarpStateStats 验证）：
1. Long Scoreboard stall：内存延迟没有完全被 warp 切换覆盖
2. 访问模式问题：w 的广播访问（所有 row 读同一个 w）可能造成 L1/L2 bank conflict
3. 写操作与读操作无法完全流水
```

**Step 5：优化建议**
```
1. 当前已是 float4 向量化版本，访存粒度已优化
2. 进一步方向：
   - Kernel fusion（fused_add_rmsnorm）：将两次访存合并为一次
   - 使用 __ldg() 加载 w（weight），走 texture cache 路径
   - persistent kernel：减少 kernel launch overhead（Waves 尾效应）
3. 63% DRAM 利用率在 reduction kernel 中属于正常水平，
   纯 RMSNorm 难以突破 ~70-75%（每行需要两次 pass，访存不规整）
```

---

## 10. 优化终止条件：什么时候可以停了

没有绝对的"优化完毕"，但以下条件满足时可以认为基本到位：

### Memory-bound kernel

```
✓ DRAM Throughput > 80%        接近硬件带宽上限
✓ Global Load Efficiency = 100% 内存访问完全合并
✓ L2 Hit Rate 合理             复用的数据能被 cache
✓ Achieved Occupancy > 60%     足够的 warp 隐藏内存延迟
✓ 与 cuBLAS/cuDNN 等库相比相差 < 10%（如有参照）
```

### Compute-bound kernel

```
✓ Compute Throughput > 80%     计算单元接近饱和
✓ 使用了合适的指令（Tensor Core for matmul）
✓ 寄存器/smem 使用合理，Occupancy 不过低
✓ IPC 接近理论最大值
```

### 通用终止条件

```
1. Roofline：kernel 点距离 Roof < 20%（处于 Roof 的 80% 以上）
2. 对比参照：与 cuBLAS / Triton autotuned 版本性能相当
3. 收益递减：优化一轮后提升 < 5%，成本不值
4. 算法已是理论最优：I/O 量无法再减少（如读写每个元素恰好一次）
```

### 实际项目中的判断标准

| 场景 | 参考目标 |
|------|---------|
| Memory-bound（向量化算子） | DRAM Throughput > 75% |
| Reduction（softmax/norm） | DRAM Throughput > 60%，Occupancy > 75% |
| Matmul（Tensor Core） | Compute Throughput > 85%，接近 cuBLAS |
| 小 kernel（< 1 µs） | 与 Triton 相当即可，launch 开销占主导 |

---

## 11. 常见瓶颈模式与对应优化手段

### Pattern 1：DRAM Throughput 低 + Load Efficiency 低

**现象**：内存带宽没跑满，但 coalescing 差
**诊断**：`Global Memory Load Efficiency < 80%`
**优化**：
- 检查访问模式：确保同一 warp 的线程访问连续地址
- 重排数据布局（AoS → SoA）
- 使用 shared memory 做 transpose（如 matrix transpose kernel）

### Pattern 2：DRAM Throughput 高 + 性能仍不理想

**现象**：DRAM 已跑满，但 kernel 还不够快
**诊断**：已到硬件带宽上限，无法再提速
**优化**：
- 减少访存量（算法优化，减少 redundant load）
- Kernel fusion（把多个 kernel 合并，减少 DRAM round-trip）
- 数学变形（如 online softmax 避免两次读取）

### Pattern 3：Long Scoreboard Stall 高

**现象**：Warp States 中 Long Scoreboard 占 50%+
**诊断**：大量时间等待全局内存
**优化**：
- 增加 Occupancy（更多 warp 可以 overlap 内存延迟）
- 使用异步内存拷贝（`cuda::memcpy_async`，CP.ASYNC 指令）
- Prefetch：提前发出内存请求，在用到之前就开始加载

### Pattern 4：Compute Throttle 高

**现象**：Warp States 中 Math Pipe Throttle 高
**诊断**：计算流水线是瓶颈（Compute-bound）
**优化**：
- 换用 Tensor Core（`wmma` API 或 CUTLASS）：FP32 matmul → TF32/FP16
- 减少指令数：用数学近似（`__expf` vs `exp`，`rsqrtf` vs `1/sqrt`）
- 避免 branch divergence（warp 内分支不一致导致串行化）

### Pattern 5：Wait (Barrier) Stall 高

**现象**：大量 `__syncthreads()` 导致等待
**诊断**：同步点太多或 block 内负载不均
**优化**：
- 用 warp-level primitives 替代 block-level（`__shfl_xor_sync` 代替 smem reduce + sync）
- 减少 sync 次数（合并多次 smem 操作为一次）
- 考虑不需要 sync 的算法（纯 warp-level reduction）

### Pattern 6：Waves Per SM < 1

**现象**：Grid 太小，部分 SM 空闲
**诊断**：batch size 小，GPU 没有跑满
**优化**：
- 合并多个小 batch 的计算（batched kernel）
- 减小 block size（让更多 block 分布到 SM）
- 在 kernel 内部做 grid-stride loop（一个 block 处理多行）

---

## 12. 分析流程 SOP

按以下顺序系统地分析一个 kernel：

```
Step 1: 跑 --set basic，看 Speed of Light
        ├── Memory% >> Compute%?  → Memory-bound，走路径 A
        ├── Compute% >> Memory%?  → Compute-bound，走路径 B
        └── 两者都低?             → Latency-bound，走路径 C

路径 A（Memory-bound）:
  Step 2: 看 DRAM Throughput 绝对值
          ├── > 80%?  → 已接近硬件上限，考虑算法层面优化
          └── < 80%?  → 继续往下找原因

  Step 3: 跑 --section MemoryWorkloadAnalysis
          ├── Load Efficiency < 90%?  → 非合并访问，优化内存布局
          ├── L2 Hit Rate 低?        → 数据复用性差，考虑 tiling
          └── 哪一层是瓶颈?          → L1/L2/DRAM 各层 throughput

  Step 4: 跑 --section WarpStateStats
          └── 主要 stall 原因是什么?
              ├── Long Scoreboard → 提高 Occupancy 或用 prefetch
              ├── MIO Throttle    → 减少内存指令数（float4 等）
              └── Wait (barrier)  → 减少 sync，用 warp primitives

路径 B（Compute-bound）:
  Step 2: 跑 --section ComputeWorkloadAnalysis
          ├── Pipe FMA Throughput 高?  → FMA 是瓶颈
          └── 指令类型分析

  Step 3: 跑 --section SpeedOfLight_HierarchicalSingleRooflineChart
          └── 在 Roofline 图上的位置，距 Compute Roof 多远

  Step 4: 考虑 Tensor Core / 更高精度指令

路径 C（Latency-bound）:
  Step 2: 看 Waves Per SM 和 Grid Size
          ├── Grid 太小?  → 增大 batch 或合并 kernel
          └── Block 太小? → 调整 Grid/Block 配置

  Step 3: 考虑 kernel fusion（多个小 kernel 合并成一个）

Step 5（所有路径）: 优化后再次 profile，验证：
  ├── 目标指标是否提升
  ├── 瓶颈是否转移（旧瓶颈消除后，新瓶颈出现）
  └── 是否达到终止条件（> 80% Roof 或与参照库相当）
```

---

## 附录：快速命令参考

```bash
# 基础诊断（第一步必跑）
bash scripts/profile.sh --op <op> --kernel <kernel>

# 内存深度分析
bash scripts/profile.sh --op <op> --kernel <kernel> \
    --section SpeedOfLight,MemoryWorkloadAnalysis,WarpStateStats

# Compute-bound 分析
bash scripts/profile.sh --op <op> --kernel <kernel> \
    --section SpeedOfLight,ComputeWorkloadAnalysis,WarpStateStats

# Roofline 图
bash scripts/profile.sh --op <op> --kernel <kernel> \
    --section SpeedOfLight_HierarchicalSingleRooflineChart

# 全量（慢 10-30x，最详细）
bash scripts/profile.sh --op <op> --kernel <kernel> --set full

# 重新查看已有报告
ncu --import results/ncu/<file>.ncu-rep
```
