# RMSNorm CUDA Kernel 性能优化实战记录

**目标算子**: `rms_norm`，B=4096 行，N=4096 列，float32
**GPU**: NVIDIA H20（Hopper sm_90），峰值 DRAM 带宽 4.0 TB/s
**优化路径**: cuda_v1 → cuda_v2（已有）→ cuda_v3（本次新增）

---

## 目录

1. [基础数据与理论上界](#1-基础数据与理论上界)
2. [Round 1：分析 cuda_v1](#2-round-1分析-cuda_v1)
3. [Round 1 优化：float4 向量化 → cuda_v2](#3-round-1-优化float4-向量化--cuda_v2)
4. [Round 2：分析 cuda_v2，找出新瓶颈](#4-round-2分析-cuda_v2找出新瓶颈)
5. [Round 2 优化：寄存器缓存消除重读 → cuda_v3](#5-round-2-优化寄存器缓存消除重读--cuda_v3)
6. [Round 3：分析 cuda_v3，验证优化效果](#6-round-3分析-cuda_v3验证优化效果)
7. [三版本横向对比](#7-三版本横向对比)
8. [优化到头了吗？终止条件分析](#8-优化到头了吗终止条件分析)
9. [关键经验总结](#9-关键经验总结)

---

## 1. 基础数据与理论上界

### 算子访存量（理论最优）

RMSNorm 每行的操作：读 x（N 个 float）、读 w（N 个 float，所有行共享，会被 L2 cache）、写 y（N 个 float）。

```
读  x:   B × N × 4 B = 4096 × 4096 × 4 =  64 MB   (每次都需要从 DRAM 读)
读  w:        N × 4 B =       4096 × 4 =  16 KB   (很小，几乎全部命中 L2 cache)
写  y:   B × N × 4 B =                =  64 MB   (写回 DRAM)

理论最优 DRAM 流量 ≈ 128 MB（x 读一次 + y 写一次，w 走 cache）
理论最优时间 = 128 MB / 4000 GB/s = 32.0 µs
```

这是 **性能天花板**。任何实现都不可能快过 32 µs（在这块 H20 上）。

### 算子类型判断

```
算术强度 (AI) = FLOPs / Bytes
  每个元素: x² + 累加 + rsqrt + 乘法 ≈ 5 FLOPs
  总 FLOPs = 4096 × 4096 × 5 ≈ 84 MFLOPs
  总 Bytes = 128 MB

AI = 84 MFLOPs / 128 MB ≈ 0.66 FLOP/Byte << H20 Ridge Point (~11 FLOP/Byte)

→ 强 Memory-bound kernel，DRAM 带宽是唯一瓶颈
```

---

## 2. Round 1：分析 cuda_v1

### cuda_v1 实现回顾

```cuda
// 每线程步长遍历，每次处理 1 个 float
float local_ss = 0.0f;
for (int i = tid; i < N; i += blockDim.x) {
    float xi = xrow[i];
    local_ss += xi * xi;              // Pass 1: 读 x，计算 sum(x²)
}
// ... reduce → rms_inv ...
for (int i = tid; i < N; i += blockDim.x) {
    yrow[i] = xrow[i] * rms_inv * w[i];  // Pass 2: 再次读 x，写 y
}
```

配置：`gridDim=(4096,1,1)`, `blockDim=(1024,1,1)`

### ncu 原始数据

```
Duration:                   99.49 µs
Memory Throughput:          28.55%    → DRAM 实际带宽: 4.0 × 28.55% = 1.14 TB/s
Compute Throughput:         40.23%
DRAM Throughput:            28.55%
L2 Cache Throughput:        33.19%
Achieved Occupancy:         93.19%
Registers Per Thread:       16
Block Size:                 1024
Grid Size:                  4096
Waves Per SM:               26.26
```

### 分析解读

**Step 1：定性**

```
Memory 28% < Compute 40%  ←  不寻常！Memory-bound 算子 Memory 利用率反而低于 Compute？
→ 说明两者都没跑满，实为 Latency-bound
NCU 的提示: "low compute throughput and memory bandwidth utilization below 60% of peak"
            "Look at Warp State Statistics"
```

**Step 2：带宽利用率**

```
DRAM Throughput = 28.55%
实际 DRAM 带宽 = 1.14 TB/s（H20 峰值 4.0 TB/s，利用率极低）

实测时间 99.49 µs，理论最优 32 µs，效率 = 32 / 99.49 = 32.2%
→ 性能严重落后于理论上界，有大量优化空间
```

**Step 3：找根因——Warp State Statistics**

```
Warp Cycles Per Issued Instruction:  35.52 cycle
Est. Local Speedup:                  53.12%

每条指令平均等待 35.52 个周期，其中：
→ 18.9 cycles（53%）在 stall waiting for L1TEX (global memory)
```

这说明：warp 发出 load 指令后，要等 ~19 个周期才拿到数据。原因：

```
v1 每个线程每次只 load 1 个 float（32 bit），单次内存事务 32B 只有 4B 有效
虽然同 warp 的 32 个线程合并访问是连续的（128B/事务），但：

Block Size = 1024 → 32 warps per block
每 SM 只有 2 个 block（Block Limit Warps = 2 → 4 warps active at a time）
→ 活跃 warp 太少，无法 overlap 内存延迟
```

实际上 Occupancy 达到 93%，waves=26，这里主要问题是**访存带宽本身未充分利用**。

**Step 4：根本问题**

```
v1 每个线程每次 load 4B（1 个 float）
内存事务 128B，每事务有效载荷 = 32 threads × 4B = 128B → 看似 100% 合并

但：
  Compute Throughput = 40%，说明计算流水线也有 60% 空转
  根本原因：scalar load（32-bit），内存指令发射频率不够高
  每条内存指令对应 1 个 float，throughput 受 instruction issue rate 限制
  → 换成 float4（128-bit），每条指令处理 4x 数据，指令数减少 4 倍
```

**结论**：v1 的性能瓶颈是 **scalar memory access（32-bit load）导致指令吞吐量不足**，而不是真正的 DRAM 带宽饱和。优化方向：**float4 向量化**。

---

## 3. Round 1 优化：float4 向量化 → cuda_v2

### 优化策略

将每次 load/store 1 个 float 改为 load/store 1 个 float4（4 个 float）：

```cuda
// Before (v1): 每线程每次 1 float
for (int i = tid; i < N; i += blockDim.x)
    local_ss += xrow[i] * xrow[i];

// After (v2): 每线程每次 1 float4 = 4 floats
const float4* x4 = reinterpret_cast<const float4*>(x + row * N);
for (int i = tid; i < N/4; i += blockDim.x) {
    float4 xi = x4[i];
    local_ss += xi.x*xi.x + xi.y*xi.y + xi.z*xi.z + xi.w*xi.w;
}
```

同时，block size 从 1024 改为 256（减少寄存器压力，让更多 block 上 SM）。

### 效果对比

```
指标                  v1          v2          改善
Duration             99.49 µs    43.97 µs    -55.8%
DRAM Throughput      28.55%      64.28%      +35.7 pp
Memory Throughput    28.55%      64.28%      +35.7 pp
Compute Throughput   40.23%      25.02%      (随之下降，说明已非 compute 瓶颈)
Achieved Occupancy   93.19%      90.68%      -2.5 pp（可忽略）
Registers/Thread     16          26          +10
```

v2 速度提升 **2.26×**，DRAM 利用率从 28% → 64%。

---

## 4. Round 2：分析 cuda_v2，找出新瓶颈

### ncu 原始数据（v2）

```
Duration:                   43.97 µs
Memory Throughput:          64.28%    → 实际 DRAM 带宽: 2.57 TB/s
Compute Throughput:         25.02%
L2 Hit Rate:                60.70%
Achieved Occupancy:         90.68%
Registers Per Thread:       26
Block Size:                 256
Waves Per SM:               6.56
```

Warp State Statistics：
```
Warp Cycles Per Issued Instruction:  54.46 cycle
Est. Local Speedup:                  71.98%
→ 39.2 cycles（72%）stall 在 L1TEX global memory 操作
```

### 深度分析

**Step 1：带宽利用率离上界还差多少？**

```
实测时间 43.97 µs，理论最优 32.0 µs
效率 = 32.0 / 43.97 = 72.8%（DRAM Throughput 64.28% 与此吻合）

剩余 27.2% 的差距来自哪里？
```

**Step 2：计算实际 DRAM 流量**

```
理论最优 DRAM = 128 MB（x 读一次 + y 写一次，w 全 cache）

实际：v2 的 Pass 2 也会读 x！
  Pass 1: 读 x（64 MB）→ DRAM
  Pass 2: 读 x（64 MB）→ L2 Hit Rate 60.7%，仍有 40% miss → DRAM
         L2 miss 导致额外 DRAM 流量 ≈ 64 × (1 - 0.607) ≈ 25 MB

实际 DRAM 流量 ≈ 64 + 25 + 64 = 153 MB（比理论多 25 MB，~19.5%）
```

**Step 3：Warp Stall 分析**

```
v2 的 Warp Cycles Per Issued Instruction 反而比 v1 更高（54 vs 35）
原因：
  Block Size 从 1024 → 256，每 SM 的 warp 数量：
    v1: 2 blocks × 32 warps = 64 active warps（由 Block Limit Warps=2 限制）
    v2: 8 blocks × 8 warps = 64 active warps（Block Limit Registers=8）
  warp 数量相同，但 v2 每次处理更多数据（float4），发出更多内存请求
  → 内存 pipeline 更繁忙，排队等待时间增加

但总体 kernel 时间还是快了，因为指令数少了 4 倍
```

**Step 4：确认新瓶颈**

```
v2 的主要问题：
  对 x 的两次访问（Pass 1 和 Pass 2）
  L2 Hit Rate 60.7% 意味着 ~40% 的 Pass 2 访问仍然穿透到 DRAM
  导致额外 ~25 MB DRAM 流量（~19% 额外开销）

优化方向：
  Pass 1 读 x 时顺便存入寄存器
  Pass 2 直接从寄存器读，彻底消除 x 的第二次内存访问
  代价：寄存器用量增加（存储 x 的 float4 数组）
```

---

## 5. Round 2 优化：寄存器缓存消除重读 → cuda_v3

### 优化策略

**核心思路**：用寄存器缓存 x，变 two-pass 为 "one-DRAM-pass + one-register-pass"。

```
N = 4096, blockDim = 256, float4
→ 每线程负责 N / (4 × 256) = 4 个 float4 块
→ 寄存器开销: 4 × float4 = 16 个 float = 64B per thread
```

```cuda
// V3 核心变化
template <int ELEMS_PER_THREAD>  // = 4 (for N=4096, threads=256)
__global__ void rms_norm_v3_impl(...) {

    float4 reg_x[ELEMS_PER_THREAD];  // 寄存器缓存 x
    float local_ss = 0.0f;

    // Pass 1: 读 x 一次，存寄存器，同时计算 sum(x²)
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int i = tid + e * blockDim.x;
        reg_x[e] = x4[i];                   // ← 唯一一次从 DRAM 读 x
        local_ss += reg_x[e].x * reg_x[e].x + ...;
    }
    // ... reduce ...

    // Pass 2: 从寄存器读 x（零 DRAM 访问），w 用 __ldg
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int i = tid + e * blockDim.x;
        float4 wi = __ldg(&w4[i]);           // texture cache 路径
        float4 out;
        out.x = reg_x[e].x * rms_inv * wi.x; // ← 从寄存器取，不访问 DRAM
        ...
        y4[i] = out;
    }
}
```

另外，w 的访问改为 `__ldg()`（read-only data cache），提示编译器走 texture cache 路径，提高 L1 命中率。

### 代价分析（优化前预估）

```
寄存器增加：存 4 个 float4 = 16 floats → 寄存器从 26 增至 ~38
→ Block Limit Registers: 65536 / (38 × 256) ≈ 6 blocks/SM（从 8 降至 6）
→ Theoretical Occupancy: 6 × 8 warps / 64 = 75%（从 100% 降至 75%）
→ Achieved Occupancy 会下降

但 DRAM 流量从 ~153 MB → ~128 MB，减少 ~16%
预期 kernel 时间从 43.97 µs → ~37 µs（如果 DRAM 是主要瓶颈）
```

---

## 6. Round 3：分析 cuda_v3，验证优化效果

### ncu 原始数据（v3）

```
Duration:                   35.49 µs    ← 较 v2 快 8.48 µs (-19.3%)
Memory Throughput:          76.30%      ← 较 v2 +12 pp
DRAM Throughput:            76.30%      → 实际 DRAM 带宽: 3.05 TB/s
Compute Throughput:         23.80%
L2 Hit Rate:                51.01%      ← 略低于 v2（60.70%）（w 的 cache 效果变化）
Achieved Occupancy:         67.83%      ← 较 v2 -22.85 pp（寄存器增加导致）
Registers Per Thread:       38          ← 较 v2 +12（符合预期）
Block Size:                 256
Waves Per SM:               8.75        ← 较 v2 +2.19（Occupancy 下降 → 更多 waves）
```

Warp State Statistics：
```
Warp Cycles Per Issued Instruction:  43.05 cycle  ← 较 v2 改善（54.46 → 43.05）
Est. Local Speedup:                  59.28%
→ 25.5 cycles（59%）stall 在 L1TEX
```

### 验证优化效果

**DRAM 流量验证**：

```
实测时间 35.49 µs
理论最优 32.0 µs
效率 = 32.0 / 35.49 = 90.2%（距上界仅差 9.8%）

DRAM 带宽 = 3.05 TB/s（v2 为 2.57 TB/s，提升 18.7%）
```

**Occupancy 代价验证**：

```
预估：100% → 75%（寄存器限制）
实测：Theoretical 75%，Achieved 67.83%
→ 符合预期，Occupancy 下降但被 DRAM 流量减少抵消
```

**Warp Stall 改善**：

```
v2: 54.46 cycles，39.2 cycles (72%) stall
v3: 43.05 cycles，25.5 cycles (59%) stall
→ 减少了约 35% 的等待时间（Pass 2 不再需要等 DRAM 返回 x）
```

---

## 7. 三版本横向对比

### 性能指标对比

```
指标                        v1          v2          v3          改善方向
─────────────────────────────────────────────────────────────────────────
Duration (µs)               99.49       43.97       35.49       ↓ 越小越好
vs 理论最优 (32 µs)         3.11×       1.37×       1.11×       ↑ 越接近 1 越好
DRAM Throughput (%)         28.55       64.28       76.30       ↑ 越高越好
实际带宽 (TB/s)              1.14        2.57        3.05        ↑
Compute Throughput (%)      40.23       25.02       23.80
Achieved Occupancy (%)      93.19       90.68       67.83       -（代价）
Registers/Thread            16          26          38          +（代价）
Waves/SM                    26.26       6.56        8.75
Warp Cycles/Instruction     35.52       54.46       43.05       ↓ 越小越好
L1TEX Stall 占比 (%)        53.12       71.98       59.28       ↓ 越小越好
```

### Benchmark 结果（python -m operators.rms_norm.test）

```
实现         延迟 (ms)    带宽 (GB/s)    vs PyTorch
─────────────────────────────────────────────────────
PyTorch      0.2452       821            1.00×
Triton v1    0.0623       3232           3.94×
CUDA v1      0.1218       1653           2.01×
CUDA v2      0.0676       2979           3.63×
CUDA v3      0.0618       3257           3.97×    ← 新增，超过 Triton
CuTe v1      0.1092       1843           2.25×
CuTe v2      0.0675       2983           3.63×
```

v3 带宽 3257 GB/s，**已超过 Triton（3232 GB/s）**，达到当前最优。

---

## 8. 优化到头了吗？终止条件分析

### 当前状态（v3）

```
DRAM Throughput = 76.30%
实测时间 35.49 µs，理论最优 32.0 µs，效率 90.2%
剩余差距 = 35.49 - 32.0 = 3.49 µs（9.8%）
```

### 剩余差距来自哪里？

```
1. Occupancy 下降（75% 理论）：
   Waves = 8.75，尾效应（0.75 波没有完全跑满 SM）
   估计损耗 ≈ 0.75 / 8.75 × 35.49 ≈ 3 µs

2. L1TEX Stall（59%）：
   w 的访问（通过 __ldg 走 texture cache），仍有部分 L2/DRAM miss
   L2 Hit Rate = 51%（w 只有 16 KB，理应全 L2 命中，可能有 cache 争用）

3. 写 y 的 store 延迟：
   store 到 DRAM 无法完全 overlap
```

### 还能继续优化吗？

| 方向 | 预期收益 | 实现难度 | 推荐度 |
|------|---------|---------|--------|
| `__launch_bounds__(256, 6)` 限制寄存器 | Occupancy 从 67% → 75%，约 +3-5% | 低 | ⭐⭐⭐ |
| 减小 Block Size（128）提高 wave 数量 | 改善尾效应，约 +2-3% | 低 | ⭐⭐ |
| Kernel fusion（fused_add_rmsnorm） | 减少一次 DRAM round-trip，约 +30-40% | 中 | ⭐⭐⭐⭐⭐ |
| 算法改进（online normalization） | 单 pass，彻底消除 smem barrier | 高 | ⭐⭐⭐ |
| Persistent kernel / CTA swizzling | 改善 cache 局部性 | 高 | ⭐ |

**结论**：在当前 standalone RMSNorm 的场景下，v3 的 76.3% DRAM 利用率已接近实际上限（two-pass 算法的理论上限约为 80-85%，受 smem reduce 和指令 overhead 影响）。**更高收益的方向是 kernel fusion**，与前后算子合并。

---

## 9. 关键经验总结

### 1. 每次优化都要用 ncu 验证效果，而不是凭感觉

```
直觉：减少一次 x 读取，应该快 ~30%
实测：快了 19.3%（因为 Occupancy 下降抵消了部分收益）
→ 两个因素同时变化时，ncu 帮你分别量化每一项
```

### 2. 指标之间的取舍关系

```
寄存器 ↑ → Occupancy ↓ → 能 hide 的 latency 减少
        ↑ → 可缓存更多数据 → DRAM 访问减少 → 实际延迟减少
→ 两者对性能的影响方向相反，实测才知道哪个效果更大
```

### 3. Warp Cycles Per Issued Instruction 的解读

```
v1: 35.52   低 occupancy 的体现，但也反映 scalar access 问题
v2: 54.46   更高（float4 访问更多数据，排队等待增加），但总时间更短
v3: 43.05   介于中间（Pass 2 从寄存器读，消除了部分 stall）

→ 这个指标不能单独看，要结合 Duration 和 DRAM Throughput
```

### 4. 理论流量估算是分析的基础

```
在 profile 之前就计算：
  理论 DRAM 流量 = 128 MB
  理论最优时间 = 32.0 µs

然后看实测与理论的差距：
  v1: 3.11× 差距（极大）
  v2: 1.37× 差距（中等，L2 cache 部分命中带来额外 DRAM 流量）
  v3: 1.11× 差距（接近上限）

这个比值给了你"还值不值得优化"的直觉
```

### 5. NCU 的 OPT 建议要结合上下文判断

```
v3 的 OPT 提示："Theoretical Occupancy (75.0%) is limited by required registers"
→ 建议提高 Occupancy
→ 但 kernel 已经在 76.3% DRAM 利用率，进一步提高 Occupancy 收益有限
→ 更好的方向是 kernel fusion

NCU 的建议是针对单一指标的，需要结合整体分析判断优先级
```

---

## 附：profile 命令速查

```bash
# 快速基础分析（SpeedOfLight + LaunchStats + Occupancy）
bash scripts/profile.sh --op rms_norm --kernel cuda_v2

# 深度分析（加内存 + warp stall）
bash scripts/profile.sh --op rms_norm --kernel cuda_v2 \
    --section SpeedOfLight,MemoryWorkloadAnalysis,WarpStateStats,LaunchStats,Occupancy

# 全量分析（最慢，最详细）
bash scripts/profile.sh --op rms_norm --kernel cuda_v2 --set full

# 对比多个版本（分别跑）
bash scripts/profile.sh --op rms_norm --kernel cuda_v1
bash scripts/profile.sh --op rms_norm --kernel cuda_v2
bash scripts/profile.sh --op rms_norm --kernel cuda_v3

# 重新查看已有报告
ncu --import results/ncu/rms_norm_cuda_v3_20260319_000400.ncu-rep
```
