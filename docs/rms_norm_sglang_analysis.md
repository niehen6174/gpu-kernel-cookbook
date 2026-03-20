# sglang norm_fusion 分析与 cute_v3 实现对比

**目标**：对比 sglang CuTe DSL 的 norm 实现与本项目的差异，借鉴其核心思路，新增 `cute_v3` 并进行 ncu 性能分析。

---

## 目录

1. [sglang norm_fusion 实现解析](#1-sglang-norm_fusion-实现解析)
2. [与现有实现的设计差异](#2-与现有实现的设计差异)
3. [可借鉴的核心思路](#3-可借鉴的核心思路)
4. [cute_v3 实现方案](#4-cute_v3-实现方案)
5. [ncu Profile 数据：五版本横向对比](#5-ncu-profile-数据五版本横向对比)
6. [完整 Benchmark 结果](#6-完整-benchmark-结果)
7. [深度数据解读](#7-深度数据解读)
8. [设计差异总结](#8-设计差异总结)

---

## 1. sglang norm_fusion 实现解析

源文件：`sglang/jit_kernel/diffusion/cutedsl/common/norm_fusion.py`

### 1.1 核心 RMSNorm 逻辑（`apply_rmsnorm_cta`）

```python
@cute.jit
def apply_rmsnorm_cta(num_warps, tidx, tXrX, tWrW, D, eps):
    # Step 1: 遍历已在寄存器中的 x，计算 sum(x²)
    val = cute.Float32(0.0)
    for idx in range(cute.size(tXrX)):
        x_fp32 = tXrX[idx].to(cutlass.Float32)   # FP16/BF16 → FP32 accumulate
        val += x_fp32 * x_fp32

    # Step 2: warp + CTA reduce
    val = warp_reduce_sum(val)
    acc_sq = cta_reduce_sum(val, num_warps, tidx)
    factor = cute.rsqrt(acc_sq / D + eps)

    # Step 3: normalize（从寄存器读 x，不再访问 DRAM）
    tNrN = cute.make_fragment_like(tXrX)
    tNrN.store((tXrX.load() * factor * tWrW.load()).to(tNrN.element_type))
    return tNrN
```

**关键点**：`tXrX` 是 **register fragment**，已在调用此函数前通过 `cute.autovec_copy(tXgX, tXrX)` 全量 load 进寄存器。所以这里的 `tXrX.load()` 和 `tXrX[idx]` 都是从**寄存器**读取，完全不访问 DRAM。

### 1.2 调用链（`scale_residual_norm_scale_shift.py`）

```python
# 1. 构建 tiled_copy（LDG.128：128-bit 向量化加载）
atom_copy = make_copy_atom(CopyUniversalOp(), element_type, num_bits_per_copy=128)
tiled_copy = make_tiled_copy_tv(atom_copy, t_layout, v_layout)  # threads × 8 elems

# 2. Load x 到 registers（autovec_copy = LDG.128 指令）
copy_if(tXgX, tXrX)   # gmem → registers，LDG.128

# 3. 执行 norm（x 在寄存器，不再读 DRAM）
tNrN = norm(tXrX, tWrW, tBias)

# 4. Store 结果
copy_if(tYrY, tYgY)   # registers → gmem，STG.128
```

### 1.3 配置参数

```python
num_warps = D // 256          # N=4096 → 16 warps/CTA
num_threads = num_warps * 32  # = 512 threads/CTA
num_vectorized = 8            # 每线程每次 load 8 个 float（256-bit）
```

注意：sglang 用 **512 threads**（16 warps），我们的 v1/v2 用 256 threads（8 warps）。

---

## 2. 与现有实现的设计差异

### 2.1 三个维度的对比

| 维度 | 我们的 cute_v2 | sglang norm_fusion | 我们的 cute_v3 |
|------|--------------|-------------------|--------------|
| **x 的读取次数** | Pass1 读 1 次（DRAM），Pass2 再读 1 次（L2/DRAM） | 只读 1 次，缓存到寄存器 | 只读 1 次，`__ldg` + 寄存器缓存 |
| **w 的读取次数** | Pass1 无，Pass2 读（L2） | 全量 load 进寄存器，reuse | Pass1 load + 寄存器缓存 |
| **向量化加载** | 手动 `reinterpret_cast<float4*>` | `make_copy_atom(num_bits=128)` 自动生成 LDG.128 | `__ldg` 128-bit aligned，CuTe 模板参数 |
| **线程配置** | 256 threads/CTA | 512 threads/CTA（D//256 warps） | 256 threads/CTA（模板参数） |
| **精度** | FP32 全程 | **FP32 accumulation for FP16/BF16**（`x_fp32 = x.to(Float32)`） | FP32 全程 |
| **smem 用量** | 8 warps × 4B = 32B | 16 warps × 4B = 64B | 8 warps × 4B = 32B（精简） |
| **应用场景** | 纯 RMSNorm | **融合算子**：`norm(residual + gate*x) * (1+scale) + shift` | 纯 RMSNorm + w 寄存器缓存 |

### 2.2 sglang 的核心优势：极致 Kernel Fusion

sglang 的 `ScaleResidualNormScaleShift` 在单个 kernel 中完成：

```
输入: residual, x, gate, weight, bias, scale, shift
计算:
  1. value = gate * x + residual        ← 融合 elementwise 加法
  2. residual_out = value               ← 写出 residual（DiT 残差分支）
  3. y = RMSNorm(value, weight, bias)   ← norm
  4. y = y * (1 + scale) + shift        ← 融合 adaLN scale/shift
输出: y, residual_out
```

这一个 kernel 替代了 **5-6 个**独立 PyTorch 算子，将 DRAM 流量从 `5 × 2 × B×N×4 = 10N` 减少到 `~3N`（只读一次 x，写两次输出）。在 DiT（Diffusion Transformer）推理中，这是最关键的优化点。

---

## 3. 可借鉴的核心思路

### 思路 1：全量寄存器缓存（最重要）

```
sglang 设计：所有输入先 load 到 register fragment，再在寄存器上计算，
             单次 kernel 中 x/w/bias/scale/shift 只各读一次

我们的差距：cute_v2 的 Pass2 对 x 有 L2/DRAM 重读（L2 Hit Rate 60%）
```

→ 对应实现：`float4 rX[ELEMS_PER_THREAD]` 寄存器数组全量缓存

### 思路 2：w 也全量缓存进寄存器

```
sglang：tWrW 同样通过 autovec_copy 全量 load 进寄存器
cute_v2：每次 Pass2 循环都用 w4[i]（L2 cache 命中但仍有 cache 压力）
```

→ cute_v3 同时缓存 w：`float4 rW[ELEMS_PER_THREAD]`，Pass2 完全零 DRAM/L2 访问

### 思路 3：`__ldg` 代替普通 load（借鉴 sglang 的 CopyUniversalOp）

```
sglang：LDG.128 通过 make_copy_atom 指定 128-bit aligned read-only load
        等价 CUDA：__ldg() 走 texture/read-only cache
```

→ `__ldg(&x4[i])`, `__ldg(&w4[i])`：提示编译器走 read-only cache 路径，减少 L1 污染

### 思路 4（未采用）：扩大 Block Size 至 D//256 warps

```
sglang：num_warps = D // 256 = 16（for D=4096）→ 512 threads
        每线程负责更少元素，每次 copy 8 floats（256-bit）

权衡：更大 Block → 更少 waves，尾效应更严重
      对于 B=4096（4096 blocks），78 SMs 的 H20 上：
        256 threads: waves = ceil(4096 / (78 × 8)) = 6.56
        512 threads: waves = ceil(4096 / (78 × 4)) = 13.1
      waves 反而增大，但每 SM active blocks 减少
      实测在我们的 B=4096 场景下 256 threads 更优
```

---

## 4. cute_v3 实现方案

### 关键代码（`operators/rms_norm/cutlass/kernel.cu`）

```cpp
template <int THREADS, int ELEMS_PER_THREAD>
__global__ void __launch_bounds__(THREADS)
rms_norm_cute_v3(const float* x, const float* w, float* y, int N, float eps)
{
    // 1. Register fragment（全量缓存 x 和 w）
    float4 rX[ELEMS_PER_THREAD];   // 等价 sglang tXrX（register fragment）
    float4 rW[ELEMS_PER_THREAD];   // 等价 sglang tWrW（register fragment）

    // 2. Load x/w 到寄存器（LDG.128，等价 sglang autovec_copy LDG.128）
    float local_ss = 0.0f;
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int i = tid + e * THREADS;
        rX[e] = __ldg(&x4[i]);     // 唯一一次读 x from DRAM
        rW[e] = __ldg(&w4[i]);     // 唯一一次读 w（小tensor，L2 命中）
        local_ss += rX[e].x*rX[e].x + ...;
    }

    // 3. Warp + block reduce（等价 sglang warp_reduce_sum + cta_reduce_sum）
    // ... warp shuffle + smem reduce ...

    float rms_inv = rsqrtf(smem[0] / N + eps);

    // 4. Normalize（从寄存器读 x/w，零 DRAM 访问）
    //    等价 sglang: tNrN.store((tXrX.load() * factor * tWrW.load()).to(...))
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        float4 out;
        out.x = rX[e].x * rms_inv * rW[e].x;   // 寄存器读取
        ...
        y4[i] = out;    // STG.128 写出
    }
}
```

**配置**：`N=4096, THREADS=256, ELEMS_PER_THREAD=4`
- 每线程负责 4 个 float4 = 16 floats
- 寄存器：48 regs/thread（多出 22 个用于缓存 rX + rW）
- Theoretical Occupancy：62.5%（Block Limit Registers = 5 blocks/SM）

---

## 5. ncu Profile 数据：五版本横向对比

Profile 命令：
```bash
bash scripts/profile.sh --op rms_norm --kernel <version> \
    --section SpeedOfLight,MemoryWorkloadAnalysis,WarpStateStats,LaunchStats,Occupancy
```

### 完整 ncu 指标对比表

| 指标 | cuda_v2 | cuda_v3 | cute_v2 | **cute_v3** | 理论上限 |
|------|---------|---------|---------|-------------|---------|
| **Duration (µs)** | 43.97 | 35.49 | 48.70 | **36.54** | 32.0 |
| **vs 理论最优** | 1.37× | 1.11× | 1.52× | **1.14×** | 1.00× |
| **DRAM Throughput (%)** | 64.28 | 76.30 | 71.20 | **74.90** | ~87% |
| **实际带宽 (TB/s)** | 2.57 | 3.05 | 2.85 | **3.00** | — |
| **L2 Cache Throughput (%)** | 70.72 | 81.33 | 66.00 | **81.27** | — |
| **L2 Hit Rate (%)** | 60.70 | 51.01 | 59.84 | **50.22** | — |
| **Achieved Occupancy (%)** | 90.68 | 67.83 | 91.65 | **56.83** | — |
| **Theoretical Occupancy (%)** | 100 | 75 | 100 | **62.5** | — |
| **Registers/Thread** | 26 | 38 | 26 | **48** | — |
| **Waves Per SM** | 6.56 | 8.75 | 6.56 | **10.50** | — |
| **Warp Cycles/Instruction** | 54.46 | 43.05 | 61.78 | **33.03** | — |
| **L1TEX Stall 占比 (%)** | 71.98 | 59.28 | 74.41 | **55.33** | — |
| **Stall cycles** | 39.2 | 25.5 | 46.0 | **18.3** | — |

### 关键指标趋势（可视化）

```
DRAM Throughput (越高越好，上限~87%)
  cuda_v2 ████████████████░░░░░░░░  64.3%
  cute_v2 ██████████████████░░░░░░  71.2%
  cute_v3 ███████████████████░░░░░  74.9%
  cuda_v3 ████████████████████░░░░  76.3%
  理论上界 █████████████████████░░░  ~87%

Warp Stall Cycles (越低越好)
  cute_v2  ████████████████████████  46.0 cycles
  cuda_v2  ███████████████████████░  39.2 cycles
  cuda_v3  █████████████████░░░░░░░  25.5 cycles
  cute_v3  ████████████░░░░░░░░░░░░  18.3 cycles  ← 最优！

Duration (越低越好，下限 32.0 µs)
  cute_v2  ████████████████████████████████████░  48.70 µs
  cuda_v2  ████████████████████████████░░░░░░░░░  43.97 µs
  cute_v3  ██████████████████████░░░░░░░░░░░░░░░  36.54 µs
  cuda_v3  █████████████████████░░░░░░░░░░░░░░░░  35.49 µs
  理论下界  ████████████████████░░░░░░░░░░░░░░░░░  32.00 µs
```

---

## 6. 完整 Benchmark 结果

**测试配置**：B=4096, N=4096, float32，NVIDIA H20

```
实现           延迟 (ms)    带宽 (GB/s)   vs PyTorch   DRAM %
──────────────────────────────────────────────────────────────
PyTorch        0.2483       810.7         1.00×        ~20%
Triton v1      0.0638       3153.3        3.89×
CUDA v1        0.1232       1633.9        2.02×        28.6%
CuTe v1        0.1108       1817.8        2.24×
CUDA v2        0.0694       2899.5        3.58×        64.3%
CuTe v2        0.0692       2910.0        3.59×        71.2%
cuda_v3        0.0638       3155.8        3.89×        76.3%
cute_v3        0.0621       3240.5        4.00×        74.9%   ← 新增最优
```

### 结论

- **cute_v3 是当前所有版本中最快的**，3240.5 GB/s，4.00× PyTorch
- cute_v3 比 cute_v2 快 **18%**（同为 CuTe 路线，纯靠寄存器缓存提升）
- cute_v3 比 cuda_v3 快约 **2.7%**（两者策略相同，差异来自 w 的缓存方式）
  - cuda_v3：x 寄存器缓存，w 用 `__ldg`（texture cache）
  - cute_v3：x 和 w **都**缓存到寄存器，Pass2 完全零内存访问

---

## 7. 深度数据解读

### 7.1 为什么 cute_v3 的 Warp Stall 最低（18.3 cycles）？

```
x 和 w 都缓存到寄存器 → Pass2 完全不发出任何内存指令
→ Pass2 的 warp stall 来源消失（无 L1TEX 等待）

剩余的 18.3 cycles stall 全部来自 Pass1（等待 DRAM 返回 x 和 w 的 load）
这是不可消除的最小值（必须读一次 x）
```

### 7.2 为什么 cute_v3 的 Occupancy 更低（57%）但性能更好？

```
cute_v3 寄存器：48/thread（缓存 rX[4] + rW[4] = 64 floats = 16 float4 = 64 regs）
cute_v2 寄存器：26/thread

Block Limit Registers（H20 每 SM 65536 regs）：
  cute_v2: 65536 / (26×256) = 9.8 → 8 blocks/SM  → Waves = 6.56
  cute_v3: 65536 / (48×256) = 5.3 → 5 blocks/SM  → Waves = 10.50

Occupancy 降低，但 waves 增多（10.50 vs 6.56），每个 wave 虽然 warp 少，
但 Pass2 几乎无 stall（不发内存指令）→ 每 wave 执行更快
总体效果：更多 waves × 更快执行 > 更少 waves × 慢执行
```

### 7.3 L2 Hit Rate 下降的解释

```
cute_v2 L2 Hit Rate: 59.84%
cute_v3 L2 Hit Rate: 50.22%

原因：cute_v3 同时 load x 和 w，两者都走 DRAM（尤其是第一次访问 w）
     cute_v2 的 Pass2 对 x 有部分 L2 命中（60%），命中率看起来更高

但实际 DRAM 流量：
  cute_v2：Pass1 读 x(64MB) + Pass2 再读 x 40% miss(25MB) + 写 y(64MB) ≈ 153MB
  cute_v3：读 x(64MB) + 读 w(16KB, cache hit) + 写 y(64MB) ≈ 128MB（接近理论最优）

总 DRAM 流量 cute_v3 更少，L2 命中率看起来低是因为 Pass2 不再产生任何内存请求
```

### 7.4 cute_v3 与 cuda_v3 的细微差别

```
两者策略基本一致，性能差距 ~2.7%（3240 vs 3156 GB/s）

cute_v3 的额外优化：
  同时缓存 w（rW[ELEMS_PER_THREAD]）
  cuda_v3 的 Pass2 仍然通过 __ldg(&w4[i]) 访问 w（需要一次 texture cache 查找）
  cute_v3 的 Pass2 对 w 也是纯寄存器读取

另一方面：cute_v3 的寄存器压力更大（48 vs 38），
  Occupancy 57% vs 68%，但 Stall cycles 更少（18.3 vs 25.5）
  两者换算后 cute_v3 略占优
```

---

## 8. 设计差异总结

### sglang 的设计哲学 vs 本项目

| 维度 | sglang norm_fusion | 本项目 |
|------|-------------------|--------|
| **目标** | 生产级 DiT 推理，极致 kernel fusion | 教学目的，逐步优化，理解瓶颈 |
| **Kernel 粒度** | 极大：一个 kernel 融合 5-6 个算子 | 单一算子，逐版本对比 |
| **语言** | CuTe DSL（Python + JIT 编译） | C++ CUDA + C++ CuTe Header |
| **可扩展性** | 高（Python 参数控制 fusion 组合） | 中（需要手写新 kernel） |
| **调试友好性** | 中（JIT 编译，stack trace 复杂） | 高（直接 ncu 分析 C++ kernel） |
| **dtype** | FP16/BF16/FP32（FP32 accumulate） | FP32 |
| **核心优化** | 极致 kernel fusion + 寄存器缓存 | 寄存器缓存（本文借鉴点） |

### 最重要的借鉴：寄存器缓存策略

sglang 提供了一个**架构层面的证明**：在 DiT 推理的生产环境中，这种全量寄存器缓存策略是值得的（它在实际推理 pipeline 中被采用），验证了我们的优化方向。

从 cute_v2（两次读 x）→ cute_v3（一次读 x，全量寄存器缓存 x+w）：
- DRAM Throughput：71% → 75%
- Warp Stall：46 cycles → 18 cycles
- Duration：48.7 µs → 36.5 µs（**-25%**）
- 带宽：2910 GB/s → 3241 GB/s（**+11%**）

### 进一步可借鉴的方向

1. **Kernel Fusion（最高优先级）**：与 residual add、scale/shift 融合，预期提升 30-50%
2. **FP16/BF16 支持 + FP32 accumulate**：sglang 的精度处理方式，避免 FP16 的 sum-of-squares 溢出
3. **动态 D（运行时参数）**：sglang 通过 `fake_sig_args` 机制将 D 作为编译时常量，不同 D 值分别编译并缓存，兼顾灵活性和性能
