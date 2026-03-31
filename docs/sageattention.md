# SageAttention — 量化 Attention (INT8 QK + FP8 PV)

## 1. 算法概述

SageAttention 是一种**量化 Attention** 算法，核心思想是：

- 将 Q、K 量化为 **INT8**，利用 INT8 Tensor Core 计算 QK^T（吞吐量是 FP16 的 2 倍）
- 将 V 量化为 **FP8 (E4M3)**，利用 FP8 Tensor Core 计算 PV（吞吐量是 FP16 的 2 倍）
- 通过 **per-warp / per-thread 量化粒度** + **K-smoothing** 保证精度

与 FlashAttention 的对比：

| 特性 | FlashAttention-2 | SageAttention v2 |
|------|-----------------|------------------|
| QK 精度 | FP16 × FP16 | INT8 × INT8 |
| PV 精度 | FP16 × FP16 | FP8 × FP8 |
| Tensor Core 吞吐量 | 1× | 2× |
| 需要量化预处理 | 否 | 是（Triton kernels） |
| 精度损失 | 无 | atol ≈ 0.03 (可接受) |

**论文**: [SageAttention: Accurate 8-Bit Attention Mechanism for Plug-and-Play Inference Acceleration](https://arxiv.org/abs/2410.02367)

---

## 2. SM90 内核架构（Hopper）

官方 SM90 内核 `qk_int_sv_f8_cuda_sm90.cu` 的核心设计：

```
CTA_Q = 64, CTA_K = 128, NUM_THREADS = 128 (1 warpgroup)
QK GEMM: wgmma_s8s8s32  (INT8 × INT8 → INT32, SS mode)
PV GEMM: wgmma_f8f8f32  (FP8 × FP8 → FP32, RS mode)
数据加载: TMA (cp.async.bulk.tensor.4d) + mbarrier phase-flip
输出: 直接 GMEM store (half2 / bfloat162)
支持: GQA, causal mask, per-warp/per-thread 量化, BF16/FP16
```

### 数据流

```
Q (FP16) ──量化──> Q_INT8 ──TMA──> sQ (smem)
                                              ├─ QK GEMM (WGMMA INT8×INT8→I32)
K (FP16) ──smooth──量化──> K_INT8 ──TMA──> sK ┘
                                              ├─ softmax (CUDA core: I32→F32→exp2→FP8)
V (FP16) ──量化──> V_FP8 ──TMA──> sV         ├─ PV GEMM (WGMMA FP8×FP8→F32)
                                              └─ Output → GMEM
```

### 主循环

```
load Q, K[0], V[0]
for iter = 1 to N-1:
    wait K       →  QK GEMM (sQ × sK → RS_i32)
    load K[next] →  INT32→FP32→softmax(exp2)→FP8 转换
    wait V       →  PV GEMM (RS_fp8 × sV → RO_temp), RO += RO_temp
    load V[next]

最后一次迭代 (含 boundary masking / causal masking)
normalize(RO / d), apply v_scale, store output
```

---

## 3. 源码来源与文件结构

源码**复制自**官方 [SageAttention v2.2.0](https://github.com/thu-ml/SageAttention) 仓库。
原始路径: `/usr/local/app/leowhzhang/worksapce/SageAttention/csrc/`

复制到本项目后，将 `csrc/qattn/` 和 `csrc/` 两级目录的头文件扁平化到同一 `csrc/` 下，并修正了 `#include` 路径（`../xxx.cuh` → `xxx.cuh`）。

```
operators/sageattention/
├── __init__.py
├── test.py                      # 全局测试（Triton / CuTe DSL / 官方对比）
│
├── triton/                      # Triton 实现 (量化 + kernel)
│   ├── kernel_v1.py             # SageAttn V1: per-block INT8 Q/K, FP16 V
│   └── kernel_v2.py             # SageAttn V2: per-warp INT8 Q, FP8 V
│
├── cute/                        # CuTe DSL 实现 (Hopper WGMMA)
│   ├── kernel.py                # v2: rs-mode, 生产版本
│   ├── kernel_v1.py             # v1: ss-mode, 参考实现
│   ├── kernel_v3.py             # v3: BLOCK_M=128, split-WG
│   ├── kernel_v4.py             # v4: FP8 V GMEM (调试中)
│   ├── quant.py                 # Triton 量化 kernels
│   └── DEVLOG.md                # CuTe DSL 开发日志
│
└── cuda/                        # ★ 官方 SM90 CUDA 内核 (本文档)
    ├── OPTIMIZATION_ANALYSIS.md  # 详细优化空间分析
    ├── build.sh                  # 一键编译
    ├── setup.py                  # torch CUDAExtension 构建
    ├── test.py                   # 正确性 + 性能测试
    └── csrc/                     # C++/CUDA 源码 (12 个文件)
        ├── qk_int_sv_f8_cuda_sm90.cu  # 主 kernel (915 行)
        ├── pybind_sm90.cpp            # pybind11 入口
        ├── attn_cuda_sm90.h           # 函数声明
        ├── attn_utils.cuh             # TMA/softmax/FP8 转换
        ├── wgmma.cuh                  # Hopper WGMMA PTX 封装
        ├── math.cuh                   # exp2/log2/rcp 等 PTX 数学
        ├── mma.cuh                    # Ampere MMA PTX (INT8/FP8/FP16)
        ├── permuted_smem.cuh          # Swizzle 共享内存封装
        ├── numeric_conversion.cuh     # FP32↔FP8↔FP16 类型转换
        ├── cp_async.cuh               # cp.async 封装
        ├── dispatch_utils.h           # 模板分发宏
        └── utils.cuh                  # CHECK_CUDA 等宏
```

---

## 4. 编译

### 环境要求

```
GPU:    Hopper 架构 (sm_90) — H100 / H200 / H20
CUDA:   >= 12.3
PyTorch: >= 2.0 (torch.utils.cpp_extension)
```

### 编译步骤

```bash
cd operators/sageattention/cuda
bash build.sh
```

编译产物：`_qattn_sm90.cpython-3xx-x86_64-linux-gnu.so`

也可以手动调用：

```bash
python setup.py build_ext --inplace
```

### 编译日志关键信息

```
ptxas info: Used 128 registers, used 1 barriers, 128 bytes smem
ptxas info: 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
```

- 128 寄存器 / 线程，0 spill — 与官方编译结果一致
- 使用 1 个 mbarrier（用于 TMA phase-flip 同步）

### 编译 flags 说明

```
-gencode arch=compute_90a,code=sm_90a   # sm_90a 才能使用 WGMMA PTX
--use_fast_math                          # 启用 exp2.approx 等近似指令
-U__CUDA_NO_HALF_OPERATORS__             # 启用 half 运算符
-lcuda                                   # 链接 CUDA Driver API (TMA)
```

---

## 5. 测试

### 运行测试

```bash
# 从项目根目录
cd operators/sageattention/cuda
python test.py
```

测试内容：

1. **正确性验证** — 对比本地编译 kernel 与官方 `sageattention` 包输出
2. **性能对比** — 含量化开销的端到端耗时 vs 官方
3. **Kernel-only 性能** — 纯 kernel 耗时（不含量化）

### 测试结果 (H20, B=2, H=40, D=128, BF16 — 视频生成场景)

**正确性**：与官方 bit-exact 一致 (`max_diff = 0.000000`)

```
N=  256: ✓ PASS  local_vs_official(max=0.000000)  local_vs_ref(max=0.043)
N=  512: ✓ PASS  local_vs_official(max=0.000000)  local_vs_ref(max=0.024)
N= 1024: ✓ PASS  local_vs_official(max=0.000000)  local_vs_ref(max=0.020)
N= 2048: ✓ PASS  local_vs_official(max=0.000000)  local_vs_ref(max=0.018)
N= 4096: ✓ PASS  local_vs_official(max=0.000000)  local_vs_ref(max=0.009)
```

**Kernel vs Kernel 公平对比**（相同量化输入，仅对比 kernel 耗时）：

| N | Local (ms) | Official (ms) | Speedup | Local TFLOPS | Official TFLOPS |
|---|-----------|-------------|---------|-------------|----------------|
| 256 | 0.029 | 0.029 | 1.002× | 94.1 | 94.0 |
| 1024 | 0.189 | 0.188 | 0.999× | 227.7 | 228.0 |
| 4096 | 2.561 | 2.561 | 1.000× | 268.3 | 268.3 |
| 8192 | 10.040 | 10.039 | 1.000× | 273.8 | 273.8 |
| 18900 | 53.057 | 53.090 | 1.001× | 275.8 | 275.6 |
| 40500 | 239.823 | 239.745 | 1.000× | **280.1** | **280.2** |

> 两者 kernel 完全一致（speedup 1.000×），验证了 local build 的正确性。

**Kernel-only 性能**（排除量化开销）：

| N | Kernel (ms) | TFLOPS |
|---|------------|--------|
| 4096 | 2.541 | 270.5 |
| 8192 | 9.957 | 276.1 |
| 18900 | 52.676 | 277.8 |
| 40500 | 239.756 | **280.2** |

**端到端性能**（含量化 + K-smoothing）：

| N | Local E2E (ms) | Official E2E (ms) | Speedup | TFLOPS |
|---|---------------|-------------------|---------|--------|
| 4096 | 3.137 | 3.140 | 1.001× | 219.1 |
| 8192 | 11.123 | 11.097 | 0.998× | 247.1 |
| 18900 | 55.982 | 56.050 | 1.001× | 261.4 |
| 40500 | 246.917 | 246.958 | 1.000× | 272.1 |

> 大 seq_len 下量化开销趋近于零（N=40500: kernel 239.8ms vs E2E 246.9ms，量化仅占 3%）。

---

## 6. Hopper 关键技术

### 6.1 TMA (Tensor Memory Accelerator)

TMA 是 Hopper 新增的硬件单元，可以直接从 GMEM 加载多维 tensor tile 到 smem：

```c
// 创建 TMA descriptor (host 端)
CUtensorMap tma_map = create_tensor_map_4D<CTA_Q, HEAD_DIM>(Q_ptr, ...);

// kernel 中使用 (仅 thread 0 发起)
if (threadIdx.x == 0) {
    expect_bytes<CTA_K * HEAD_DIM>(&barrier_K);          // 告知 mbarrier 预期字节数
    load_async_4D(sK, &tma_map_K, &barrier_K, ...);      // TMA 异步加载
}
wait(&barrier_K, phase);  // 所有线程等待加载完成
```

**优势**：相比 `cp.async`，TMA 无需每线程计算地址，硬件自动处理多维索引。

### 6.2 WGMMA (Warpgroup MMA)

WGMMA 是 Hopper 的 128 线程 warpgroup 级别 MMA 指令：

```c
// INT8 QK GEMM (SS mode: 两个操作数都来自 smem)
wgmma::wgmma_s8s8s32<CTA_K, ScaleD=0, head_dim>(RS, sQ, sK);

// FP8 PV GEMM (RS mode: A 来自寄存器, B 来自 smem)
wgmma::wgmma_f8f8f32<head_dim, ScaleD=0, CTA_K>(RO_temp, RS_f8, sV);
```

| 指令 | 操作数 A | 操作数 B | 累加器 | Tile |
|------|---------|---------|--------|------|
| `wgmma_s8s8s32` | INT8 (smem) | INT8 (smem) | INT32 | m64n128k32 |
| `wgmma_f8f8f32` | FP8 (reg) | FP8 (smem) | FP32 | m64n64k32 |

### 6.3 mbarrier Phase-Flip

用于 K/V 双缓冲的同步机制：

```c
int p = 1;
for (iter = 1; iter < N; iter++) {
    p ^= 1;                    // 翻转 phase
    wait(&barrier_K, p);        // 等待 phase p 的 TMA 完成
    // ... compute ...
    load_K(sK, iter + 1);      // 发起下一次 TMA (phase = p^1)
}
```

---

## 7. NCU Profile 分析

使用 Nsight Compute 对 kernel 进行了详细 profiling，定位真实瓶颈。

### 7.1 Profiling 方法

```bash
cd operators/sageattention/cuda

# 1. 准备 profiling 脚本 (profile_ncu.py)
#    配置: B=2, H=40, D=128, N=4096 (ncu 下较大 N 运行太慢)
#    warmup 3 次 + 1 次 profiling run

# 2. Full profile (需要 sudo 获取 GPU perf counter 权限)
sudo ncu --kernel-name regex:qk_int8 --launch-skip 3 --launch-count 1 \
    --set full --csv $(which python) profile_ncu.py

# 3. 自定义 metrics (pipe utilization, warp stall 细分)
sudo ncu --kernel-name regex:qk_int8 --launch-skip 3 --launch-count 1 \
    --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,\
sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_active,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active \
    --csv $(which python) profile_ncu.py
```

> **注意**: NCU 需要 `sudo` 或 `/proc/driver/nvidia/params` 中启用 perf counter 权限。
> SM90 (Hopper) 上 `smsp__warp_issue_stalled_*` 细分指标大部分返回 `n/a`（架构限制）。

### 7.2 Roofline 定位

**结论：纯 Compute Bound，非 Memory Bound**

| 指标 | N=4096 | N=18900 | 含义 |
|------|--------|---------|------|
| SM Throughput | 94.3% | **96.8%** | SM 几乎满载 |
| DRAM Throughput | 1.8% | **0.4%** | 显存带宽完全没用满 |
| L2 Hit Rate | 95.8% | — | L2 缓存命中率极高 |
| Compute Memory Throughput | 18.9% | 18.9% | 内存子系统远未饱和 |

```
  Roofline 示意:

  TFLOPS
    ^
    |         ╱ Compute Ceiling (296 TFLOPS FP8 peak)
    |        ╱  ← 280 TFLOPS achieved (94.6%)
    |       ╱
    |      ╱ ← 我们在这里 (AI=128 FLOP/B)
    |     ╱
    |    ╱
    |   ╱ Memory Ceiling
    |  ╱
    | ╱
    +──────────────────→ Arithmetic Intensity (FLOP/Byte)
          ↑ridge (~8)    ↑128
```

Arithmetic Intensity = 128 FLOP/B（远超 ridge point ~8），kernel 完全处于 roofline 的计算 ceiling 区域。

### 7.3 Grid & Occupancy

```
Grid:       (64, 40, 2) = 5120 CTAs       # Q-blocks × heads × batch
Block:      128 threads = 1 warpgroup (4 warps)
Registers:  167 / thread                   # → 限制 3 blocks/SM
Shared Mem: 40960 B / block (dynamic)      # → 限制 3 blocks/SM
            + 128 B (static, mbarrier)
```

| Occupancy 指标 | 值 | 说明 |
|---------------|-----|------|
| Theoretical Occupancy | 18.75% | 12 warps / 64 max = 3 blocks/SM |
| Achieved Occupancy | 18.57% | ≈ theoretical（几乎完美） |
| Block Limit (Registers) | **3** | 167 regs × 128 threads = 21376 > 65536/4 |
| Block Limit (Shared Mem) | **3** | 41088 B × 3 = 123264 < 228KB |
| Block Limit (Warps) | 16 | 4 warps/block, 64 max warps/SM |
| Block Limit (Barriers) | 32 | 1 barrier/block |

**关键发现**：Occupancy 同时受寄存器和共享内存限制为 3 blocks/SM。虽然 occupancy 只有 18.75%，但这对 compute-bound kernel 不一定是问题——**前提是有足够的 ILP 来隐藏延迟**。

### 7.4 Pipe Utilization（核心瓶颈）

```
                    N=4096    N=18900     说明
Tensor Core pipe:   23.8%     24.2%      WGMMA (INT8 QK + FP8 PV)
  └─ HMMA subset:   11.9%     12.1%      纯矩阵乘部分
Shared Mem pipe:    23.8%     24.2%      WGMMA 从 smem 读取操作数
FMA pipe:            9.6%      9.5%      softmax (exp2, rescale O)
ALU pipe:           17.7%     17.5%      地址计算, int→float, 比较
```

**⚠ 核心问题：Tensor Core 只有 ~24% 的活跃周期在工作！**

这意味着 **76% 的时间 Tensor Core 在空闲**。SM 虽然 97% busy，但大部分时间花在非 WGMMA 操作上。

### 7.5 调度器分析

| 指标 | 值 | 含义 |
|------|-----|------|
| No Eligible Warps | **72.8%** | 超过七成时间 scheduler 没有可发射的 warp |
| Eligible Warps/Scheduler | 0.37 | 远低于理想值 (>1) |
| Active Warps/Scheduler | 2.97 | 约 3 warps active，但几乎都在等待 |
| Issue Slots Busy | 27.2% | 只有 27% 的时间在实际发射指令 |
| Executed IPC (active) | 1.09 | 单发射（正常 warpgroup 行为） |
| Warp Cycles/Instruction | 10.93 | 每条指令需等待 ~11 cycles |

**根因分析**：

```
为什么 72.8% 时间 scheduler 无 eligible warp？

1 个 CTA = 1 个 warpgroup = 4 warps（锁步执行）
3 个 CTA/SM = 12 warps = 3 个 warpgroup

每个 warpgroup 的主循环:
  ┌─ wait mbarrier (K ready)    ← 阻塞! TMA 可能还没到
  │  QK WGMMA (async)           ← tensor core 工作
  │  wgmma_commit + wait         ← fence + drain
  │  INT32→FP32 dequant         ← ALU/FMA
  │  softmax (exp2, rescale)     ← FMA
  │  FP32→FP8 convert           ← ALU
  │  wait mbarrier (V ready)    ← 阻塞!
  │  PV WGMMA (async)           ← tensor core 工作
  │  wgmma_commit + wait         ← fence + drain
  │  RO += RO_temp              ← FMA
  │  issue TMA load K[next]     ← 单线程发 TMA
  │  issue TMA load V[next]
  └─ loop

关键: 3 个 warpgroup 执行相同的 pattern，同时碰到 mbarrier wait 时
     → 全部 12 warps 阻塞 → No Eligible = 72.8%
```

### 7.6 Memory Traffic

| 指标 | 实测值 | 理论最小值 | 比值 |
|------|--------|-----------|------|
| DRAM Read | 126 MB | 83.9 MB (K+V, INT8) | 1.50× |
| DRAM Write | 70.5 MB | 83.9 MB (O, BF16) | 0.84× |
| L2 Traffic | 5.58 GB | — | L2 hit 95.8% |
| L1 Traffic | 200 MB | — | — |

内存访问效率很好：L2 命中率 95.8%（40 heads 共享 K/V，L2 自然复用），DRAM 读取仅 1.5× 理论最小值。

### 7.7 时间分解估算

基于 pipe utilization 数据，估算 kernel 时间的大致分解：

```
┌─────────────────────────────────────────────────────┐
│               Kernel 时间分解 (估算)                  │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ███████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  24%  WGMMA (QK+PV)      │
│  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  10%  softmax (FMA)       │
│  ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  18%  ALU (dequant/cvt)   │
│  ██████████████████████████████░░░░░░░░░  48%  mbarrier/TMA wait   │
│                                          + WGMMA fence/drain       │
└─────────────────────────────────────────────────────┘
```

**~48% 的时间浪费在等待上**，这是单 warpgroup 串行架构的本质局限。

---

## 8. 优化计划（基于 NCU Profile）

基于 NCU profiling 的实际数据，重新评估优化方向。**核心瓶颈是 Tensor Core 利用率仅 24%，~48% 时间浪费在 mbarrier/TMA 等待和 WGMMA fence 上。**

### 8.1 P0 已完成：Quick Wins

| 优化 | 结果 |
|------|------|
| `#pragma unroll` 拼写修复 | ✅ 完成。编译器对 constexpr 循环已自动展开，**无可测量提升** |
| L2 Cache Promotion (L2_128B) | ✅ 完成。L2 hit 已 95.8%，DRAM 仅 0.4%，**无可测量提升** |

> 教训：P0 优化基于代码审查的猜测，实际上编译器和 L2 cache 已经做得很好。
> 后续优化应以 NCU 数据为导向。

### 8.2 不值得做的方向

| 方向 | 放弃原因（NCU 数据支撑）|
|------|----------------------|
| L2 优化 | DRAM 0.4%，L2 hit 95.8%，内存子系统完全不是瓶颈 |
| TMA Store 输出 | 输出写仅在循环末尾一次（非热路径），global store 在 pipe 中占比极小 |
| Softmax 优化 | FMA pipe 仅 9.5%，且 exp2.approx PTX 已是最快实现 |
| RO_temp 消除 | WGMMA 需要独立输出寄存器空间，add-back 是必须的（占比 <2%） |

### 8.3 真正有价值的优化方向

根据 **72.8% No Eligible + Tensor Core 24% 利用率** 这两个核心发现，优化必须解决 **TMA 等待 / WGMMA fence 造成的流水线气泡**。

#### [P1] Deep Software Pipeline（TMA Prefetch）— ❌ 失败

**实际结果: -1% ~ -3% 性能下降** | 原预期: +15-25%

```
优化方案: 双缓冲 K/V smem + 提前 2 步 TMA prefetch
失败原因: smem 40KB → 72KB, 占用率从 5 → 3 CTAs/SM (下降 40%)

关键发现:
  - L2 hit rate 95.8%, TMA 从 L2 加载延迟很低 (~几十cycles)
  - 当前 single-buffer parity-flip 已经有效重叠了 TMA 和计算
  - 额外的 smem 使用 (32KB for double-buffer K+V) 导致占用率大幅下降
  - 占用率下降完全抵消了 TMA 重叠收益
```

正确性验证: max_diff < 0.5 ULP BF16, 仅 0.002% 元素有差异 (FP32 指令顺序不同导致的舍入差异)。

#### [P2] CTA_Q=128（双 Q Tile）— ❌ 失败

**实际结果: -6% ~ -16% 性能下降** | 原预期: +10-20%

```
优化方案: CTA_Q=64 → 128, 每次迭代做 2x WGMMA 工作
失败原因: 寄存器 168 → 255 (硬件上限) + 264-304 bytes 栈溢出

资源分析 (CTA_Q=128, HEAD_DIM=128):
  RS[2][8][8]     = 128 regs (vs 64)   ← doubled
  RO[2][8][8]     = 128 regs (vs 64)   ← doubled
  RO_temp[2][8][8] = 128 regs (vs 64)  ← doubled
  RS_f32[2][8][8]  = 128 regs (vs 64)  ← doubled
  总增加 ~256 regs → 超过 255 硬件限制 → 栈溢出

结果:
  CTA_Q=64:  168 regs, 0 spill, 3 CTAs/SM → 126 TFLOPS
  CTA_Q=128: 255 regs, 264B spill, 1-2 CTAs/SM → 106-118 TFLOPS
```

正确性验证: PASS (max_diff=0.003418, per-fq dequant vs fused-sm_scale 顺序差异)。

### 8.3 核心约束分析

两次优化失败揭示了该 kernel 的**核心约束**:

```
┌──────────────────────────────────────────────────────┐
│              资源约束分析                               │
├──────────────────────────────────────────────────────┤
│                                                       │
│  寄存器: 168/thread × 128 threads = 21504/CTA         │
│          65536/SM ÷ 21504 = 3 CTAs/SM (寄存器限制)     │
│                                                       │
│  Smem:   40KB/CTA                                     │
│          228KB/SM ÷ 40KB = 5 CTAs/SM (smem 限制)       │
│                                                       │
│  实际占用: 3 CTAs/SM (寄存器瓶颈)                       │
│                                                       │
│  结论: 任何增加 per-CTA 资源使用的优化都会降低占用率，    │
│        而占用率下降的性能损失 > 优化带来的收益            │
└──────────────────────────────────────────────────────┘
```

### 8.4 真正有价值的优化方向 (更新)

基于 P1/P2 失败的经验，优化方向必须是**资源中性**或**资源节约**的：

#### [P3] Warp Specialization（生产者-消费者）— 仍然可行

**预期提升: 20-40%** | 难度: 高 | 时间: 2-3 周

```
当前: 1 warpgroup 做所有事              优化后: 2 warpgroups 分工
┌──────────────────────┐              ┌───── Producer WG ──────┐
│ wait TMA             │              │ TMA load K/V           │
│ QK WGMMA             │              │ mbarrier signal        │
│ softmax              │              │ (loop independently)   │
│ PV WGMMA             │              └────────────────────────┘
│ load TMA next        │              ┌───── Consumer WG ──────┐
│ ... (串行)           │              │ wait mbarrier          │
└──────────────────────┘              │ QK WGMMA               │
                                      │ softmax                │
                                      │ PV WGMMA               │
                                      │ (loop independently)   │
                                      └────────────────────────┘
```

这是 FlashAttention-3 的核心架构。Producer warpgroup 持续发 TMA，Consumer warpgroup 持续做 WGMMA+softmax，两者通过 mbarrier 同步，TMA latency 被完全隐藏。

代价：
- 256 threads/CTA，2× register pressure
- 4× smem（deep pipeline + double buffer）
- 架构大改，调试复杂

#### [P4] Persistent Kernel + Tile Scheduler

**预期提升: 5-10%** | 难度: 高 | 时间: 1-2 周

跨 Q-block 复用 K/V 在 L2 中的缓存，减少 grid launch 开销。在 B×H 较小但 N 很大时收益明显。对 B=2, H=40 场景收益有限（5120 CTAs 已足够填满 66 SMs）。

### 8.4 实施路线

```
Phase 1 (1-2 周): Deep Software Pipeline
  ├─ 双缓冲 K/V smem + 提前 2 步 TMA prefetch
  ├─ 验证正确性 (bit-exact with baseline)
  └─ 目标: Tensor Core 利用率 24% → 35-40%, TFLOPS +15-25%

Phase 2 (1-2 周): CTA_Q=128
  ├─ 模板参数扩展 + smem layout 调整
  ├─ 寄存器预算评估 (是否降到 2 blocks/SM)
  └─ 目标: 摊薄 softmax/TMA 开销, TFLOPS +10-20%

Phase 3 (2-3 周): Warp Specialization
  ├─ Producer + Consumer warpgroup 分离
  ├─ 参考 FlashAttention-3 架构
  └─ 目标: Tensor Core 利用率 → 60%+, TFLOPS +20-40%

Phase 4 (可选): Persistent Kernel
  └─ 针对超长序列 (N>40K) 的 Tile Scheduler
```

### 8.5 收益估算

```
当前基线:  280 TFLOPS (H20, B=2, H=40, D=128, N=40500, kernel-only)
           Tensor Core 利用率: 24%

Phase 1:  ~320-350 TFLOPS  (+14-25%)  TC 利用率 → ~35%
Phase 2:  ~350-390 TFLOPS  (+10-20%)  TC 利用率 → ~42%
Phase 3:  ~420-490 TFLOPS  (+20-40%)  TC 利用率 → ~60%

理论极限: 592 TOPS INT8 / 296 TFLOPS FP8 (H20 peak)
          实际极限约 ~500 TFLOPS (考虑 softmax 不可消除的开销)
```

> 详细的 P0 优化分析和代码级改动见 `operators/sageattention/cuda/OPTIMIZATION_ANALYSIS.md`

---

## 9. 与其他实现的关系

本项目中 SageAttention 有三套实现：

| 实现 | 框架 | 峰值 TFLOPS | 状态 |
|------|------|-----------|------|
| **Triton** | Triton JIT | ~150 | 生产可用 |
| **CuTe DSL** (v2) | CuTe Python DSL | ~18.7 (D=64) | 生产可用 |
| **CUDA** (本目录) | Raw CUDA + PTX | **~280** | 生产可用, 准备优化 |

CUDA 版本性能最高（因为使用了 TMA + WGMMA 原生 PTX），是后续优化的基线。

CuTe DSL 版本使用 `cp.async` 而非 TMA，这是其与 CUDA 版本的主要性能差距来源（~25%）。

---

## 10. 参考资料

- [SageAttention 论文](https://arxiv.org/abs/2410.02367)
- [SageAttention GitHub](https://github.com/thu-ml/SageAttention)
- [FlashAttention-3 论文](https://arxiv.org/abs/2407.08691) — Warp Specialization + Deep Pipeline 参考
- [NVIDIA Hopper 白皮书](https://resources.nvidia.com/en-us-tensor-core) — TMA / WGMMA 架构
- [CUTLASS 3.x](https://github.com/NVIDIA/cutlass) — PipelineAsync / TileScheduler 参考实现
