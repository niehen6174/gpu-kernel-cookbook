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

### 测试结果 (H20, B=2, H=32, D=64, BF16)

**正确性**：与官方 bit-exact 一致

```
N=  256: ✓ PASS  local_vs_official(max=0.000000)  local_vs_ref(max=0.035)
N=  512: ✓ PASS  local_vs_official(max=0.000000)  local_vs_ref(max=0.027)
N= 1024: ✓ PASS  local_vs_official(max=0.000000)  local_vs_ref(max=0.025)
N= 2048: ✓ PASS  local_vs_official(max=0.000000)  local_vs_ref(max=0.016)
N= 4096: ✓ PASS  local_vs_official(max=0.000000)  local_vs_ref(max=0.012)
```

**性能**：

| N | Kernel (ms) | TFLOPS | 含量化 (ms) | vs 官方 |
|---|------------|--------|------------|---------|
| 256 | 0.019 | 56.9 | 0.020 | 0.26× (量化开销主导) |
| 512 | 0.033 | 129.6 | 0.034 | 0.34× |
| 1024 | 0.088 | 195.8 | 0.089 | 0.48× |
| 2048 | 0.288 | 238.6 | 0.291 | 0.67× |
| 4096 | 1.077 | 255.2 | 1.078 | 0.81× |
| 8192 | 4.203 | **261.6** | 4.210 | 0.91× |

> 注：`vs 官方` 列是含量化的端到端对比。local 测试跳过了量化步骤（使用官方量化结果），
> 而官方 API 包含完整的 K-smoothing + 量化。因此 local 比 official 快。
> Kernel-only 性能两者完全一致。

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

## 7. 优化计划

基于对官方 SM90 kernel 的代码分析，识别出以下优化方向：

### 优化总览

| 优先级 | 方向 | 预期收益 | 时间 | 说明 |
|-------|------|---------|------|------|
| P0 | `#pragma unroll` 拼写修复 | +1-3% | 10 min | line 314/445 `unrol` → `unroll` |
| P0 | L2 Cache Promotion | +1-3% | 10 min | TMA descriptor 开启 L2 promotion |
| P2 | TMA Store 输出 | +3-5% | 2-3 天 | 替换逐元素 GMEM store |
| P2 | RO_temp 累加器优化 | +3-5% | 1-2 天 | 减少寄存器压力 |
| P2 | CTA_Q=128 | +5-10% | 1 周 | 更大 Q tile, 减少 CTA 数量 |
| P1 | 深度流水线 (3-5 stage) | +5-15% | 1-2 周 | 多 inflight TMA 请求 |
| P1 | Warp Specialization | +10-20% | 2-3 周 | Load/Compute WG 分离 |
| P3 | Persistent Kernel | +1-5% | 1 周 | Tile scheduler, 减少 launch 开销 |

### 实施路径

```
Phase 0 (< 1 天): Quick Wins
  ├─ 修复 #pragma unroll 拼写 → cuobjdump 验证
  └─ 开启 L2 Promotion → benchmark 验证

Phase 1 (1-2 周): 中等改进
  ├─ TMA Store 输出
  ├─ RO_temp 优化
  └─ CTA_Q=128 实验

Phase 2 (3-5 周): 架构级改进
  ├─ Warp Specialization (1 LoadWG + 1-2 MmaWG)
  └─ Deep Pipeline (3-5 stage)

Phase 3: 长期
  ├─ Persistent Kernel + Tile Scheduler
  └─ Causal mask 优化
```

### 收益估算

```
当前:     ~261 TFLOPS (H20, kernel-only)
Phase 0:  ~269-275 TFLOPS (+2-6%)
Phase 1:  ~290-310 TFLOPS (+11-20%)
Phase 2:  ~330-360 TFLOPS (+15-35%)
```

> 详细的优化分析见 `operators/sageattention/cuda/OPTIMIZATION_ANALYSIS.md`

---

## 8. 与其他实现的关系

本项目中 SageAttention 有三套实现：

| 实现 | 框架 | 峰值 TFLOPS | 状态 |
|------|------|-----------|------|
| **Triton** | Triton JIT | ~150 | 生产可用 |
| **CuTe DSL** (v2) | CuTe Python DSL | ~18.7 (D=64) | 生产可用 |
| **CUDA** (本目录) | Raw CUDA + PTX | **~261** | 生产可用, 准备优化 |

CUDA 版本性能最高（因为使用了 TMA + WGMMA 原生 PTX），是后续优化的基线。

CuTe DSL 版本使用 `cp.async` 而非 TMA，这是其与 CUDA 版本的主要性能差距来源（~25%）。

---

## 9. 参考资料

- [SageAttention 论文](https://arxiv.org/abs/2410.02367)
- [SageAttention GitHub](https://github.com/thu-ml/SageAttention)
- [FlashAttention-3 论文](https://arxiv.org/abs/2407.08691) — Warp Specialization + Deep Pipeline 参考
- [NVIDIA Hopper 白皮书](https://resources.nvidia.com/en-us-tensor-core) — TMA / WGMMA 架构
- [CUTLASS 3.x](https://github.com/NVIDIA/cutlass) — PipelineAsync / TileScheduler 参考实现
