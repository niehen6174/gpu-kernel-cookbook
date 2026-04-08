# GPU Kernel 开发与优化流程

> 本 skill 总结了 gpu-kernel-lab 项目中算子从实现到优化的完整方法论。
> 以 sageattention (INT8 QK + FP8 PV, SM90 Hopper) 的 P0-P5 六轮优化为主线，
> 结合 fp8_quant、svdquant、spargeattn 等算子的经验。

---

## 0. 项目约定

```
GPU:         NVIDIA H20 (sm_90, Hopper), CUDA 12.9
构建:        CUDA_ARCH=sm_90 bash scripts/build_all.sh
单算子测试:   python -m operators.<name>.test
全量测试:    bash scripts/run_all_tests.sh
基准测试:    python benchmarks/benchmark.py
NCU Profile: bash scripts/profile.sh --op <name> --kernel <kernel>
```

---

## 1. 标准目录结构

新建算子时遵循如下布局：

```
operators/<op_name>/
├── pytorch/
│   └── baseline.py          # PyTorch 参考实现 (FP32)
├── cuda/
│   ├── csrc/
│   │   ├── kernel.cu         # CUDA kernel (v1 naive → v2/v3 optimized)
│   │   ├── kernel.h          # 函数声明
│   │   └── pybind.cpp        # pybind11 绑定
│   ├── setup.py              # torch.utils.cpp_extension 构建
│   ├── build.sh              # 或 CMake 构建脚本
│   └── test.py               # 独立 kernel 测试 (含 correctness + benchmark)
├── triton/
│   └── kernel.py             # Triton DSL 实现
├── cute/
│   └── kernel.py             # CuTe DSL / CUTLASS 实现
├── test.py                   # 统一测试入口 (所有 backend)
├── benchmark.py              # 性能基准 (可选)
└── OPTIMIZATION_ANALYSIS.md  # 优化分析文档 (可选，复杂算子必备)
```

---

## 2. 开发流程 (Phase 0: Baseline)

### 2.1 实现顺序

1. **PyTorch baseline** — 正确性参考，用于后续所有对比
2. **Triton** — 快速验证算法，自动 autotune
3. **CUDA naive (v1)** — 手写 kernel，先保证正确
4. **CUDA optimized (v2/v3)** — 逐步优化
5. **CuTe / CUTLASS** — Hopper 架构优化 (TMA, WGMMA)

### 2.2 正确性测试

使用 `common/check.py` 的统一接口：

```python
from common.check import check_correctness

ref = pytorch_baseline(inputs)
out = my_kernel(inputs)
check_correctness(out, ref, atol=1e-3, rtol=1e-3, name="cuda_v1")
```

**误差容忍度选择指南:**

| 算子类型 | atol | rtol | 原因 |
|----------|------|------|------|
| 逐元素 (vector_add, relu) | 1e-6 | 1e-6 | 无精度损失 |
| 规约 (softmax, layernorm) | 1e-3 | 1e-3 | 浮点累加顺序 |
| 在线算法 (flash attention) | 5e-3 | 1e-3 | running max/sum 顺序 |
| Tensor Core FP32 累加 | 1e-3 | 1e-3 | TF32 中间精度 |
| 量化 (FP8 per-tensor) | 0.1 | 0.01 | 3-bit mantissa |
| 量化 (FP8 per-block) | 0.3 | 0.05 | 多 scale 误差 |
| 多级量化 (INT4→INT8) | 1.0 | 0.1 | 误差叠加 |
| 近似/稀疏 (tile skip) | 自适应 | — | 取决于 skip threshold |

**量化算子额外指标** (参考 fp8_quant):
- SNR > 25 dB
- Cosine similarity > 0.999
- RMSE < σ × 7% (per-tensor) / 5% (per-block)
- Max error < amax / 16 (FP8 E4M3 理论上界)

### 2.3 Benchmark 模板

使用 `common/utils.py` 的统一接口：

```python
from common.utils import benchmark_func, compute_tflops, compute_bandwidth

# 性能测量
result = benchmark_func(my_kernel, *args, warmup=10, repeat=100)
ms = result["median_ms"]

# 计算指标
flops = 2 * M * K * N  # GEMM 例子
tflops = compute_tflops(flops, ms)
bw = compute_bandwidth(bytes_accessed, ms)

# 或用 CUDA events（sageattention 风格）
def benchmark_fn(fn, warmup=5, repeat=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]  # median
```

**标准 Benchmark 表格格式:**

```
  Impl              | Mean(ms) | TFLOPS | vs_baseline | Notes
  fp16_baseline     |   1.234  | 123.5  | 1.00x       |
  cuda_v1           |   1.100  | 138.8  | 1.12x       |
  cuda_v2           |   0.950  | 160.9  | 1.30x       |
  triton            |   1.050  | 145.6  | 1.18x       |
```

---

## 3. Profile 分析 (Phase 1+: 性能诊断)

### 3.1 NCU Profile

```bash
# 快速 SOL 概览
bash scripts/profile.sh --op <name> --kernel <kernel>

# 完整分析 (慢 10-30x)
bash scripts/profile.sh --op <name> --kernel <kernel> --set full

# Roofline 分析
bash scripts/profile.sh --op <name> --kernel <kernel> --section roofline

# 自定义 section
bash scripts/profile.sh --op <name> --kernel <kernel> --section SpeedOfLight,MemoryWorkloadAnalysis
```

如果算子不在 `profiling/profile_driver.py` 的 REGISTRY 中，需要先注册。
对于 Python extension 类型的算子 (如 sageattention)，也可以直接用 ncu：

```bash
sudo ncu --set full --target-processes all \
    -o results/ncu/sageattention_$(date +%Y%m%d) \
    python operators/sageattention/cuda/test.py
```

### 3.2 关键 NCU 指标及解读

| 指标 | 健康值 | 含义 |
|------|--------|------|
| SM Throughput (SOL%) | >60% | Tensor Core / CUDA Core 利用率 |
| Memory Throughput (SOL%) | >60% | 带宽利用率 |
| Warp Stall - Barrier | <30% | 同步等待占比 |
| Warp Stall - MIO Throttle | <20% | 内存指令阻塞 |
| Warp Stall - Math Pipe Throttle | <10% | 计算指令阻塞 (好事！说明 compute-bound) |
| Register/Thread | <255 | 超过则 spill 到 local memory |
| Shared Memory/Block | <100KB | SM90 最大 228KB |
| Occupancy | >50% | 但低 occupancy 不一定差 (latency hiding) |

### 3.3 Roofline 分析模板

```python
# 1. 计算 Arithmetic Intensity (AI)
flops = 2 * B * H * N * N * D  # attention 例子
bytes = (Q + K + V + O) * element_size
AI = flops / bytes  # FLOP/byte

# 2. 对比硬件 ridge point
# H20: FP8 峰值 ~148 TFLOPS, BW ~4.0 TB/s
ridge_point = 148e12 / 4.0e12  # ≈ 37 FLOP/byte

if AI > ridge_point:
    print("Compute-bound → 优化 Tensor Core 利用率")
    # 关注: WGMMA pipeline, register pressure, occupancy
else:
    print("Memory-bound → 优化带宽利用率")
    # 关注: coalescing, TMA, shared memory, vectorized load
```

---

## 4. 迭代优化流程 (Phase 1-N)

### 4.1 方法论: 每轮优化都遵循

```
1. Profile → 找到当前瓶颈 (NCU 指标)
2. 假设 → 提出优化方案 (附预期收益)
3. 实现 → 修改 kernel
4. 验证:
   a. 正确性 — 对比 baseline (threshold=0 时 bit-exact)
   b. 性能 — 对比上一版本
   c. NCU 指标 — 确认瓶颈是否改善
5. 记录 → 写入 OPTIMIZATION_ANALYSIS.md
```

### 4.2 常见优化方向 (按收益排序)

**Memory-bound 算子:**
1. Vectorized load (float4) — 4x bandwidth
2. Shared memory tiling + padding — 消除 bank conflict
3. TMA (SM90) — 硬件异步加载
4. Coalesced access pattern — 合并全局内存访问

**Compute-bound 算子:**
1. Tensor Core (WMMA/WGMMA) — 硬件矩阵乘
2. 流水线 (pipeline K/V loads) — 隐藏延迟
3. Warp Specialization — 生产者/消费者分离
4. Register pressure 管理 — 避免 spill
5. 算法优化 (tile skip, 稀疏) — 减少无效计算

**通用:**
1. 减少同步 (__syncthreads) 次数
2. 编译期常量 (template) 消除分支
3. Loop unrolling (#pragma unroll)
4. 融合相邻 kernel

### 4.3 SageAttention 优化历程 (参考案例)

| Phase | 优化 | 结果 | 教训 |
|-------|------|------|------|
| P0 | Baseline 性能锚定 | 124 TFLOPS | 需要公平对比 (kernel-vs-kernel) |
| P1 | 循环展开/指令调度 | 失败 (已被编译器优化) | 先 profile 确认瓶颈 |
| P2 | Smem 布局优化 | 失败 (寄存器溢出) | Hopper WGMMA 寄存器压力大 |
| P3 | Warp Specialization | 失败 (寄存器 >255) | SM90 named barrier 复杂性 |
| P4 | Deep Pipeline | 失败 (-2% 慢了) | Per-iteration overhead 抵消 |
| P5 | Tile Skip | 成功实现但目标场景无效 | **算法优化需要匹配数据特征** |

**P5 的核心教训**: 论文中的优化方案 (SpargeAttn 的 tile skip) 在 LLM 上有效但在 DiT 视频生成上无效，因为 DiT 的注意力分布接近均匀、不稀疏。**必须用真实数据验证假设。**

---

## 5. 真实数据验证 (可选 — 量化/稀疏算子)

> **适用场景**: 算子行为或收益高度依赖输入数据分布时才需要。
> 典型例子: 量化算子 (FP8/INT4)、稀疏注意力 (tile skip)、pruning、MoE routing。
> **不需要**: 逐元素算子 (ReLU, GELU)、标准 GEMM、LayerNorm、Softmax 等 —
> 这类算子用随机数据即可充分验证正确性和性能。

### 5.1 什么时候需要真实数据

- **量化算子**: 随机数据通常近似均匀/正态分布，但真实激活值可能有 outlier、长尾、
  通道间量级差异等特征，影响量化精度 (per-tensor vs per-channel 的选择)
- **稀疏/跳过算子**: 优化收益取决于注意力分布的稀疏程度 —
  LLM 的 causal attention 天然稀疏，但 DiT 视频生成接近均匀分布 (skip 无效)
- **数据相关的算法选择**: 如动态量化的 scale 计算策略、block size 选择等

### 5.2 数据采集方法

在模型推理代码中 dump 中间张量：

```python
# 在模型 attention 层中插入:
torch.save(q, f"run000/step{step:02d}_block{block_id}.q.pt")
torch.save(k, f"run000/step{step:02d}_block{block_id}.k.pt")
torch.save(v, f"run000/step{step:02d}_block{block_id}.v.pt")
```

### 5.3 真实数据 Benchmark 脚本模板

参考 `operators/sageattention/cuda/test_real_data.py` 的结构：

```python
# 1. 发现数据文件
blocks = discover_blocks(data_dir)  # glob + sort

# 2. 加载 + 布局转换
q, k, v = load_block(q_path, k_path, v_path)
# 注意 layout: NHD [B,N,H,D] → HND [B,H,N,D] 视 kernel 需要

# 3. 量化 (如果 kernel 需要)
q_int8, q_scale, ... = quantize_for_kernel(q, k, v)

# 4. 数据特征分析 (关键！)
analyze_attention_sparsity(q, k, sm_scale)
# - logit 分布统计
# - tile 间 gap 分析
# - skip rate 预测

# 5. Benchmark 多组参数
for threshold in [0.0, 5.0, 10.0, 16.0]:
    ms = benchmark_fn(lambda: kernel(..., threshold=threshold))
    # 精度对比 + 速度对比

# 6. 释放 GPU 内存
del q, k, v; torch.cuda.empty_cache()
```

### 5.4 精度 vs 速度权衡报告

```
  Threshold  Time(ms)  TFLOPS  Speedup  MaxDiff  MeanDiff
  ─────────────────────────────────────────────────────────
       0.0   522.370  126.84  1.000x   0.000000  0.000000
       5.0   517.270  128.10  1.010x   0.109375  0.000234
      10.0   518.530  127.78  1.007x   0.031250  0.000012
      16.0   518.700  127.74  1.007x   0.015625  0.000001
```

---

## 6. 文档规范

### 6.1 OPTIMIZATION_ANALYSIS.md 结构

复杂算子 (需要多轮优化的) 应维护优化分析文档：

```markdown
# <算子名> 优化分析

## 1. 架构概览
- 数据流图
- 主循环结构
- 关键参数 (tile size, thread count, etc.)

## 2. 优化空间分析
- 高/中/低收益分类
- 每个方向的预期收益、实现难度、时间估计

## 3. NCU Profile Baseline
- SOL%, stall 分析, 寄存器/smem 使用
- Roofline 定位 (compute/memory bound)

## N. Phase X: <优化名>
### N.1 方案设计
### N.2 实现细节
### N.3 测试结果 (正确性 + 性能)
### N.4 NCU 对比 (优化前 vs 后)
### N.5 结论 + 失败原因 (如果失败)

## 最终结论
- 汇总表格
- 经验教训
```

### 6.2 失败也要记录

失败的优化尝试同样重要 — 记录**为什么失败**，避免重复踩坑:

```markdown
### P3: Warp Specialization — ❌ 失败

**原因**: 寄存器压力。SM90 WGMMA 要求大量寄存器 (RS_f32[4][4][8] = 128 regs),
加上 warp specialization 的额外状态, 总寄存器 >255, 无法编译。

**教训**: Hopper 架构上 warp specialization 需要配合 setmaxnreg 动态调整寄存器分配,
且需要仔细规划哪些数据 spill 到 smem。
```

---

## 7. 构建与调试

### 7.1 Python Extension 构建 (推荐)

```python
# setup.py
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    ext_modules=[CUDAExtension(
        name='_my_kernel',
        sources=['csrc/kernel.cu', 'csrc/pybind.cpp'],
        extra_compile_args={
            'nvcc': ['-arch=sm_90a', '-O2', '--use_fast_math',
                     '-U__CUDA_NO_HALF_OPERATORS__']
        }
    )],
    cmdclass={'build_ext': BuildExtension}
)
```

**清理重建** (修改 kernel 后必须):
```bash
rm -rf build/ *.so && python setup.py build_ext --inplace
```

### 7.2 常见编译问题

| 错误 | 原因 | 解决 |
|------|------|------|
| `ptxas : error : Entry function uses too much register` | 寄存器 >255 | 减少 local 变量, 用 `__launch_bounds__`, 或 smem 替代 |
| `undefined reference to __nv_...` | CUDA math 函数缺失 | 添加 `-lm` 或检查 arch flag |
| `no kernel image is available` | arch 不匹配 | 确认 `-arch=sm_90a` 与实际 GPU 一致 |
| `CUDA error: misaligned address` | 内存对齐问题 | 检查 tensor contiguity, 使用 `.contiguous()` |

### 7.3 调试 Tip

```python
# 用小尺寸 + print 调试
if threadIdx.x == 0 and blockIdx.x == 0:
    printf("m[0][0] = %f\n", m[0][0]);

# 检查 NaN/Inf
assert not torch.isnan(out).any(), "Output contains NaN"
assert not torch.isinf(out).any(), "Output contains Inf"

# 对比每个中间步骤
# 将 kernel 拆分为多步, 每步输出中间结果对比 PyTorch
```

---

## 8. Checklist: 算子完成标准

### 基本 (所有算子)
- [ ] PyTorch baseline 实现
- [ ] 至少一个优化实现 (CUDA / Triton / CuTe)
- [ ] 正确性测试通过 (适当 atol/rtol)
- [ ] 性能 benchmark (latency + TFLOPS/bandwidth)
- [ ] test.py 可独立运行

### 进阶 (复杂算子)
- [ ] NCU profile baseline 记录
- [ ] 多尺寸测试 (小/中/大)
- [ ] 多参数测试 (如不同 head_dim, causal/non-causal)
- [ ] 与 official/reference 实现对比
- [ ] OPTIMIZATION_ANALYSIS.md 文档

### 完整 (量化/稀疏等数据分布敏感的算子，可选)
- [ ] 量化精度指标 (SNR, RMSE, cosine similarity)
- [ ] 真实模型数据验证 (dump 中间张量，见 §5)
- [ ] 精度 vs 速度权衡分析
- [ ] 多 threshold/参数 sweep
- [ ] 失败尝试记录

---

## 9. 快速参考

### H20 硬件参数
```
SM90 (Hopper), 60 SMs
FP8 Tensor Core: 148 TFLOPS (理论峰值)
INT8 Tensor Core: 148 TOPS
BF16 Tensor Core: 74 TFLOPS
FP32 CUDA Core: 37 TFLOPS
HBM Bandwidth: ~4.0 TB/s
L2 Cache: 60 MB
Shared Memory: 228 KB/SM (max)
Registers: 65536 per SM, 255 per thread (max)
```

### 常用命令
```bash
# 构建全部
CUDA_ARCH=sm_90 bash scripts/build_all.sh

# 单算子测试
python -m operators.<name>.test

# 单算子 CUDA extension 构建
cd operators/<name>/cuda && rm -rf build/ *.so && python setup.py build_ext --inplace

# NCU profile
bash scripts/profile.sh --op <name> --kernel <kernel> --set full

# 查看 GPU 信息
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_properties(0))"
```
