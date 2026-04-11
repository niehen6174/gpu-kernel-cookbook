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
1. Vectorized load (float4 / bf16x2 / half2) — 2-4x bandwidth
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

### 4.4 关键优化模式 (代码级)

#### 向量化内存访问 (Vectorized Load/Store)

BF16/FP16 使用 `x2` 类型一次读写 2 个元素，FP32 使用 `float4` 一次读写 4 个。
实测 RMSNorm 上 BF16 向量化带来 **~3x** 加速。

```cuda
// BF16: 2 元素/次 (32-bit transaction)
const __nv_bfloat162* vec_in = reinterpret_cast<const __nv_bfloat162*>(row_input);
#pragma unroll 4
for (int i = tid; i < hidden_size / 2; i += stride) {
    __nv_bfloat162 v = vec_in[i];
    float v0 = __bfloat162float(v.x);
    float v1 = __bfloat162float(v.y);
    // ... process v0, v1 in FP32 ...
}

// FP16: 同理
const __half2* vec_in = reinterpret_cast<const __half2*>(row_input);
__half2 v = vec_in[i];  // 2 元素/次

// FP32: 4 元素/次 (128-bit transaction, 一个 warp 的一条事务)
const float4* vec_in = reinterpret_cast<const float4*>(row_input);
float4 v = vec_in[i];  // v.x, v.y, v.z, v.w
```

**注意**: 数组长度必须是向量宽度的倍数 (或处理尾部元素)，指针必须对齐。

#### Bank Conflict 消除

Shared memory 有 32 个 bank (每 bank 4 bytes)。Stride-32 访问 = 32 路冲突:

```cuda
// ❌ 32-stride: 全部命中同一 bank
__shared__ float data[1024];
float val = data[threadIdx.x * 32];

// ✅ 连续访问: 无 bank conflict
float val = data[threadIdx.x];

// ✅ +1 padding: 消除列访问的 bank conflict (Tiled Matmul 关键技巧)
__shared__ float Bs[TILE][TILE + 1];  // 33 instead of 32
float val = Bs[k][threadIdx.x];       // 不同 bank
```

#### Warp Shuffle 规约 (替代 Shared Memory)

Warp 内通信最快方式 — 无内存访问、无 bank conflict:

```cuda
// Sum reduction
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// Max reduction
template <typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

// Broadcast (lane 0 → all)
float broadcast = __shfl_sync(0xffffffff, val, 0);
```

#### 混合精度累加模式

输入用低精度 (BF16/FP16) 减少带宽，累加用 FP32 保证精度:

```cuda
float sum = 0.0f;  // FP32 累加
for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = to_float(input[i]);  // BF16→FP32
    sum += val * val;
}
sum = block_reduce_sum(sum);  // FP32 规约
output[i] = from_float(result, (scalar_t*)nullptr);  // FP32→BF16
```

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

---

## 10. Kernel 代码模板

> 精简可复用的模板。新 kernel 从对应模板出发，替换 `// Your computation here` 即可。
> 所有模板支持 FP32 / FP16 / BF16 三种精度。

### 10.1 类型转换 Helpers (每个 .cu 文件都要包含)

```cuda
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float from_float(float x, float*) { return x; }
__device__ __forceinline__ __half from_float(float x, __half*) { return __float2half(x); }
__device__ __forceinline__ __nv_bfloat16 from_float(float x, __nv_bfloat16*) { return __float2bfloat16(x); }
```

> **注意**: PyTorch 禁用了隐式 FP16/BF16 转换，必须用显式 `to_float()` / `from_float()`。

### 10.2 逐元素模板 (Element-wise)

适用: vector_add, relu, gelu, rope, 等。

```cuda
constexpr int BLOCK_SIZE = 256;

template <typename scalar_t>
__global__ void elementwise_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int total_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float val = to_float(input[idx]);
        float result = val;  // Your computation here
        output[idx] = from_float(result, (scalar_t*)nullptr);
    }
}

// Launch: grid = (total_elements + 255) / 256, block = 256
```

### 10.3 行规约模板 (Row-wise Reduction)

适用: rmsnorm, layernorm, softmax, 等。每 block 处理一行。

```cuda
constexpr int WARP_SIZE = 32;

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

template <typename T>
__device__ __forceinline__ T block_reduce_sum(T val) {
    __shared__ T shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : T(0);
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

template <typename scalar_t>
__global__ void reduction_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int hidden_size, const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const scalar_t* row_in = input + row * hidden_size;

    // Pass 1: 规约 (以 RMSNorm 为例: sum of squares)
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float v = to_float(row_in[i]);
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float s_factor;
    if (tid == 0) s_factor = rsqrtf(sum_sq / hidden_size + eps);
    __syncthreads();

    // Pass 2: 逐元素应用
    scalar_t* row_out = output + row * hidden_size;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float norm = to_float(row_in[i]) * s_factor;
        row_out[i] = from_float(norm * to_float(weight[i]), (scalar_t*)nullptr);
    }
}

// Launch: grid = num_rows, block = min(hidden_size, 1024) 对齐到 WARP_SIZE
```

**向量化版本** (BF16, ~3x 加速): 将 `scalar_t*` 重新解释为 `__nv_bfloat162*`，循环步长 /2，
每次处理 2 元素。参考 §4.4 向量化模式。

### 10.4 Tiled Matmul 模板

适用: GEMM, batched matmul。C = A×B, A[M,K], B[K,N]。
`+1` padding 消除 B 矩阵列访问的 bank conflict。

```cuda
constexpr int TILE = 32;

__global__ void matmul_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE + 1];  // +1 padding 防 bank conflict

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// Launch: grid = ((N+31)/32, (M+31)/32), block = (32, 32)
```

### 10.5 全局归约模板 (Two-Pass Reduction)

适用: 大数组求和/最大值。Phase 1 分块归约 → Phase 2 最终归约。

```cuda
__device__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void reduce_sum(const float* input, float* output, int n) {
    __shared__ float warp_results[32];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + tid;

    // 每线程加载 2 个元素 (消除首步空闲线程)
    float sum = 0.0f;
    if (idx < n)               sum += input[idx];
    if (idx + blockDim.x < n)  sum += input[idx + blockDim.x];

    sum = warp_reduce(sum);
    if (tid % 32 == 0) warp_results[tid / 32] = sum;
    __syncthreads();

    sum = (tid < blockDim.x / 32) ? warp_results[tid] : 0.0f;
    if (tid / 32 == 0) sum = warp_reduce(sum);
    if (tid == 0) output[blockIdx.x] = sum;
}

// Host: pass 1 (N → grid_size), pass 2 (grid_size → 1)
```

### 10.6 Stream Overlap Pipeline 模板

适用: 大数据量 + 简单 kernel。4 路 stream 流水线重叠 H2D/Compute/D2H。

```cuda
void stream_pipeline(const float* h_in, float* h_out, int total) {
    constexpr int NSTREAMS = 4;
    int chunk = (total + NSTREAMS - 1) / NSTREAMS;

    cudaStream_t streams[NSTREAMS];
    for (int i = 0; i < NSTREAMS; ++i) cudaStreamCreate(&streams[i]);

    float *d_in, *d_out;
    cudaMalloc(&d_in,  total * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));

    for (int i = 0; i < NSTREAMS; ++i) {
        int off = i * chunk, cnt = min(chunk, total - off);
        size_t bytes = cnt * sizeof(float);
        cudaMemcpyAsync(d_in + off, h_in + off, bytes, cudaMemcpyH2D, streams[i]);
        my_kernel<<<(cnt+255)/256, 256, 0, streams[i]>>>(d_in+off, d_out+off, cnt);
        cudaMemcpyAsync(h_out + off, d_out + off, bytes, cudaMemcpyD2H, streams[i]);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < NSTREAMS; ++i) cudaStreamDestroy(streams[i]);
    cudaFree(d_in); cudaFree(d_out);
}
// 前提: h_in/h_out 必须是 pinned memory (cudaMallocHost)
// 验证: nsys profile 查看 timeline，确认 copy 与 compute 并行
```

---

## 11. SM90 (Hopper) 优化速查

> 参数以 **H20** (60 SMs, ~4.0 TB/s HBM) 为准。H100 差异: 132 SMs, 3.35 TB/s。

### 11.1 硬件参数对照

| 参数 | H20 | H100 | 优化影响 |
|------|-----|------|---------|
| SMs | 60 | 132 | Grid 大小对齐 SM 数的倍数 |
| Threads/SM | 2048 | 2048 | 最多 16 blocks × 128 threads |
| Shared Memory | 228 KB/SM | 192 KB/SM | H20 可用更大 tile |
| L2 Cache | 60 MB | 50 MB | H20 更大，跨 block 复用更多 |
| HBM BW | ~4.0 TB/s | 3.35 TB/s | Coalesced 访问至关重要 |
| Warp Size | 32 | 32 | 所有 reduction 用 shuffle |
| Registers/SM | 65536 | 65536 | >255/thread 则 spill |

### 11.2 Shared Memory 配置

H20 每 SM 支持可配置 smem/L1 拆分:

```cuda
// 请求最大 dynamic shared memory
cudaFuncSetAttribute(my_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    228 * 1024);  // H20: 228KB max
```

### 11.3 Occupancy 快速计算

```
Occupancy = Active Warps / Max Warps(64)

限制因素 (取最小):
  1. 寄存器: 65536 / (threads_per_block × regs_per_thread)
  2. Shared memory: 228KB / smem_per_block
  3. 线程数: 2048 / threads_per_block
```

查看实际寄存器用量:
```bash
nvcc --ptxas-options=-v -arch=sm_90 my_kernel.cu
# 输出: "Used X registers, Y bytes smem"
```

### 11.4 Block Size 选择指南

| Kernel 类型 | Threads/Block | Warps | 理由 |
|-------------|---------------|-------|------|
| 逐元素 | 256 | 8 | 高 occupancy, 简单 |
| 行规约 | 512-1024 | 16-32 | 需要足够线程完成规约 |
| Attention | 256 | 8 | 平衡 smem 和寄存器 |
| GEMM (Tiled) | 256 (16×16) or 1024 (32×32) | 8/32 | 取决于 tile 大小 |

### 11.5 NCU / Nsys Profiling 命令

```bash
# Nsys: 系统级 timeline (kernel 耗时、内存传输、GPU 空闲)
nsys profile -o report python test.py

# NCU: kernel 级分析 (occupancy, throughput, stall 原因)
ncu --set full -o metrics.ncu-rep python test.py

# NCU 特定指标
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed python test.py
```

### 11.6 NVCC 编译 Flags

```bash
nvcc -arch=sm_90 -O3 kernel.cu           # 基本
nvcc --ptxas-options=-v kernel.cu         # 查看寄存器/smem 用量
nvcc -maxrregcount=128 kernel.cu          # 限制寄存器上限
nvcc --use_fast_math kernel.cu            # 快速但低精度数学函数
nvcc -lineinfo kernel.cu                  # 添加调试行号信息
```

---

## 12. SM89 (Ada Lovelace) 优化速查

当需要在 L40S / RTX 4090 / RTX 4070 等 Ada 架构上运行时参考本节。

### 12.1 SM89 vs SM90 关键差异

| 特性 | SM89 (Ada) | SM90 (H20/Hopper) |
|------|-----------|-------------------|
| Shared Memory/SM | **100 KB** | 228 KB (H20) / 192 KB (H100) |
| L1+Shared 合计 | 128 KB | 256 KB |
| Threads/SM | 1536 | 2048 |
| Warps/SM | 48 | 64 |
| Max Blocks/SM | **24** | 16 |
| Registers/SM | 65536 | 65536 |
| TMA | ❌ 无 | ✅ |
| Thread Block Clusters | ❌ 无 | ✅ |
| Distributed Shared Memory | ❌ 无 | ✅ |
| 显存类型 | GDDR6/GDDR6X | HBM3/HBM3e |
| Async Copy | **cp.async** | TMA |

### 12.2 典型设备参数

| 设备 | SMs | 显存带宽 | L2 Cache | 显存 |
|------|-----|---------|----------|------|
| L40S | 142 | 864 GB/s | **96 MB** | 48 GB GDDR6 |
| RTX 4090 | 128 | 1.01 TB/s | 72 MB | 24 GB GDDR6X |
| RTX 4070 | 46 | 504 GB/s | 36 MB | 12 GB GDDR6X |
| RTX 6000 Ada | 142 | 960 GB/s | **96 MB** | 48 GB GDDR6 |

### 12.3 Tile Sizing — 100 KB 限制

SM89 最大 smem = 100 KB，直接影响 tile 大小选择:

```cuda
// SM90 可用: 128×64 FP16 tile = 16 KB, 双缓冲 = 32 KB, 192 KB 绰绰有余
// SM89 必须缩小: 100 KB 总预算

// Attention on SM89:
// Q: 64×64 × 2B = 8 KB
// K: 64×64 × 2B = 8 KB
// V: 64×64 × 2B = 8 KB  → 共 24 KB, 双缓冲 48 KB < 100 KB ✓

// GEMM on SM89 (双缓冲):
// A: 64×32 × 2B = 4 KB × 2 = 8 KB
// B: 32×64 × 2B = 4 KB × 2 = 8 KB → 共 16 KB < 100 KB ✓

cudaFuncSetAttribute(kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    100 * 1024);  // SM89 max: 100 KB
```

### 12.4 cp.async 双缓冲 (替代 TMA)

SM89 无 TMA，使用 cp.async 实现异步 global→shared 拷贝:

```cuda
#include <cuda_pipeline.h>

__shared__ float buf[2][TILE_K][TILE_N];
int stage = 0;

// 预取第一个 tile
__pipeline_memcpy_async(&buf[0][ty][tx], &A[...], sizeof(float));
__pipeline_commit();

for (int k = 0; k < K; k += TILE_K) {
    // 预取下一个 tile 到另一个 buffer
    if (k + TILE_K < K) {
        __pipeline_memcpy_async(&buf[1-stage][ty][tx], &A[...], sizeof(float));
        __pipeline_commit();
    }
    __pipeline_wait_prior(1);  // 等待当前 tile 就绪
    __syncthreads();

    compute(buf[stage]);       // 计算当前 tile
    stage = 1 - stage;
}
```

### 12.5 L2 Cache 利用策略

Ada 的 L2 (72-96 MB) 远大于 Hopper (50 MB)，是弥补 smem 不足的关键:

```cuda
// 核心策略: 小 smem tile + 依赖 L2 做 inter-tile 重用
// 而非 Hopper 的 大 smem tile + 中等 L2

// L2 persistence hints (Ada 支持)
asm volatile("ld.global.ca.b32 %0, [%1];" : "=r"(val) : "l"(ptr));  // 保留在 L2
asm volatile("ld.global.cs.b32 %0, [%1];" : "=r"(val) : "l"(ptr));  // 流式,不污染 L2
```

### 12.6 Occupancy 与 Block Size

```
SM89 Occupancy = Active Warps / 48

 64 threads (2 warps) × 24 blocks = 48 warps → 100% ✓
128 threads (4 warps) × 12 blocks = 48 warps → 100% ✓
256 threads (8 warps) ×  6 blocks = 48 warps → 100% ✓  ← 推荐
512 threads (16 warps) × 3 blocks = 48 warps → 100% ✓
1024 threads (32 warps) × 1 block = 32 warps → 67%  ⚠️ 避免!
```

**规则**: 优先 256 threads/block; **禁止** 1024 threads/block (occupancy 上限 67%)。

### 12.7 SM89 优化优先级

1. **Fusion 最重要** — GDDR 带宽低 (864 GB/s vs H20 ~4.0 TB/s), 每次 kernel launch 的内存代价更高
2. **向量化访问** — GDDR 对 coalescing 更敏感,必须用 `__nv_bfloat162`/`float4`
3. **L2 重用** — 小 tile 高频迭代,数据留在 L2
4. **cp.async 双缓冲** — 异步搬运 + 计算重叠
5. **FP32 双发射** — Ada 有 128 FP32 cores/SM (dual datapath)，FP32 计算几乎免费

### 12.8 SM89 编译 Flags

```bash
nvcc -arch=sm_89 -O3 kernel.cu             # Ada 专用

# 多架构构建 (Ada + Hopper)
nvcc -gencode arch=compute_89,code=sm_89 \
     -gencode arch=compute_90,code=sm_90 \
     -O3 kernel.cu

# Grid sizing: 动态获取 SM 数
cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
// L40S=142, RTX 4090=128, RTX 4070=46
```

### 12.9 SM89 Profiling 要点

```bash
ncu --set full -o metrics.ncu-rep python test.py
```

重点关注指标 (与 SM90 不同):
- **L2 hit rate** — Ada 优化的核心指标,低于 80% 说明 tile 策略需调整
- **Memory throughput** — 占 GDDR peak 百分比 (通常 30-40%)
- **Achieved occupancy** — 上限 48 warps, 检查是否被 1024-thread block 限制
- **FP32 dual-issue utilization** — Ada 特有,衡量双发射利用率
