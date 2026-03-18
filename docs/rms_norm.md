# RMSNorm

## 1. 数学定义

给定输入 $x \in \mathbb{R}^N$，可学习参数 $w$（weight，无 bias）：

$$\text{rms}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \varepsilon}$$

$$y_i = \frac{x_i}{\text{rms}(x)} \cdot w_i$$

其中 $\varepsilon$ 通常为 `1e-6`，防止除零。

---

## 2. 与 LayerNorm 的区别

```
LayerNorm:
    μ  = mean(x)
    σ² = mean((x - μ)²)
    y  = (x - μ) / sqrt(σ² + ε) * γ + β
    → 去均值（中心化）+ 缩放 + 偏移，需要 γ 和 β 两组参数

RMSNorm:
    rms = sqrt(mean(x²) + ε)
    y   = x / rms * w
    → 不中心化，只缩放，仅需 w 一组参数
    → 计算更简单（省去均值计算），LLM 推理更快
```

**RMSNorm 的优势**：
- 去掉了均值计算（少一次 reduction pass）
- 去掉了 bias，减少参数量
- 实验表明，对 Transformer 模型效果与 LayerNorm 相当

**LLM 使用情况**：LLaMA、Mistral、Gemma、Qwen 等主流开源大模型均使用 RMSNorm。

---

## 3. Fused Add + RMSNorm

在 Transformer 解码器中，残差连接和归一化通常连续出现：

```python
# 标准写法（两次内存读写）
residual = x + residual
y = rms_norm(residual)

# Fused kernel（一次内存读写）
y, residual = fused_add_rms_norm(x, residual, weight)
# 等价于:
#   residual = x + residual  (inplace)
#   y = rms_norm(residual)
```

**Fused 的好处**：把两次全局内存访问合并为一次，节省约 33% 带宽（读 x + 读旧 residual + 写新 residual + 写 y → 4次，合并后只需读 x 和旧 residual，写新 residual 和 y，但实际上读旧 residual 和写新 residual 共享同一地址，真正节省了一次大读）。

---

## 4. GPU 并行策略

```
输入: (B, N) — B 行，每行长度 N

每个 CUDA block 处理一行：
  1 block = 1 row = blockIdx.x

Pass 1: 计算 sum(x²)
  每个线程处理 ceil(N / blockDim.x) 个元素
  两级归约：
    Level 1 (Warp Reduce):   32 threads → 1 float（__shfl_xor_sync）
    Level 2 (Block Reduce):  warp 结果写 smem → 第一个 warp 归约 → smem[0]

  rms_inv = rsqrtf(smem[0] / N + eps)

Pass 2: 写出 y = x * rms_inv * w
  每个线程写 ceil(N / blockDim.x) 个元素
```

---

## 5. Warp Shuffle Reduce

```cuda
__device__ __forceinline__
float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}
```

`__shfl_xor_sync(mask, val, delta)` 是 warp 内线程间直接交换寄存器值的指令，**不需要经过 shared memory**，延迟极低（1-2 个时钟周期）。

蝶形规约过程（32 个 lanes）：

```
第1轮 (mask=16): lane 0 ↔ lane 16, lane 1 ↔ lane 17, ...
第2轮 (mask= 8): lane 0 ↔ lane  8, lane 1 ↔ lane  9, ...
第3轮 (mask= 4): lane 0 ↔ lane  4, ...
第4轮 (mask= 2): lane 0 ↔ lane  2, ...
第5轮 (mask= 1): lane 0 ↔ lane  1, ...

结果：每个 lane 都持有所有 32 个值的和
5 轮 = log2(32) 轮
```

---

## 6. 两级归约架构（Block Reduce）

```
行长度 N = 4096，block = 256 threads = 8 warps

Level 1 (Warp Reduce)：
  每个 warp 的 32 threads 各负责 4096/256 = 16 个元素
  每个 warp 内做 5 轮 shfl → 得到本 warp 的局部 sum(x²)
  lane 0 将结果写入 smem[warp_id]  → smem[0..7]

__syncthreads()

Level 2 (Block Reduce by warp 0)：
  第一个 warp 的 lane 0..7 从 smem 读取 8 个值
  再做一轮 warp reduce → lane 0 得到全局 sum(x²)
  写入 smem[0]

__syncthreads()

所有 thread 读取 smem[0]，计算 rms_inv = rsqrtf(smem[0] / N + eps)
```

---

## 7. float4 向量化（V2）

```cuda
// 把行数据视作 float4 数组
const float4* x4 = reinterpret_cast<const float4*>(x + row * N);
int N4 = N / 4;

// 每次加载 4 个连续元素（128-bit 事务）
for (int i = tid; i < N4; i += blockDim.x) {
    float4 xi = x4[i];
    local_ss += xi.x*xi.x + xi.y*xi.y + xi.z*xi.z + xi.w*xi.w;
}
```

**为什么 float4 更快？**

| 属性 | float | float4 |
|------|-------|--------|
| 每次 load 字节 | 4 B | 16 B |
| 内存事务数 | N | N/4 |
| 地址计算次数 | N | N/4 |
| 峰值带宽利用率 | ~50% | ~90%+ |

GPU 全局内存最小读取粒度为 32 B（cache line 128 B）。float4 的 16 B 对齐访问能更好地利用内存事务，减少地址计算开销，让 SM 的 load/store 单元吞吐最大化。

---

## 8. `rsqrtf`

```cuda
float rms_inv = rsqrtf(smem[0] / N + eps);
// 计算 1 / sqrt(x)，GPU 专用硬件指令
```

对比：
- `1.0f / sqrtf(x)`：需要 sqrt 指令（~10 cycles）+ div 指令（~20 cycles）
- `rsqrtf(x)`：单条硬件指令（~4 cycles），精度约 23 bit

后续用乘法代替除法：`y = x * rms_inv * w`（乘法比除法快 4-6×）。

---

## 9. Triton 实现解析

```python
@triton.jit
def rms_norm_kernel(X_ptr, W_ptr, Y_ptr, N, stride, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)                       # 每个 program 处理一行
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

    ss = tl.sum(x * x, axis=0)                  # 向量点积 → 标量
    rms_inv = tl.rsqrt(ss / N + eps)            # 1 / rms

    w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    tl.store(Y_ptr + row * stride + cols, x * rms_inv * w, mask=mask)
```

**关键点**：
- `BLOCK_SIZE = triton.next_power_of_2(N)` — 必须是 2 的幂（Triton 约束）
- `tl.sum(x * x, axis=0)` — Triton 自动用最优 reduce 指令
- 整行数据放在寄存器里，**一次 load 完成 pass1 + pass2**，实际只需 1 次读 + 1 次写（比 CUDA 两趟少 1 次读）

---

## 10. CuTe 实现解析

```cpp
// V1: 用 CuTe Tensor 包装行数据
auto xrow = make_tensor(make_gmem_ptr(x + row * N), make_layout(N));
auto yrow = make_tensor(make_gmem_ptr(y + row * N), make_layout(N));
auto wvec = make_tensor(make_gmem_ptr(w),            make_layout(N));

// 用 CuTe 索引访问元素
for (int i = tid; i < N; i += blockDim.x) {
    float xi = xrow(i);     // 等价于 x[row*N + i]
    local_ss += xi * xi;
}
// Pass 2
for (int i = tid; i < N; i += blockDim.x) {
    yrow(i) = xrow(i) * rms_inv * wvec(i);
}
```

```cpp
// V2: 用 CuTe 包装 float4 视图
auto xrow4 = make_tensor(make_gmem_ptr(reinterpret_cast<const float4*>(x + row * N)),
                         make_layout(N4));
// xrow4(i) 返回 float4，等价于 reinterpret_cast<float4*>[row*N4 + i]
```

CuTe 在此处主要提供**类型安全的内存视图抽象**，实际访问模式与手写 CUDA 相同，没有额外开销。

---

## 11. 编译与运行

```bash
cd gpu-kernel-lab

# 编译 CUDA kernel
CUDA_ARCH=sm_90 bash operators/rms_norm/cuda/build.sh

# 编译 CuTe kernel
CUDA_ARCH=sm_90 bash operators/rms_norm/cutlass/build.sh

# 正确性测试 + Benchmark
python -m operators.rms_norm.test
```

---

## 12. Benchmark 结果（H20，float32）

H20 理论峰值带宽：~4.0 TB/s

### B=4096，N=4096

| 实现 | 延迟 | 带宽 | vs PyTorch |
|------|------|------|------------|
| PyTorch | 0.245 ms | 822 GB/s | 1.00× |
| Triton | 0.064 ms | 3149 GB/s | 3.83× |
| CUDA v1 (two-pass) | 0.121 ms | 1665 GB/s | 2.03× |
| CUDA v2 (float4) | 0.063 ms | 3186 GB/s | 3.88× |
| CuTe v1 | 0.104 ms | 1941 GB/s | 2.36× |
| CuTe v2 (float4) | 0.063 ms | 3198 GB/s | **3.89×** |

### B=4096，N=8192

| 实现 | 延迟 | 带宽 | vs PyTorch |
|------|------|------|------------|
| PyTorch | 0.483 ms | 835 GB/s | 1.00× |
| Triton | 0.115 ms | 3488 GB/s | 4.18× |
| CUDA v1 (two-pass) | 0.198 ms | 2037 GB/s | 2.44× |
| CUDA v2 (float4) | 0.110 ms | 3652 GB/s | 4.38× |
| CuTe v1 | 0.188 ms | 2144 GB/s | 2.57× |
| CuTe v2 (float4) | 0.110 ms | 3650 GB/s | **4.37×** |

---

## 13. 性能分析

### 内存访问量

```
输入: (B, N) float32
读:  x (B×N×4 B)  +  w (N×4 B，通常在 L2 cache 中）
写:  y (B×N×4 B)
总计: ≈ 3 × B × N × 4 字节（x 两次读 + y 一次写，w 忽略）
```

**Triton/CuTe v2 为何大幅领先 PyTorch？**

PyTorch 的 `torch.Tensor.pow(2).mean()` 没有针对 RMSNorm 专门融合，会产生多次内核调用和中间张量：
1. `x.float()` — 可能触发类型转换 kernel
2. `.pow(2)` — 写出 x² 中间张量
3. `.mean(-1)` — reduction kernel
4. `.add(eps).sqrt()` — element-wise kernel
5. `x / rms * weight` — element-wise kernel

而 fused kernel 把这些全部合并成一次 pass，**内存读写从 5+ 次降到 2 次（读 x、写 y）**。

**v1 vs v2（CUDA/CuTe）**：
- v1 (two-pass)：读 x 两次（pass1 算 sum、pass2 算 y）
- v2 (float4)：同样两趟，但每次 load/store 4 个元素，减少内存事务数量，带宽利用率从 ~50% 提升到 ~90%

### 理论带宽利用率

```
N=4096, B=4096:
  数据量 = 3 × 4096 × 4096 × 4 B = 201 MB
  CUDA v2 延迟 0.063 ms
  带宽利用率 = 3186 GB/s / 4000 GB/s = 79.7%
```

接近理论峰值，说明实现质量已经很好。

---

## 14. 关键学习点

1. **RMSNorm 比 LayerNorm 简单**：不需要计算均值，少一次 reduction
2. **Warp Shuffle Reduce**：`__shfl_xor_sync` 无需 shared memory，比写 smem 再读快
3. **两级归约**：warp→smem→warp，处理任意 block 大小
4. **float4 向量化**：128-bit 对齐加载，减少内存事务，提升带宽利用率 ~2×
5. **rsqrtf**：GPU 专用倒数平方根指令，比 `1/sqrt` 快约 4×
6. **Fused Add + RMSNorm**：LLM 推理常见 pattern，合并残差加和归一化，节省一次全局内存读
7. **Triton 自动融合**：整行一次 load 完成 pass1+pass2，减少一次全局内存读
