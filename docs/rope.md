# RoPE (Rotary Position Embedding)

## 1. 为什么需要位置编码

Transformer 的注意力机制本身对序列顺序无感知——将所有 token 打乱，注意力得分不变。
位置编码（Positional Encoding）的作用是把每个 token 的位置信息注入到表示中，让模型能够区分"第 3 个词"和"第 7 个词"。

**绝对位置编码（Sinusoidal / Learned PE）**：直接把位置向量加到 token embedding 上。缺点：不能很好地泛化到训练时未见过的序列长度。

**RoPE（Su et al., 2021）**：改为旋转 Q/K 向量，让注意力点积天然编码相对位置差，泛化性极好。现已成为 LLM 标配（LLaMA、Mistral、Qwen、Gemma、DeepSeek 等）。

---

## 2. 数学推导

### 2.1 核心思想

RoPE 的目标：对 Q、K 施加与位置相关的变换，使得注意力点积 $\langle q_m, k_n \rangle$ 只依赖于**相对位置** $m - n$，而不是绝对位置 $m$ 和 $n$。

数学上，对 2D 向量的旋转矩阵满足：

$$\langle R_m x, R_n y \rangle = \langle R_{m-n} x, y \rangle$$

即旋转 $m$ 角的 $x$ 与旋转 $n$ 角的 $y$ 的内积，等价于旋转 $(m-n)$ 角后的内积。

### 2.2 旋转频率（Frequency）

将 $d$ 维向量分为 $d/2$ 个二维子空间，每个子空间 $i$ 对应一个旋转频率：

$$\theta_i = \frac{1}{b^{2i/d}}, \quad i = 0, 1, \ldots, \frac{d}{2} - 1$$

其中 $b = 10000$（RoPE 原始论文），近年扩展上下文的方法（LLaMA3、Yarn 等）会修改 $b$ 或引入额外缩放因子。

对位置 $p$，第 $i$ 个子空间的旋转角度为 $p \cdot \theta_i$。

### 2.3 应用方式：非交错（GPT-NeoX / rotate_half）

将 head 的 $d$ 维向量分为前半 $x_1 = x[:d/2]$ 和后半 $x_2 = x[d/2:]$：

$$x'[:d/2] = x_1 \cdot \cos(p \cdot \theta) - x_2 \cdot \sin(p \cdot \theta)$$
$$x'[d/2:] = x_2 \cdot \cos(p \cdot \theta) + x_1 \cdot \sin(p \cdot \theta)$$

其中 $\cos(p \cdot \theta)$ 和 $\sin(p \cdot \theta)$ 是长度 $d/2$ 的向量（逐元素）。

等价矩阵形式（以 $d=4$，$i=0,1$ 为例）：

$$\begin{pmatrix} x'_0 \\ x'_1 \\ x'_2 \\ x'_3 \end{pmatrix} = \begin{pmatrix} \cos\theta_0 & 0 & -\sin\theta_0 & 0 \\ 0 & \cos\theta_1 & 0 & -\sin\theta_1 \\ \sin\theta_0 & 0 & \cos\theta_0 & 0 \\ 0 & \sin\theta_1 & 0 & \cos\theta_1 \end{pmatrix} \begin{pmatrix} x_0 \\ x_1 \\ x_2 \\ x_3 \end{pmatrix}$$

### 2.4 应用方式：交错（Llama-style / interleaved）

将相邻两个元素 $(x_{2i}, x_{2i+1})$ 作为一个二维子空间，直接旋转：

$$x'_{2i}   = x_{2i} \cdot \cos\theta_i - x_{2i+1} \cdot \sin\theta_i$$
$$x'_{2i+1} = x_{2i+1} \cdot \cos\theta_i + x_{2i} \cdot \sin\theta_i$$

两种风格数学等价（均为相同的旋转变换），只是维度排列不同。**GPT-NeoX / HuggingFace 默认非交错，原始 LLaMA 代码用交错。**

### 2.5 预计算 cos/sin 缓存

```python
half_dim = head_dim // 2
# θ_i = 1 / (10000^(2i/head_dim)), shape (half_dim,)
inv_freq = 1.0 / (10000 ** (arange(0, half_dim) / half_dim))
# positions × frequencies → (max_seq_len, half_dim)
freqs = outer(arange(max_seq_len), inv_freq)
cos_cache = cos(freqs)   # (max_seq_len, head_dim//2)
sin_cache = sin(freqs)   # (max_seq_len, head_dim//2)
```

推理时通过 `positions` 数组索引（支持 KV Cache 场景下的任意位置）。

---

## 3. 输入输出格式

```
Q, K: (seq_len, num_heads, head_dim)  contiguous float32
cos_cache: (max_seq_len, head_dim//2)  预计算
sin_cache: (max_seq_len, head_dim//2)  预计算
positions: (seq_len,)  int32，每个 token 在序列中的绝对位置

输出: Q', K' 与输入形状相同，inplace 更新
```

`positions` 支持非连续场景（prefill 批量多序列、decode 阶段单步推理等）。

---

## 4. GPU 并行策略

```
输入: (seq_len, num_heads, head_dim)
总工作量: seq_len × num_heads 个 head，每个 head 做 head_dim/2 次旋转

映射：
  blockIdx.x  = token * num_heads + head    （1 block = 1 head）
  threadIdx.x = 0 .. head_dim/2 - 1         （1 thread = 1 旋转对）

每个 thread 的工作：
  1. 读 positions[token_idx] → pos
  2. 读 cos_cache[pos, tx], sin_cache[pos, tx]
  3. 读 q[token, head, tx], q[token, head, tx + half_dim]
  4. 写 q'[...] = q1*c - q2*s, q'[...+half] = q2*c + q1*s
  5. 对 K 重复 3-4
```

**无 reduction，无 shared memory**：RoPE 是纯 element-wise 操作，每个 thread 独立，没有通信开销。

---

## 5. float2 向量化（V2）

```cuda
// 一次 64-bit load：加载两个相邻 cos/sin 值
float2 c2 = *reinterpret_cast<const float2*>(cos_cache + pos * half_dim + idx);
float2 s2 = *reinterpret_cast<const float2*>(sin_cache + pos * half_dim + idx);

// 一次 64-bit load：加载 q 的两个元素
float2 q1 = *reinterpret_cast<float2*>(q + q_base + idx);
float2 q2 = *reinterpret_cast<float2*>(q + q_base + idx + half_dim);

// 向量化旋转
q1_out.x = q1.x * c2.x - q2.x * s2.x;
q1_out.y = q1.y * c2.y - q2.y * s2.y;
q2_out.x = q2.x * c2.x + q1.x * s2.x;
q2_out.y = q2.y * c2.y + q1.y * s2.y;
```

每个线程处理 2 个旋转对，线程数减半（`threads = half_dim / 2`），内存事务数减半。

**注意**：float2 向量化对 RoPE 的性能提升有限（相比 RMSNorm 的 float4），原因在于 RoPE 的数据访问本身已经不连续（`q[..., tx]` 和 `q[..., tx + half_dim]` 之间有 `half_dim` 步长），访问模式不如 RMSNorm 规整。

---

## 6. Triton 实现解析

```python
@triton.jit
def rope_kernel(Q_ptr, K_ptr, COS_ptr, SIN_ptr, POS_ptr, ...,
                HALF_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    token_idx = pid // num_heads
    head_idx  = pid % num_heads

    pos  = tl.load(POS_ptr + token_idx)                  # 标量
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < HALF_DIM

    cos = tl.load(COS_ptr + pos * cos_stride + cols, mask=mask)
    sin = tl.load(SIN_ptr + pos * cos_stride + cols, mask=mask)

    q_base = Q_ptr + token_idx * q_stride_s + head_idx * q_stride_h
    q1 = tl.load(q_base + cols,            mask=mask)
    q2 = tl.load(q_base + cols + HALF_DIM, mask=mask)
    tl.store(q_base + cols,            q1 * cos - q2 * sin, mask=mask)
    tl.store(q_base + cols + HALF_DIM, q2 * cos + q1 * sin, mask=mask)
    # K 同理 ...
```

**关键点**：
- `BLOCK_SIZE = triton.next_power_of_2(HALF_DIM)`，整个 half-dim 一次加载入寄存器
- 每个 program 处理一个 `(token, head)` 对，并行度 = `seq_len × num_heads`
- `tl.load(POS_ptr + token_idx)` 加载标量 position，Triton 自动广播

---

## 7. CuTe 实现解析

```cpp
// V1: CuTe Tensor 包装 head 视图
int q_offset = (token_idx * num_heads + head_idx) * head_dim;
auto qhead   = make_tensor(make_gmem_ptr(q + q_offset), make_layout(head_dim));
auto cos_row = make_tensor(make_gmem_ptr(cos_cache + pos * half_dim), make_layout(half_dim));

float c = cos_row(tx);   // 等价于 cos_cache[pos * half_dim + tx]
float q1 = qhead(tx);    // 等价于 q[..offset.. + tx]
float q2 = qhead(tx + half_dim);
qhead(tx)           = q1 * c - q2 * s;
qhead(tx + half_dim) = q2 * c + q1 * s;
```

```cpp
// V2: CuTe + float2 视图
int half2 = half_dim / 2;
auto q1v = make_tensor(make_gmem_ptr(reinterpret_cast<float2*>(q + q_offset)),
                       make_layout(half2));
// q1v(tx) 返回 float2，包含 q[tx*2] 和 q[tx*2+1]（连续内存）
float2 q1 = q1v(tx);
```

CuTe 在 RoPE 场景中主要用于**提供类型安全的 2D/3D 内存视图**，计算逻辑与 CUDA 版本相同。

---

## 8. 非交错 vs 交错风格对比

| 特征 | 非交错（GPT-NeoX） | 交错（Llama-style） |
|------|------|------|
| 分组方式 | $[x_0 \ldots x_{d/2-1}]$ 和 $[x_{d/2} \ldots x_{d-1}]$ | $(x_0, x_1), (x_2, x_3), \ldots$ |
| 内存访问 | 两段连续，stride = `half_dim` | 相邻两元素，stride = 2 |
| float2 向量化 | 前半段连续，后半段连续，各自可用 float2 | 奇偶交替，不适合 float2 |
| 数学等价性 | ✓ 相同旋转，不同排列 | ✓ |
| 主流使用 | HuggingFace, vLLM, SGLang | 原始 LLaMA C++ 代码 |

---

## 9. 与 Attention 的关系

```
Multi-Head Attention 中，RoPE 的位置：

Token Embedding
    ↓
Linear Projection → Q(seq, heads, d)  K(seq, heads, d)  V(seq, heads, d)
                         ↓                  ↓
                    apply_rope(Q, pos)  apply_rope(K, pos)     （本算子）
                         ↓                  ↓
                    Q'                  K'
                         ↓
                    Q' @ K'^T / sqrt(d)  → softmax → @ V → output
```

**KV Cache 场景（推理加速）**：
- Prefill 阶段：对整个 prompt 的 Q/K 批量旋转
- Decode 阶段：每步只生成一个新 token，仅对新 token 的 Q/K 旋转
- `positions` 数组允许为每个 token 指定不同位置，支持批量不等长序列

---

## 10. 编译与运行

```bash
cd gpu-kernel-lab

# 编译 CUDA kernel
CUDA_ARCH=sm_90 bash operators/rope/cuda/build.sh

# 编译 CuTe kernel
CUDA_ARCH=sm_90 bash operators/rope/cutlass/build.sh

# 正确性测试 + Benchmark
python -m operators.rope.test
```

---

## 11. Benchmark 结果（H20，float32）

H20 理论峰值带宽：~4.0 TB/s

### seq=2048，num_heads=32，head_dim=64

| 实现 | 延迟 | 带宽 | vs PyTorch |
|------|------|------|------------|
| PyTorch | 0.228 ms | 294 GB/s | 1.00× |
| Triton | 0.105 ms | 639 GB/s | 2.17× |
| CUDA v1 (scalar) | 0.092 ms | 728 GB/s | 2.47× |
| CUDA v2 (float2) | 0.093 ms | 724 GB/s | 2.46× |
| CuTe v1 | 0.091 ms | 737 GB/s | 2.50× |
| CuTe v2 (float2) | 0.091 ms | **738 GB/s** | **2.51×** |

### seq=4096，num_heads=32，head_dim=64

| 实现 | 延迟 | 带宽 | vs PyTorch |
|------|------|------|------------|
| PyTorch | 0.431 ms | 312 GB/s | 1.00× |
| Triton | 0.220 ms | 610 GB/s | 1.96× |
| CUDA v1 (scalar) | 0.157 ms | 856 GB/s | 2.75× |
| CUDA v2 (float2) | 0.157 ms | 853 GB/s | 2.74× |
| CuTe v1 | 0.157 ms | **857 GB/s** | **2.75×** |
| CuTe v2 (float2) | 0.158 ms | 847 GB/s | 2.72× |

---

## 12. 性能分析

### 内存访问量

```
每个 token × head 的访问：
  读: q (head_dim) + k (head_dim) + cos (half_dim) + sin (half_dim)
    = 2*head_dim + 2*half_dim = 2d + d = 3d 个 float
  写: q' (head_dim) + k' (head_dim) = 2d 个 float
  合计: 5d × 4 字节

对于 seq=4096, heads=32, d=64:
  总数据量 = 4096 × 32 × 5 × 64 × 4 B = 160 MB
  CUDA v1 延迟 0.157 ms → 带宽利用率 = 856 / 4000 = 21.4%
```

### 为什么带宽利用率较低？

RoPE 的工作粒度很小：每个 block 只处理 `head_dim = 64` 个 float，即 256 B。单个 block 的数据量远小于一个 cache line（128 B），**属于 latency-bound 而非 bandwidth-bound**。

具体瓶颈：
1. **kernel launch overhead**：`seq=4096, heads=32` → 131072 个 block，每个只做 32 次浮点运算（`half_dim = 32`）。Block 启动/调度开销不可忽略
2. **非连续访问**：读 `q[..., :half_dim]` 和 `q[..., half_dim:]` 两段，各自连续但之间有跳跃
3. **cos/sin cache 访问**：`half_dim` 个 float = 128 B，恰好一个 cache line，但每个 block 独立加载，L2 cache 命中率取决于 seq_len

**PyTorch 落后的原因**：同 RMSNorm 一样，多次 kernel 调用和中间张量（`torch.cat([cos, cos])`、broadcast 操作）带来的额外开销。

### v1 vs v2（scalar vs float2）

对于 `head_dim=64`（`half_dim=32`），float2 向量化的提升边际：
- v1：32 threads，每线程 1 个 float load
- v2：16 threads，每线程 1 个 float2 load（64-bit）

由于 `half_dim=32` 很小，两个版本的内存事务数差异不大，性能基本持平。若 `head_dim` 更大（如 128+），float2 的优势会更明显。

---

## 13. 关键学习点

1. **RoPE 的数学直觉**：旋转矩阵的内积只依赖相对角度，自然编码相对位置差
2. **频率设计**：$\theta_i = 1/b^{2i/d}$ 形成从高频到低频的多尺度编码，低维度旋转快（短程依赖），高维度旋转慢（长程依赖）
3. **非交错 vs 交错**：数学等价，排列不同，float2 向量化友好度不同
4. **预计算 cos/sin 缓存**：避免每次推理重新计算三角函数（代价高），用 position index 查表
5. **并行策略**：1 block = 1 head，无 reduction，纯 element-wise，极易并行
6. **float2 向量化**：64-bit load 减少内存事务，但对小 `head_dim` 提升有限
7. **Latency-bound**：RoPE 的 block 工作量很小（64 floats），在高并发场景下属于 launch-overhead sensitive 操作，可考虑将多个 head 合并到一个 block
