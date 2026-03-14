# LayerNorm

## 1. 数学定义

给定输入 $x \in \mathbb{R}^N$，可学习参数 $\gamma$（weight）和 $\beta$（bias）：

$$\mu = \frac{1}{N} \sum_{i=1}^{N} x_i$$

$$\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2$$

$$y_i = \gamma_i \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta_i$$

其中 $\varepsilon$ 通常为 `1e-5`，防止除零。

---

## 2. 与 BatchNorm 的区别

```
BatchNorm:  在 batch 维度上归一化（每个 feature 维度独立）
             适合 CNN（有固定的 spatial 结构）

LayerNorm:  在 feature 维度上归一化（每个样本独立）
             适合 NLP/Transformer（序列长度可变）

形状：
  输入: (B, T, C)，B=batch, T=seq, C=hidden
  LayerNorm 对最后一维 C 做归一化
  每个 (b, t) 样本独立计算 μ 和 σ²
```

---

## 3. GPU 并行策略

```
输入: (B, N) — B 行，每行长度 N

每个 CUDA block 处理一行：
  Step 1: 计算 μ（block-level sum reduction）
  Step 2: 计算 σ²（block-level sum reduction）
  Step 3: 归一化 + 仿射变换

Three-pass 实现 → 两次全局内存读取
```

---

## 4. Welford 在线算法

传统两趟计算方差：
```
mean = sum(x) / N
var  = sum((x - mean)^2) / N
     = E[x^2] - E[x]^2   ← 数值不稳定！大数相减
```

**Welford 算法**（Welford, 1962）在线计算均值和方差，数值稳定：

```python
n = 0
mean = 0.0
M2 = 0.0    # Σ(x_i - mean)²

for x in data:
    n += 1
    delta = x - mean
    mean += delta / n
    delta2 = x - mean      # 注意：使用更新后的 mean
    M2 += delta * delta2

var = M2 / n
```

**数值稳定性分析：**
- 传统方法：`E[x^2] - E[x]^2`，当 x 的均值很大时，两个大数相减精度损失严重
- Welford：增量更新，中间量始终是"当前偏差"，数值量级小

### 4.1 Welford 状态合并

两个独立的 Welford 状态 $(n_a, \mu_a, M2_a)$ 和 $(n_b, \mu_b, M2_b)$ 可以合并：

$$n_c = n_a + n_b$$

$$\delta = \mu_b - \mu_a$$

$$\mu_c = \mu_a + \delta \cdot \frac{n_b}{n_c}$$

$$M2_c = M2_a + M2_b + \delta^2 \cdot \frac{n_a \cdot n_b}{n_c}$$

这使得 Welford 可以并行化——每个 thread/warp 独立计算自己的状态，最后合并。

---

## 5. Warp-level Welford Reduction

```cuda
struct WelfordState { float mean, m2, count; };

__device__ WelfordState warp_reduce_welford(WelfordState s) {
    for (int mask = 16; mask > 0; mask >>= 1) {
        float other_mean  = __shfl_xor_sync(0xffffffff, s.mean,  mask);
        float other_m2    = __shfl_xor_sync(0xffffffff, s.m2,    mask);
        float other_count = __shfl_xor_sync(0xffffffff, s.count, mask);

        // 合并两个 Welford 状态
        float total = s.count + other_count;
        float delta = other_mean - s.mean;
        s.mean  = s.mean + delta * other_count / total;
        s.m2   += other_m2 + delta * delta * s.count * other_count / total;
        s.count = total;
    }
    return s;
}
```

注意：Welford 状态的合并不能直接使用简单的 `sum` reduce，
需要使用上面的合并公式，因此需要传输 3 个值（mean, m2, count）。

---

## 6. 两级规约架构

```
输入行长度 N = 2048，block 有 8 个 warp（256 threads）

Level 1: 每个 warp 对 N/8 = 256 个元素做 Welford
         → 8 个 (mean_w, M2_w, count_w) 状态

Level 2: 通过 shared memory 把 8 个状态写入，
         第一个 warp 用 warp_reduce_welford 合并
         → (global_mean, global_var)

Final: 所有 thread 并行计算归一化：
       y_i = γ_i * (x_i - global_mean) / sqrt(global_var + ε) + β_i
```

---

## 7. `rsqrtf` vs `1.0f / sqrtf`

```cuda
float inv_std = rsqrtf(var + eps);   // GPU 硬件指令，单条
// vs
float inv_std = 1.0f / sqrtf(var + eps);  // 需要 sqrt + div，两条指令
```

`rsqrtf(x)` 计算 $1/\sqrt{x}$，GPU 有专用硬件支持，比 `1/sqrt(x)` 快约 2×。

---

## 8. Triton 实现解析

```python
@triton.jit
def layernorm_kernel(X_ptr, W_ptr, B_ptr, Y_ptr, N, stride, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # 加载整行
    x = tl.load(X_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

    # 计算均值
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N

    # 计算方差（用 (x-mean)^2）
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N

    # 归一化
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = diff * inv_std

    # 仿射变换
    if W_ptr is not None:
        w = tl.load(W_ptr + cols, mask=mask)
        x_norm = x_norm * w
    if B_ptr is not None:
        b = tl.load(B_ptr + cols, mask=mask)
        x_norm = x_norm + b

    tl.store(Y_ptr + row * stride + cols, x_norm, mask=mask)
```

---

## 9. 反向传播（BackProp）简述

LayerNorm 的前向只需 mean 和 std，反向需要：

$$\frac{\partial L}{\partial x_i} = \frac{\gamma_i}{\sigma} \left(
    \frac{\partial L}{\partial y_i} - \frac{1}{N} \sum_j \frac{\partial L}{\partial y_j}
    - \frac{x_i - \mu}{N \sigma^2} \sum_j \frac{\partial L}{\partial y_j} (x_j - \mu)
\right)$$

前向需要保存 $\mu$ 和 $\sigma$（而不是 $x$），用于反向计算。
Triton 的 fused LayerNorm 可以同时输出 mean 和 rstd（reciprocal std），
供反向 kernel 使用。

---

## 10. 性能分析

| 指标 | 值 |
|------|-----|
| 读操作 | 2 × B × N float（读 x 两次：pass1 + pass2） |
| 写操作 | B × N float |
| 额外读 | W, B（共 2N float，但通常在 L1 cache 中） |
| Arithmetic Intensity | 低，memory-bound |

**Fused kernel 的优势：**
- 将 2-pass 融合到 1-pass（Welford）
- 全局内存读取从 2× 降低到 1×
- 写入仍然是 1×
- 实际节省约 25-35% 时间

---

## 11. 关键学习点

1. **均值和方差的 GPU 计算**：必须用 reduction 操作
2. **Welford 在线算法**：数值稳定的单趟均值+方差计算
3. **Welford 状态合并**：使得并行计算成为可能
4. **rsqrtf**：利用 GPU 硬件的倒数平方根指令
5. **Fused Kernel**：把多个操作合并，减少内存带宽压力
6. **数值精度**：FP32 vs FP16，eps 的作用
