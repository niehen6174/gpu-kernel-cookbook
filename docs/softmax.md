# Softmax

## 1. 数学定义

给定向量 $x \in \mathbb{R}^N$：

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}$$

**数值稳定版本**（防止 exp 溢出）：

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^{N} e^{x_j - \max(x)}}$$

**证明等价性：**

$$\frac{e^{x_i - M}}{\sum_j e^{x_j - M}} = \frac{e^{x_i} \cdot e^{-M}}{\sum_j e^{x_j} \cdot e^{-M}} = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

其中 $M = \max(x)$。

---

## 2. GPU 并行策略

### 朴素实现（三趟扫描）

```
输入: (B, N) 矩阵，每行独立做 softmax

对每行 x[i, :]：
  Pass 1: max_val = max(x[i, :])
  Pass 2: sum_exp = sum(exp(x[i, :] - max_val))
  Pass 3: output[i, :] = exp(x[i, :] - max_val) / sum_exp

每个 CUDA block 处理一行：
  - 每趟都是 block-level reduction
  - 需要 3 次全局内存访问
```

**问题：** 3 次扫描 = 3× 读取 x + 2× 写 exp → 内存带宽浪费

---

## 3. Reduction 操作详解

Softmax 的核心是 **reduction**（规约）：将 N 个数归约为 1 个（max 或 sum）。

### 3.1 Naive Reduction（2× 慢于最优）

```
N = 8, values = [3, 1, 4, 1, 5, 9, 2, 6]

Round 1 (stride=4):
  s[0] += s[4]  →  8
  s[1] += s[5]  →  10
  s[2] += s[6]  →  6
  s[3] += s[7]  →  7

Round 2 (stride=2):
  s[0] += s[2]  →  14
  s[1] += s[3]  →  17

Round 3 (stride=1):
  s[0] += s[1]  →  31  (= sum)
```

代码：
```cuda
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        shared[tid] += shared[tid + s];
    }
    __syncthreads();
}
```

时间复杂度：`O(log N)` 步骤，每步 `N/2` 个并行操作。

### 3.2 Warp Shuffle Reduction（更快）

```cuda
// 使用 __shfl_xor_sync 在 warp 内无需 shared memory 就能通信
__device__ float warp_reduce_sum(float val) {
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}
```

**`__shfl_xor_sync` 工作原理：**
```
mask=16:  lane i 和 lane i^16 交换数据
  lane 0  ↔ lane 16
  lane 1  ↔ lane 17
  ...

mask=8:   lane i 和 lane i^8 交换
  lane 0  ↔ lane 8
  ...

经过 5 次操作（mask: 16→8→4→2→1），lane 0 包含所有 32 个 lane 的 sum
```

优势：
- 寄存器-寄存器通信（无 shared memory 延迟）
- 约 4 cycle/次 vs shared memory ~20+ cycle/次
- 适合 warp 大小（32）内的规约

---

## 4. Online Softmax 算法

来自论文：[Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)（Milakov & Gimelshein, 2018）

### 4.1 核心思想

一趟扫描同时维护 `(max, sum)` 状态，避免多趟扫描：

```python
m = -∞    # running max
d = 0     # running sum of exp

for x_i in sequence:
    if x_i > m:
        d = d * exp(m - x_i) + 1   # 校正旧的 sum
        m = x_i
    else:
        d += exp(x_i - m)
```

**为什么可以校正？**

当遇到新的更大值 `x_new > m_old` 时：
$$d_{\text{old}} = \sum_{j \text{ 已处理}} e^{x_j - m_{\text{old}}}$$

需要更新到以 `m_new = x_new` 为基准：
$$d_{\text{new}} = d_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + e^{x_{\text{new}} - m_{\text{new}}}$$
$$= d_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + 1$$

### 4.2 Warp 间合并 Online 状态

两个独立的 online state `(m_a, d_a)` 和 `(m_b, d_b)` 可以合并：

```python
# 假设 m_a >= m_b（不失一般性）
d_merged = d_a + d_b * exp(m_b - m_a)
m_merged = m_a
```

这使得 online softmax 可以并行化：每个 thread/warp 独立维护状态，最后合并。

---

## 5. 两级规约架构（V3 Kernel）

对于大 N，需要两级结构：

```
输入行长度 N = 4096

Level 1: 每个 warp（32 threads）处理 N/num_warps 个元素
         → (warp_max, warp_sum)
         使用 __shfl_xor_sync

Level 2: 第一个 warp 汇总所有 warp 的 (max, sum)
         → (global_max, global_sum)
         通过 shared memory 通信

Final:   所有 thread 并行计算输出
         output[i] = exp(input[i] - global_max) / global_sum
```

代码结构：
```cuda
// Level 1: warp 内规约
float m = ..., d = ...;  // 每个 thread 的局部 state
m = warp_reduce_max(m);
d = warp_reduce_sum(d * expf(local_m - m));  // 校正后 sum

// Warp 代表写入 shared memory
if (lane == 0) { smem_m[warp_id] = m; smem_d[warp_id] = d; }
__syncthreads();

// Level 2: 第一个 warp 做 block-level 规约
if (warp_id == 0) {
    float gm = ..., gd = ...;  // 读取所有 warp 的状态
    // 再次 warp reduce
    gm = warp_reduce_max(gm);
    gd = warp_reduce_sum(gd * expf(local_wm - gm));
    if (lane == 0) { smem[0] = gm; smem[1] = gd; }
}
__syncthreads();
```

---

## 6. Triton 实现解析

```python
@triton.jit
def softmax_kernel(input_ptr, output_ptr, B, N, stride_b, stride_n, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # 加载整行（BLOCK_SIZE >= N，超出部分用 -inf）
    x = tl.load(input_ptr + row_idx * stride_b + cols * stride_n,
                mask=mask, other=-float("inf"))

    x_max = tl.max(x, axis=0)              # Triton 内置 reduce
    exp_x = tl.exp(x - x_max)
    exp_sum = tl.sum(tl.where(mask, exp_x, 0.0), axis=0)

    tl.store(output_ptr + row_idx * stride_b + cols * stride_n, exp_x / exp_sum, mask=mask)
```

关键点：
- `BLOCK_SIZE = triton.next_power_of_2(N)`：必须 >= N，保证整行在一个 program 内处理
- `tl.max(x, axis=0)`：Triton 编译器自动生成最优的 reduction 指令（warp shuffle + shared memory）
- `other=-float("inf")`：越界位置不影响 max 计算

---

## 7. 性能分析

| 指标 | 值 |
|------|-----|
| 读操作 | 2 × B × N float（naive 3 pass）|
| 写操作 | B × N float |
| FLOPs | ~5N（max, exp, sum, div） |
| Arithmetic Intensity | ~5N / (3N × 4B) ≈ 0.42 FLOP/byte |

Softmax 是 **memory-bound** 操作，优化目标是减少内存访问次数。

**Online softmax 的优势：**
- 将 3 pass 减少到 2 pass（1 pass 读 + 1 pass 写）
- 内存访问减少 ~33%

参考性能（A100，B=4096，N=2048）：

| 实现 | 延迟 | 带宽 |
|------|------|------|
| PyTorch（cuDNN）| ~0.15 ms | ~850 GB/s |
| Triton | ~0.18 ms | ~710 GB/s |
| CUDA v1（3-pass）| ~0.25 ms | ~510 GB/s |
| CUDA v3（online）| ~0.16 ms | ~800 GB/s |

---

## 8. Flash Attention 的连接

Softmax 中的 online softmax 算法是 Flash Attention 的核心：

```
Flash Attention 中，对每个 Q tile：
  遍历所有 K tile，维护 (m, l, O) 状态
  这正是 online softmax + 输出累积

m = running max of attention scores
l = running sum of exp(scores - m)
O = running sum of softmax_weights @ V
```

详见 [attention.md](./attention.md)。

---

## 9. 关键学习点

1. **Reduction 算法**：tree reduction 和 warp shuffle reduction
2. **数值稳定性**：减去 max 避免 exp 溢出
3. **Online 算法**：在单次扫描中维护统计量
4. **两级规约**：warp 内 + warp 间
5. **`__shfl_xor_sync`**：warp 内无需 shared memory 的通信原语
6. **Memory-bound 优化**：减少 pass 数，减少全局内存访问
