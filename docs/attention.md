# Flash Attention

## 1. 数学定义

**Scaled Dot-Product Attention：**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

其中：
- $Q \in \mathbb{R}^{N \times d}$：Query
- $K \in \mathbb{R}^{N \times d}$：Key
- $V \in \mathbb{R}^{N \times d}$：Value
- $N$：序列长度
- $d$：head dimension
- $1/\sqrt{d}$：缩放因子，防止点积值过大导致 softmax 梯度消失

**多头版本（Multi-Head Attention）：**

$$\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

---

## 2. 朴素 Attention 的问题

### 2.1 显存复杂度

```
朴素实现：
  S = Q K^T         → (B, H, N, N) 矩阵
  P = softmax(S)    → (B, H, N, N)
  O = P V           → (B, H, N, D)

以 Llama-2-7B 的典型配置为例：
  B=4, H=32, N=4096, D=128
  S 的大小 = 4 × 32 × 4096 × 4096 × 4 bytes ≈ 8.6 GB！

这在单卡（如 A100 80GB）中虽然能放下，但对于：
  N=32768（长文本）：S ≈ 536 GB → 无法存储
```

### 2.2 内存访问复杂度

朴素 Attention 的 HBM 读写量：

```
写 S:   B×H×N²×4 bytes
读 S:   B×H×N²×4 bytes（softmax 需要读整行）
写 P:   B×H×N²×4 bytes
读 P:   B×H×N²×4 bytes（PV 乘法）

总计：O(N²) 的 HBM 读写 → 非常 memory-intensive
```

---

## 3. Flash Attention 核心思想

论文：[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)（Dao et al., NeurIPS 2022）

### 3.1 IO-Aware 算法设计

**关键洞察：** Attention 不是 compute-bound，而是 memory-bound。
瓶颈在于 S 矩阵的 HBM 读写，而非矩阵乘法的 FLOPs。

**解决方案：** 不把 S 矩阵写到 HBM，完全在 SRAM（shared memory）中完成计算。

```
SRAM（GPU shared memory）：~192 KB/SM，非常快（~1 TB/s）
HBM（GPU DRAM）：~80 GB，较慢（~2 TB/s，但每次访问延迟高）

Flash Attention 目标：
  - 永远不在 HBM 中物化 S = QK^T
  - 分块计算，每块只在 SRAM 中处理
  - 最终只写一次输出 O 到 HBM
```

### 3.2 分块策略

```
Q 按行分块：Q₁, Q₂, ..., Q_{Tr}  （每块 Br 行）
K, V 按行分块：K₁, K₂, ..., K_{Tc}  （每块 Bc 行）

Br, Bc 的选择：使 Br×Bc×4 ≤ SRAM 大小

外层循环 i: 1..Tr  （Q tile）
  内层循环 j: 1..Tc  （KV tile）
    1. 加载 Q_i, K_j, V_j 到 SRAM
    2. 计算 S_ij = Q_i K_j^T × scale  (SRAM 内完成)
    3. Online softmax 更新状态 (m_i, l_i, O_i)
  写回 O_i 到 HBM
```

---

## 4. Online Softmax 推导

### 4.1 标准 Softmax 的分块问题

标准 softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
需要知道整行的 max → 无法分块。

### 4.2 Online Softmax 递推

设当前已处理到第 j 个 KV tile，维护状态：
- $m_i^{(j)}$：到目前为止见过的最大 attention score
- $l_i^{(j)}$：到目前为止的 exp 之和（以 $m_i^{(j)}$ 为基准）
- $O_i^{(j)}$：到目前为止的加权 V 之和

处理新的 KV tile $j+1$ 时：

$$S_{ij} = Q_i K_j^T / \sqrt{d} \quad \in \mathbb{R}^{B_r \times B_c}$$

$$m_i^{(j+1)} = \max(m_i^{(j)}, \text{rowmax}(S_{ij+1}))$$

$$\tilde{P}_{ij+1} = \exp(S_{ij+1} - m_i^{(j+1)})$$

$$l_i^{(j+1)} = e^{m_i^{(j)} - m_i^{(j+1)}} \cdot l_i^{(j)} + \text{rowsum}(\tilde{P}_{ij+1})$$

$$O_i^{(j+1)} = \text{diag}(e^{m_i^{(j)} - m_i^{(j+1)}}) \cdot O_i^{(j)} + \tilde{P}_{ij+1} V_{j+1}$$

最终归一化：

$$O_i = O_i^{(T_c)} / l_i^{(T_c)}$$

### 4.3 推导验证

展开完整公式：
$$O_i = \frac{\sum_{j} \exp(S_{ij} - m^*) V_j}{\sum_j \sum_k \exp(S_{ijk} - m^*)} = \text{softmax}(S_i) V$$

其中 $m^* = m_i^{(T_c)}$ 是全局最大值。✓ 与标准 Attention 完全等价。

---

## 5. IO 复杂度分析

| 方法 | HBM 读写量 | 中间存储 |
|------|-----------|---------|
| 朴素 Attention | $O(N^2 d)$ | $O(N^2)$ |
| Flash Attention | $O(N d^2 / M)$ | $O(Nd)$ |

其中 $M$ 是 SRAM 大小。

**实际数字（A100, N=4096, d=128, M=192KB）：**
```
朴素：~8.6 GB × 4 次读写 ≈ 34.4 GB
Flash：~O(Nd) ≈ 4096×128×4×4 ≈ 8 MB（仅 Q, K, V, O）

内存节省：约 4000×！
```

---

## 6. Flash Attention V2 改进

论文：[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)（Dao, 2023）

主要改进：

1. **减少非矩阵乘法 FLOPs**：
   - V1 中每个 KV tile 需要 rescale 整个 O accumulator
   - V2 推迟 rescale，到最后一次性归一化

2. **更好的并行性**：
   - V1：外层循环 KV，内层循环 Q → 适合 KV 并行
   - V2：外层循环 Q，内层循环 KV → Q tile 可以并行（无数据依赖）

3. **Warp 分工**：
   - 更好地分配 warp 处理 Q 和 KV

---

## 7. Causal Mask

在 decoder-only 模型（如 GPT/Llama）中，每个 token 只能 attend to 之前的 token：

```
N=4 的 causal mask：
┌─────────────┐
│ 1  0  0  0  │  token 0 只看自己
│ 1  1  0  0  │  token 1 看 0, 1
│ 1  1  1  0  │  token 2 看 0, 1, 2
│ 1  1  1  1  │  token 3 看所有
└─────────────┘

实现：
  S_{ij} = -∞  if j > i
  这等价于：if position(k) > position(q): score = -inf
```

在 Flash Attention 中，causal mask 对右上角的 tile 直接跳过（全为 -∞），
对对角线上的 tile 部分 mask。实际节省了约一半的计算量。

---

## 8. Triton 实现解析

```python
@triton.jit
def flash_attn_fwd_kernel(..., BLOCK_M, BLOCK_N, BLOCK_D, CAUSAL):
    # 每个 program 处理一个 (batch, head, q_tile) 组合
    start_m = tl.program_id(0)   # Q tile 索引
    off_bh  = tl.program_id(1)   # batch * head

    # 加载 Q tile
    q = tl.load(Q_ptrs, mask=q_mask)  # [BLOCK_M, BLOCK_D]

    # 初始化状态
    m_i = tl.full([BLOCK_M], float("-inf"))  # running max
    l_i = tl.zeros([BLOCK_M])                # running sum
    acc = tl.zeros([BLOCK_M, BLOCK_D])       # output

    # 遍历 KV tiles
    for start_n in range(0, kv_end, BLOCK_N):
        k = tl.load(K_ptrs)  # [BLOCK_N, BLOCK_D]
        v = tl.load(V_ptrs)  # [BLOCK_N, BLOCK_D]

        # S = Q K^T * scale: [BLOCK_M, BLOCK_N]
        s = tl.dot(q, tl.trans(k)) * softmax_scale

        # Causal mask（可选）
        if CAUSAL:
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float("-inf"))

        # Online softmax 更新
        m_ij = tl.max(s, axis=1)          # [BLOCK_M]
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)       # 校正因子
        p = tl.exp(s - m_new[:, None])    # [BLOCK_M, BLOCK_N]
        l_new = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m_i, l_i = m_new, l_new

    # 归一化并写回
    acc = acc / l_i[:, None]
    tl.store(O_ptrs, acc)
```

---

## 9. 测试方法

### 9.1 运行正确性测试 + Benchmark

Attention 的 CUDA kernel 目前为模板头文件，主要通过 Python 调用 Triton 实现运行：

```bash
# 从项目根目录运行
cd gpu-kernel-lab
python -m operators.attention.test
```

---

## 10. Benchmark 结果（H20，B=4 H=8 N=1024 D=64，float32）

H20 理论峰值 FP32：~44 TFLOPS

| 实现 | 延迟 | TFLOPS | vs PyTorch naive |
|------|------|--------|-----------------|
| PyTorch (naive) | 0.6303 ms | 13.63 T | 1.00x |
| PyTorch SDPA | 0.7481 ms | 11.48 T | 0.84x |
| Triton Flash Attention | 0.2960 ms | 29.02 T | 2.13x |

**不同序列长度的 Scaling（B=1 H=8 D=64）：**

| N | PyTorch naive | Triton Flash | 加速比 |
|---|---------------|--------------|--------|
| 256 | 0.074 ms (1.82T) | 0.050 ms (2.69T) | 1.48x |
| 512 | 0.084 ms (6.41T) | 0.064 ms (8.35T) | 1.30x |
| 1024 | 0.198 ms (10.87T) | 0.105 ms (20.48T) | 1.88x |
| 2048 | 0.645 ms (13.32T) | 0.302 ms (28.45T) | 2.14x |
| 4096 | 2.602 ms (13.21T) | 1.089 ms (31.54T) | 2.39x |

注：
- Flash Attention 在所有序列长度下均优于 naive 实现
- 加速比随 N 增大而提升（N=4096 时达到 2.39x），因为 naive 的 O(N²) HBM 访问随 N 放大
- PyTorch SDPA 在此配置下比 naive 慢（0.84x），可能是小 batch 下 kernel launch 开销占比高
- Triton Flash Attention 在 N=4096 时达到 31.54 TFLOPS，约为 H20 峰值的 72%

---

## 11. 性能分析

**计算密度分析：**
```
Flash Attention FLOPs：
  QK^T：2 × N² × d
  PV：  2 × N² × d
  Total：4 × N² × d × B × H

以 N=4096, d=64, B=4, H=8：
  FLOPs = 4 × 4096² × 64 × 4 × 8 ≈ 137 GFLOP

HBM bytes（Flash Attention）：
  读 Q, K, V：3 × B × H × N × d × 4 ≈ 3 × 4 × 8 × 4096 × 64 × 4 ≈ 1 GB
  写 O：       B × H × N × d × 4 ≈ 0.34 GB
  总计：≈ 1.34 GB

Arithmetic Intensity = 137 GFLOP / 1.34 GB ≈ 102 FLOP/byte
远大于 Ridge Point（9.75）→ Compute-bound！
```

**与朴素实现对比：**

| 方法 | HBM 访问量 | 中间存储 | 速度提升 |
|------|-----------|---------|---------|
| 朴素 | ~34.4 GB | ~8.6 GB | 1× |
| Flash V1 | ~1.3 GB | ~8 MB | ~7× |
| Flash V2 | ~1.3 GB | ~8 MB | ~10× |

---

## 12. 关键学习点

1. **Memory vs Compute Bound**：根据 Arithmetic Intensity 判断瓶颈
2. **IO-Aware 算法**：从全局内存访问次数角度设计算法
3. **Online Softmax**：允许分块计算 Attention 的关键
4. **Tiling 策略**：根据 SRAM 大小选择 block size
5. **Flash Attention 等价性**：分块计算与全矩阵完全等价
6. **Causal Mask 优化**：跳过上三角 KV tiles，节省约 50% FLOPs

---

## 13. 扩展阅读

- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 论文](https://arxiv.org/abs/2307.08691)
- [ELI5: FlashAttention 博客](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)
- [Triton Flash Attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- `../LeetGPU/32-multi-head-self-attention/Triton/native.py`
