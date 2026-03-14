# Matrix Multiplication (Matmul)

## 1. 数学定义

$$C = A \times B$$

其中 $A \in \mathbb{R}^{M \times K}$，$B \in \mathbb{R}^{K \times N}$，$C \in \mathbb{R}^{M \times N}$

$$C_{i,j} = \sum_{k=0}^{K-1} A_{i,k} \cdot B_{k,j}$$

**FLOPs：** $2MNK$（每个输出元素 K 次乘法 + K 次加法）

---

## 2. Roofline 分析

Matmul 是 **compute-bound** 操作，与大多数 elementwise 操作相反。

```
以 A100 为例（M=N=K=4096，float32）：

FLOPs = 2 × 4096³ ≈ 137.4 GFLOP
bytes = (M×K + K×N + M×N) × 4 ≈ 201 MB
Arithmetic Intensity = 137.4G / 201M ≈ 684 FLOP/byte

A100 Ridge Point：
  Peak FP32 TFLOPS：19.5 TFLOPS
  Peak BW：2000 GB/s
  Ridge Point = 19.5T / 2000G = 9.75 FLOP/byte

684 >> 9.75  →  Compute-bound！

优化目标：最大化 TFLOPS（计算效率），而非带宽利用率
```

---

## 3. V1: Naive Kernel

```cuda
__global__ void matmul_naive(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}
```

**性能分析：**

```
对 C[row][col]，需要读：
  A[row][0..K-1]：连续 K 个 float（coalesced）
  B[0..K-1][col]：K 个分散的 float（每次间隔 N 个元素，NOT coalesced）

对整个 block(32×32 threads)：
  每 thread 读 B 的一列 → 每步 k 中，32 个 thread 读同一行的 32 个连续元素 ✓
  但 A 的读取：32 个 thread 行号相同，读同一行（broadcasting from L1/L2）

实际问题：每个 C 元素独立读 B 的一列
  K=4096 时，B 的总读取量 = M×N×K×4 bytes（大量重复！）

等效：B 被读取 M 次，A 被读取 N 次
     vs 理想：每个矩阵只读 1 次
```

Naive 实现的 TFLOPS：通常只有峰值的 0.5-2%。

---

## 4. V2: Shared Memory Tiling（核心优化）

### 4.1 核心思想

将 K 维度分成大小为 `TILE` 的块，每次把 A 和 B 的当前块加载到 shared memory：

```
C = A × B 的分块计算：
C[M, N] = Σ_{t} A[M, TILE] × B[TILE, N]    (分块迭代 K 维)

对于 block (bm, bn)，计算 C 的 TILE×TILE 子块：
  C[bm, bn] = Σ_t A_tile[bm, t] × B_tile[t, bn]
```

### 4.2 图解

```
K = 4096, TILE = 32

A (M×K):
┌──────────────────────────────┐
│  [TILE]  [TILE]  ...  [TILE] │  ← M 行
└──────────────────────────────┘
   t=0      t=1         t=127

B (K×N):
┌──────┐
│[TILE]│  t=0
│[TILE]│  t=1
│  ... │
│[TILE]│  t=127
└──────┘

对一个 CUDA block (bm=0, bn=0)：
C[0:32, 0:32] = A[0:32, 0:32] × B[0:32, 0:32]
              + A[0:32, 32:64] × B[32:64, 0:32]
              + ...
              + A[0:32, 4064:4096] × B[4064:4096, 0:32]
```

### 4.3 内存复用计算

```
没有 tiling：
  计算 C[i][j]：读 A[i][0..K-1] + B[0..K-1][j]
  总读取量 = M × N × 2K × 4 bytes

有 tiling（TILE=32）：
  每个 tile 被 block 中所有 32² = 1024 个 thread 共享
  A tile 的 32×32 个数被 32（N 方向）个 thread 复用 → 复用 32×
  B tile 的 32×32 个数被 32（M 方向）个 thread 复用 → 复用 32×
  总读取量 ≈ M × N × 2K × 4 / TILE bytes

带宽需求降低 TILE = 32 倍！
```

### 4.4 代码解析

```cuda
__global__ void matmul_v2_tiled(const float* A, const float* B, float* C, int M, int K, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;  // C 的行
    int col = blockIdx.x * TILE + tx;  // C 的列

    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // Phase 1: 协同加载（每个 thread 加载一个元素）
        // sA[ty][tx] = A[row][t*TILE + tx]  ← 同一 warp 读连续列 → coalesced
        // sB[ty][tx] = B[t*TILE + ty][col]  ← 同一 warp 读连续列 → coalesced
        sA[ty][tx] = (row < M && t*TILE+tx < K) ? A[row*K + t*TILE+tx] : 0.0f;
        sB[ty][tx] = (t*TILE+ty < K && col < N) ? B[(t*TILE+ty)*N + col] : 0.0f;
        __syncthreads();

        // Phase 2: 从 shared memory 计算（无全局内存访问）
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();  // 防止 race condition（保护 sA, sB 被下一轮覆盖）
    }

    if (row < M && col < N) C[row*N + col] = acc;
}
```

---

## 5. V3: Thread Coarsening

**问题：** V2 中每个 thread 只计算 C 的 1 个元素，寄存器使用率不足。

**思想：** 让每个 thread 计算 C 的多个元素（`TH×TH`），
提高每个 thread 的算术强度，更充分利用寄存器文件。

```
TH=4: 每个 thread 计算 4×4 = 16 个 C 元素
block 大小: (TILE/TH) × (TILE/TH) = (8×8) = 64 threads（vs V2 的 1024）
每个 block 仍处理 TILE×TILE = 32×32 的 C 子块

好处：
  - 寄存器复用更高（A 的 4 个值服务 4 列，B 的 4 个值服务 4 行）
  - 减少 thread 数量 → 减少 block 同步开销
  - 更多的寄存器用于 accumulator（减少 shared memory 访问）
```

**Occupancy 考虑：**
```
V2: 1024 threads/block
    SM 可以有多少个 block？受 shared memory 和寄存器限制
    shared memory: 2 × 32×32×4 = 8 KB/block
    A100: 192 KB/SM → 最多 24 个 block，但受寄存器限制

V3: 64 threads/block，但每 thread 更多寄存器
    SM 可以有更多 block → 更高 occupancy（如果 SM 资源够）
```

---

## 6. Triton 实现：L2 Cache 优化（Swizzle）

Triton 官方教程中的一个重要优化：重新排列 program 的执行顺序以提高 L2 cache 命中率。

### 6.1 为什么需要？

```
默认行优先排布：
  program (0,0), (0,1), (0,2), ... → 处理 C 的第一行 tiles
  program (1,0), (1,1), (1,2), ... → 处理 C 的第二行 tiles

当处理 program (1,0) 时：
  需要 A 的第 2 行 tiles（新数据）
  需要 B 的第 1 列 tiles（与 (0,0) 相同，可能还在 L2 中）

但如果 SM 很多，(0,0)...(0,N) 和 (1,0)...(1,N) 同时运行：
  B 的列 tiles 不能复用 → L2 未命中

GROUP_SIZE_M 分组（L2-aware swizzle）：
  先完成 GROUP_SIZE_M 行的 tiles（固定几列），
  使 B 的列 tiles 在 L2 中保持热
```

### 6.2 示意图

```
默认顺序（行优先）：         GROUP_SIZE_M=4 后的顺序：
(0,0) (0,1) (0,2) ...       (0,0) (1,0) (2,0) (3,0)  ← 同列，共享 B tile
(1,0) (1,1) (1,2) ...       (0,1) (1,1) (2,1) (3,1)
(2,0) (2,1) (2,2) ...       (0,2) (1,2) (2,2) (3,2)
...                          ...

右图中：4 个连续 program 共享相同的 B 列 tile → L2 命中率更高
```

---

## 7. CUTLASS 方式

CUTLASS（CUDA Templates for Linear Algebra Subroutines）提供了
更高级的矩阵乘法抽象，支持：

1. **Tensor Core（wmma/mma.sync）**：利用 Tensor Core 指令，FP16/BF16/FP8
2. **Pipeline（软件流水）**：多阶段 double buffering，隐藏内存延迟
3. **Warp-level 编程**：更细粒度的控制

### CUTLASS Tile 层次

```
Problem (M×K×N)
  ↓ CTA Tile (block level)
    ↓ Warp Tile
      ↓ Thread Block Tile
        ↓ MMA Tile（Tensor Core 指令级别）
```

简单 CUTLASS 示例（cutlass Python API）：
```python
import cutlass
from cutlass import Gemm

plan = Gemm(element_A=cutlass.Float32, element_B=cutlass.Float32,
            element_C=cutlass.Float32, element_D=cutlass.Float32,
            layout_A=cutlass.RowMajor, layout_B=cutlass.RowMajor,
            layout_C=cutlass.RowMajor)

A = torch.randn(M, K, device='cuda')
B = torch.randn(K, N, device='cuda')
C = plan.run(A, B)
```

---

## 8. Tensor Core 简介

NVIDIA Volta+ GPU 引入了 Tensor Core，专门加速矩阵乘法：

```
FP32 CUDA Core: 1 FMA/clock = 2 FLOP/clock/core
Tensor Core (A100):
  wmma 指令: 16×16×16 = 4096 FMA = 8192 FLOP/clock/tensor core
  A100 有 512 个 Tensor Core → 8192×512 = 4M FLOP/clock
  peak FP16: ~312 TFLOPS（vs FP32 ~19.5 TFLOPS）
```

使用方式：
- CUDA：`nvcuda::wmma` 指令（FP16/BF16 输入，FP32 累积）
- Triton：`tl.dot()` 自动调用 Tensor Core（当输入是 FP16 时）
- CUTLASS：通过 MmaAtom 抽象

---

## 9. 性能参考（A100 80GB，M=N=K=4096）

| 实现 | 延迟 | TFLOPS | 峰值利用率 |
|------|------|--------|----------|
| Naive CUDA | ~180 ms | 0.76 T | 4% |
| Shared Memory Tiling | ~1.8 ms | 76 T | 19% |
| Thread Coarsening | ~1.2 ms | 114 T | 29% |
| Triton（autotuned）| ~0.45 ms | 305 T | 78% |
| PyTorch (cuBLAS) | ~0.42 ms | 327 T | 84% |

注：以上数据使用 FP32，使用 FP16/BF16 + Tensor Core 可以再快 10-16×。

---

## 10. 关键学习点

1. **Compute-bound vs Memory-bound**：Arithmetic Intensity 分析
2. **Shared Memory Tiling**：核心优化，TILE 倍的带宽节省
3. **Thread 协同**：block 内的 thread 合作完成数据加载
4. **Thread Coarsening**：每 thread 处理多元素，提高寄存器利用率
5. **L2 Cache 优化（Swizzle）**：重排 program 执行顺序
6. **Tensor Core**：利用 GPU 专用矩阵乘指令
7. **两次 `__syncthreads`**：第一次确保加载完毕，第二次防止下轮覆盖
