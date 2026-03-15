# Matrix Transpose

## 1. 数学定义

$$B_{j,i} = A_{i,j}$$

给定 M×N 矩阵 A（行主序），计算 N×M 矩阵 B = A^T。

---

## 2. 内存布局与访问模式

GPU 内存以行主序（row-major）存储：

```
A[M×N]（行主序）：
A[0][0], A[0][1], ..., A[0][N-1],   ← 连续存储
A[1][0], A[1][1], ..., A[1][N-1],
...
A[M-1][0], ..., A[M-1][N-1]

地址：A[i][j] = A + i*N + j
```

转置操作：`B[j][i] = A[i][j]`

---

## 3. V1: Naive Kernel 的问题

```cuda
__global__ void transpose_v1_naive(...) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // 列
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // 行
    output[x * rows + y] = input[y * cols + x];      // 读连续，写跳跃
}
```

**访问模式分析（以 32-thread warp 为例）：**

```
warp 中 thread 0..31 的 threadIdx.x = 0..31（列方向）

读 input：
  thread 0 → input[y * cols + 0]
  thread 1 → input[y * cols + 1]
  ...
  thread 31 → input[y * cols + 31]
  → 连续地址！1 个 cache line 服务 32 个 thread ✓ COALESCED

写 output：
  thread 0 → output[0 * rows + y]
  thread 1 → output[1 * rows + y]
  ...
  thread 31 → output[31 * rows + y]
  → 地址间隔 rows（数千个元素）→ 32 个 cache line！✗ NOT COALESCED
```

非 coalesced 写入的后果：
- 每个 thread 独立触发一次内存事务
- 32 个 thread → 32 次内存事务（vs 理想的 1 次）
- 内存带宽利用率极低（约 1/32 = 3%）

---

## 4. V2: Shared Memory Tiling

核心思想：用 shared memory 作为中转，使读写都 coalesced。

### 4.1 算法流程

```
Step 1: 从全局内存 COALESCED 读取 32×32 tile 到 shared memory
         thread(tx, ty) 读 input[(by*32+ty)*N + (bx*32+tx)]
         → warp 中 tx=0..31 → 连续列 → COALESCED

Step 2: __syncthreads() 等待所有数据加载完毕

Step 3: 从 shared memory 读取，COALESCED 写回全局内存
         thread(tx, ty) 写 output[(bx*32+ty)*M + (by*32+tx)]
         → warp 中 tx=0..31 → 连续列 → COALESCED
         → 但从 shared memory 读 sdata[tx][ty] → 列读取！
```

### 4.2 示意图

```
输入矩阵 A:                    输出矩阵 B = A^T:
┌─────────────────────┐       ┌─────────────────────┐
│  tile(0,0)  tile(0,1)│       │  tile^T(0,0) ...    │
│  tile(1,0)  tile(1,1)│       │  tile^T(0,1) ...    │
└─────────────────────┘       └─────────────────────┘

一个 CUDA block 的工作：
1. 加载 tile(by, bx) 到 sdata[TILE][TILE]  ← COALESCED 读
2. 写 sdata 的转置到 tile^T(bx, by)         ← COALESCED 写
   (注意：写的 block 位置是 (bx, by)，行列互换)
```

---

## 5. Bank Conflict：关键细节

### 什么是 Bank Conflict？

GPU shared memory 分为 32 个 bank（bank width = 4 bytes），
地址为 `addr` 的数据在 bank `(addr / 4) % 32` 中。

一个 warp 内所有 thread 同时访问 shared memory 时：
- 访问不同 bank → **无冲突，一个时钟周期**
- 访问同一 bank 不同地址 → **bank conflict，串行化**
- 访问同一 bank 同一地址 → **广播，无冲突**

### 32×32 转置中的 Bank Conflict

```
sdata[32][32]：按行存储，每行 32 个 float = 128 bytes = 32 banks

写入 sdata（连续列写）：sdata[ty][tx]
  thread 0..31：sdata[0..31][0]
  → 地址：0×128, 1×128, ..., 31×128
  → bank：(0/4)%32, (128/4)%32, ..., (31×128/4)%32
           = 0, 0, 0, ..., 0     ← 全部在 bank 0！

32 路 bank conflict！每次写操作被串行化为 32 个时钟周期！
```

### 解决方案：Padding

```c
__shared__ float sdata[TILE][TILE + 1];  // +1 列 padding
```

```
sdata[32][33]：每行 33 个 float = 132 bytes

写入 sdata（连续列写）：sdata[ty][tx]
  thread 0..31：sdata[0..31][0]
  → 地址：0×132, 1×132, ..., 31×132
  → bank：(0/4)%32, (132/4)%32, ..., (31×132/4)%32
           = 0, 1, 2, ..., 31   ← 全部在不同 bank！✓

无 bank conflict！
```

**为什么 +1 有效？**
- 每行从 32 个 float 变为 33 个 float
- 行跨度变为 33 × 4 = 132 bytes，不再是 128 的整数倍
- 相邻行的起始地址不再对齐到同一 bank

---

## 6. V2 完整代码解析

```cuda
template <int BLOCK_SZ, int NUM_PER_THREAD>
__global__ void transpose_v2_shared(const float* input, float* output, int rows, int cols) {
    // BLOCK_SZ+1: padding 消除 bank conflict
    __shared__ float sdata[BLOCK_SZ][BLOCK_SZ + 1];

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    // 每个 thread 处理 NUM_PER_THREAD 行
    // block size: (BLOCK_SZ, BLOCK_SZ/NUM_PER_THREAD) = (32, 8)
    constexpr int ROW_STRIDE = BLOCK_SZ / NUM_PER_THREAD;  // 8

    // Phase 1: 读 input，写 sdata（COALESCED 读）
    int x = bx * BLOCK_SZ + tx;   // 列索引
    int y = by * BLOCK_SZ + ty;   // 行索引（起始）
    if (x < cols) {
        #pragma unroll
        for (int y_off = 0; y_off < BLOCK_SZ; y_off += ROW_STRIDE) {
            // 同一 warp(ty固定，tx=0..31)：读 input[y*cols + x]，x连续 → COALESCED
            sdata[ty + y_off][tx] = input[(y + y_off) * cols + x];
        }
    }
    __syncthreads();

    // Phase 2: 读 sdata，写 output（COALESCED 写）
    x = by * BLOCK_SZ + tx;  // 注意：bx/by 互换，实现转置
    y = bx * BLOCK_SZ + ty;
    if (x < rows) {
        #pragma unroll
        for (int y_off = 0; y_off < BLOCK_SZ; y_off += ROW_STRIDE) {
            // 写 output[(y+y_off)*rows + x]，x 连续 → COALESCED
            // 读 sdata[tx][ty+y_off]：列读取，bank conflict 被 padding 消除
            output[(y + y_off) * rows + x] = sdata[tx][ty + y_off];
        }
    }
}
```

---

## 7. 测试方法

### 7.1 编译 CUDA Kernel

```bash
cd gpu-kernel-lab/operators/transpose/cuda && bash build.sh
# 或指定架构
CUDA_ARCH=sm_90 bash build.sh
```

### 7.2 运行正确性测试 + Benchmark

```bash
# 从项目根目录运行
cd gpu-kernel-lab
python -m operators.transpose.test
```

---

## 8. Benchmark 结果（H20，4096×4096，float32）

H20 理论峰值带宽：~4.0 TB/s

| 实现 | 延迟 | 带宽 | vs PyTorch |
|------|------|------|------------|
| PyTorch | 0.1418 ms | 946 GB/s | 1.00x |
| Triton | 0.0756 ms | 1776 GB/s | 1.88x |
| CUDA v1 (naive) | 0.2913 ms | 461 GB/s | 0.49x |
| CUDA v2 (shared mem) | 0.0683 ms | 1965 GB/s | 2.08x |

注：
- CUDA v2 通过 shared memory tiling + bank conflict padding，带宽比 naive 提升 4.3×
- Triton 的 transpose kernel 同样使用了类似的优化策略，性能接近 CUDA v2
- PyTorch 使用 cuBLAS，内部有额外的内存管理开销，反而比手写 CUDA v2 慢

---

## 9. 性能对比（vs A100 理论）

```
操作量：4096² × 2 × 4 bytes ≈ 134 MB
H20 峰值带宽：~4000 GB/s
理论最优延迟 = 134 MB / 4000 GB/s = 0.034 ms

CUDA v2 实测：0.068 ms，带宽效率 = 1965 / 4000 ≈ 49%
Triton：0.076 ms，带宽效率 = 1776 / 4000 ≈ 44%
```

转置永远无法达到 100% 带宽利用率，因为写方向无法与硬件缓存行对齐。

---

## 9. 关键学习点

1. **Non-Coalesced 写入是性能杀手**：32× 带宽浪费
2. **Shared Memory 作为 scratchpad**：CPU 的 L1 cache 概念在 GPU 的显式化
3. **Bank Conflict**：shared memory 的并发访问限制
4. **Padding 技巧**：用 +1 列消除 bank conflict
5. **Thread Coarsening**：每个 thread 处理多行，减少 block 数量
6. **`#pragma unroll`**：提示编译器展开循环，减少循环开销

---

## 10. 扩展阅读

- [How to Access Global Memory Efficiently in CUDA (NVIDIA Blog)](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
- [Shared Memory Bank Conflict](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- 参考实现：`../LeetGPU/03-matrix-transpose/CUDA/use_shared.cu`
