# Vector Add

## 1. 数学定义

$$C_i = A_i + B_i, \quad i = 0, 1, \ldots, N-1$$

向量加法是最简单的并行计算：每个输出元素独立于其他元素，完全可以并行计算。

---

## 2. GPU 并行策略

### 线程模型

```
N = 1024, BLOCK_SIZE = 256

线程 ID:   0   1   2   ...  255 | 256  257  ...  511 | ...
           ─────────────────────  ─────────────────────
Block 0:  [0..255]              Block 1: [256..511]  ...

每个线程：idx = blockIdx.x * blockDim.x + threadIdx.x
           C[idx] = A[idx] + B[idx]
```

核心计算：
- Grid 大小：`ceil(N / BLOCK_SIZE)`
- Block 大小：`BLOCK_SIZE = 256`
- 每个 thread 处理 1 个元素
- 边界检查：`if (idx < N)`

### CUDA 线程层次

```
Grid
├── Block 0  (256 threads)
│   ├── Warp 0  (threads 0..31)   → 同时执行
│   ├── Warp 1  (threads 32..63)
│   └── ...
├── Block 1  (256 threads)
└── ...
```

---

## 3. 内存访问分析

Vector Add 是一个典型的 **memory-bound** 操作：

| 指标 | 值 |
|------|-----|
| 读操作 | 2N float（读 A、B）|
| 写操作 | N float（写 C）|
| FLOPs  | N（1 次加法/元素）|
| Arithmetic Intensity | N / (3N × 4B) = 0.083 FLOP/byte |
| A100 Ridge Point | ~9.75 FLOP/byte |

结论：Vector Add 的 Arithmetic Intensity (0.083) << Ridge Point (9.75)，
所以是 **memory-bound**（受内存带宽限制，而非计算能力）。

这意味着优化方向是**提升内存带宽利用率**。

---

## 4. V1: Naive Kernel

```cuda
__global__ void vector_add_v1(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
```

**内存访问模式分析：**
- 同一 warp 中的 thread 0..31 访问 `A[0], A[1], ..., A[31]`
- 这是连续的内存地址 → **Coalesced 访问** ✓
- 一次内存事务（128 字节 cache line）服务 32 个 float

**Coalesced vs Non-Coalesced：**
```
Coalesced（好）：                    Non-Coalesced（差）：
thread 0 → A[0]                     thread 0 → A[0]
thread 1 → A[1]                     thread 1 → A[128]
thread 2 → A[2]     1 cache line    thread 2 → A[256]    32 cache lines!
...                                  ...
thread 31 → A[31]                   thread 31 → A[31*128]
```

---

## 5. V2: 向量化 Kernel（float4）

```cuda
const float4* A4 = reinterpret_cast<const float4*>(A);
// 每个 thread 加载 4 个 float（一个 128-bit 内存事务）
float4 a = A4[i];
```

**float4 的好处：**
- 一条 LD 指令加载 4 个 float（vs 4 条 LD 指令）
- 减少地址计算开销
- 更好地利用 128-bit 内存总线宽度
- 在某些架构上，128-bit load 比 4 个 32-bit load 更高效

**性能提升：**
- 理论上减少了指令发射次数（4×）
- 实际加速：~5-15%（取决于 GPU 和问题规模）

---

## 6. Triton 实现解析

```python
@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)        # 当前 program 的 ID
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 向量化地址
    mask = offsets < n_elements        # 边界 mask
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)    # 向量化 load
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(c_ptr + offsets, a + b, mask=mask)            # 向量化 store
```

Triton 与 CUDA 的对应关系：
| Triton | CUDA |
|--------|------|
| `tl.program_id(0)` | `blockIdx.x` |
| `tl.arange(0, BLOCK_SIZE)` | 向量化的 `threadIdx.x` 范围 |
| `tl.load(ptr + offsets, mask)` | 向量化的 `__global__` 内存读取 |
| `BLOCK_SIZE: tl.constexpr` | 编译期常量（用于向量化分析）|

Triton 自动完成的优化：
- 自动选择最优的内存访问指令（float4 / float2 等）
- 自动处理内存对齐
- 自动生成 PTX/SASS 指令

---

## 7. CuTe DSL 实现解析

CuTe DSL 是 CUTLASS 库的 Python 接口，提供更接近数学的编程模型：

```python
@cute.kernel
def vector_add_kernel(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    thread_idx = bidx * bdim + tidx
    if thread_idx < N:
        C[thread_idx] = A[thread_idx] + B[thread_idx]
```

CuTe 的核心抽象：
- `cute.Tensor`：带 Layout 信息的张量（描述内存布局）
- `cute.kernel`：编译为 GPU kernel 的装饰器
- `cute.jit`：JIT 编译的 host 调度函数

---

## 8. 测试方法

### 8.1 编译 CUDA Kernel

```bash
# 在项目根目录下
cd gpu-kernel-lab

# 编译 vector_add 的 CUDA kernel（sm_90 = H20/H200）
cd operators/vector_add/cuda && bash build.sh
# 或指定架构
CUDA_ARCH=sm_90 bash build.sh
```

### 8.2 运行正确性测试 + Benchmark

```bash
# 从项目根目录运行（确保 common/ 模块可以被导入）
cd gpu-kernel-lab
python -m operators.vector_add.test
```

### 8.3 单独运行 Benchmark

```bash
python benchmarks/benchmark.py --op vector_add
```

---

## 9. Benchmark 结果（H20，N = 16M，float32）

H20 理论峰值带宽：~4.0 TB/s（Hopper 架构，sm_90）

| 实现 | 延迟 | 带宽 | vs PyTorch |
|------|------|------|------------|
| PyTorch | 0.0637 ms | 3158 GB/s | 1.00x |
| Triton | 0.0746 ms | 2698 GB/s | 0.85x |
| CUDA v1 (naive) | 0.1108 ms | 1818 GB/s | 0.58x |
| CUDA v2 (float4) | 0.0729 ms | 2762 GB/s | 0.87x |

注：
- H20 理论峰值带宽约 4000 GB/s，实际利用率约 70-80%（已相当高效）
- PyTorch/Triton/CUDA v2 性能接近，差异来自编译器优化和 kernel launch 开销
- CUDA v1 naive 的带宽利用率偏低，主要因为 float4 指令减少了指令开销

---

## 10. 关键学习点

1. **CUDA 线程层次**：Grid → Block → Warp → Thread
2. **全局线程 ID 计算**：`blockIdx.x * blockDim.x + threadIdx.x`
3. **Coalesced 内存访问**：同一 warp 访问连续内存地址
4. **Memory-bound vs Compute-bound**：Arithmetic Intensity 决定瓶颈
5. **向量化访问（float4）**：提高内存事务效率
6. **边界检查**：处理 N 不是 BLOCK_SIZE 整数倍的情况

---

## 11. 进一步学习

- [CUDA 内存模型](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)
- [Coalesced Memory Access](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
- [Triton 教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
