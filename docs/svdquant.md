# SVDQuant 算子实现文档

> **工作目录**: `gpu-kernel-svdquant/operators/svdquant/`
> **参考实现**: `nunchaku/`（SM 90/H20 不支持，已跳过安装）
> **最后更新**: 2026-03-28

---

## 目录

1. [SVDQuant 数学原理与动机](#1-svdquant-数学原理与动机)
2. [W4A4 量化格式详解](#2-w4a4-量化格式详解)
3. [实现策略（各版本设计思路）](#3-实现策略)
4. [遇到的问题与解决方法](#4-遇到的问题与解决方法)
5. [Benchmark 结果](#5-benchmark-结果)
6. [与 nunchaku 的对比分析](#6-与-nunchaku-的对比分析)

---

## 1. SVDQuant 数学原理与动机

### 1.1 动机

大模型推理时，线性层的权重访存成为瓶颈。将权重量化到 INT4/INT8 可以：
- 减少 4x（W4）内存占用
- 提高 2-4x 内存带宽利用率
- 在 Tensor Core 支持下加速 GEMM

**核心挑战**：激活值（activation）存在大量 **outlier**（离群值），直接量化精度损失严重。

### 1.2 SVDQuant 方案

SVDQuant 通过两步解决 outlier 问题：

**Step 1 - 平滑迁移（Smooth Transfer）**：
```
x̂ = x * smooth⁻¹                    # 激活变"平滑"
Ŵ = diag(smooth) * W                 # outlier 迁移到权重
```
smooth factor 通常选为 `max(|x|)` 的适当幂次，使激活/权重间 outlier 均衡。

**Step 2 - SVD 低秩分解**：
Ŵ 经过平滑后包含从激活迁移来的 outlier，仍难以直接量化。
通过 SVD 提取主要成分到低秩 LoRA 分支：
```
[U, S, Vᵀ] = SVD(Ŵ)                  # 离线
lora_down = U[:, :r] * √S[:r]         # (K, r)
lora_up   = Vᵀ[:r, :] * √S[:r]       # (r, N)
residual  = Ŵ - lora_down @ lora_up   # (K, N)，outlier 已被低秩分支"吸收"
```

**Step 3 - 量化残差**：
residual 的分布更平滑，可以高精度量化：
```
Q_residual, wscales = INT4_quantize(residual)
```

**Step 4 - 前向传播**：
```
y = dequant(Q_x) @ dequant(Q_residual) + (x @ lora_down) @ lora_up + bias
```
- 主路径：W4A4 GEMM（量化激活 × 量化残差）
- 低秩路径：FP16 LoRA（精确补偿 outlier 信息）

### 1.3 为什么 SVD 能解决 outlier？

SVD 的奇异值按大小排序，前 r 个奇异值捕捉了矩阵的"主要能量"。
平滑后的权重 Ŵ 的 outlier 往往集中在少数几个主方向，
因此低秩分支可以精确表达，而残差分布则更均匀，更易量化。

---

## 2. W4A4 量化格式详解

### 2.1 对称 INT4 per-group 量化

**格式规格**：
- 值域：[-8, 7]（非对称，但量化方案对称于 0）
- Per-group：每 `group_size=64` 个元素共享一个 scale
- Scale 计算：`s = max(|x_group|) / 7`
- 量化：`q = round(x / s).clamp(-8, 7)`
- 反量化：`x̂ = q * s`

**理论精度**：
最大量化误差 = `s / 2 = max(|x|) / 14`，约为原始值的 7% 误差

### 2.2 Packing 格式（INT4 → uint8）

每两个 INT4 值打包进一个 uint8：
```
packed[i] = (q[2i] + 8) | ((q[2i+1] + 8) << 4)
```
- 低 nibble（bit 0-3）：偶数索引
- 高 nibble（bit 4-7）：奇数索引
- `+8` 将 [-8, 7] 映射到 [0, 15]

解包：
```
q_even = (packed & 0x0F) - 8
q_odd  = (packed >> 4) - 8
```

### 2.3 Per-group Scale 布局

激活 scales 形状：`(M, K // group_size)`
权重 scales 形状：`(K // group_size, N)`（内部表示为列主序 `(N, K//G)` 便于访问）

在 Triton kernel 中，按 `g = k // GROUP_SIZE` 索引 scale，
每个 BLOCK_K tile 对应一个 group。

---

## 3. 实现策略

### 3.1 目录结构

```
operators/svdquant/
├── pytorch/
│   ├── baseline.py          # FP16 基准（cuBLAS）
│   └── svdquant_torch.py    # PyTorch 参考实现
├── triton/
│   └── kernel.py            # Triton 融合 kernel
├── cute/
│   └── kernel.py            # CuTe DSL 实现
├── test.py                  # 正确性测试（5 组测试）
└── benchmark.py             # 独立 benchmark
```

### 3.2 PyTorch 实现

**文件**: `pytorch/svdquant_torch.py`

核心函数：
- `int4_quantize(x, group_size=64)` → `(q, scales)` - 对称 INT4 量化
- `int4_dequantize(q, scales)` → `x_fp16` - 反量化
- `int4_pack_uint8(q)` / `int4_unpack_uint8(packed)` - nibble 打包/解包
- `create_svdquant_params(W, rank, group_size, smooth)` - 离线参数构建（含 SVD）
- `svdquant_forward_torch(x, q_w, wscales, lora_down, lora_up, smooth, bias)` - 完整前向

**精度验证**：
- 量化误差 ≤ `max(|x|) / 14`（理论上界）
- SVDQuant vs FP16 baseline：max_err ≈ 0.11~0.28（量化误差范围内）

### 3.3 Triton 实现

**文件**: `triton/kernel.py`

主 kernel：`svdquant_simple_kernel`

设计要点：
- **融合策略**：smooth + 量化反量化 + W4A4 GEMM + LoRA 全部融合进一个 kernel
- **BLOCK_K = GROUP_SIZE = 64**：每个 K-tile 对应一个量化 group，简化 scale 索引
- **量化模拟**：在 kernel 内对激活做 round + clamp 模拟 INT4 精度（非真正 INT4 MMA）
- **LoRA 计算**：在主 GEMM 外额外做两次 tl.dot（lora_down 和 lora_up）

关键配置：
```python
BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 64 (= group_size)
```

**注意事项**：
- `lora_down` 从 `torch.linalg.svd` 返回的 U 矩阵切片，默认非连续（stride 非行主序）
- 必须调用 `.contiguous()` 确保 row-major 内存布局，否则 kernel 地址计算出错
- `tl.dot` 的第三参数 accumulator 语义在 Triton 3.x 中已变化，需用 `acc += tl.dot(a, b)`

### 3.4 CuTe Python DSL 实现

**文件**: `cute/kernel.py`

实现了两个版本：
- **V1**：Naive per-element，串行 K 循环，展示 CuTe 张量索引 API
- **V2**：共享内存分块，在线反量化，减少 HBM 读取

**环境限制**：
当前 H20（SM 9.0）环境存在 `cudaErrorInsufficientDriver` 问题（CUDA driver 版本不兼容 cutlass Python DSL 运行时）。kernel 代码实现完整，环境修复后即可运行。

相关设置：
```bash
export CUDA_TOOLKIT_PATH=/usr/local/cuda
export CUTLASS_TARGET_ARCH=sm_90a
```

---

## 4. 遇到的问题与解决方法

### Problem 1: nunchaku 不支持 SM 90 (H20)

**现象**：
```
AssertionError: Unsupported SM 90
```
**原因**：nunchaku setup.py 的 `get_sm_targets()` 仅支持 SM 75/80/86/89/120a/121a，不含 SM 90。

**解法**：跳过 nunchaku 安装，用 PyTorch 参考实现作为正确性基准。
参考代码仍可阅读 `nunchaku/nunchaku/ops/quantize.py` 学习 API 设计。

### Problem 2: Triton kernel 无法 import（包名冲突）

**现象**：
```
No module named 'triton.language'
```
**原因**：`operators/svdquant/triton/__init__.py` 遮蔽了系统的 `triton` 包。
当 Python 解析 `operators.svdquant.triton.kernel` 时，将 `operators/svdquant/triton/` 视为 `triton` 包，
`kernel.py` 中 `import triton.language as tl` 就找不到系统的 triton 了。

**解法**：删除 `__init__.py`，让 Python 以 namespace package 方式处理，系统 triton 照常可用。
（参考：项目中其他 operator 的 `triton/` 目录也无 `__init__.py`）

### Problem 3: Triton kernel LoRA 结果错误（最大差 ~0.6）

**现象**：GEMM 部分正确（max_err ≈ 0.02），但加上 LoRA 后误差跳到 0.6。

**排查过程**：
1. 隔离 GEMM 和 LoRA 分别测试 → LoRA 部分出错
2. 单独测试 `X @ lora_down` 和 `lora_act @ lora_up^T` → 均正确
3. 打印 `lora_down.stride()` → 发现 stride = **(1, 512) 而非 (32, 1)**！

**根本原因**：
`torch.linalg.svd` 返回的 `U` 矩阵是 `(K, K)` 的列主序张量。
`lora_down = U[:, :r] * S_sqrt` 在切列时保持非连续布局：
```
lora_down.shape  = (512, 32)
lora_down.stride = (1, 512)   ← 列主序！
```
Triton kernel 假设 row-major 布局（stride = (32, 1)），地址计算 `k * R + r` 出错。

**解法**：在 Python wrapper 中调用 `.contiguous()`：
```python
lora_down = lora_down.contiguous()  # 确保 row-major: stride=(rank, 1)
```

### Problem 4: `tl.clamp` 和 `tl.extra.cuda.libdevice.llrint` 问题

**现象**：
```
at 69:14: x_q = tl.clamp(x_q, -8, 7)  →  compilation error
```
**原因**：Triton 3.4 中 `tl.clamp` 有一个额外的 `propagate_nan` 参数，`llrint` 返回 int64 tensor，
与后续 FP32 操作不兼容。

**解法**：
```python
# 用 floor + 0.5 实现 round（避免 libdevice）
x_q = tl.floor(x_scaled + 0.5)
# 用 minimum/maximum 实现 clamp
x_q = tl.minimum(tl.maximum(x_q, -8.0), 7.0)
```

### Problem 5: `tl.dot(a, b, acc)` accumulator 语义变化

**现象**：使用三参数 `tl.dot(x_blk, ld_blk, lora_acc)` 时，结果比预期少约 10-20%。

**原因**：Triton 3.x 中三参数 `tl.dot(a, b, acc)` 的行为与旧版不同。
实测使用时累加不正确（原因可能是 dtype 转换时机）。

**解法**：改用标准 `+=` 累加模式：
```python
lora_acc += tl.dot(x_blk, ld_blk)  # 替代 lora_acc = tl.dot(a, b, lora_acc)
```

### Problem 6: cuda-python 版本不兼容（CuTe DSL 无法运行）

**现象**：
```
cutlass.base_dsl.common.DSLCudaRuntimeError: cudaErrorInsufficientDriver (error code: 35)
- CUDA_TOOLKIT_PATH: not set
- Target SM ARCH: not set
```

**根本原因**：cuda-python 包版本（13.2.0）与系统 NVIDIA 驱动（535.161.08，对应 CUDA 12.8）不兼容。
cuda-python 13.x 要求 CUDA driver ≥ 13.0（驱动 ≥ 545），当前驱动版本不够新。
错误信息中的"SM ARCH unknown"是误导性的，实际失败在 `cuLibraryLoadData()` 步骤。

**解法**：降级 cuda-python 到 ≤12.9，与驱动 CUDA 12.8 匹配：
```bash
pip install "cuda-python<=12.9" --force-reinstall
```

### Problem 7: CuTe DSL kernel launch 语法错误

**现象**：
```
TypeError: 'function' object is not subscriptable
```
**原因**：错误使用了 Triton 风格的 `kernel[grid, block](args)` launch 语法。
CuTe DSL 的正确语法是：
```python
kernel(args).launch(grid=(...), block=(...))
```

**解法**：参照项目中已有的 `operators/matmul/cute/kernel.py` 实现，
在 `@cute.jit` 函数内调用 `.launch()`。

### Problem 8: CuTe DSL 类型不匹配

**现象**：
```
ValueError: Type mismatch, store Float32 (-> Float32) to Tensor with element type Float16
```
**原因**：Python `0.0` 字面量和 `acc` 累加器在 CuTe DSL 中默认是 Float32，
无法直接存储到 Float16 tensor，也不能直接赋值给 Float16 smem。

**解法**：
```python
# 存储到 FP16 输出 tensor
Y[row, col] = cute.Float16(acc + lora_sum)

# 零初始化 FP16 smem
sX[tidy, tidx] = cute.Float16(0.0)
```

### Problem 9: CuTe V2 共享内存索引错误（sW 写入维度混乱）

**现象**：V2 kernel 运行结果错误（max_err ≈ 1.22），V1 正确（max_err ≈ 0.11）。

**原因**：V2 中 sW 的写入时用了错误的 thread 维度：
- 原代码：`sW[tidy, tidx] = Q_W[col, k_abs]`（col 依赖 `tidx`，但 smem 行用了 `tidy`）
- 正确逻辑：每行 thread（`tidx` 轴）加载同一 k 的不同 col 的权重，
  应让 `col_w = bidx*BN + tidy` 作为权重的列索引，`tidx` 作为 K 偏移

**解法**：
```python
col_w = bidx * BLOCK_N + tidy       # W 的列 = 当前 block 列 + tidy
sW[tidy, tidx] = dequant(Q_W[col_w, k_start + tidx])
# 读时：acc += sX[tidy, k] * sW[tidx, k]
#       x 行用 tidy，w 列用 tidx
```

---

## 5. Benchmark 结果

**测试环境**：NVIDIA H20, SM 9.0 (Hopper), 102GB HBM3

### 5.1 延迟对比（ms）

| Config | FP16 Baseline | PyTorch | Triton | CuTe V1 | CuTe V2 |
|--------|:---:|:---:|:---:|:---:|:---:|
| M=64, K=512, N=512 | 0.025 | 0.306 | 0.077 | 0.504 | 0.216 |
| M=256, K=2048, N=2048 | 0.041 | 0.410 | 0.133 | 11.82 | 0.937 |
| M=1024, K=4096, N=4096 | 0.323 | 2.072 | 0.951 | 230.7 | 11.10 |

### 5.2 TFLOPS 对比

| Config | FP16 Baseline | PyTorch | Triton | CuTe V1 | CuTe V2 |
|--------|:---:|:---:|:---:|:---:|:---:|
| M=64, K=512, N=512 | 1.48 | 0.12 | 0.49 | 0.07 | 0.18 |
| M=256, K=2048, N=2048 | 54.10 | 5.41 | 16.63 | 0.19 | 2.36 |
| M=1024, K=4096, N=4096 | 108.10 | 16.84 | 36.69 | 0.15 | 3.14 |

### 5.3 Triton vs PyTorch 加速比

| Config | 加速比 |
|--------|:---:|
| M=64, K=512, N=512 | **3.96x** |
| M=256, K=2048, N=2048 | **3.08x** |
| M=1024, K=4096, N=4096 | **2.18x** |

### 5.4 内存压缩率

| Config | FP16 内存 | SVDQuant 内存 | 压缩率 |
|--------|:---:|:---:|:---:|
| K=512, N=512 | 0.6 MB | 0.3 MB | 1.85x |
| K=2048, N=2048 | 10.0 MB | 4.6 MB | 2.15x |
| K=4096, N=4096 | 48.0 MB | 27.1 MB | 1.77x |

### 5.5 性能分析

**为什么 CuTe V1 比 PyTorch 还慢？**
- V1 是 naive per-element 实现，每个 thread 串行遍历所有 K（无 Tensor Core，无内存合并）
- K=4096 时每 thread 做 4096 次迭代，严重的串行瓶颈
- 每次 `Y[col, k]` 访问都是非合并 HBM read，带宽利用极差

**为什么 CuTe V2 比 V1 快 20x？**
- 共享内存分块后，K 维分成 32-element tile，内层串行循环仅 32 次
- sX/sW 存入 smem 后，所有 thread 复用，HBM 读取降低 32x

**CuTe V2 为什么还比 Triton 慢？**
- CuTe V2 不使用 Tensor Core（串行 FMA，不是 `mma.sync`）
- Triton `tl.dot` 会被编译器映射到硬件 Tensor Core（WMMA/MMA 指令）
- 真正发挥 CuTe 优势需要显式使用 `cute.arch.mma_*` 指令（需要更复杂的 layout 配合）
1. 当前实现用 FP16 模拟 INT4 GEMM（量化→反量化→FP16 tl.dot），没有使用真正的 INT4 MMA 指令
2. LoRA 额外增加了两次矩阵乘法（`X @ lora_down` 和 `lora_act @ lora_up`）
3. 真正的 W4A4 INT4 MMA 需要 CUTLASS 3.x 的 `SM90 WGMMA` 指令（见 nunchaku CUDA kernel）

**Triton 比 PyTorch 快 2-3x 的原因**：
- Kernel fusion 减少了 HBM 往返（smooth + quant + GEMM 合并）
- tl.dot 利用 Tensor Core（FP16 tl.dot 有 SM 加速）
- 减少了中间张量分配和内存复制

**进一步优化方向**：
- 使用 inline PTX `wmma.mma.sync.aligned.m16n8k16.s8.s8.s32.s32` 真正的 INT4 MMA
- 异步流水线：LDGSTS + WarpGroup MMA（Hopper-specific）
- 使用 CUTLASS 3.x Python binding 的 INT4 GEMM tile

---

## 6. 与 nunchaku 的对比分析

### 6.1 nunchaku API 设计

**量化 API** (`nunchaku/ops/quantize.py`)：
```python
def svdq_quantize_w4a4_act_fuse_lora_cuda(
    input,          # (M, K) BF16/FP16
    output,         # (M_pad, K//2) uint8 packed INT4
    oscales,        # (K//64, M_pad) FP16 activation scales
    lora_down,      # (K, R) FP16
    lora_act_out,   # (M_pad, R) FP32 LoRA activation output
    smooth,
    fuse_glu, fp4, pad_size
) -> (output, oscales, lora_act_out)
```

关键设计点：
- **M_pad**: batch size 向上取整为 `pad_size=256` 的倍数（coalesced memory access）
- **Scales 布局**: `(K//G, M)` 列主序（激活），便于按 K 维 coalesced 读取
- **融合 GLU**: 可选 gate-linear unit 融合（用于 SwiGLU 激活）
- **支持 NVFP4**: 除 INT4 外支持 group_size=16 的 FP4 量化

### 6.2 我们的实现 vs nunchaku

| 特性 | 我们的实现 | nunchaku |
|------|:---:|:---:|
| 量化位宽 | INT4 (模拟) | INT4 + NVFP4 |
| Padding | 无 | M_pad = ceil(M/256)*256 |
| Scales 布局 | (M, K//G) 行主序 | (K//G, M) 列主序 |
| INT4 MMA | FP16 tl.dot 模拟 | 真正 INT4 MMA（SM 80-89） |
| 融合 GLU | 无 | 支持 |
| SM 支持 | 全平台 | 75/80/86/89/120a/121a |
| LoRA 精度 | FP16/FP32 | FP32 (lora_act_out) |

### 6.3 nunchaku CUDA kernel 关键技术

参考 `nunchaku/src/kernels/zgemm/gemm_w4a4.cuh`：
- 使用 CUTLASS 3.x 的 `CollectiveMma` 实现 W4A4 GEMM
- SM 80+: 使用 `wmma::mma_sync` INT8 MMA (4-bit 通过 packing 模拟)
- SM 89: 可能支持原生 INT4 MMA 指令
- Scale dequant 与 GEMM 流水线并行（online dequant strategy）

### 6.4 学习收获

通过本次实现，深入理解了：
1. SVDQuant 的数学设计（为什么 SVD 能有效处理 outlier）
2. INT4 per-group 量化的实现细节（scale 计算、打包格式）
3. Triton kernel 的内存布局陷阱（contiguous 的重要性）
4. tl.dot 的正确使用（accumulator 语义、float32 累加）
5. Kernel fusion 的收益分析（2-3x 相比 PyTorch eager）

---

## 附录：运行命令

```bash
# 工作目录：gpu-kernel-svdquant/

# 正确性测试
python operators/svdquant/test.py -v

# 大矩阵测试
python operators/svdquant/test.py --large -v

# 独立 benchmark
python operators/svdquant/benchmark.py --warmup 10 --repeat 100

# 全局 benchmark（仅 svdquant）
python benchmarks/benchmark.py --op svdquant
```
