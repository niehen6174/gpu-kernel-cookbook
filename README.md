# GPU Kernel Lab

一个系统化的 **GPU Kernel 学习项目**，实现常见深度学习算子并对比多种 GPU 编程框架。

## 项目结构

```
gpu-kernel-lab/
├── README.md
├── docs/                    # 详细的算子设计文档
│   ├── vector_add.md        # CUDA 线程模型、coalesced 访问
│   ├── transpose.md         # Memory coalescing、shared memory、bank conflict
│   ├── softmax.md           # Reduction、online softmax、warp shuffle
│   ├── layernorm.md         # Welford 算法、两级规约
│   ├── matmul.md            # Shared memory tiling、Roofline 分析
│   └── attention.md         # Flash Attention、IO-Aware 算法
│
├── common/
│   ├── utils.py             # benchmark 工具、性能指标计算
│   ├── tensor_utils.py      # 张量生成工具
│   └── check.py             # 正确性验证
│
├── benchmarks/
│   └── benchmark.py         # 统一 benchmark 入口
│
├── operators/
│   ├── vector_add/          # ⭐ GPU 线程模型
│   ├── transpose/           # ⭐⭐ Memory Coalescing
│   ├── softmax/             # ⭐⭐⭐ Reduction
│   ├── layernorm/           # ⭐⭐⭐ Warp Reduction
│   ├── matmul/              # ⭐⭐⭐⭐ Shared Memory Tiling
│   └── attention/           # ⭐⭐⭐⭐⭐ Fused Kernel (Flash Attention)
│
└── scripts/
    ├── build_all.sh
    └── run_all_tests.sh
```

## 算子列表

| 算子 | 难度 | 核心技术 | 文档 |
|------|------|---------|------|
| Vector Add | ⭐ | CUDA 线程模型，float4 向量化 | [vector_add.md](docs/vector_add.md) |
| Transpose | ⭐⭐ | Shared Memory，Bank Conflict | [transpose.md](docs/transpose.md) |
| Softmax | ⭐⭐⭐ | Reduction，Online Softmax | [softmax.md](docs/softmax.md) |
| LayerNorm | ⭐⭐⭐ | Welford 算法，Warp Reduction | [layernorm.md](docs/layernorm.md) |
| Matmul | ⭐⭐⭐⭐ | Shared Memory Tiling，Roofline | [matmul.md](docs/matmul.md) |
| Attention | ⭐⭐⭐⭐⭐ | Flash Attention，IO-Aware | [attention.md](docs/attention.md) |

## 每个算子的实现

每个算子包含：
- **CUDA**：从 naive 到优化的手写 kernel（含详细注释）
- **Triton**：Python DSL 实现
- **CuTe DSL**：CUTLASS 的 Python 接口（部分算子）
- **PyTorch**：baseline（用于正确性验证和性能对比）

## 快速开始

### 环境要求

```bash
CUDA >= 11.8
PyTorch >= 2.0
Triton >= 2.1
cutlass (可选，用于 CuTe DSL)
```

### 编译 CUDA Kernels

```bash
# 编译所有（默认 sm_80 = A100/RTX3090）
CUDA_ARCH=sm_80 bash scripts/build_all.sh

# 编译指定算子
cd operators/matmul/cuda && bash build.sh
```

常见 CUDA 架构：
| GPU | 架构 | `CUDA_ARCH` |
|-----|------|-------------|
| V100 | Volta | `sm_70` |
| A100 | Ampere | `sm_80` |
| RTX 30xx | Ampere | `sm_86` |
| H100 | Hopper | `sm_90` |
| RTX 40xx | Ada Lovelace | `sm_89` |

### 运行测试

```bash
# 测试所有算子
bash scripts/run_all_tests.sh

# 测试单个算子
python operators/matmul/test.py
python operators/attention/test.py
```

### 运行 Benchmark

```bash
# 所有算子
python benchmarks/benchmark.py

# 指定算子
python benchmarks/benchmark.py --op matmul

# 保存结果
python benchmarks/benchmark.py --save
```

## 学习路径

```
1. vector_add   →  理解 CUDA 线程层次（grid/block/thread/warp）
2. transpose    →  理解内存访问模式（coalescing、shared memory、bank conflict）
3. softmax      →  理解 GPU reduction（tree reduce、warp shuffle）
4. layernorm    →  复合 reduction（Welford 算法、两级规约）
5. matmul       →  理解 compute-bound 优化（tiling、数据复用）
6. attention    →  理解 IO-aware 算法（Flash Attention）
```

编程框架学习路径：
```
CUDA (手写) → Triton (Python DSL) → CuTe DSL → CUTLASS
```

## 文档说明

每份文档包含：
1. 算子数学定义
2. GPU 并行策略
3. 各版本 kernel 设计与分析
4. Roofline / 性能分析
5. 关键概念详解（含代码示例）
6. 性能 benchmark 参考数据
7. 学习要点总结

## 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Triton 教程](https://triton-lang.org/main/getting-started/tutorials/)
- [CUTLASS 文档](https://github.com/NVIDIA/cutlass)
- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/)

## 相关项目

- [`../LeetGPU`](../LeetGPU)：参考实现（vector add、matmul、transpose、softmax、attention）
- [`../cutlass`](../cutlass)：CUTLASS 库源码
