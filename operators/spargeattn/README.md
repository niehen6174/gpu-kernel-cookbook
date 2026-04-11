# SpargeAttention — Block-Sparse Quantized Attention

SpargeAttention (Sparse + SageAttention) 在 SageAttention 的 INT8/FP8 量化注意力基础上，增加了**块级稀疏**机制。
通过预测哪些 K/V 块对注意力输出贡献最大，跳过不重要的块，在保持精度的同时实现显著加速。

> **论文**: [SpargeAttn: Accurate and Training-free Sparse Attention Accelerating Any Model Inference](https://arxiv.org/abs/2502.18137) (ICML 2025)

## 核心思想

```
传统 Attention:  对所有 K 块计算 → O(N²)
SpargeAttention: 预测重要块 → 仅计算 top-K 块 → O(topk × N²)
```

**三阶段流程**:
1. **稀疏预测** (Triton): 块级均值池化 + 相似度评分 + Top-K 选择 → 生成 LUT
2. **量化** (Triton + CUDA): Q/K → INT8, V → FP8 (SM90) / FP16 (SM80)
3. **块稀疏注意力** (CUDA): 只计算 LUT 指定的 K/V 块

## 架构支持

| 架构 | QK 精度 | SV 精度 | 块大小 | 内存加载 |
|------|---------|---------|--------|----------|
| SM80 (A100) | INT8 (MMA m16n16k32) | FP16 (MMA m16n16k16) | CTA_Q=128, CTA_K=64 | cp.async |
| SM90 (H100/H20) | INT8 (WGMMA s8s8s32) | FP8 (WGMMA f8f8f32) | CTA_Q=64, CTA_K=128 | TMA |

## 快速开始

### 编译

```bash
cd operators/spargeattn/cuda
pip install ninja   # 加速编译
python setup.py build_ext --inplace
```

### 使用

```python
from spargeattn import spas_sage2_attn_meansim_topk_cuda

# 即插即用，替代 torch.nn.functional.scaled_dot_product_attention
attn_output = spas_sage2_attn_meansim_topk_cuda(q, k, v, topk=0.5, is_causal=False)
```

### API 参考

#### `spas_sage2_attn_meansim_topk_cuda` (推荐)

```python
spas_sage2_attn_meansim_topk_cuda(
    q, k, v,              # [B, H, N, D] 或 [B, N, H, D]
    topk=0.5,             # 保留的 K 块比例 (0~1)，越小越快但精度越低
    is_causal=False,      # 是否使用因果掩码
    scale=None,           # softmax 缩放因子，默认 1/sqrt(D)
    smooth_k=True,        # 是否减去 K 均值以改善量化
    pvthreshd=50,         # PV 阈值，跳过不重要的 P*V 计算
    tensor_layout="HND",  # "HND" 或 "NHD"
    return_sparsity=False # 是否返回稀疏度
)
```

#### `block_sparse_sage2_attn_cuda` (自定义掩码)

```python
block_sparse_sage2_attn_cuda(
    q, k, v,
    mask_id=mask,   # [B, H, ⌈N/BLKQ⌉, ⌈N/BLKK⌉] 的 0/1 掩码
    tensor_layout="HND"
)
```

## 运行测试

```bash
cd operators
python spargeattn/test.py            # 快速正确性测试 + 基准
python spargeattn/benchmark.py       # 完整性能基准测试
```

## 项目结构

```
spargeattn/
├── __init__.py           # 导出 API
├── core.py               # 高层 Python API (自动检测 SM80/SM90)
├── utils.py              # Triton 内核: 稀疏预测 + 量化 + LUT 生成
├── test.py               # 快速测试
├── benchmark.py          # 完整基准测试
├── README.md             # 本文档
└── cuda/
    ├── setup.py           # 构建脚本
    ├── build.sh           # 便捷构建
    ├── csrc/
    │   ├── *.cuh          # 共享 CUDA 头文件
    │   ├── qattn/         # 块稀疏注意力内核
    │   │   ├── qk_int_sv_f16_cuda_sm80.{cu,cuh}  # SM80 内核
    │   │   ├── qk_int_sv_f8_cuda_sm90.{cu,cuh}   # SM90 内核
    │   │   ├── attn_utils.cuh                      # 在线 softmax / PV 阈值
    │   │   ├── decl.cuh                            # 模板前向声明
    │   │   ├── instantiations_sm80/autogen.py      # SM80 模板实例化
    │   │   └── instantiations_sm90/autogen.py      # SM90 模板实例化
    │   └── fused/         # V 量化内核 (SM90 FP8 路径)
    │       ├── fused.cu   # 转置 + 填充 + FP8 量化
    │       └── fused.h
    └── test.py            # 内核级测试
```

## 算法细节

### 稀疏预测 (Triton)

对每个 Q/K 块执行三个操作 (融合到单个 Triton 内核):
1. **均值池化**: 块内所有向量取平均 → 单个代表向量
2. **块内相似度**: 计算块内向量的平均余弦相似度 (高相似度 = 均匀 = 可能不重要)
3. **INT8 量化**: 顺便完成 per-block INT8 量化

然后:
4. 计算块级注意力分数: `pooled_q @ pooled_k.T * scale`
5. Softmax → 排序 → 选择 top-K
6. 生成 delta 编码的 LUT (查找表)

### LUT 编码

LUT 使用**增量编码** (delta encoding)，存储距离上一个选中块的偏移量，而非绝对索引:
```
block_map: [1, 0, 1, 0, 0, 1]
LUT:       [0, 2, 3]  → 第0块, 第0+2=2块, 第2+3=5块
valid_block_num: 3
```

### PV 阈值优化

在计算完 QK^T 后，通过检测当前块的 max score 变化量来决定是否值得执行 P*V:
- 如果 `max_diff + threshold > 0`，执行 P*V
- 否则跳过 (因为 attention 权重太小，贡献可忽略)

## 引用

```bibtex
@inproceedings{zhang2025spargeattn,
  title={Spargeattn: Accurate sparse attention accelerating any model inference},
  author={Zhang, Jintao and Xiang, Chendong and Huang, Haofeng and Wei, Jia and Xi, Haocheng and Zhu, Jun and Chen, Jianfei},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```
