"""
FP8 量化算子包

包含两种量化方案：
  Scheme A：Per-Tensor FP8（整个 tensor 共享一个 scale）
  Scheme B：Per-Block FP8（128-element 分组量化，每组独立 scale）

各 Backend：
  - pytorch/     : PyTorch baseline
  - triton/      : Triton quant kernel + FP8 GEMM
  - cute/        : CuTe Python DSL（SM90）
  - cutlass_fp8/ : CUTLASS 3.x C++（V1 per-tensor, V2 per-block）
"""
