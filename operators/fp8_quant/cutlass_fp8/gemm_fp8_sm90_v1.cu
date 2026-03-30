/**
 * CUTLASS 3.x SM90 FP8 GEMM — V1 (Per-Tensor Scale, Scalar Epilogue)
 *
 * FP8 E4M3 x FP8 E4M3 → BF16，使用 Hopper WGMMA 指令。
 * Epilogue: LinearCombination，D = alpha * acc
 * 其中 alpha = inv_scale_a * inv_scale_b（Python 侧预先计算好的标量）
 *
 * Per-Tensor 方案：整个矩阵共享一对 scale，epilogue 只需乘一个标量。
 *
 * 编译要求：
 *   -gencode arch=compute_90a,code=sm_90a
 *   -DCUTLASS_ARCH_MMA_SM90_SUPPORTED
 *   -DCUTE_ARCH_MMA_SM90A_ENABLED
 *   -std=c++17
 */

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"

#include <torch/extension.h>

// ---------------------------------------------------------------------------
// 类型与布局别名
// ---------------------------------------------------------------------------

using ElementA_v1    = cutlass::float_e4m3_t;   // FP8 E4M3
using ElementB_v1    = cutlass::float_e4m3_t;   // FP8 E4M3
using ElementC_v1    = cutlass::bfloat16_t;     // C（beta=0，不读）
using ElementD_v1    = cutlass::bfloat16_t;     // 输出 BF16
using ElementAcc_v1  = float;                   // FP8 → FP32 accumulator
using ElementComp_v1 = float;                   // Epilogue 计算精度

using LayoutA_v1 = cutlass::layout::RowMajor;
using LayoutB_v1 = cutlass::layout::ColumnMajor;   // B 存为 (N,K) row-major = col-major (K,N)
using LayoutC_v1 = cutlass::layout::RowMajor;
using LayoutD_v1 = cutlass::layout::RowMajor;

using ArchTag_v1  = cutlass::arch::Sm90;
using OpClass_v1  = cutlass::arch::OpClassTensorOp;

// FP8 TileShape sweet-spot on Hopper: 128x256x64
// FP8 accumulation path: K tile = 64 (vs INT8 = 128)
using TileShape_v1    = cute::Shape<cute::_128, cute::_256, cute::_64>;
using ClusterShape_v1 = cute::Shape<cute::_1,   cute::_1,   cute::_1>;

// FP8 = 1 byte → 128bit / 8bit = 16 elements alignment
constexpr int AlignA_v1 = 128 / cutlass::sizeof_bits<ElementA_v1>::value;  // 16
constexpr int AlignB_v1 = 128 / cutlass::sizeof_bits<ElementB_v1>::value;  // 16
constexpr int AlignC_v1 = 128 / cutlass::sizeof_bits<ElementC_v1>::value;  // 8
constexpr int AlignD_v1 = 128 / cutlass::sizeof_bits<ElementD_v1>::value;  // 8

// ---------------------------------------------------------------------------
// CollectiveBuilder：FP8 SS 路径
// ---------------------------------------------------------------------------

using CollectiveEpilogue_v1 = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag_v1, OpClass_v1,
    TileShape_v1, ClusterShape_v1,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc_v1, ElementComp_v1,
    ElementC_v1, LayoutC_v1, AlignC_v1,
    ElementD_v1, LayoutD_v1, AlignD_v1,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop_v1 = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag_v1, OpClass_v1,
    ElementA_v1, LayoutA_v1, AlignA_v1,
    ElementB_v1, LayoutB_v1, AlignB_v1,
    ElementAcc_v1,
    TileShape_v1, ClusterShape_v1,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue_v1::SharedStorage))
    >,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel_v1 = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop_v1,
    CollectiveEpilogue_v1
>;

using Gemm_v1 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_v1>;

// ---------------------------------------------------------------------------
// Python 接口
// ---------------------------------------------------------------------------

/**
 * fp8_per_tensor_gemm_v1
 *
 * FP8 Per-Tensor GEMM，输出 BF16。
 * Epilogue: D[i,j] = alpha * acc[i,j]
 * 其中 alpha = inv_scale_a * inv_scale_b（Python 侧传入）
 *
 * @param a          (M, K)  float8_e4m3fn  row-major  激活
 * @param b          (N, K)  float8_e4m3fn  row-major（= col-major (K,N)，供 CUTLASS B operand）
 * @param out        (M, N)  bfloat16  pre-allocated 输出
 * @param alpha      float   inv_scale_a * inv_scale_b（Python 侧预乘）
 * @param bias_opt   (N,) bfloat16 or None — per-col bias（加到 epilogue 输出上）
 */
void fp8_per_tensor_gemm_v1(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor out,
    float alpha,
    c10::optional<torch::Tensor> bias_opt
) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && out.is_cuda(),
                "fp8_per_tensor_gemm_v1: all tensors must be on CUDA");
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn, "a must be float8_e4m3fn");
    TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn, "b must be float8_e4m3fn");
    TORCH_CHECK(out.dtype() == torch::kBFloat16,    "out must be bfloat16");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && out.is_contiguous(),
                "fp8_per_tensor_gemm_v1: tensors must be contiguous");

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(0));

    TORCH_CHECK(b.size(1) == K, "b K-dim mismatch");
    TORCH_CHECK(out.size(0) == M && out.size(1) == N, "out shape mismatch");

    using StrideA = typename Gemm_v1::GemmKernel::StrideA;
    using StrideB = typename Gemm_v1::GemmKernel::StrideB;
    using StrideC = typename Gemm_v1::GemmKernel::StrideC;
    using StrideD = typename Gemm_v1::GemmKernel::StrideD;

    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    typename Gemm_v1::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            reinterpret_cast<ElementA_v1*>(a.data_ptr()), stride_a,
            reinterpret_cast<ElementB_v1*>(b.data_ptr()), stride_b
        },
        {
            {alpha, 0.0f},      // alpha = inv_scale_a * inv_scale_b, beta = 0
            nullptr, stride_c,  // C not used (beta=0)
            reinterpret_cast<ElementD_v1*>(out.data_ptr()), stride_d
        }
    };

    // Optional: add bias via simple post-GEMM op (not fused in V1)
    // bias handling done in Python layer

    Gemm_v1 gemm_op;

    auto status = gemm_op.can_implement(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "FP8 V1 SM90: can_implement failed, status=",
                static_cast<int>(status));

    const size_t ws_bytes = Gemm_v1::get_workspace_size(args);
    auto workspace = torch::empty(
        {static_cast<int64_t>(ws_bytes)},
        a.options().dtype(torch::kUInt8)
    );

    status = gemm_op.initialize(args, workspace.data_ptr());
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "FP8 V1 SM90: initialize failed, status=",
                static_cast<int>(status));

    status = gemm_op.run();
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "FP8 V1 SM90: run failed, status=",
                static_cast<int>(status));
}

// ---------------------------------------------------------------------------
// pybind11 模块注册
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUTLASS SM90 FP8 GEMM V1: Per-Tensor Scale (scalar epilogue) → BF16";

    m.def("fp8_per_tensor_gemm_v1",
          &fp8_per_tensor_gemm_v1,
          "FP8 Per-Tensor GEMM (CUTLASS 3.x SM90 WGMMA).\n"
          "  a    : (M, K) float8_e4m3fn row-major\n"
          "  b    : (N, K) float8_e4m3fn row-major (= col-major for CUTLASS B)\n"
          "  out  : (M, N) bfloat16 pre-allocated\n"
          "  alpha: float = inv_scale_a * inv_scale_b\n"
          "  bias : (N,) bfloat16 or None\n"
          "Computes: out[i,j] = bf16(alpha * sum_k a[i,k] * b[j,k])",
          py::arg("a"),
          py::arg("b"),
          py::arg("out"),
          py::arg("alpha") = 1.0f,
          py::arg("bias")  = c10::nullopt);
}
