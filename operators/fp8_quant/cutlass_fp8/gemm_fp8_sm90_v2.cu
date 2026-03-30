/**
 * CUTLASS 3.x SM90 FP8 GEMM — V2 (Per-Block Scale, EVT per-row/col 近似)
 *
 * FP8 E4M3 x FP8 E4M3 → BF16，使用 Hopper WGMMA 指令。
 *
 * Per-Block 方案精确计算需要按 group 乘 scale 对，但 EVT 直接支持 per-row + per-col。
 * 近似方法：
 *   act_scale[i]  = mean(act_inv_scales[i, :])   ← per-row（ColBroadcast）
 *   wgt_scale[j]  = mean(wgt_inv_scales[:, j])   ← per-col（RowBroadcast）
 *   D[i,j] = bf16(act_scale[i] * acc[i,j] * wgt_scale[j] + bias[j])
 *
 * 精度略低于精确 per-block，但 epilogue 性能最优（无 group 展开）。
 *
 * EVT 结构与 W8A8 V2 完全一致，仅 ElementA/B 改为 float_e4m3_t：
 *   inner: Sm90ColBroadcast(act_scale) × acc × Sm90RowBroadcast(wgt_scale) + bias
 *   outer: 0×C + inner  (实际输出 inner，cast to BF16)
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
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/numeric_conversion.h"
#include "cute/tensor.hpp"

#include <torch/extension.h>

// ---------------------------------------------------------------------------
// 类型与布局别名
// ---------------------------------------------------------------------------

using ElementA_v2    = cutlass::float_e4m3_t;
using ElementB_v2    = cutlass::float_e4m3_t;
using ElementC_v2    = cutlass::bfloat16_t;
using ElementD_v2    = cutlass::bfloat16_t;
using ElementAcc_v2  = float;
using ElementComp_v2 = float;

using LayoutA_v2 = cutlass::layout::RowMajor;
using LayoutB_v2 = cutlass::layout::ColumnMajor;
using LayoutC_v2 = cutlass::layout::RowMajor;
using LayoutD_v2 = cutlass::layout::RowMajor;

using ArchTag_v2  = cutlass::arch::Sm90;
using OpClass_v2  = cutlass::arch::OpClassTensorOp;

// FP8 TileShape sweet-spot
using TileShape_v2    = cute::Shape<cute::_128, cute::_256, cute::_64>;
using ClusterShape_v2 = cute::Shape<cute::_1,   cute::_1,   cute::_1>;

constexpr int AlignA_v2 = 128 / cutlass::sizeof_bits<ElementA_v2>::value;  // 16
constexpr int AlignB_v2 = 128 / cutlass::sizeof_bits<ElementB_v2>::value;  // 16
constexpr int AlignC_v2 = 128 / cutlass::sizeof_bits<ElementC_v2>::value;  // 8
constexpr int AlignD_v2 = 128 / cutlass::sizeof_bits<ElementD_v2>::value;  // 8

static constexpr cutlass::FloatRoundStyle RoundStyle_v2 =
    cutlass::FloatRoundStyle::round_to_nearest;

// ---------------------------------------------------------------------------
// Custom EVT（与 W8A8 V2 完全相同结构，复用框架）
//
// D[i,j] = bf16( act_scale[i] * acc[i,j] * wgt_scale[j] + bias[j] )
//
// 注意：InnerEVT 用 multiply_add 而非 homogeneous_multiply_add，
// 因为三个操作数类型不同：
//   child0: float (act_scale)
//   child1: float (acc)
//   child2: float (wgt_scale, RowBroadcast 输出 float)
// Inner compute: Sm90Compute<multiply_add, float, float>
//   = act_scale * acc + 0 → 再 × wgt_scale
//
// 简化方案：使用两步 EVT：
//   step1: inner_scale = act_scale[i] * wgt_scale[j]      (per-row × per-col)
//   step2: D[i,j] = bf16(inner_scale * acc + bias[j])
//
// CUTLASS EVT 嵌套结构（bottom-up）：
//   InnerProduct: homogeneous_multiply_add(float)
//     child0: Sm90ColBroadcast<act_scale>  ← per-row float32
//     child1: Sm90AccFetch                 ← accumulator
//     child2: Sm90RowBroadcast<wgt_scale>  ← per-col float32 (bias 用 BF16)
//   Outer: 0×C + inner → cast to BF16
// ---------------------------------------------------------------------------

using EpilogueSchedule_v2 = cutlass::epilogue::TmaWarpSpecializedCooperative;

using EpilogueDescriptor_v2 = cutlass::epilogue::collective::detail::EpilogueDescriptor<
    TileShape_v2,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementC_v2,
    ElementD_v2,
    EpilogueSchedule_v2
>;

using EpilogueTile_v2 = typename EpilogueDescriptor_v2::EpilogueTile;
static constexpr int RowBcastStages_v2 =
    cute::ceil_div(EpilogueDescriptor_v2::StagesC,
        cute::size(cute::shape_div(
            cute::take<0,2>(TileShape_v2{}),
            EpilogueTile_v2{}
        ))
    ) + 1;

// Per-row act_scale: ColBroadcast（每行一个 float32）
using ActScaleBroadcast_v2 = cutlass::epilogue::fusion::Sm90ColBroadcast<
    0,
    TileShape_v2,
    float,
    cute::Stride<cute::_1, cute::_0, cute::_0>,
    128 / cutlass::sizeof_bits<float>::value
>;

// Per-col wgt_scale: RowBroadcast（每列一个 float32）
using WgtScaleBroadcast_v2 = cutlass::epilogue::fusion::Sm90RowBroadcast<
    RowBcastStages_v2,
    TileShape_v2,
    float,
    cute::Stride<cute::_0, cute::_1, cute::_0>,
    128 / cutlass::sizeof_bits<float>::value
>;

// Per-col bias: RowBroadcast（每列一个 BF16，可选）
using BiasBroadcast_v2 = cutlass::epilogue::fusion::Sm90RowBroadcast<
    RowBcastStages_v2,
    TileShape_v2,
    cutlass::bfloat16_t,
    cute::Stride<cute::_0, cute::_1, cute::_0>,
    128 / cutlass::sizeof_bits<cutlass::bfloat16_t>::value
>;

// Inner EVT: act_scale[i] * acc * wgt_scale[j]
// 使用两级：先 act_scale × acc（ColBroadcast × AccFetch），再乘 wgt_scale
// 等效：(act_scale * acc + 0) * wgt_scale + bias
// 简化为：act_scale[i] * acc[i,j]   —— 第一 EVT
//   然后 × wgt_scale[j] + bias[j]    —— 第二 EVT（外层）

// 第一 Inner：act_scale × acc（homogeneous_multiply_add with zero bias）
using AccScaleEVT_v2 = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies,   // act_scale * acc（无 add）
        float,
        float,
        RoundStyle_v2
    >,
    ActScaleBroadcast_v2,    // child0: act_scale[i]
    cutlass::epilogue::fusion::Sm90AccFetch   // child1: accumulator
>;

// 外层：(act_scale * acc) * wgt_scale + bias → cast to BF16
using InnerEVT_v2 = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::homogeneous_multiply_add,
        float,
        float,
        RoundStyle_v2
    >,
    WgtScaleBroadcast_v2,    // child0: wgt_scale[j]
    AccScaleEVT_v2,          // child1: act_scale * acc
    BiasBroadcast_v2         // child2: bias[j]
>;

// 最终 EVT：0×C + inner → BF16
using CustomEVT_v2 = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::homogeneous_multiply_add,
        cutlass::bfloat16_t,
        float,
        RoundStyle_v2
    >,
    cutlass::epilogue::fusion::Sm90ScalarBroadcast<float>,  // beta=0
    cutlass::epilogue::fusion::Sm90SrcFetch,                // C (zero-weighted)
    InnerEVT_v2
>;

// ---------------------------------------------------------------------------
// CollectiveEpilogue & Mainloop
// ---------------------------------------------------------------------------

using CollectiveEpilogue_v2 = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag_v2, OpClass_v2,
    TileShape_v2, ClusterShape_v2,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc_v2, ElementComp_v2,
    ElementC_v2, LayoutC_v2, AlignC_v2,
    ElementD_v2, LayoutD_v2, AlignD_v2,
    EpilogueSchedule_v2,
    CustomEVT_v2
>::CollectiveOp;

using CollectiveMainloop_v2 = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag_v2, OpClass_v2,
    ElementA_v2, LayoutA_v2, AlignA_v2,
    ElementB_v2, LayoutB_v2, AlignB_v2,
    ElementAcc_v2,
    TileShape_v2, ClusterShape_v2,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue_v2::SharedStorage))
    >,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel_v2 = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop_v2,
    CollectiveEpilogue_v2
>;

using Gemm_v2 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_v2>;

// ---------------------------------------------------------------------------
// Python 接口
// ---------------------------------------------------------------------------

/**
 * fp8_per_block_gemm_v2
 *
 * FP8 Per-Block GEMM（EVT per-row/col 近似），输出 BF16。
 * Epilogue: D[i,j] = bf16(act_scale[i] * acc[i,j] * wgt_scale[j] + bias[j])
 *
 * @param a           (M, K)  float8_e4m3fn  row-major  激活
 * @param b           (N, K)  float8_e4m3fn  row-major（= col-major）权重
 * @param out         (M, N)  bfloat16  pre-allocated 输出
 * @param act_scale   (M,)    float32   per-row act scale（mean of per-group scales）
 * @param wgt_scale   (N,)    float32   per-col wgt scale（mean of per-group scales）
 * @param bias_opt    (N,)    bfloat16 or None
 */
void fp8_per_block_gemm_v2(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor out,
    torch::Tensor act_scale,
    torch::Tensor wgt_scale,
    c10::optional<torch::Tensor> bias_opt
) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && out.is_cuda(),
                "fp8_per_block_gemm_v2: a, b, out must be on CUDA");
    TORCH_CHECK(act_scale.is_cuda() && wgt_scale.is_cuda(),
                "fp8_per_block_gemm_v2: act_scale, wgt_scale must be on CUDA");
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn, "a must be float8_e4m3fn");
    TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn, "b must be float8_e4m3fn");
    TORCH_CHECK(out.dtype() == torch::kBFloat16,    "out must be bfloat16");
    TORCH_CHECK(act_scale.dtype() == torch::kFloat32, "act_scale must be float32");
    TORCH_CHECK(wgt_scale.dtype() == torch::kFloat32, "wgt_scale must be float32");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && out.is_contiguous() &&
                act_scale.is_contiguous() && wgt_scale.is_contiguous(),
                "fp8_per_block_gemm_v2: tensors must be contiguous");

    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(0));

    TORCH_CHECK(b.size(1) == K, "b K-dim mismatch");
    TORCH_CHECK(out.size(0) == M && out.size(1) == N, "out shape mismatch");
    TORCH_CHECK(act_scale.size(0) == M, "act_scale size mismatch: expected M=", M);
    TORCH_CHECK(wgt_scale.size(0) == N, "wgt_scale size mismatch: expected N=", N);

    const float* act_scale_ptr = reinterpret_cast<float const*>(act_scale.data_ptr());
    const float* wgt_scale_ptr = reinterpret_cast<float const*>(wgt_scale.data_ptr());

    cutlass::bfloat16_t const* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined() && bias_opt.value().numel() > 0) {
        torch::Tensor bias = bias_opt.value();
        TORCH_CHECK(bias.is_cuda() && bias.dtype() == torch::kBFloat16 && bias.is_contiguous(),
                    "bias must be CUDA bfloat16 contiguous");
        TORCH_CHECK(bias.size(0) == N, "bias size mismatch");
        bias_ptr = reinterpret_cast<cutlass::bfloat16_t const*>(bias.data_ptr());
    }

    using StrideA = typename Gemm_v2::GemmKernel::StrideA;
    using StrideB = typename Gemm_v2::GemmKernel::StrideB;
    using StrideC = typename Gemm_v2::GemmKernel::StrideC;
    using StrideD = typename Gemm_v2::GemmKernel::StrideD;

    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    // EVT arguments（与 W8A8 V2 结构一致，只换 tensor 指针）
    // CustomEVT_v2 = Sm90EVT<outer_compute, ScalarBcast, SrcFetch, InnerEVT>
    // InnerEVT_v2  = Sm90EVT<inner_compute, WgtScaleBcast, AccScaleEVT, BiasBcast>
    // AccScaleEVT  = Sm90EVT<multiplies, ActScaleBcast, AccFetch>
    typename CollectiveEpilogue_v2::Arguments epilogue_args{
        {   // thread: CustomEVT_v2 arguments
            {{0.0f}},   // child0: ScalarBroadcast beta=0
            {},         // child1: SrcFetch
            {           // child2: InnerEVT_v2
                {wgt_scale_ptr, 0.0f, {}},  // child0: WgtScaleBroadcast
                {                           // child1: AccScaleEVT
                    {act_scale_ptr, 0.0f, {}},  // child0: ActScaleBroadcast
                    {},                         // child1: AccFetch
                    {}                          // multiplies op args
                },
                {bias_ptr, cutlass::bfloat16_t(0), {}},  // child2: BiasBroadcast
                {}      // inner compute op args
            },
            {}          // outer compute op args
        },
        nullptr, stride_c,
        reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr()), stride_d
    };

    typename Gemm_v2::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            reinterpret_cast<ElementA_v2*>(a.data_ptr()), stride_a,
            reinterpret_cast<ElementB_v2*>(b.data_ptr()), stride_b
        },
        epilogue_args
    };

    Gemm_v2 gemm_op;

    auto status = gemm_op.can_implement(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "FP8 V2 SM90: can_implement failed, status=",
                static_cast<int>(status));

    const size_t ws_bytes = Gemm_v2::get_workspace_size(args);
    auto workspace = torch::empty(
        {static_cast<int64_t>(ws_bytes)},
        a.options().dtype(torch::kUInt8)
    );

    status = gemm_op.initialize(args, workspace.data_ptr());
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "FP8 V2 SM90: initialize failed, status=",
                static_cast<int>(status));

    status = gemm_op.run();
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "FP8 V2 SM90: run failed, status=",
                static_cast<int>(status));
}

// ---------------------------------------------------------------------------
// pybind11 模块注册
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUTLASS SM90 FP8 GEMM V2: Per-Block Scale (EVT per-row/col approx) → BF16";

    m.def("fp8_per_block_gemm_v2",
          &fp8_per_block_gemm_v2,
          "FP8 Per-Block GEMM (CUTLASS 3.x SM90 WGMMA, EVT per-row/col approx).\n"
          "  a          : (M, K) float8_e4m3fn\n"
          "  b          : (N, K) float8_e4m3fn (row-major = col-major)\n"
          "  out        : (M, N) bfloat16 pre-allocated\n"
          "  act_scale  : (M,)   float32 per-row activation scale (mean of group scales)\n"
          "  wgt_scale  : (N,)   float32 per-col weight scale (mean of group scales)\n"
          "  bias       : (N,)   bfloat16 or None\n"
          "Computes: out[i,j] = bf16(act_scale[i] * acc[i,j] * wgt_scale[j] + bias[j])",
          py::arg("a"),
          py::arg("b"),
          py::arg("out"),
          py::arg("act_scale"),
          py::arg("wgt_scale"),
          py::arg("bias") = c10::nullopt);
}
