/**
 * CUTLASS 3.x SM90 W8A8 WGMMA GEMM
 *
 * INT8 x INT8 → FP16，使用 Hopper WGMMA（warpgroup-level MMA）指令。
 * 通过 CollectiveBuilder SS 路径自动选择 MainloopSm90TmaGmmaWarpSpecialized。
 *
 * 编译要求：
 *   -gencode arch=compute_90a,code=sm_90a
 *   -DCUTLASS_ARCH_MMA_SM90_SUPPORTED
 *   -DCUTE_ARCH_MMA_SM90A_ENABLED
 *   -std=c++17
 */

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

using ElementA    = int8_t;           // 激活，RowMajor (M, K)
using ElementB    = int8_t;           // 权重，ColMajor (K, N) = RowMajor (N, K)
using ElementC    = float;            // output C (pass-through)，FP32 避免溢出
using ElementD    = float;            // output D，FP32 避免溢出（Python 侧再转 FP16）
using ElementAcc  = int32_t;          // INT8 累加器
using ElementComp = float;            // epilogue 计算精度

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using ArchTag     = cutlass::arch::Sm90;
using OpClass     = cutlass::arch::OpClassTensorOp;

// TileShape: MxNxK = 128x256x128（INT8 SS 路径标准 sweet-spot）
using TileShape    = cute::Shape<cute::_128, cute::_256, cute::_128>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

// 对齐要求（128bit / element_bits）
constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 16
constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementB>::value;  // 16
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;  // 4

// ---------------------------------------------------------------------------
// CollectiveBuilder：A 和 B 同位宽（均 8bit）→ IsMixedWidth=false → SS 路径
// → 真正 WGMMA wgmma.mma_async s32.s8.s8
// ---------------------------------------------------------------------------

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OpClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementComp,
    ElementC, LayoutC, AlignC,
    ElementD, LayoutD, AlignC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// ---------------------------------------------------------------------------
// Python 接口
// ---------------------------------------------------------------------------

/**
 * w8a8_gemm_sm90 — INT8 W8A8 WGMMA GEMM（SM90 Hopper）
 *
 * @param act   (M, K) int8 contiguous row-major — 激活
 * @param wgt   (N, K) int8 contiguous row-major（= col-major (K, N)）— 权重
 * @param out   (M, N) float32 pre-allocated — 输出（FP32 避免 INT32→FP16 溢出）
 * @param alpha scale factor（INT32 → FP32 统一缩放，默认 1.0）
 *
 * 注意：dequant scale 乘法在 Python 侧完成，此处输出未缩放的 FP32 值。
 */
void w8a8_gemm_sm90(
    torch::Tensor act,   // (M, K) int8
    torch::Tensor wgt,   // (N, K) int8，row-major 存储 = col-major (K, N)
    torch::Tensor out,   // (M, N) float32
    float alpha          // default 1.0
) {
    TORCH_CHECK(act.is_cuda() && wgt.is_cuda() && out.is_cuda(),
                "w8a8_gemm_sm90: all tensors must be on CUDA");
    TORCH_CHECK(act.dtype() == torch::kInt8,  "act must be int8");
    TORCH_CHECK(wgt.dtype() == torch::kInt8,  "wgt must be int8");
    TORCH_CHECK(out.dtype() == torch::kFloat32, "out must be float32");
    TORCH_CHECK(act.is_contiguous(), "act must be contiguous");
    TORCH_CHECK(wgt.is_contiguous(), "wgt must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    int M = act.size(0), K = act.size(1);
    int N = wgt.size(0);

    TORCH_CHECK(wgt.size(1) == K, "wgt K-dim mismatch: act.K=", K, " wgt.K=", wgt.size(1));
    TORCH_CHECK(out.size(0) == M && out.size(1) == N,
                "out shape mismatch: expected (", M, ",", N, ")");

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    // RowMajor A (M, K): packed stride = (K, 1, M*K)
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    // ColMajor B: wgt stored as (N, K) row-major = (K, N) col-major
    // packed stride for ColMajor (N, K, L) = (1, N, N*K)
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    // RowMajor C/D (M, N): packed stride = (N, 1, M*N)
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        // Mainloop: A ptr + stride, B ptr + stride
        {reinterpret_cast<ElementA*>(act.data_ptr()), stride_a,
         reinterpret_cast<ElementB*>(wgt.data_ptr()), stride_b},
        // Epilogue: {alpha, beta}, C ptr + stride, D ptr + stride
        {{alpha, 0.0f},
         nullptr, stride_c,
         reinterpret_cast<ElementD*>(out.data_ptr()), stride_d}
    };

    Gemm gemm_op;

    // can_implement 检查
    auto status = gemm_op.can_implement(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS W8A8 SM90: can_implement failed, status=",
                static_cast<int>(status));

    // workspace 分配
    size_t ws_bytes = Gemm::get_workspace_size(args);
    auto workspace = torch::empty(
        {static_cast<int64_t>(ws_bytes)},
        act.options().dtype(torch::kUInt8)
    );

    status = gemm_op.initialize(args, workspace.data_ptr());
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS W8A8 SM90: initialize failed, status=",
                static_cast<int>(status));

    status = gemm_op.run();
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS W8A8 SM90: run failed, status=",
                static_cast<int>(status));
}

// ---------------------------------------------------------------------------
// pybind11 模块注册
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("w8a8_gemm_sm90", &w8a8_gemm_sm90,
          "INT8 W8A8 WGMMA SM90 GEMM (CUTLASS 3.x)",
          py::arg("act"),
          py::arg("wgt"),
          py::arg("out"),
          py::arg("alpha") = 1.0f);
}
