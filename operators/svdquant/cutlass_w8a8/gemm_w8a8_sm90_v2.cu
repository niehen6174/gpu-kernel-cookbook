/**
 * CUTLASS 3.x SM90 W8A8 WGMMA GEMM — V2 (Fused Epilogue)
 *
 * 相对 V1 的改进：
 *   1. Part 1: Fused smooth+quantize CUDA kernel
 *      将 activation smooth 除法 + 动态 per-group INT8 量化合并到单个 kernel，
 *      消除 V1 中分开的 smooth 和量化步骤的 overhead。
 *
 *   2. Part 2: INT8 WGMMA + fused epilogue (custom EVT: per-row alpha + per-col bias)
 *      使用 CUTLASS 3.x EVT（Epilogue Visitor Tree）在 epilogue 中直接完成：
 *        D[i,j] = half(alpha[i] * acc[i,j] + bias[j])
 *      输出直接为 FP16，消除 V1 中单独的 dequant scale 乘法 kernel。
 *      EpilogueSchedule 显式指定 TmaWarpSpecialized（EVT 要求）。
 *
 * 编译要求：
 *   -gencode arch=compute_90a,code=sm_90a
 *   -DCUTLASS_ARCH_MMA_SM90_SUPPORTED
 *   -DCUTE_ARCH_MMA_SM90A_ENABLED
 *   -std=c++17
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// CUTLASS 头文件
// ---------------------------------------------------------------------------
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
// EVT 原语：Sm90RowBroadcast / Sm90ColBroadcast / Sm90AccFetch / Sm90SrcFetch
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/numeric_conversion.h"
#include "cute/tensor.hpp"

#include <torch/extension.h>

// ============================================================
// Part 1: Fused Smooth + Quantize Kernel
// ============================================================

/**
 * fused_smooth_quantize_kernel
 *
 * 每个 block 处理一个 (row, group) 对。
 * gridDim  = (M, K/group_size)
 * blockDim = (group_size,)  — 要求 group_size 为 2 的幂，<= 1024
 * shared   = group_size * sizeof(float)
 *
 * 步骤：
 *   1. 读取 x[row, col] / smooth[col] → 寄存器 val（float）
 *   2. 将 |val| 写入 smem，tree-reduce 求 max(|val|) over group
 *   3. scale = max_val / 127.0f，clamp >= 1e-8
 *   4. thread 0 写 group_scales[row, grp] = scale
 *   5. q[row, col] = round(val / scale).clamp(-128, 127)
 */
__global__ void fused_smooth_quantize_kernel(
    const __half* __restrict__ x,            // (M, K) fp16, row-major
    const __half* __restrict__ smooth,       // (K,)   fp16
    int8_t*       __restrict__ q,            // (M, K) int8, row-major  [output]
    float*        __restrict__ group_scales, // (M, K/group_size) float32 [output]
    int K,
    int group_size
) {
    const int row = blockIdx.x;
    const int grp = blockIdx.y;
    const int tid = threadIdx.x;
    const int col = grp * group_size + tid;

    extern __shared__ float smem[];  // group_size * sizeof(float)

    // 1. Load and divide by smooth
    const float val = __half2float(x[row * K + col]) / __half2float(smooth[col]);

    // 2. Store |val| in smem for tree-reduce
    smem[tid] = fabsf(val);
    __syncthreads();

    // Parallel tree-reduce to find max(|val|) in this group
    for (int stride = group_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }

    // 3. scale = max_val / 127.0f, clamp >= 1e-8
    const float scale = fmaxf(smem[0] / 127.0f, 1e-8f);

    // 4. Write group scale (thread 0 only)
    if (tid == 0) {
        group_scales[row * (K / group_size) + grp] = scale;
    }

    // 5. Quantize: round(val / scale), clamp to [-128, 127]
    q[row * K + col] = static_cast<int8_t>(
        fmaxf(-128.0f, fminf(127.0f, rintf(val / scale)))
    );
}

/**
 * fused_smooth_quantize（内部版，接受预分配输出）
 */
void fused_smooth_quantize_impl(
    torch::Tensor x,
    torch::Tensor smooth,
    torch::Tensor q,
    torch::Tensor group_scales,
    int group_size
) {
    TORCH_CHECK(x.is_cuda() && smooth.is_cuda() && q.is_cuda() && group_scales.is_cuda(),
                "fused_smooth_quantize: all tensors must be on CUDA");
    TORCH_CHECK(x.dtype()            == torch::kFloat16,  "x must be fp16");
    TORCH_CHECK(smooth.dtype()       == torch::kFloat16,  "smooth must be fp16");
    TORCH_CHECK(q.dtype()            == torch::kInt8,     "q must be int8");
    TORCH_CHECK(group_scales.dtype() == torch::kFloat32,  "group_scales must be float32");
    TORCH_CHECK(x.is_contiguous() && smooth.is_contiguous() &&
                q.is_contiguous() && group_scales.is_contiguous(),
                "fused_smooth_quantize: all tensors must be contiguous");

    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(x.size(1));
    TORCH_CHECK(K % group_size == 0,
                "K (", K, ") must be divisible by group_size (", group_size, ")");
    TORCH_CHECK((group_size & (group_size - 1)) == 0,
                "group_size must be a power of 2, got ", group_size);
    TORCH_CHECK(group_size <= 1024,
                "group_size must be <= 1024, got ", group_size);

    const int ngroups = K / group_size;
    const size_t smem_bytes = static_cast<size_t>(group_size) * sizeof(float);

    fused_smooth_quantize_kernel<<<dim3(M, ngroups), dim3(group_size), smem_bytes>>>(
        reinterpret_cast<const __half*>(x.data_ptr()),
        reinterpret_cast<const __half*>(smooth.data_ptr()),
        reinterpret_cast<int8_t*>(q.data_ptr()),
        reinterpret_cast<float*>(group_scales.data_ptr()),
        K, group_size
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "fused_smooth_quantize_kernel launch failed: ", cudaGetErrorString(err));
}

/**
 * fused_smooth_quantize（Python-facing，自动分配输出并返回）
 *
 * @param x          (M, K) fp16
 * @param smooth     (K,)   fp16
 * @param group_size int（默认 128）
 * @return {q: (M,K) int8,  group_scales: (M, K/group_size) float32}
 */
std::pair<torch::Tensor, torch::Tensor> fused_smooth_quantize(
    torch::Tensor x,
    torch::Tensor smooth,
    int group_size = 128
) {
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(x.size(1));
    TORCH_CHECK(K % group_size == 0,
                "K (", K, ") must be divisible by group_size (", group_size, ")");
    auto q            = torch::empty({M, K},           x.options().dtype(torch::kInt8));
    auto group_scales = torch::empty({M, K / group_size}, x.options().dtype(torch::kFloat32));
    fused_smooth_quantize_impl(x, smooth, q, group_scales, group_size);
    return {q, group_scales};
}


// ============================================================
// Part 2: INT8 WGMMA + Custom EVT Epilogue → FP16
//
// EVT computes:
//   D[i,j] = half( alpha[i] * acc[i,j] + bias[j] )
// where
//   alpha[i] = ascale_mean[i] * wscale_mean  (per-row float32, precomputed in Python)
//   bias[j]  = per-col FP16 bias (optional, nullptr → 0)
// ============================================================

// ---------------------------------------------------------------------------
// Element / layout type aliases
// ---------------------------------------------------------------------------
using ElementA_v2    = int8_t;
using ElementB_v2    = int8_t;
using ElementC_v2    = cutlass::half_t;   // 传入 C（不使用，beta=0）
using ElementD_v2    = cutlass::half_t;   // 输出 FP16
using ElementAcc_v2  = int32_t;
using ElementComp_v2 = float;             // epilogue 中间计算精度

using LayoutA_v2 = cutlass::layout::RowMajor;
using LayoutB_v2 = cutlass::layout::ColumnMajor;
using LayoutC_v2 = cutlass::layout::RowMajor;
using LayoutD_v2 = cutlass::layout::RowMajor;

using ArchTag_v2  = cutlass::arch::Sm90;
using OpClass_v2  = cutlass::arch::OpClassTensorOp;

// INT8 SS TileShape sweet-spot on Hopper: 128x256x128
using TileShape_v2    = cute::Shape<cute::_128, cute::_256, cute::_128>;
using ClusterShape_v2 = cute::Shape<cute::_1,   cute::_1,   cute::_1>;

constexpr int AlignA_v2 = 128 / cutlass::sizeof_bits<ElementA_v2>::value;  // 16
constexpr int AlignB_v2 = 128 / cutlass::sizeof_bits<ElementB_v2>::value;  // 16
constexpr int AlignC_v2 = 128 / cutlass::sizeof_bits<ElementC_v2>::value;  // 8
constexpr int AlignD_v2 = 128 / cutlass::sizeof_bits<ElementD_v2>::value;  // 8

static constexpr cutlass::FloatRoundStyle RoundStyle_v2 =
    cutlass::FloatRoundStyle::round_to_nearest;

// ---------------------------------------------------------------------------
// Custom EVT: D[i,j] = half( alpha[i] * acc[i,j] + bias[j] )
//
// Tree structure (bottom-up):
//   inner: homogeneous_multiply_add(float, float) = alpha[i] * acc + bias[j]
//     child0: Sm90ColBroadcast<alpha> — per-row alpha, stride (1,0,0) over (M,N,L)
//     child1: Sm90AccFetch            — accumulator
//     child2: Sm90RowBroadcast<bias>  — per-col bias, stride (0,1,0) over (M,N,L)
//   outer: homogeneous_multiply_add(half_t, float) = 0*C + inner  [= inner → cast to half]
//     child0: Sm90ScalarBroadcast<float>(beta=0) — scalar zero
//     child1: Sm90SrcFetch — C (not read when beta=0 ptr is null)
//     child2: inner above
//
// Sm90ColBroadcast: Stages=0 (col broadcast doesn't use smem)
//   stride <_1,_0,_0> = col-vector = one value per row (per M)
// Sm90RowBroadcast:  Stages calculated via EpilogueDescriptor
//   stride <_0,_1,_0> = row-vector = one value per col (per N)
// ---------------------------------------------------------------------------

// First compute the EpilogueDescriptor to get the right Stages for RowBroadcast
// (must be computed at template instantiation time)
// Mainloop KernelScheduleAuto picks KernelTmaWarpSpecializedCooperative for M=128
// → epilogue must match: TmaWarpSpecializedCooperative
using EpilogueSchedule_v2 = cutlass::epilogue::TmaWarpSpecializedCooperative;

using EpilogueDescriptor_v2 = cutlass::epilogue::collective::detail::EpilogueDescriptor<
    TileShape_v2,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementC_v2,
    ElementD_v2,
    EpilogueSchedule_v2
>;

// Stages for RowBroadcast:
//   ceil_div(StagesC, epi_tiles_per_cta) + 1
//   where epi_tiles_per_cta = size(shape_div(take<0,2>(TileShape), EpilogueTile))
using EpilogueTile_v2 = typename EpilogueDescriptor_v2::EpilogueTile;
static constexpr int RowBcastStages_v2 =
    cute::ceil_div(EpilogueDescriptor_v2::StagesC,
        cute::size(cute::shape_div(
            cute::take<0,2>(TileShape_v2{}),
            EpilogueTile_v2{}
        ))
    ) + 1;

// Per-row alpha: ColBroadcast, stride (M-dim=1, N-dim=0, L-dim=0)
using AlphaBroadcast_v2 = cutlass::epilogue::fusion::Sm90ColBroadcast<
    0,              // Stages = 0 (col broadcast uses no smem)
    TileShape_v2,
    float,          // ElementScalar = float32
    cute::Stride<cute::_1, cute::_0, cute::_0>,
    128 / cutlass::sizeof_bits<float>::value  // alignment = 4
>;

// Per-col bias: RowBroadcast, stride (M-dim=0, N-dim=1, L-dim=0)
using BiasBroadcast_v2 = cutlass::epilogue::fusion::Sm90RowBroadcast<
    RowBcastStages_v2,   // stages from descriptor
    TileShape_v2,
    cutlass::half_t,     // ElementBias = FP16
    cute::Stride<cute::_0, cute::_1, cute::_0>,
    128 / cutlass::sizeof_bits<cutlass::half_t>::value  // alignment = 8
>;

// Inner EVT: alpha[i] * acc + bias[j]  (float arithmetic)
using InnerEVT_v2 = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::homogeneous_multiply_add,
        float,      // ElementOutput of inner (stays float until outer cast to half)
        float,      // ElementCompute
        RoundStyle_v2
    >,
    AlphaBroadcast_v2,   // child0: alpha[i] per-row
    cutlass::epilogue::fusion::Sm90AccFetch,   // child1: accumulator (int32 → float)
    BiasBroadcast_v2     // child2: bias[j] per-col
>;

// Outer EVT: 0*C + inner  — effectively: cast inner result to FP16
// beta=0 means C is not loaded (Sm90SrcFetch is present but zeroed out)
using CustomEVT_v2 = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        cutlass::homogeneous_multiply_add,
        cutlass::half_t,   // ElementOutput = FP16 (final output)
        float,             // ElementCompute
        RoundStyle_v2
    >,
    cutlass::epilogue::fusion::Sm90ScalarBroadcast<float>,  // beta (scalar = 0)
    cutlass::epilogue::fusion::Sm90SrcFetch,                // C
    InnerEVT_v2                                            // inner result
>;

// ---------------------------------------------------------------------------
// CollectiveMainloop — INT8 SS path → wgmma.mma_async s32.s8.s8
// Use StageCountAutoCarveout to account for epilogue SharedStorage
// ---------------------------------------------------------------------------

// We need forward declaration of CollectiveEpilogue for its SharedStorage size.
// Do this in two passes:
// 1. Define epilogue first (no dependency on mainloop stages)
// 2. Use epilogue::SharedStorage size in mainloop StageCountAutoCarveout

using CollectiveEpilogue_v2 = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag_v2, OpClass_v2,
    TileShape_v2, ClusterShape_v2,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc_v2, ElementComp_v2,
    ElementC_v2, LayoutC_v2, AlignC_v2,
    ElementD_v2, LayoutD_v2, AlignD_v2,
    EpilogueSchedule_v2,    // explicit TmaWarpSpecialized (EVT requires this)
    CustomEVT_v2            // custom EVT as FusionOpOrCallbacks
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

// ---------------------------------------------------------------------------
// GEMM kernel & device adapter
// ---------------------------------------------------------------------------

using GemmKernel_v2 = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop_v2,
    CollectiveEpilogue_v2
>;

using Gemm_v2 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_v2>;

// ---------------------------------------------------------------------------
// Python-facing function
// ---------------------------------------------------------------------------

/**
 * w8a8_gemm_sm90_v2
 *
 * INT8 W8A8 WGMMA SM90 GEMM，epilogue 直接融合 per-row dequant scale 和 per-col bias，
 * 输出 FP16。消除 V1 中单独的 dequant scale 乘法 kernel。
 *
 * Computes:  out[i,j] = half( alpha_row[i] * Σ_k act[i,k]*wgt[k,j] + bias[j] )
 *
 * @param act       (M, K)  int8  row-major 激活
 * @param wgt       (N, K)  int8  row-major（= col-major (K, N)），权重
 * @param out       (M, N)  fp16  pre-allocated 输出
 * @param alpha_row (M,)    float32 per-row scale = ascale_mean[i] * wscale_mean
 * @param bias      (N,)    fp16 or undefined Tensor (None from Python) — per-col bias
 */
void w8a8_gemm_sm90_v2(
    torch::Tensor act,       // (M, K) int8
    torch::Tensor wgt,       // (N, K) int8, row-major = col-major (K, N)
    torch::Tensor out,       // (M, N) fp16, pre-allocated
    torch::Tensor alpha_row, // (M,)   float32 per-row scale
    c10::optional<torch::Tensor> bias_opt  // (N,) fp16 or nullopt (None from Python)
) {
    // ------------------------------------------------------------------
    // Type / device / shape checks
    // ------------------------------------------------------------------
    TORCH_CHECK(act.is_cuda() && wgt.is_cuda() && out.is_cuda() && alpha_row.is_cuda(),
                "w8a8_gemm_sm90_v2: act, wgt, out, alpha_row must be on CUDA");
    TORCH_CHECK(act.dtype() == torch::kInt8,        "act must be int8");
    TORCH_CHECK(wgt.dtype() == torch::kInt8,        "wgt must be int8");
    TORCH_CHECK(out.dtype() == torch::kFloat16,     "out must be fp16");
    TORCH_CHECK(alpha_row.dtype() == torch::kFloat32, "alpha_row must be float32");
    TORCH_CHECK(act.is_contiguous() && wgt.is_contiguous() &&
                out.is_contiguous() && alpha_row.is_contiguous(),
                "w8a8_gemm_sm90_v2: act/wgt/out/alpha_row must be contiguous");

    const int M = static_cast<int>(act.size(0));
    const int K = static_cast<int>(act.size(1));
    const int N = static_cast<int>(wgt.size(0));

    TORCH_CHECK(wgt.size(1) == K,
                "wgt K-dim mismatch: act.K=", K, " wgt.K=", wgt.size(1));
    TORCH_CHECK(out.size(0) == M && out.size(1) == N,
                "out shape mismatch: expected (", M, ",", N, ")");
    TORCH_CHECK(alpha_row.size(0) == M,
                "alpha_row size mismatch: expected M=", M, " got ", alpha_row.size(0));

    // ------------------------------------------------------------------
    // Bias pointer (optional, null when not provided)
    // ------------------------------------------------------------------
    cutlass::half_t const* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt.value().defined() && bias_opt.value().numel() > 0) {
        torch::Tensor bias = bias_opt.value();
        TORCH_CHECK(bias.is_cuda(),                    "bias must be on CUDA");
        TORCH_CHECK(bias.dtype() == torch::kFloat16,   "bias must be fp16");
        TORCH_CHECK(bias.is_contiguous(),              "bias must be contiguous");
        TORCH_CHECK(bias.size(0) == N,
                    "bias size mismatch: expected N=", N, " got ", bias.size(0));
        bias_ptr = reinterpret_cast<cutlass::half_t const*>(bias.data_ptr());
    }

    const float* alpha_ptr = reinterpret_cast<float const*>(alpha_row.data_ptr());

    // ------------------------------------------------------------------
    // Stride construction
    // ------------------------------------------------------------------
    using StrideA = typename Gemm_v2::GemmKernel::StrideA;
    using StrideB = typename Gemm_v2::GemmKernel::StrideB;
    using StrideC = typename Gemm_v2::GemmKernel::StrideC;
    using StrideD = typename Gemm_v2::GemmKernel::StrideD;

    // A: RowMajor (M, K)
    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    // B: ColMajor (K, N); wgt stored as (N, K) row-major
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    // C/D: RowMajor (M, N)
    StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    // ------------------------------------------------------------------
    // Epilogue arguments
    //
    // CustomEVT_v2 is an Sm90EVT tree; arguments follow the nested
    // {child0_args, ..., childN_args, op_args} structure (see example 49).
    //
    // CustomEVT_v2 = Sm90EVT<outer_compute,
    //   Sm90ScalarBroadcast<float>,  // child0: beta
    //   Sm90SrcFetch,                // child1: C
    //   InnerEVT_v2>                 // child2: alpha*acc+bias
    //
    // InnerEVT_v2 = Sm90EVT<inner_compute,
    //   AlphaBroadcast_v2,           // child0: alpha[i]
    //   Sm90AccFetch,                // child1: acc
    //   BiasBroadcast_v2>            // child2: bias[j]
    //
    // So the outermost thread args init:
    //   {
    //     {{0.0f}},    // child0: beta scalar = 0
    //     {},          // child1: C (Sm90SrcFetch has no args)
    //     {            // child2: InnerEVT
    //       {alpha_ptr, 0.0f, {}},   // AlphaBroadcast: ptr_col, null_default, stride
    //       {},                      // Sm90AccFetch
    //       {bias_ptr, 0.0_h, {}},   // BiasBroadcast: ptr_row, null_default, stride
    //       {}                       // inner compute args
    //     },
    //     {}           // outer compute args
    //   }
    // ------------------------------------------------------------------
    // CollectiveEpilogue::Arguments = {thread, ptr_C, dC, ptr_D, dD}
    // thread = FusionCallbacks::Arguments = nested EVT args
    typename CollectiveEpilogue_v2::Arguments epilogue_args{
        {   // thread: FusionCallbacks/EVT arguments
            // CustomEVT_v2 = Sm90EVT<outer_compute, ScalarBcast<float>, SrcFetch, InnerEVT>
            // Structure: {child0, child1, child2, op_args}
            // outer: homogeneous_multiply_add(half, float) = beta*C + inner
            {{0.0f}},  // child0: Sm90ScalarBroadcast<float> — beta scalar = 0
            {},        // child1: Sm90SrcFetch — C (zero-weighted, not accessed)
            {          // child2: InnerEVT_v2
                // inner: homogeneous_multiply_add(float, float) = alpha[i]*acc + bias[j]
                {alpha_ptr, 0.0f, {}},   // child0: AlphaBroadcast (ptr_col, null_default, stride)
                {},                      // child1: Sm90AccFetch (no args)
                {bias_ptr, cutlass::half_t(0), {}},  // child2: BiasBroadcast (ptr_row, null_default, stride)
                {}                       // inner compute op args
            },
            {}         // outer compute op args
        },
        nullptr, stride_c,  // ptr_C (null — C not used, beta=0) + dC
        reinterpret_cast<cutlass::half_t*>(out.data_ptr()), stride_d   // ptr_D + dD
    };

    // ------------------------------------------------------------------
    // GEMM arguments
    // ------------------------------------------------------------------
    typename Gemm_v2::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {   // MainloopArguments
            reinterpret_cast<ElementA_v2*>(act.data_ptr()), stride_a,
            reinterpret_cast<ElementB_v2*>(wgt.data_ptr()), stride_b
        },
        epilogue_args
    };

    Gemm_v2 gemm_op;

    auto status = gemm_op.can_implement(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS W8A8 V2 SM90: can_implement failed, status=",
                static_cast<int>(status));

    const size_t ws_bytes = Gemm_v2::get_workspace_size(args);
    auto workspace = torch::empty(
        {static_cast<int64_t>(ws_bytes)},
        act.options().dtype(torch::kUInt8)
    );

    status = gemm_op.initialize(args, workspace.data_ptr());
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS W8A8 V2 SM90: initialize failed, status=",
                static_cast<int>(status));

    status = gemm_op.run();
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS W8A8 V2 SM90: run failed, status=",
                static_cast<int>(status));
}


// ============================================================
// pybind11 module
// ============================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUTLASS SM90 W8A8 GEMM V2: fused smooth-quant + fused dequant/bias epilogue";

    // fused_smooth_quantize(x, smooth, group_size=128) -> (q, group_scales)
    m.def("fused_smooth_quantize",
          &fused_smooth_quantize,
          "Fused smooth-divide + per-group INT8 quantization.\n"
          "  x          : (M, K) fp16 — activation\n"
          "  smooth     : (K,)   fp16 — smooth factor (divide x by this before quant)\n"
          "  group_size : int         — quant group size (default 128, power-of-2)\n"
          "Returns: (q: (M,K) int8, group_scales: (M, K/group_size) float32)",
          py::arg("x"),
          py::arg("smooth"),
          py::arg("group_size") = 128);

    // w8a8_gemm_sm90_v2(act, wgt, out, alpha_row, bias=None)
    m.def("w8a8_gemm_sm90_v2",
          &w8a8_gemm_sm90_v2,
          "INT8 W8A8 WGMMA SM90 GEMM with fused per-row dequant scale + per-col bias → FP16.\n"
          "  act       : (M, K)  int8    — quantized activation\n"
          "  wgt       : (N, K)  int8    — quantized weight (row-major = col-major)\n"
          "  out       : (M, N)  fp16    — pre-allocated output\n"
          "  alpha_row : (M,)    float32 — per-row scale = ascale_mean * wscale_mean\n"
          "  bias      : (N,)    fp16 or None — per-col bias\n"
          "Computes: out[i,j] = half(alpha_row[i] * Σ_k act[i,k]*wgt[k,j] + bias[j])",
          py::arg("act"),
          py::arg("wgt"),
          py::arg("out"),
          py::arg("alpha_row"),
          py::arg("bias") = c10::nullopt);
}
