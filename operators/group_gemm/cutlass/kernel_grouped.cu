/*
 * Group GEMM — CUTLASS 3.x 高层 API 实现 (sm90a / Hopper)
 *
 * 调用 CUTLASS 官方 Ptr-Array / Grouped GEMM，利用：
 *   - wgmma 指令（Tensor Core）
 *   - TMA (Tensor Memory Accelerator) 异步加载
 *   - Persistent thread block + ProblemVisitor 动态调度
 *   - Warp-specialized 软件流水线
 *   - 真正的 variable-size group（每组 M_g, K, N 可不同，本实现固定 K/N）
 *
 * =========================================================================
 * CUTLASS 3.x PtrArray/Grouped GEMM 的布局约束（FP32 TF32 SS 路径）：
 *
 *   CUTLASS 3.x 将 GEMM 分为两条路径：
 *     SS (smem-smem): A 和 B 都走 TMA 加载到 smem → 适合 Grouped/PtrArray
 *     RS (rmem-smem): A 从 gmem 直读寄存器，B 走 TMA → 不支持 PtrArray
 *
 *   触发 SS 路径的条件（FP32 同类型输入）：
 *     is_k_major_A(LayoutA) && is_k_major_B(LayoutB)
 *
 *   对应的 Layout Tags：
 *     LayoutA = RowMajor   → TagToStrideA = (stride_M=int64, stride_K=Int<1>)
 *                            即 A(M,K) 行主序，K 是最快维度 ✓
 *     LayoutB = ColMajor   → TagToStrideB = (stride_N=int64, stride_K=Int<1>)
 *                            即 B(N,K) 行主序，K 是最快维度 ✓
 *
 *   注意：CUTLASS 约定 B 的维度是 (N, K)，ColMajor 表示 K 是最快维度，
 *         对应 PyTorch 中的 B^T = B.T(N,K)。
 *         因此：Python 调用侧需传入 B.transpose(1,2).contiguous()！
 *
 *   LayoutC = RowMajor：C(M,N) 行主序，stride_M=N, stride_N=1 ✓
 *
 * =========================================================================
 * 两个版本（与 matmul/cutlass/kernel_highlevel.cu 对应）：
 *
 * V1: KernelPtrArrayTmaWarpSpecializedCooperative + TileShape 128×128×32
 *     两个 warp group 协同完成一个 tile，适合大 M/N
 *
 * V2: KernelPtrArrayTmaWarpSpecializedPingpong + TileShape 64×128×32
 *     乒乓调度（一个 warp group load，另一个 compute）
 * =========================================================================
 */

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/dispatch_policy.hpp>
#include <cutlass/gemm/group_array_problem_shape.hpp>

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace cutlass;
using namespace cutlass::gemm;

// =========================================================================
// 类型别名
//
// 关键：LayoutA=RowMajor, LayoutB=ColMajor 触发 SS 路径（PtrArray 必须）
//   A(M,K) 行主序 → RowMajor：stride_M=K, stride_K=1 (K 最快)
//   B^T(N,K) 行主序 → ColMajor：stride_N=K, stride_K=1 (K 最快)
//   C(M,N) 行主序 → RowMajor：stride_M=N, stride_N=1
//
// 与普通 GEMM 的 "列主序技巧" 不同，这里不需要交换 A/B，
// 直接传 A 和 B^T 即可，CUTLASS 计算 C = A × B^T^T = A × B。
// =========================================================================
using ElementA   = float;
using ElementB   = float;
using ElementC   = float;
using ElementAcc = float;

// PtrArray/Grouped GEMM 用 pointer-type Layout Tags（带 *）
using LayoutA = cutlass::layout::RowMajor *;
using LayoutB = cutlass::layout::ColumnMajor *;  // B^T(N,K) row-major → ColumnMajor* → k-major ✓
using LayoutC = cutlass::layout::RowMajor *;

using ArchTag = cutlass::arch::Sm90;
using OpClass = cutlass::arch::OpClassTensorOp;

// GroupProblemShape: G 个独立问题，每个形状 = (M_g, N, K)
using UnderlyingProblemShape = Shape<int, int, int>;   // (M, N, K)
using ProblemShape = cutlass::gemm::GroupProblemShape<UnderlyingProblemShape>;


// =========================================================================
// V1: Cooperative warp-specialized，TileShape 128×128×32
// =========================================================================
namespace v1 {

using TileShape    = Shape<_128, _128, _32>;
using ClusterShape = Shape<_1, _1, _1>;

// Epilogue（Ptr-Array 调度：PtrArrayNoSmemWarpSpecialized，与 PtrArray mainloop 对应）
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OpClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, float,
    ElementC, LayoutC, 4,
    ElementC, LayoutC, 4,
    cutlass::epilogue::PtrArrayNoSmemWarpSpecialized
>::CollectiveOp;

// Mainloop（KernelPtrArrayTmaWarpSpecializedCooperative 触发 Grouped GEMM SS 路径）
using CollectiveMma = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, LayoutA, 4,
    ElementB, LayoutB, 4,
    ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMma,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

} // namespace v1


// =========================================================================
// V2: Pingpong warp-specialized，TileShape 64×128×32
// =========================================================================
namespace v2 {

using TileShape    = Shape<_64, _128, _32>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OpClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, float,
    ElementC, LayoutC, 4,
    ElementC, LayoutC, 4,
    cutlass::epilogue::PtrArrayNoSmemWarpSpecialized
>::CollectiveOp;

// Mainloop（KernelPtrArrayTmaWarpSpecializedPingpong 触发 Grouped GEMM SS 路径）
using CollectiveMma = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, LayoutA, 4,
    ElementB, LayoutB, 4,
    ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMma,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

} // namespace v2


// =========================================================================
// 通用 launch 函数
//
// 输入（注意 B 已经是转置后的 B^T(N,K)）：
//   A_ptrs[g] → A_g(M_g, K) 行主序
//   BT_ptrs[g] → B_g^T(N, K) 行主序（= B_g.T.contiguous()）
//   C_ptrs[g] → C_g(M_g, N) 行主序
//   problem_sizes[g] = {M_g, N, K}
// =========================================================================
template <typename GemmType>
static void run_grouped_gemm(
    const float** A_ptrs_host,
    const float** BT_ptrs_host,
    float**       C_ptrs_host,
    const int* M_sizes_host,     // M_g for each group
    int G, int K, int N)
{
    using StrideA = typename GemmType::GemmKernel::StrideA;
    using StrideB = typename GemmType::GemmKernel::StrideB;
    using StrideC = typename GemmType::GemmKernel::StrideC;
    using StrideD = typename GemmType::GemmKernel::StrideD;

    // StrideA underlying = Stride<int64, Int<1>, Int<0>> for RowMajor*
    //   stride_M = K_g (runtime), stride_K = 1 (static)
    // StrideB underlying = Stride<int64, Int<1>, Int<0>> for ColMajor*
    //   stride_N = K_g (runtime), stride_K = 1 (static)  [B^T is (N, K) row-major]
    // StrideC/D underlying = Stride<int64, Int<1>, Int<0>> for RowMajor*
    //   stride_M = N (runtime), stride_N = 1 (static)

    // Use the stride types directly from GemmKernel's deduced strides (pointer types for PtrArray)
    using StrideA_ptr = typename GemmType::GemmKernel::StrideA;  // UnderlyingType* = Stride<int64,Int<1>,Int<0>>*
    using StrideB_ptr = typename GemmType::GemmKernel::StrideB;
    using StrideC_ptr = typename GemmType::GemmKernel::StrideC;
    using StrideD_ptr = typename GemmType::GemmKernel::StrideD;

    // Dereference pointer type to get the underlying stride per-group
    using UnderlyingStrideA = cute::remove_pointer_t<StrideA_ptr>;
    using UnderlyingStrideB = cute::remove_pointer_t<StrideB_ptr>;
    using UnderlyingStrideC = cute::remove_pointer_t<StrideC_ptr>;
    using UnderlyingStrideD = cute::remove_pointer_t<StrideD_ptr>;

    std::vector<UnderlyingStrideA> stride_A_vec(G);
    std::vector<UnderlyingStrideB> stride_B_vec(G);
    std::vector<UnderlyingStrideC> stride_C_vec(G);
    std::vector<UnderlyingStrideD> stride_D_vec(G);
    std::vector<UnderlyingProblemShape> host_problem_shapes(G);

    for (int g = 0; g < G; ++g) {
        int M_g = M_sizes_host[g];
        // A_g (M_g, K): stride_M = K, stride_K = 1
        stride_A_vec[g] = UnderlyingStrideA{(int64_t)K, cute::Int<1>{}, cute::Int<0>{}};
        // B_g^T (N, K): stride_N = K, stride_K = 1  (ColMajor* B[N,K] row-major)
        stride_B_vec[g] = UnderlyingStrideB{(int64_t)K, cute::Int<1>{}, cute::Int<0>{}};
        // C_g (M_g, N): stride_M = N, stride_N = 1
        stride_C_vec[g] = UnderlyingStrideC{(int64_t)N, cute::Int<1>{}, cute::Int<0>{}};
        stride_D_vec[g] = UnderlyingStrideC{(int64_t)N, cute::Int<1>{}, cute::Int<0>{}};
        // problem shape: (M_g, N, K)
        host_problem_shapes[g] = make_shape(M_g, N, K);
    }

    // Upload everything to device
    const float** d_A_ptrs;
    const float** d_BT_ptrs;
    const float** d_C_ptrs_const;  // for epilogue source (beta=0, but API requires it)
    float**       d_D_ptrs;        // for epilogue output
    UnderlyingStrideA* d_stride_A;
    UnderlyingStrideB* d_stride_B;
    UnderlyingStrideC* d_stride_C;
    UnderlyingStrideD* d_stride_D;
    UnderlyingProblemShape* d_problem_shapes;

    cudaMalloc(&d_A_ptrs,        G * sizeof(float*));
    cudaMalloc(&d_BT_ptrs,       G * sizeof(float*));
    cudaMalloc(&d_C_ptrs_const,  G * sizeof(float*));
    cudaMalloc(&d_D_ptrs,        G * sizeof(float*));
    cudaMalloc(&d_stride_A,      G * sizeof(UnderlyingStrideA));
    cudaMalloc(&d_stride_B,      G * sizeof(UnderlyingStrideB));
    cudaMalloc(&d_stride_C,      G * sizeof(UnderlyingStrideC));
    cudaMalloc(&d_stride_D,      G * sizeof(UnderlyingStrideD));
    cudaMalloc(&d_problem_shapes, G * sizeof(UnderlyingProblemShape));

    cudaMemcpy(d_A_ptrs,       A_ptrs_host,  G * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_BT_ptrs,      BT_ptrs_host, G * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_ptrs_const, C_ptrs_host,  G * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D_ptrs,       C_ptrs_host,  G * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stride_A, stride_A_vec.data(), G * sizeof(UnderlyingStrideA), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stride_B, stride_B_vec.data(), G * sizeof(UnderlyingStrideB), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stride_C, stride_C_vec.data(), G * sizeof(UnderlyingStrideC), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stride_D, stride_D_vec.data(), G * sizeof(UnderlyingStrideD), cudaMemcpyHostToDevice);
    cudaMemcpy(d_problem_shapes, host_problem_shapes.data(),
               G * sizeof(UnderlyingProblemShape), cudaMemcpyHostToDevice);

    // Build GroupProblemShape
    ProblemShape problem_shape;
    problem_shape.num_groups         = G;
    problem_shape.problem_shapes     = d_problem_shapes;
    problem_shape.host_problem_shapes = host_problem_shapes.data();

    typename GemmType::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        problem_shape,
        {   // MainloopArguments: ptr_A, stride_A*, ptr_B, stride_B*
            reinterpret_cast<ElementA const**>(d_A_ptrs),  d_stride_A,
            reinterpret_cast<ElementB const**>(d_BT_ptrs), d_stride_B,
        },
        {   // EpilogueArguments: {alpha,beta}, ptr_C, stride_C*, ptr_D, stride_D*
            {1.0f, 0.0f},
            d_C_ptrs_const, d_stride_C,
            d_D_ptrs,       d_stride_D,
        },
    };

    GemmType gemm;

    cutlass::Status status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "[CUTLASS GroupedGemm] can_implement failed: "
                  << cutlass::cutlassGetStatusString(status) << "\n";
        goto cleanup;
    }
    {
        size_t ws_size = GemmType::get_workspace_size(args);
        void* workspace = nullptr;
        if (ws_size > 0) cudaMalloc(&workspace, ws_size);

        status = gemm.initialize(args, workspace);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "[CUTLASS GroupedGemm] initialize failed: "
                      << cutlass::cutlassGetStatusString(status) << "\n";
            if (workspace) cudaFree(workspace);
            goto cleanup;
        }

        status = gemm.run();
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "[CUTLASS GroupedGemm] run failed: "
                      << cutlass::cutlassGetStatusString(status) << "\n";
        }
        cudaDeviceSynchronize();
        if (workspace) cudaFree(workspace);
    }

cleanup:
    cudaFree(d_A_ptrs);      cudaFree(d_BT_ptrs);
    cudaFree(d_C_ptrs_const); cudaFree(d_D_ptrs);
    cudaFree(d_stride_A); cudaFree(d_stride_B);
    cudaFree(d_stride_C); cudaFree(d_stride_D);
    cudaFree(d_problem_shapes);
}


// =========================================================================
// extern "C" 接口（供 Python ctypes 调用）
//
// A:  (G, M, K) float32, 行主序
// BT: (G, N, K) float32, 行主序（= B.transpose(1,2).contiguous()，Python 端处理）
// C:  (G, M, N) float32, 行主序（输出）
// =========================================================================
extern "C" {

void group_gemm_cutlass_hl_v1(
    const float* A,   // (G, M, K)
    const float* BT,  // (G, N, K)  B^T
    float*       C,   // (G, M, N)
    int G, int M, int K, int N)
{
    std::vector<const float*> A_ptrs(G);
    std::vector<const float*> BT_ptrs(G);
    std::vector<float*>       C_ptrs(G);
    std::vector<int>          M_sizes(G, M);

    for (int g = 0; g < G; ++g) {
        A_ptrs[g]  = A  + (long long)g * M * K;
        BT_ptrs[g] = BT + (long long)g * N * K;
        C_ptrs[g]  = C  + (long long)g * M * N;
    }

    run_grouped_gemm<v1::Gemm>(
        A_ptrs.data(), BT_ptrs.data(), C_ptrs.data(),
        M_sizes.data(), G, K, N);
}

void group_gemm_cutlass_hl_v2(
    const float* A,
    const float* BT,
    float*       C,
    int G, int M, int K, int N)
{
    std::vector<const float*> A_ptrs(G);
    std::vector<const float*> BT_ptrs(G);
    std::vector<float*>       C_ptrs(G);
    std::vector<int>          M_sizes(G, M);

    for (int g = 0; g < G; ++g) {
        A_ptrs[g]  = A  + (long long)g * M * K;
        BT_ptrs[g] = BT + (long long)g * N * K;
        C_ptrs[g]  = C  + (long long)g * M * N;
    }

    run_grouped_gemm<v2::Gemm>(
        A_ptrs.data(), BT_ptrs.data(), C_ptrs.data(),
        M_sizes.data(), G, K, N);
}

} // extern "C"
