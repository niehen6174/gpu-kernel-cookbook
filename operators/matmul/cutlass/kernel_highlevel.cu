/*
 * Matmul — CUTLASS 4.x 高层 API 实现 (sm90 / Hopper)
 *
 * 这是与 kernel.cu (CuTe 低层) 的核心区别所在：
 *
 * CuTe (kernel.cu):
 *   - 你手写每一行 kernel 代码
 *   - 手动管理 shared memory、线程分工、tile 循环
 *   - 不使用 Tensor Core（我们的 CuTe 版本用的是 FFMA，不是 wgmma）
 *
 * CUTLASS 高层 API (本文件):
 *   - 声明式：只描述"我要算什么"（element type, layout, tile size）
 *   - 硬件特性全自动：
 *       wgmma 指令（Tensor Core）
 *       TMA (Tensor Memory Accelerator) 异步加载
 *       Warp-specialized 软件流水线（producer/consumer 分工）
 *       Shared memory swizzle（消除 bank conflict）
 *   - 你不需要写任何 __global__ 函数
 *
 * =========================================================================
 * CUTLASS 3.x/4.x 高层 API 的四层结构：
 *
 *  1. CollectiveEpilogue  (先建！因为 mainloop 需要知道它占多少 smem)
 *     - 负责 D = alpha * AB + beta * C
 *     - 由 CollectiveBuilder 从模板参数自动推导
 *
 *  2. CollectiveMma  (mainloop，建立时用 StageCountAutoCarveout 扣除 epilogue smem)
 *     - 负责 A × B 的分块矩阵乘
 *     - sm90 自动选：TMA 加载 + wgmma 指令 + Warp-specialized pipeline
 *     - Stage 数 = (剩余 smem) / (每 stage A+B tile 大小)
 *
 *  3. GemmKernel  (组装 mainloop + epilogue)
 *     - cutlass::gemm::kernel::GemmUniversal
 *
 *  4. GemmUniversalAdapter  (设备级 API，自动分配 workspace 并 launch)
 *     - 提供 can_implement / initialize / run 接口
 *
 * =========================================================================
 * 布局约定（"列主序技巧"）：
 *
 * 问题：输入 A(M,K) 和 B(K,N) 均为行主序（PyTorch 默认）。
 *       CUTLASS 3.x 中若 LayoutA=RowMajor + LayoutB=RowMajor（即 A K-major,
 *       B N-major），会触发 SwapAB=true（仅 float32），导致计算错误。
 *
 * 解法："列主序技巧" —— 计算 C^T = B^T × A^T，输出写成列主序即 C 行主序。
 *
 *   C(M,N) row-major = C^T(N,M) col-major  [同一块内存！]
 *   C^T = B^T(N,K) × A^T(K,M)
 *
 *   映射到 CUTLASS GemmUniversal(M_new=N, N_new=M, K_new=K)：
 *     new A = B^T 的数据 = B(K,N) row-major ptr
 *       → 视作 (N,K) ColMajor：stride_N=1(static), stride_K=N_orig(dynamic)
 *       → LayoutA=ColMajor, stride_A=(Int<1>{}, N_orig, N_orig*K)
 *     new B = A^T 的数据 = A(M,K) row-major ptr
 *       → 视作 (M,K) 对应 CUTLASS B(N_new,K)，K-major
 *       → TagToStrideB<ColMajor>=Stride<int64,Int<1>,int64>:[0]=stride_N_new=K(dyn),[1]=stride_K=1(static)
 *       → LayoutB=ColMajor, stride_B=(K, Int<1>{}, M*K)
 *     new C/D = C 的指针 (C^T(N,M) ColMajor)
 *       → TagToStrideC<ColMajor>=Stride<Int<1>,int64,int64>:[0]=stride_M_new=1(static),[1]=N(dyn)
 *       → LayoutC=ColMajor, stride_C=(Int<1>{}, N_orig, M*N_orig)
 *
 *   结果：StrideToLayoutTagA=ColMajor, StrideToLayoutTagB=ColMajor
 *         → IsLayoutAkBmn=false, SwapAB=false ✓
 *
 * =========================================================================
 * V1: TileShape 128×128×32，KernelTmaWarpSpecializedCooperative
 *     - 2 warp group 协同完成一个 128×128 tile
 *     - 适合大矩阵（M, N 均 >= 128）
 *
 * V2: TileShape 64×128×32，KernelTmaWarpSpecializedPingpong
 *     - 2 warp group 乒乓交替执行（一个 load，一个 compute）
 *     - 适合 M 较小或需要更小 latency 的场景
 *
 * 对比 CuTe v2 (kernel.cu):
 *   CuTe v2  tile=32×32，手写 shared memory tiling，FFMA（无 Tensor Core）
 *   本文件   tile=128×128，自动 TMA + wgmma + 流水线，接近 cuBLAS 性能
 * =========================================================================
 */

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>

#include <cuda_runtime.h>
#include <iostream>

using namespace cutlass;
using namespace cutlass::gemm;

// =========================================================================
// 类型别名
// =========================================================================
using ElementA   = float;
using ElementB   = float;
using ElementC   = float;
using ElementAcc = float;

// "列主序技巧"：A=ColMajor, B=ColMajor, C=ColMajor
// （详见文件头注释）
using LayoutA = cutlass::layout::ColumnMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;

using ArchTag = cutlass::arch::Sm90;
using OpClass = cutlass::arch::OpClassTensorOp;  // 使用 Tensor Core (wgmma)

// =========================================================================
// 构建 GEMM 类型
//
// 关键：必须先建 CollectiveEpilogue，然后把它的 SharedStorage 大小传给
// CollectiveMma 的 StageCountAutoCarveout，这样 builder 才能正确算出
// 多少 stage 能装进剩余的 shared memory。
// =========================================================================

// =========================================================================
// V1: TileShape 128×128×32 (Cooperative warp-specialized)
// =========================================================================
namespace v1 {

using TileShape    = Shape<_128, _128, _32>;
using ClusterShape = Shape<_1, _1, _1>;

// Step 1: 先建 Epilogue（决定它占多少 smem）
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OpClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, float,          // accumulator type, compute type
    ElementC,   LayoutC, 4,     // source C (beta=0 时不读，但仍需指定)
    ElementC,   LayoutC, 4,     // destination D
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

// Step 2: 建 Mainloop，用 StageCountAutoCarveout 扣除 epilogue 的 smem
using CollectiveMma = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, LayoutA, 4,
    ElementB, LayoutB, 4,
    ElementAcc,
    TileShape, ClusterShape,
    // StageCountAutoCarveout<bytes>: 从 smem 总量中扣除 epilogue 占用的 bytes，
    // 剩余部分用于 mainloop stage buffer，自动算出最大 stage 数
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int>,   // problem shape: M, N, K 作为运行时整数
    CollectiveMma,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

} // namespace v1


// =========================================================================
// V2: TileShape 64×128×32 (Pingpong warp-specialized)
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
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMma = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, LayoutA, 4,
    ElementB, LayoutB, 4,
    ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int>,
    CollectiveMma,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

} // namespace v2


// =========================================================================
// 通用 launch 函数（两个版本复用）
// =========================================================================
// "列主序技巧"说明（在 run_gemm 中）：
//
//   调用方传入的是 A(M,K) 行主序 和 B(K,N) 行主序。
//   我们计算 C^T = B^T × A^T，等价于 CUTLASS GemmUniversal(M_new=N, N_new=M, K_new=K)。
//
//   new A (CUTLASS 视角)  = B data，作为 (N,K) ColMajor
//     TagToStrideA<ColMajor> = Stride<Int<1>, int64, int64> → [0]=stride_N=1, [1]=stride_K=N_orig
//     stride_A = make_stride(Int<1>{}, N_orig, N_orig*K)
//
//   new B (CUTLASS 视角)  = A data，作为 (M,K) ColMajor（K-major）
//     TagToStrideB<ColMajor> = Stride<int64, Int<1>, int64> → [0]=stride_N_new=K, [1]=stride_K=1
//     stride_B = make_stride(K, Int<1>{}, M*K)
//
//   new C/D (CUTLASS 视角) = C ptr，作为 (N,M) ColMajor
//     TagToStrideC<ColMajor> = Stride<Int<1>, int64, int64> → [0]=stride_N=1, [1]=stride_M=N_orig
//     stride_C = make_stride(Int<1>{}, N_orig, M*N_orig)
//
//   C^T(N,M) col-major 与 C(M,N) row-major 占用完全相同的内存 ✓
//
template <typename GemmType>
static void run_gemm(const float* A, const float* B, float* C,
                     int M, int K, int N)
{
    using StrideA = typename GemmType::GemmKernel::StrideA;
    using StrideB = typename GemmType::GemmKernel::StrideB;
    using StrideC = typename GemmType::GemmKernel::StrideC;
    using StrideD = typename GemmType::GemmKernel::StrideD;

    // "列主序技巧": 交换 A/B 指针，交换 M/N 维度
    // new A = B data (N,K) ColMajor: stride_N=1(static), stride_K=N(dynamic)
    StrideA stride_A = cute::make_stride(cute::Int<1>{}, (int64_t)N, (int64_t)(N * K));
    // new B = A data (M,K) ColMajor: stride_N_new=K(dynamic), stride_K=1(static)
    StrideB stride_B = cute::make_stride((int64_t)K, cute::Int<1>{}, (int64_t)(M * K));
    // C/D = C ptr (N,M) ColMajor: stride_N=1(static), stride_M=N(dynamic)
    StrideC stride_C = cute::make_stride(cute::Int<1>{}, (int64_t)N, (int64_t)(M * N));
    StrideD stride_D = cute::make_stride(cute::Int<1>{}, (int64_t)N, (int64_t)(M * N));

    typename GemmType::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {N, M, K},              // problem_shape: M_new=N, N_new=M, K_new=K
        {                       // mainloop args: new A = B, new B = A
            const_cast<ElementA*>(B), stride_A,
            const_cast<ElementB*>(A), stride_B,
        },
        {                       // epilogue args
            {1.0f, 0.0f},       // alpha=1, beta=0
            C, stride_C,        // source ptr (beta=0 所以不读，但 API 需要)
            C, stride_D,        // destination D ptr
        }
    };

    GemmType gemm;

    cutlass::Status status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "[CUTLASS] can_implement failed: "
                  << cutlass::cutlassGetStatusString(status) << "\n";
        return;
    }

    size_t ws_size = GemmType::get_workspace_size(args);
    void* workspace = nullptr;
    if (ws_size > 0) {
        cudaMalloc(&workspace, ws_size);
    }

    status = gemm.initialize(args, workspace);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "[CUTLASS] initialize failed: "
                  << cutlass::cutlassGetStatusString(status) << "\n";
        if (workspace) cudaFree(workspace);
        return;
    }

    status = gemm.run();
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "[CUTLASS] run failed: "
                  << cutlass::cutlassGetStatusString(status) << "\n";
    }

    cudaDeviceSynchronize();
    if (workspace) cudaFree(workspace);
}


// =========================================================================
// extern "C" 接口（供 Python ctypes 调用）
// =========================================================================
extern "C" {

void matmul_cutlass_hl_v1(const float* A, const float* B, float* C,
                           int M, int K, int N) {
    run_gemm<v1::Gemm>(A, B, C, M, K, N);
}

void matmul_cutlass_hl_v2(const float* A, const float* B, float* C,
                           int M, int K, int N) {
    run_gemm<v2::Gemm>(A, B, C, M, K, N);
}

} // extern "C"
