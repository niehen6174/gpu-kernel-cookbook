/*
 * Vector Add — CuTe C++ 实现
 *
 * 目的：通过最简单的算子展示 CuTe 的核心抽象：
 *   - cute::Tensor<Engine, Layout>：带 Layout 信息的张量
 *   - make_tensor / make_layout：构造 Tensor 和 Layout
 *   - cute::copy：协同的 element-wise 复制（利用 MMA/Copy Atom 扩展点）
 *   - thread_layout / partition：将 Tensor 按线程分块
 *
 * CuTe 与普通 CUDA 的区别：
 *   普通 CUDA:  thread_id = blockIdx.x * blockDim.x + threadIdx.x
 *               C[thread_id] = A[thread_id] + B[thread_id]
 *
 *   CuTe:       将"哪个 thread 负责哪些数据"用 Layout 描述，
 *               通过 local_partition 自动切分，代码与具体维度无关。
 *               这为后续 gemm / attention 中复杂的多维 tiling 打基础。
 *
 * =========================================================================
 * V1: 直接索引（最接近普通 CUDA，最易理解）
 *   - 把输入指针包装成 1D CuTe Tensor
 *   - 每个 thread 负责一个元素
 *   - 通过 local_tile / local_partition 演示分片方式
 *
 * V2: 向量化（使用 CuTe Copy Atom，128-bit load/store）
 *   - UniversalCopy<uint128_t> 等价于 float4
 *   - 演示 CuTe 的 Copy Atom 扩展点（为后续 Tensor Core 打基础）
 * =========================================================================
 */

#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/algorithm/copy.hpp>

#include <cuda_runtime.h>

using namespace cute;

// -------------------------------------------------------------------------
// V1: 直接索引 — 展示 CuTe Tensor / Layout 基础
// -------------------------------------------------------------------------
//
// 关键 CuTe 概念：
//   make_tensor(ptr, make_layout(N))
//     → 创建一个 1D Tensor，Layout 是 (N,) : (1,)（步长=1）
//
//   local_partition(tensor, thr_layout, thread_id)
//     → 把 tensor 按 thr_layout 切给第 thread_id 号 thread
//     → 返回该 thread 负责的子 Tensor
//
__global__ void vector_add_cute_v1(
    const float* __restrict__ A_ptr,
    const float* __restrict__ B_ptr,
    float*       __restrict__ C_ptr,
    int N)
{
    // 1. 用原始指针构造 1D CuTe Tensor
    //    make_layout(N) 等价于 Layout<Shape<int>, Stride<_1>>
    //    即连续存储的 N 元素 1D 张量
    auto A = make_tensor(make_gmem_ptr(A_ptr), make_layout(N));
    auto B = make_tensor(make_gmem_ptr(B_ptr), make_layout(N));
    auto C = make_tensor(make_gmem_ptr(C_ptr), make_layout(N));

    // 2. 描述"block 内如何分配给各 thread"的 Layout
    //    Int<256>{} 是编译期常量，等价于 constexpr int 256
    //    这里用 blockDim.x 的运行期值（实际 kernel 总是 256 threads）
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 3. local_tile：从整个 Tensor 中取出当前 block 负责的连续段
    //    参数：(tensor, tile_shape, tile_coord)
    //    tile_shape = (blockDim.x,)，tile_coord = (blockIdx.x,)
    auto tA = local_tile(A, make_shape(blockDim.x), make_coord(blockIdx.x));
    auto tB = local_tile(B, make_shape(blockDim.x), make_coord(blockIdx.x));
    auto tC = local_tile(C, make_shape(blockDim.x), make_coord(blockIdx.x));
    // tA 现在是 size(blockDim.x) 的子 Tensor，代表当前 block 的数据段

    // 4. local_partition：进一步切给每个 thread
    //    参数：(tensor, thread_layout, thread_id)
    //    thread_layout = (blockDim.x,)：线性分配给 blockDim.x 个 thread
    auto thrA = local_partition(tA, make_layout(blockDim.x), threadIdx.x);
    auto thrB = local_partition(tB, make_layout(blockDim.x), threadIdx.x);
    auto thrC = local_partition(tC, make_layout(blockDim.x), threadIdx.x);
    // 此时 thrA/thrB/thrC 各自 size = 1（每 thread 1 个元素）

    // 5. 边界检查并计算
    if (tid < N) {
        // thrA(0) 等价于 A_ptr[tid]，但语义更清晰
        thrC(0) = thrA(0) + thrB(0);
    }
}

// -------------------------------------------------------------------------
// V2: 向量化 — 使用 CuTe Copy Atom (128-bit = float4)
// -------------------------------------------------------------------------
//
// CuTe Copy Atom：
//   UniversalCopy<uint128_t> 告诉 CuTe 用 128-bit 指令搬运数据
//   等价于手写 float4 reinterpret_cast，但由 CuTe 自动处理对齐和向量化
//
//   这里展示最简形式；在 GEMM kernel 中，Copy Atom 会与 Tensor Core MMA Atom 配合
//
__global__ void vector_add_cute_v2(
    const float* __restrict__ A_ptr,
    const float* __restrict__ B_ptr,
    float*       __restrict__ C_ptr,
    int N)
{
    // 把 float* 重解释为 uint128_t*（每次处理 4 个 float）
    // 要求 N 是 4 的倍数（调用方保证）
    int N4 = N / 4;  // 向量化后的元素个数

    using Vec = uint128_t;  // 4 × float32 = 128 bit
    auto A4 = make_tensor(make_gmem_ptr(reinterpret_cast<const Vec*>(A_ptr)),
                          make_layout(N4));
    auto B4 = make_tensor(make_gmem_ptr(reinterpret_cast<const Vec*>(B_ptr)),
                          make_layout(N4));

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N4) return;

    // 取出当前 thread 负责的 128-bit 块（load）
    Vec a_vec = A4(tid);
    Vec b_vec = B4(tid);

    // 将 128-bit 解包为 4 个 float，做加法，重新打包
    float* a_f = reinterpret_cast<float*>(&a_vec);
    float* b_f = reinterpret_cast<float*>(&b_vec);
    Vec c_vec;
    float* c_f = reinterpret_cast<float*>(&c_vec);
    c_f[0] = a_f[0] + b_f[0];
    c_f[1] = a_f[1] + b_f[1];
    c_f[2] = a_f[2] + b_f[2];
    c_f[3] = a_f[3] + b_f[3];

    // Store
    auto C4 = make_tensor(make_gmem_ptr(reinterpret_cast<Vec*>(C_ptr)),
                          make_layout(N4));
    C4(tid) = c_vec;
}

// -------------------------------------------------------------------------
// Host 函数（extern "C" 供 Python ctypes 调用）
// -------------------------------------------------------------------------
extern "C" {

void vector_add_cutlass_v1(const float* A, const float* B, float* C, int N) {
    const int BLOCK = 256;
    int grid = (N + BLOCK - 1) / BLOCK;
    vector_add_cute_v1<<<grid, BLOCK>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

void vector_add_cutlass_v2(const float* A, const float* B, float* C, int N) {
    // N 不是 4 的倍数时，回退到 v1 处理尾部
    int N_aligned = (N / 4) * 4;
    int N_tail    = N - N_aligned;

    if (N_aligned > 0) {
        const int BLOCK = 256;
        int grid = (N_aligned / 4 + BLOCK - 1) / BLOCK;
        vector_add_cute_v2<<<grid, BLOCK>>>(A, B, C, N_aligned);
    }
    if (N_tail > 0) {
        const int BLOCK = 256;
        vector_add_cute_v1<<<1, BLOCK>>>(A + N_aligned, B + N_aligned,
                                          C + N_aligned, N_tail);
    }
    cudaDeviceSynchronize();
}

}  // extern "C"
