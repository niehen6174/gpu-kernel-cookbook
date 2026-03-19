# NCU Profiling 指南

本项目提供统一的 NCU（NVIDIA Nsight Compute）profiling 基础设施，用于对所有算子的 CUDA kernel 进行性能分析。

---

## 文件结构

```
gpu-kernel-lab/
├── profiling/
│   └── profile_driver.py    # 统一 driver：注册所有 kernel，控制 profiler start/stop
├── scripts/
│   └── profile.sh           # 封装 ncu 命令行调用的入口脚本
└── results/
    └── ncu/                 # profile 报告保存目录（.ncu-rep，被 .gitignore 忽略）
```

---

## 快速开始

```bash
# 从项目根目录运行
cd gpu-kernel-lab

# 1. 查看所有可 profile 的 op 和 kernel
bash scripts/profile.sh --list

# 2. profile 指定 kernel（最常用）
bash scripts/profile.sh --op rms_norm --kernel cuda_v2

# 3. 报告保存在 results/ncu/，命令行也会实时打印结果
```

---

## scripts/profile.sh 参数说明

```
bash scripts/profile.sh [选项]

必选：
  --op <op>            算子名，如 rms_norm、matmul、rope

可选：
  --kernel <name>      指定 kernel，如 cuda_v2。不指定则逐个 profile 所有 kernel
  --set <level>        ncu 预设分析集（默认 basic）
                         basic   速度快，适合日常分析
                         full    所有指标，约慢 10-30x
  --section <list>     逗号分隔的 section 名，覆盖 --set
                         如 SpeedOfLight,MemoryWorkloadAnalysis
  --warmup <n>         warmup 次数（默认 3）
  --iters <n>          profile 次数（默认 1）
  --no-export          不保存 .ncu-rep 报告
  --list               列出所有可用 op/kernel，不运行 ncu
```

### 示例

```bash
# 基础分析（Speed of Light + Launch Stats + Occupancy）
bash scripts/profile.sh --op rms_norm --kernel cuda_v2

# 详细分析，保存报告
bash scripts/profile.sh --op rms_norm --kernel cuda_v2 --set full

# 自定义 section 组合
bash scripts/profile.sh --op rms_norm --kernel cuda_v2 \
    --section SpeedOfLight,MemoryWorkloadAnalysis,OccupancyAnalysis

# Roofline 图分析
bash scripts/profile.sh --op matmul --kernel cutlass_hl_v1 \
    --section SpeedOfLight_HierarchicalSingleRooflineChart

# 对比同一算子的两个版本（分两次跑）
bash scripts/profile.sh --op rms_norm --kernel cuda_v1
bash scripts/profile.sh --op rms_norm --kernel cuda_v2

# profile 某个 op 的全部 kernel
bash scripts/profile.sh --op softmax

# 不保存报告，只看命令行输出
bash scripts/profile.sh --op rope --kernel cuda_v1 --no-export
```

---

## 可用 Section 参考

| Section 标识符 | 显示名 | 说明 |
|---|---|---|
| `SpeedOfLight` | GPU Speed Of Light Throughput | SM/DRAM/L1/L2 利用率，最常用 |
| `MemoryWorkloadAnalysis` | Memory Workload Analysis | 内存层次带宽、coalescing 效率 |
| `MemoryWorkloadAnalysis_Chart` | Memory Workload Analysis Chart | 内存层次可视化图 |
| `ComputeWorkloadAnalysis` | Compute Workload Analysis | 计算流水线利用率、指令混合 |
| `Occupancy` | Occupancy | 理论/实际 occupancy、block limit 原因 |
| `LaunchStats` | Launch Statistics | grid/block 配置、寄存器/smem 用量 |
| `WarpStateStats` | Warp State Statistics | warp stall 原因分布 |
| `SchedulerStats` | Scheduler Statistics | warp 调度效率 |
| `InstructionStats` | Instruction Statistics | 指令类型统计 |
| `SourceCounters` | Source Counters | 源码级热点（需编译时加 `-lineinfo`） |
| `SpeedOfLight_RooflineChart` | Roofline Chart | Roofline 模型图 |
| `SpeedOfLight_HierarchicalSingleRooflineChart` | Hierarchical Roofline (FP32) | 分层 Roofline（FP32） |
| `SpeedOfLight_HierarchicalTensorRooflineChart` | Hierarchical Roofline (Tensor) | 分层 Roofline（Tensor Core） |

`--set basic` 默认启用：`SpeedOfLight` + `LaunchStats` + `Occupancy` + `WorkloadDistribution`

`--set full` 启用全部 section。

---

## 当前已注册的 Op / Kernel

| Op | Kernel | 测试形状 |
|---|---|---|
| `rms_norm` | `cuda_v1`, `cuda_v2`, `cute_v1`, `cute_v2` | B=4096, N=4096 |
| `rope` | `cuda_v1`, `cuda_v2`, `cute_v1`, `cute_v2` | seq=4096, heads=32, d=64 |
| `softmax` | `cuda_v1`, `cuda_v2`, `cuda_v3`, `cute_v1`, `cute_v2` | B=4096, N=4096 |
| `layernorm` | `cuda_v1`, `cuda_v2`, `cute_v1`, `cute_v2` | B=4096, N=4096 |
| `matmul` | `cuda_v1~v3`, `cute_v1~v2`, `cutlass_hl_v1~v2` | 4096×4096×4096 |
| `transpose` | `cuda_v1`, `cuda_v2`, `cute_v1`, `cute_v2` | 4096×4096 |
| `vector_add` | `cuda_v1`, `cuda_v2`, `cute_v1`, `cute_v2` | N=64M |

---

## 实现原理

### 为什么需要专用 driver？

直接用 `ncu python my_script.py` 会 capture 进程生命周期内的**所有** CUDA kernel，包括 PyTorch 初始化、cuBLAS warm-up、JIT 编译等无关调用，导致报告噪音大、难以定位目标 kernel。

### cudaProfilerStart / Stop

`profiling/profile_driver.py` 在每次目标 kernel 调用前后插入 `cudaProfilerStart()` / `cudaProfilerStop()`：

```python
def profiler_start():
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()

def profiler_stop():
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
```

配合 ncu 的 `--profile-from-start no` 参数，ncu 只 capture profiler active 期间的 kernel：

```
进程启动
  ↓
PyTorch 初始化、tensor 分配（ncu 不 capture）
  ↓
warmup × N 次（profiler 关闭，ncu 不 capture）
  ↓
cudaProfilerStart()     ← ncu 开始 capture
  ↓
目标 kernel 调用 × iters 次   ← 这里的 kernel 被 profile
  ↓
cudaProfilerStop()      ← ncu 停止 capture
  ↓
进程退出
```

### Registry 模式

每个 op 对应一个 `_kernels_xxx()` 函数，返回 `dict[str, callable]`：

```python
def _kernels_rms_norm():
    # 准备输入数据（只初始化一次）
    B, N = 4096, 4096
    x = torch.randn(B, N, device="cuda").contiguous()
    ...
    # 返回 {kernel_name: 无参可调用对象}
    return {
        "cuda_v1": lambda: lib.rms_norm_cuda_v1(...),
        "cuda_v2": lambda: lib.rms_norm_cuda_v2(...),
        ...
    }

REGISTRY = {
    "rms_norm": _kernels_rms_norm,
    "matmul":   _kernels_matmul,
    ...
}
```

这种设计的优势：
- 新增算子只需添加一个函数，不改动框架代码
- 输入数据在 registry 函数内部初始化，测试形状集中管理
- callable 捕获了输入 tensor 的引用，调用时零额外参数

### sudo 权限

ncu 读取 GPU 硬件性能计数器（PMU）需要 root 权限（或设置 `/proc/driver/nvidia/params` 中的 `RmProfilingAdminOnly=0`）。`profile.sh` 自动使用 `sudo` 并传入绝对路径，因为 `sudo` 不继承普通用户的 `PATH`：

```bash
NCU_BIN=$(which ncu)        # 记录绝对路径
PYTHON_BIN=$(which python)  # 记录绝对路径
sudo "$NCU_BIN" ... "$PYTHON_BIN" profiling/profile_driver.py ...
```

---

## 如何新增算子

在 `profiling/profile_driver.py` 中添加一个函数并注册：

```python
def _kernels_my_op():
    # 1. 准备输入
    x = torch.randn(1024, 1024, device="cuda").contiguous()
    lib = _load_so("operators/my_op/cuda/my_op.so", {
        "my_op_cuda_v1": ([ctypes.c_void_p, ctypes.c_int], None),
    })

    # 2. 返回 {name: callable}
    kernels = {}
    if lib:
        kernels["cuda_v1"] = lambda: lib.my_op_cuda_v1(
            x.data_ptr(), ctypes.c_int(1024)
        )
    return kernels

# 3. 注册
REGISTRY = {
    ...
    "my_op": _kernels_my_op,
}
```

然后：

```bash
bash scripts/profile.sh --op my_op --kernel cuda_v1
```

---

## 报告文件

报告保存在 `results/ncu/` 目录，文件名格式为：

```
results/ncu/<op>_<kernel>_<YYYYMMDD_HHMMSS>.ncu-rep
```

`.ncu-rep` 是 Nsight Compute 的二进制报告格式，可用以下方式查看：

```bash
# 命令行（profile 时已实时打印，这是重新查看）
ncu --import results/ncu/rms_norm_cuda_v2_20260318_232054.ncu-rep

# GUI（需要本地安装 Nsight Compute）
ncu-ui results/ncu/rms_norm_cuda_v2_20260318_232054.ncu-rep
```

`results/ncu/` 目录已加入 `.gitignore`，报告文件不会被提交。

---

## 常见问题

**Q: `ERR_NVGPUCTRPERM` 权限错误**

ncu 需要 root 权限读取 PMU 计数器。`profile.sh` 已自动加 `sudo`，但如果系统未给 `sudo` 权限，可以以 root 身份运行，或由管理员执行：

```bash
# 永久解除限制（需要 root，重启失效）
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia.conf
```

**Q: ncu 捕获到了额外的 kernel（如 cuBLAS）**

正常现象，只要 driver 正确使用了 `cudaProfilerStart/Stop`，ncu 报告中第一个（或唯一一个）就是目标 kernel。也可以在 ncu 命令中加 `--kernel-name <kernel_function_name>` 进一步过滤：

```bash
# 直接在 profile.sh 的 ncu 命令中指定函数名
# kernel 函数名可从报告输出的第一行 "Profiling "xxx"" 中读取
ncu --kernel-name rms_norm_v2 ...
```

**Q: 如何修改测试形状？**

在 `profiling/profile_driver.py` 对应的 `_kernels_xxx()` 函数中修改即可，形状定义在函数开头。
