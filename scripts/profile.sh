#!/bin/bash
# =============================================================================
# scripts/profile.sh — 统一 NCU Profile 入口
#
# 用法：
#   bash scripts/profile.sh --op rms_norm --kernel cuda_v2
#   bash scripts/profile.sh --op rms_norm --kernel cuda_v2 --set full
#   bash scripts/profile.sh --op rms_norm                  # profile 所有 kernel
#   bash scripts/profile.sh --op matmul  --kernel cutlass_hl_v1 --section roofline
#   bash scripts/profile.sh --list                         # 列出所有 op/kernel
#
# 常用 --section 值：
#   basic       速度快，看 SOL（Speed of Light）和带宽
#   full        所有指标，慢 10-30x
#   roofline    Roofline 图所需指标
#   memory      只看内存子系统
#
# 输出报告默认保存到 results/ncu/<op>_<kernel>_<timestamp>.ncu-rep
# =============================================================================

set -e
cd "$(dirname "$0")/.."

# 获取绝对路径（sudo 不继承 PATH，必须使用全路径）
NCU_BIN=$(which ncu)
PYTHON_BIN=$(which python)

# ---------- 默认参数 ----------
OP=""
KERNEL_ARG=""        # --kernel <name>，可空
SET="basic"          # 默认 set
SECTION_ARGS=""      # 额外 --section 参数（用逗号分隔多个）
EXPORT=1             # 是否保存 .ncu-rep
WARMUP=3
ITERS=1
LIST=0

# ---------- 解析参数 ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --op)        OP="$2";      shift 2 ;;
        --kernel)    KERNEL_ARG="--kernel $2"; shift 2 ;;
        --set)       SET="$2";     shift 2 ;;
        --section)   SECTION_ARGS="$2"; shift 2 ;;  # 逗号分隔，如 "SpeedOfLight,MemoryWorkloadAnalysis"
        --warmup)    WARMUP="$2";  shift 2 ;;
        --iters)     ITERS="$2";   shift 2 ;;
        --no-export) EXPORT=0;     shift ;;
        --list)      LIST=1;       shift ;;
        *) echo "[ERROR] 未知参数: $1"; exit 1 ;;
    esac
done

# ---------- 只列出时不需要 ncu ----------
if [[ $LIST -eq 1 ]]; then
    python profiling/profile_driver.py --list
    exit 0
fi

if [[ -z "$OP" ]]; then
    echo "[ERROR] 请指定 --op"
    echo "运行 bash scripts/profile.sh --list 查看可用 op/kernel"
    exit 1
fi

# ---------- 构建 ncu section 参数 ----------
if [[ -n "$SECTION_ARGS" ]]; then
    # 用逗号分隔的 section 列表，转成多个 --section
    NCU_SECTIONS=""
    IFS=',' read -ra SECS <<< "$SECTION_ARGS"
    for s in "${SECS[@]}"; do
        NCU_SECTIONS="$NCU_SECTIONS --section $s"
    done
else
    NCU_SECTIONS="--set $SET"
fi

# ---------- 输出路径 ----------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OP_KERNEL="${OP}"
if [[ -n "$KERNEL_ARG" ]]; then
    # 从 "--kernel cuda_v2" 提取 "cuda_v2"
    KNAME=$(echo "$KERNEL_ARG" | awk '{print $2}')
    OP_KERNEL="${OP}_${KNAME}"
fi

mkdir -p results/ncu
REPORT_PATH="results/ncu/${OP_KERNEL}_${TIMESTAMP}"

# ---------- 构建 driver 参数 ----------
DRIVER_ARGS="--op $OP $KERNEL_ARG --warmup $WARMUP --iters $ITERS"

# ---------- 打印信息 ----------
echo "============================================================"
echo "  NCU Profile"
echo "  op:      $OP"
echo "  kernel:  ${KNAME:-all}"
echo "  set/sections: ${SECTION_ARGS:-set=$SET}"
echo "  report:  ${REPORT_PATH}.ncu-rep"
echo "============================================================"

# ---------- 构建 export 参数 ----------
EXPORT_ARG=""
if [[ $EXPORT -eq 1 ]]; then
    EXPORT_ARG="--export ${REPORT_PATH}"
fi

# ---------- 运行 ncu ----------
sudo "$NCU_BIN" \
    --target-processes all \
    --profile-from-start no \
    $NCU_SECTIONS \
    $EXPORT_ARG \
    "$PYTHON_BIN" profiling/profile_driver.py $DRIVER_ARGS

echo ""
if [[ $EXPORT -eq 1 ]]; then
    echo "报告已保存: ${REPORT_PATH}.ncu-rep"
    echo "用 ncu-ui 打开: ncu-ui ${REPORT_PATH}.ncu-rep"
fi
