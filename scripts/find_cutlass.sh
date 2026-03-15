#!/bin/bash
# find_cutlass.sh — 统一检测 CUTLASS 路径，供各算子 build.sh source 使用
#
# 使用方式（在 build.sh 开头）：
#   source "$(dirname "$0")/../../../scripts/find_cutlass.sh"
#   nvcc -I${CUTLASS_INCLUDE} ...
#
# 输出变量：
#   CUTLASS_ROOT    — CUTLASS 根目录
#   CUTLASS_INCLUDE — CUTLASS include 路径（含 cutlass/ 和 cute/）
#
# 检索顺序：
#   1. 环境变量 CUTLASS_PATH（用户手动指定）
#   2. flashinfer Python 包自带的 CUTLASS（推荐，无需单独安装）
#   3. 系统常见路径 /usr/local/cutlass、~/cutlass
#   4. 仍未找到则报错退出

_find_cutlass() {
    # 1. 用户手动指定
    if [ -n "${CUTLASS_PATH}" ] && [ -d "${CUTLASS_PATH}/include/cute" ]; then
        echo "${CUTLASS_PATH}"
        return 0
    fi

    # 2. flashinfer 包自带（当前环境已验证可用）
    local flashinfer_cutlass
    flashinfer_cutlass=$(python3 -c "
import importlib.util, os, sys
for finder in sys.meta_path:
    pass
spec = importlib.util.find_spec('flashinfer')
if spec and spec.origin:
    p = os.path.join(os.path.dirname(spec.origin), 'data', 'cutlass')
    if os.path.isdir(p):
        print(p)
" 2>/dev/null)
    if [ -n "${flashinfer_cutlass}" ] && [ -d "${flashinfer_cutlass}/include/cute" ]; then
        echo "${flashinfer_cutlass}"
        return 0
    fi

    # 3. 系统常见路径（fallback）
    for candidate in /usr/local/cutlass ~/cutlass /opt/cutlass; do
        if [ -d "${candidate}/include/cute" ]; then
            echo "${candidate}"
            return 0
        fi
    done

    # 4. 暴力搜索（最慢，最后手段）
    local found
    found=$(find /usr/local /opt "$HOME" -maxdepth 8 -name "cute" -type d 2>/dev/null \
            | grep "include/cute$" | head -1 | sed 's|/include/cute||')
    if [ -n "${found}" ]; then
        echo "${found}"
        return 0
    fi

    return 1
}

CUTLASS_ROOT=$(_find_cutlass)
if [ -z "${CUTLASS_ROOT}" ]; then
    echo "[ERROR] CUTLASS not found. Options:"
    echo "  1. Set CUTLASS_PATH=/path/to/cutlass before building"
    echo "  2. Install flashinfer: pip install flashinfer"
    echo "  3. Clone CUTLASS: git clone https://github.com/NVIDIA/cutlass.git"
    exit 1
fi

CUTLASS_INCLUDE="${CUTLASS_ROOT}/include"
export CUTLASS_ROOT CUTLASS_INCLUDE
