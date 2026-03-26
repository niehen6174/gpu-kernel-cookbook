#!/bin/bash
# 运行所有算子的测试

set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

echo "Running all operator tests..."
echo ""

for op in vector_add transpose softmax layernorm matmul attention rms_norm rope group_gemm; do
    test_py="$ROOT/operators/$op/test.py"
    if [ -f "$test_py" ]; then
        echo "========================================"
        echo "Testing $op..."
        echo "========================================"
        python "$test_py"
        echo ""
    fi
done

echo "All tests complete!"
