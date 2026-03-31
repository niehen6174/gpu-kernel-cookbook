#!/bin/bash
# Build SageAttention SM90 kernel as a standalone torch extension
#
# Usage:
#   cd operators/sageattention/cuda
#   bash build.sh
#
# Requires: CUDA >= 12.3, PyTorch, SM90 GPU (Hopper)

set -e
cd "$(dirname "$0")"

echo "Building SageAttention SM90 kernel..."
echo "CUDA_HOME: ${CUDA_HOME:-$(python -c 'from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)')}"

python setup.py build_ext --inplace 2>&1 | tail -20

# Verify the built .so exists
SO_FILE=$(ls _qattn_sm90*.so 2>/dev/null | head -1)
if [ -n "$SO_FILE" ]; then
    echo ""
    echo "Build successful: $SO_FILE ($(du -h "$SO_FILE" | cut -f1))"
    echo ""
    echo "Quick test:"
    python -c "
import torch
import sys
sys.path.insert(0, '.')
import _qattn_sm90
print('Module loaded:', _qattn_sm90)
print('Functions:', [f for f in dir(_qattn_sm90) if not f.startswith('_')])
"
else
    echo "ERROR: Build failed, no .so file found"
    exit 1
fi
