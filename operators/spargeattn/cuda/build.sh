#!/bin/bash
# Build SpargeAttention CUDA extensions
set -e
cd "$(dirname "$0")"
python setup.py build_ext --inplace
echo "Build complete!"
