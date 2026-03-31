"""
Build script for SageAttention SM90 kernel (standalone copy).

Usage:
    cd operators/sageattention/cuda
    python setup.py build_ext --inplace

This builds _qattn_sm90.cpython-*.so in the current directory.
"""

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Compiler flags - match official SageAttention build
ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0

CXX_FLAGS = [
    "-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17",
    "-DENABLE_BF16",
    f"-D_GLIBCXX_USE_CXX11_ABI={ABI}",
]

NVCC_FLAGS = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--use_fast_math",
    "--threads=8",
    "-Xptxas=-v",
    "-diag-suppress=174",
    f"-D_GLIBCXX_USE_CXX11_ABI={ABI}",
    # SM90a required for WGMMA PTX instructions
    "-gencode", "arch=compute_90a,code=sm_90a",
]

ext_modules = [
    CUDAExtension(
        name="_qattn_sm90",
        sources=[
            "csrc/pybind_sm90.cpp",
            "csrc/qk_int_sv_f8_cuda_sm90.cu",
        ],
        include_dirs=["csrc"],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
        },
        extra_link_args=['-lcuda'],
    )
]

setup(
    name='sageattention_sm90_standalone',
    version='0.1.0',
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
