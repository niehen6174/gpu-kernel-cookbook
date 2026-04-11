"""
Build script for SpargeAttention CUDA kernels.

Supports SM80 (Ampere) and SM90 (Hopper).

Usage:
    cd operators/spargeattn/cuda
    python setup.py build_ext --inplace
"""

import os
from pathlib import Path
import subprocess
from packaging.version import parse, Version
from typing import Set
import warnings

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

HAS_SM90 = False

def run_instantiations(src_dir: str):
    base_path = Path(src_dir)
    py_files = [
        path for path in base_path.rglob('*.py')
        if path.is_file()
    ]
    for py_file in py_files:
        print(f"Running: {py_file}")
        os.system(f"python {py_file}")


def get_instantiations(src_dir: str):
    base_path = Path(src_dir)
    return [
        os.path.join(src_dir, str(path.relative_to(base_path)))
        for path in base_path.rglob('*')
        if path.is_file() and path.suffix == ".cu"
    ]


# Supported architectures
SUPPORTED_ARCHS = {"8.0", "8.6", "8.7", "8.9", "9.0"}

# Compiler flags
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
    # "-Xcompiler", "-include,cassert",  # disabled - causes redefinition errors on some systems
    f"-D_GLIBCXX_USE_CXX11_ABI={ABI}",
]

if CUDA_HOME is None:
    raise RuntimeError("Cannot find CUDA_HOME. CUDA must be available to build.")


def get_nvcc_cuda_version(cuda_dir: str) -> Version:
    nvcc_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def get_torch_arch_list() -> Set[str]:
    env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    if env_arch_list is None:
        return set()
    torch_arch_list = set(env_arch_list.replace(" ", ";").split(";"))
    if not torch_arch_list:
        return set()
    valid_archs = SUPPORTED_ARCHS.union({s + "+PTX" for s in SUPPORTED_ARCHS})
    arch_list = torch_arch_list.intersection(valid_archs)
    if not arch_list:
        raise RuntimeError(
            f"None of the CUDA architectures in `TORCH_CUDA_ARCH_LIST` ({env_arch_list}) is supported. "
            f"Supported: {valid_archs}.")
    return arch_list


# Detect compute capabilities
compute_capabilities = get_torch_arch_list()
if not compute_capabilities:
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        major, minor = torch.cuda.get_device_capability(i)
        if major < 8:
            raise RuntimeError("GPUs with compute capability below 8.0 are not supported.")
        compute_capabilities.add(f"{major}.{minor}")

nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
if not compute_capabilities:
    raise RuntimeError("No GPUs found.")

if nvcc_cuda_version < Version("12.0"):
    raise RuntimeError("CUDA 12.0 or higher is required.")

# Add target compute capabilities to NVCC flags
for capability in compute_capabilities:
    num = capability.replace(".", "").replace("+PTX", "")
    if num == '90':
        num = '90a'
        HAS_SM90 = True
        CXX_FLAGS += ["-DHAS_SM90"]
    NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
    if capability.endswith("+PTX"):
        NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=compute_{num}"]

ext_modules = []

# =====================================================================
# qattn extension: SM80 + optional SM90 block-sparse attention kernels
# =====================================================================

# Generate template instantiations
run_instantiations("csrc/qattn/instantiations_sm80")
if HAS_SM90:
    run_instantiations("csrc/qattn/instantiations_sm90")

sources = [
    "csrc/qattn/pybind.cpp",
    "csrc/qattn/qk_int_sv_f16_cuda_sm80.cu",
] + get_instantiations("csrc/qattn/instantiations_sm80")

if HAS_SM90:
    sources += ["csrc/qattn/qk_int_sv_f8_cuda_sm90.cu"]
    sources += get_instantiations("csrc/qattn/instantiations_sm90")

qattn_extension = CUDAExtension(
    name="_qattn",
    sources=sources,
    include_dirs=["csrc"],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
    extra_link_args=['-lcuda'],
)
ext_modules.append(qattn_extension)

# =====================================================================
# fused extension: V transpose + pad + permute + FP8 quantization
# =====================================================================

fused_extension = CUDAExtension(
    name="_fused",
    sources=["csrc/fused/pybind.cpp", "csrc/fused/fused.cu"],
    include_dirs=["csrc"],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(fused_extension)

setup(
    name='spargeattn_cuda',
    version='0.1.0',
    description='SpargeAttention block-sparse CUDA kernels (SM80 + SM90)',
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
