"""Build script for the fxpr_vllm CUDA extension.

`pip install -e .` invokes nvcc once at install time. Fat binary covers
Turing through Hopper (sm_75 -> sm_90); narrow it via TORCH_CUDA_ARCH_LIST.
"""

import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6;8.9;9.0")


setup(
    name="fxpr_vllm",
    version="0.1.0",
    packages=find_packages(include=["fxpr_vllm*"]),
    ext_modules=[
        CUDAExtension(
            name="fxpr_vllm._cuda",
            sources=[
                "csrc/bindings.cpp",
                "csrc/ops.cpp",
                "csrc/casts.cu",
                "csrc/rms_norm.cu",
                "csrc/softmax.cu",
                "csrc/gemm.cu",
                "csrc/attention.cu",
            ],
            include_dirs=["csrc"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "-Xptxas",
                    "-O3",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch>=2.6", "vllm"],
)
