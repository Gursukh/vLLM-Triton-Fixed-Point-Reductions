"""Build script for the fxpr_vllm CUDA extension.

Run `pip install -e .` from the repo root. The setuptools-driven
`BuildExtension` invokes nvcc once at install time; there is no JIT path
and no pre-built wheel.

The fat-binary covers Turing through Hopper (sm_75 -> sm_90). Override
TORCH_CUDA_ARCH_LIST in the environment to narrow it.
"""

import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6;8.9;9.0")


setup(
    name="fxpr_vllm",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="fxpr_vllm._cuda",
            sources=[
                "csrc/bindings.cpp",
                "csrc/casts.cu",
                "csrc/scales.cu",
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
                    # Determinism: no FMA reordering. nvcc's default is
                    # already no --use_fast_math; --fmad=false additionally
                    # disables auto-FMA fusion of separate * and + ops.
                    "--fmad=false",
                    "-Xptxas",
                    "-O3",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch>=2.6", "vllm"],
)
