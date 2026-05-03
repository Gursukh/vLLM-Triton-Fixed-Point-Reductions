"""Build script for the fxpr_vllm CUDA extension.

Default fat binary spans sm_75..sm_100; narrow via TORCH_CUDA_ARCH_LIST.
sm_75 minimum (mma.sync). bf16/fp32 inputs need sm_80+.
"""

import os
import re

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6;8.9;9.0;10.0;12.0;12.0+PTX")

_arch_list = os.environ["TORCH_CUDA_ARCH_LIST"]
for _entry in _arch_list.replace(",", ";").split(";"):
    _entry = _entry.strip()
    if not _entry:
        continue
    _m = re.match(r"^(\d+)\.(\d+)", _entry)
    if not _m:
        raise RuntimeError(f"Unrecognised TORCH_CUDA_ARCH_LIST entry: {_entry!r}")
    _major, _minor = int(_m.group(1)), int(_m.group(2))
    if (_major, _minor) < (7, 5):
        raise RuntimeError(
            f"fxpr_vllm requires compute capability >= 7.5 (Turing). "
            f"TORCH_CUDA_ARCH_LIST contains {_entry!r}; remove it or set "
            f"TORCH_CUDA_ARCH_LIST to the architectures you want to target."
        )


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
