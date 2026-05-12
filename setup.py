from setuptools import find_packages, setup


setup(
    name="fxpr_vllm",
    version="0.1.0",
    packages=find_packages(include=["fxpr_vllm*"]),
    install_requires=["torch>=2.6", "triton>=3.0", "vllm"],
)
