import os

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


def _nvcc_arch_flags() -> list[str]:
    archs = os.environ.get("FLASHINFER_CUDA_ARCHS", "80,90")
    flags: list[str] = []
    for arch in (a.strip() for a in archs.split(",")):
        if not arch:
            continue
        flags.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
    return flags

setup(
    name="custom_flash_attention_v1",
    ext_modules=[
        CUDAExtension(
            name="custom_flash_attention_v1",
            sources=["attention_kernel.cu", "pv_wmma_debug.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--ptxas-options=-v",   # show register/shared mem usage
                ] + _nvcc_arch_flags(),
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
