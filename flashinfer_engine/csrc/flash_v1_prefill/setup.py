from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="custom_flash_attention_v1",
    ext_modules=[
        CUDAExtension(
            name="custom_flash_attention_v1",
            sources=["attention_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-arch=sm_90",          # H100/H200; use sm_80 for A100
                    "--ptxas-options=-v",   # show register/shared mem usage
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
