import os
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Define the C++ extension
ext_modules = [
    CUDAExtension(
        name="sam2._C",
        sources=[
            "sam2/csrc/connected_components.cu"
        ],
        extra_compile_args={
            "cxx": [],
            "nvcc": [
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_CONVERSIONS__",
            ],
        },
    )
]

setup(
    name="sam2",
    version="1.0",
    packages=find_packages(),
    package_data={
        "sam2": ["configs/**/*.yaml"],
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
