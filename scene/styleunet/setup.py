from setuptools import setup
from torch.utils import cpp_extension

# From Gaussian-Garments: pip install --no-build-isolation ./scene/styleunet
setup(
    name="gaussian-garments-styleunet",
    ext_modules=[
        cpp_extension.CUDAExtension(
            "upfirdn2d", ["upfirdn2d.cpp", "upfirdn2d_kernel.cu"]
        ),
        cpp_extension.CUDAExtension(
            "fused", ["fused_bias_act.cpp", "fused_bias_act_kernel.cu"]
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
