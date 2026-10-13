from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="greenlet-torch-tls",
    packages=[],
    ext_modules=[
        CppExtension(
            "_greenlet_torch_tls",
            [str(Path(__file__).with_name("torch_tls.cpp"))],
            extra_compile_args=["-O0"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
