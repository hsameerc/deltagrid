import os
import sys
import shutil
from setuptools import setup, find_packages

# 1. Ensure ninja is on PATH
try:
    import ninja
    ninja_dir = ninja.BIN_DIR
    if ninja_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ninja_dir + os.pathsep + os.environ.get("PATH", "")
except ImportError:
    pass

# 2. Ensure MSVC cl.exe is in environment on Windows
if sys.platform == 'win32' and not shutil.which("cl"):
    try:
        import setuptools._distutils._msvccompiler as msvc
        vc_env = msvc._get_vc_env("x64")
        for k, v in vc_env.items():
            k_upper = k.upper()
            if k_upper == "PATH":
                os.environ["PATH"] = v + os.pathsep + os.environ.get("PATH", "")
            elif k_upper in ("INCLUDE", "LIB", "LIBPATH"):
                os.environ[k_upper] = v
    except Exception as e:
        print(f"Warning: could not automatically configure MSVC environment: {e}")

if sys.platform == 'win32':
    os.environ["CXXFLAGS"] = os.environ.get("CXXFLAGS", "") + " /std:c++17 /Zc:preprocessor"
    os.environ["CFLAGS"] = os.environ.get("CFLAGS", "") + " /std:c++17 /Zc:preprocessor"
    os.environ["CL"] = "/std:c++17 /Zc:preprocessor"

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        if self.compiler.compiler_type == 'msvc':
            if hasattr(self.compiler, '_compile'):
                old_compile = self.compiler._compile
                def new_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
                    if cc_args is None:
                        cc_args = []
                    if '/std:c++17' not in cc_args:
                        cc_args.insert(0, '/std:c++17')
                    if '/Zc:preprocessor' not in cc_args:
                        cc_args.insert(1, '/Zc:preprocessor')
                    return old_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
                self.compiler._compile = new_compile
            
            if hasattr(self.compiler, 'compile'):
                old_compile_outer = self.compiler.compile
                def new_compile_outer(sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
                    if extra_preargs is None:
                        extra_preargs = []
                    if isinstance(extra_preargs, list):
                        if '/std:c++17' not in extra_preargs:
                            extra_preargs.insert(0, '/std:c++17')
                        if '/Zc:preprocessor' not in extra_preargs:
                            extra_preargs.insert(1, '/Zc:preprocessor')
                    return old_compile_outer(sources, output_dir, macros, include_dirs, debug, extra_preargs, extra_postargs, depends)
                self.compiler.compile = new_compile_outer

        super().build_extensions()

ext_modules = []
cmdclass = {}

try:
    # Bypass the strict CUDA version check (e.g. 13.3 vs 12.8)
    torch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None
    BuildExtension._check_abi = lambda *args, **kwargs: ("cl", "19.51")

    # =========================================================================
    # DELTAGRID CUDA EXTENSION
    # =========================================================================
    if torch.cuda.is_available():
        print("CUDA detected! Building high-performance DeltaGrid CUDA extension...")
        
        abi_flag = f"-D_GLIBCXX_USE_CXX11_ABI={1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0}"
        
        ext_modules.append(
            CUDAExtension(
                name='hsru_delta_cuda',
                sources=[
                    'src/cpp/binding.cpp',
                    'src/cpp/delta_kernel.cu', 
                ],
                extra_compile_args={
                    'cxx': ['-O3', '-g', abi_flag] if sys.platform != 'win32' else ['/Zc:preprocessor', '/std:c++17'],
                    'nvcc': ['-O3', abi_flag] if sys.platform != 'win32' else [
                        '-O3',
                        '-Xcompiler', '/Zc:preprocessor',
                        '-std=c++17',
                        '-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING'
                    ]
                }
            )
        )
    else:
        print("CUDA not found. Skipping CUDA extension. Core Python framework will still be installed.")

    cmdclass['build_ext'] = CustomBuildExtension

except ImportError:
    print("torch not found during setup; skipping C++ extension build.")

setup(
    name='hsru-delta',
    version='0.2.0',
    description='HSRU DeltaGrid: Zero-KV Fast-Weight Associative Recurrent Architecture',
    author="Sameer Humagain",
    author_email="im@hsameer.com.np",
    license="MIT",
    packages=[
        'hsru_delta',
        'hsru_delta.ops',
        'hsru_delta.modules',
        'hsru_delta.models',
        'hsru_delta.core',
    ],
    package_dir={
        'hsru_delta': 'src',
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    include_package_data=True,
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.22.0',
        'matplotlib>=3.5.0',
        'ninja>=1.10.0',
    ]
)
