from setuptools import setup, Extension,find_packages
from setuptools.command.build_ext import build_ext
from setuptools_rust import RustExtension
import os
import numpy as np
from Cython.Build import cythonize
import numpy

# Use absolute path to avoid confusion
base_dir = os.path.dirname(os.path.abspath(__file__))
cython_dir = os.path.join(base_dir, "utils")




# Use full module names
ac_extension = Extension(
    "AnalysisGUI.utils.ac_utils",
    [ "AnalysisGUI/utils/ac_utils2.pyx"],
    language="c++",
    extra_compile_args=["-O3","-ffast-math"],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
)

seg_extension = Extension(
    "AnalysisGUI.utils.seg_utils",
    ["AnalysisGUI/utils/seg_utils.pyx"],
    language="c++",
    extra_compile_args=["-O3","-ffast-math"],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
)

processor_extension = Extension(
    "AnalysisGUI.utils.cellprocessor",
    ["AnalysisGUI/utils/cellprocessor.pyx"],
    language="c++",
    extra_compile_args=["-O3","-ffast-math"],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
)

extensions = [ac_extension, seg_extension,processor_extension]
cython_modules = cythonize(extensions, include_path=[cython_dir], compiler_directives = {"language_level":3})





setup(
    ext_modules=cython_modules,
    
)