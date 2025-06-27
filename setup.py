from setuptools import setup, Extension,find_packages
from setuptools.command.build_ext import build_ext
from setuptools_rust import RustExtension
import os
import numpy as np
from Cython.Build import cythonize
import numpy

class BuildWithCythonAndRust(build_ext):
    def build_extensions(self):
        # Let setuptools_rust do its job (RustExtensions are already supported natively)
        build_ext.build_extensions(self)
# Use absolute paths to avoid confusion
base_dir = os.path.dirname(os.path.abspath(__file__))
cython_dir = os.path.join(base_dir, "utils")

# Check if files exist and print their paths for debugging
ac_utils_path = os.path.join(cython_dir, "ac_utils2.pyx")
seg_utils_path = os.path.join(cython_dir, "seg_utils.pyx")
processor_path = os.path.join(cython_dir,"cellprocessor.pyx")

# print(f"Checking if ac_utils.pyx exists at: {ac_utils_path} -> {os.path.exists(ac_utils_path)}")
# print(f"Checking if seg_utils.pyx exists at: {seg_utils_path} -> {os.path.exists(seg_utils_path)}")

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


# Rust extension (PyO3 crate)
# rust_module = [ RustExtension(
#         "AnalysisGUI.utils.ac_processor",
#         path="Cargo.toml",
#         binding="pyo3",
#         debug=False
#     )]


setup(
    ext_modules=cython_modules,
    # rust_extensions = rust_module,
    # cmdclass={"build_ext": BuildWithCythonAndRust},
    # zip_safe=False,
    # include_package_data=True
)