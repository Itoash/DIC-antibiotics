import os
import shutil
from setuptools import Distribution, Extension
import numpy as np
from Cython.Build import build_ext, cythonize
import platform
import subprocess
from pathlib import Path
rust_dir = Path(__file__).parent / "src" / "rust_modules" / "ac_processing"
if os.path.exists(rust_dir):
    subprocess.run(['maturin', 'build', '--release'], cwd=rust_dir, check=True)
    subprocess.run(['maturin', 'develop', '--release'], cwd=rust_dir, check=True)

system = platform.system()
if system == "Windows":
    subprocess.check_call(["pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu126"])
    subprocess.check_call(["pip", "install", "cupy-cuda12x"])
else:
    subprocess.check_call(["pip", "install", "torch", "torchvision"])
subprocess.check_call(("pip","install","pyqt5"))
# Use absolute paths to avoid confusion
base_dir = os.path.dirname(os.path.abspath(__file__))
cython_dir = os.path.join(base_dir, "src", "AnalysisGUI", "utils")

# Check if files exist and print their paths for debugging
ac_utils_path = os.path.join(cython_dir, "ac_utils2.pyx")
seg_utils_path = os.path.join(cython_dir, "seg_utils.pyx")
processor_path = os.path.join(cython_dir,"cellprocessor.pyx")

# print(f"Checking if ac_utils.pyx exists at: {ac_utils_path} -> {os.path.exists(ac_utils_path)}")
# print(f"Checking if seg_utils.pyx exists at: {seg_utils_path} -> {os.path.exists(seg_utils_path)}")

# Use full module names
ac_extension = Extension(
    "AnalysisGUI.utils.ac_utils",
    [ac_utils_path],
    language="c++",
    extra_compile_args=["-O3","-ffast-math"],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
)

seg_extension = Extension(
    "AnalysisGUI.utils.seg_utils",
    [seg_utils_path],
    language="c++",
    extra_compile_args=["-O3","-ffast-math"],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
)

processor_extension = Extension(
    "AnalysisGUI.utils.cellprocessor",
    [processor_path],
    language="c++",
    extra_compile_args=["-O3","-ffast-math"],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
)

extensions = [ac_extension, seg_extension,processor_extension]
ext_modules = cythonize(extensions, include_path=[cython_dir], compiler_directives = {"language_level":3})
print(f"Cythonized modules: {ext_modules}")

dist = Distribution({"ext_modules": ext_modules})
cmd = build_ext(dist)
cmd.ensure_finalized()
cmd.run()

# # Copy the built extensions to the src directory structure
for output in cmd.get_outputs():
    # Extract just the filename
    filename = os.path.basename(output)
    
    # Determine the target directory within src
    module_path = output.split(os.path.join('AnalysisGUI','utils'))[0]
    target_dir = os.path.join(base_dir, "src", "AnalysisGUI", "utils")
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy to the target location
    target_path = os.path.join(target_dir, filename)
    print(f"Copying {output} -> {target_path}")
    shutil.copyfile(output, target_path)

