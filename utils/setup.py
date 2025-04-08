from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
extensions = [
    Extension(
        "ac_utils",
        ["ac_utils.pyx"],
        include_dirs=[np.get_include()]
    ),
    Extension(
        "seg_utils",["seg_utils.pyx"],
        include_dirs=[np.get_include()]
        ),
    Extension(
        "track_utils",["track_utils.pyx"],
        include_dirs=[np.get_include()]
        )
]

setup(
    
    ext_modules=cythonize(extensions,
                          annotate=True,
                         compiler_directives={
                             'language_level': "3",
                             'boundscheck': False,
                             'wraparound': False,
                             'nonecheck': False,
                             'cdivision': True,
                         })
)