from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

extensions = [
    Extension("*", ["**/*.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=[
        '-fopenmp',
        '-Ofast',
        '-pipe',
        '-march=native',
        '-mtune=native',
        '-qopt-report=5',
        '-fargument-noalias-global',
    ],
        extra_link_args=['-fopenmp'],
        language='c++',
    ),
]

setup(
    name="Ground Template Filter Array",
    ext_modules=cythonize(extensions),  # accepts a glob pattern
)
