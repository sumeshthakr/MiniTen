# run.py
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("vec_multi", ["vec_multi.pyx"],
              include_dirs=[numpy.get_include()])
]

setup(
    ext_modules=cythonize(extensions)
)