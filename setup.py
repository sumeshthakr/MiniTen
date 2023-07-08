# setup.py

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
from setuptools.command.test import test as TestCommand
import numpy
import sys


class PyTest(TestCommand):
    def run_tests(self):
        import pytest
        errno = pytest.main(["--cov=./", "--cov-report=term-missing"])
        sys.exit(errno)


cython_sources = [
    "backprop.pyx",
    "vector_operations.pyx"
]

# List of test files
test_files = [
    "test_backprop.pyx",
    "test_vector_operations.pyx"
]

extensions = []
for source_file in cython_sources:
    module_name = source_file.split(".")[0]
    extension = Extension(module_name, [source_file], include_dirs=[numpy.get_include()])
    extensions.append(extension)

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
    py_modules=test_files
)
