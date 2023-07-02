# setup.py

from distutils.core import setup
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

setup(
    ext_modules=cythonize("vector_operations.pyx"),
    include_dirs=[numpy.get_include()],
    cmdclass={'test': PyTest}
)