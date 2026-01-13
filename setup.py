# setup.py
"""
MiniTen Setup Script

Builds Cython extensions and installs the MiniTen package.
"""

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy
import sys
import os


def get_extensions():
    """
    Collect all Cython source files and create extensions.
    """
    extensions = []
    
    # Determine OpenMP flags
    if sys.platform == "win32":
        openmp_compile_args = ["/openmp"]
        openmp_link_args = []
        other_compile_args = []
    elif sys.platform == "darwin":
        # macOS: Try to use libomp if available
        openmp_compile_args = ["-Xpreprocessor", "-fopenmp"]
        openmp_link_args = ["-lomp"]
        other_compile_args = ["-O3", "-march=native"]
    else:
        # Linux and others
        openmp_compile_args = ["-fopenmp"]
        openmp_link_args = ["-fopenmp"]
        other_compile_args = ["-O3", "-march=native"]
    
    # Core module extensions (with OpenMP)
    core_sources = [
        "miniten/core/operations.pyx",
        "miniten/core/backprop.pyx",
    ]
    
    for source_file in core_sources:
        if os.path.exists(source_file):
            module_name = source_file.replace("/", ".").replace(".pyx", "")
            extension = Extension(
                module_name,
                [source_file],
                include_dirs=[numpy.get_include()],
                extra_compile_args=other_compile_args + openmp_compile_args,
                extra_link_args=openmp_link_args,
            )
            extensions.append(extension)
    
    # Neural network module extensions (with OpenMP)
    nn_sources = [
        "miniten/nn/layers_impl.pyx",
        "miniten/nn/cnn_impl.pyx",
        "miniten/nn/rnn_impl.pyx",
        "miniten/nn/advanced_impl.pyx",
    ]
    
    for source_file in nn_sources:
        if os.path.exists(source_file):
            module_name = source_file.replace("/", ".").replace(".pyx", "")
            extension = Extension(
                module_name,
                [source_file],
                include_dirs=[numpy.get_include()],
                extra_compile_args=other_compile_args + openmp_compile_args,
                extra_link_args=openmp_link_args,
            )
            extensions.append(extension)
    
    # Root level extensions (for backward compatibility with tests)
    root_sources = [
        "vector_operations.pyx",
        "backprop.pyx",
    ]
    
    for source_file in root_sources:
        if os.path.exists(source_file):
            module_name = source_file.replace(".pyx", "")
            extension = Extension(
                module_name,
                [source_file],
                include_dirs=[numpy.get_include()],
                extra_compile_args=other_compile_args,
            )
            extensions.append(extension)
    
    # Optimizer extensions (with OpenMP)
    optim_sources = [
        "miniten/optim/optim_impl.pyx",
    ]
    
    for source_file in optim_sources:
        if os.path.exists(source_file):
            module_name = source_file.replace("/", ".").replace(".pyx", "")
            extension = Extension(
                module_name,
                [source_file],
                include_dirs=[numpy.get_include()],
                extra_compile_args=other_compile_args + openmp_compile_args,
                extra_link_args=openmp_link_args,
            )
            extensions.append(extension)
    
    # Utils extensions
    utils_sources = [
        "miniten/utils/deployment.pyx",
    ]
    
    for source_file in utils_sources:
        if os.path.exists(source_file):
            module_name = source_file.replace("/", ".").replace(".pyx", "")
            extension = Extension(
                module_name,
                [source_file],
                include_dirs=[numpy.get_include()],
                extra_compile_args=other_compile_args,
            )
            extensions.append(extension)
    
    return extensions


def read_requirements():
    """Read requirements from requirements.txt"""
    try:
        with open("requirements.txt", "r") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        # Fallback to minimal requirements
        return ["numpy>=1.19", "cython>=0.29"]


# Read version from miniten/__init__.py safely
import re
version = "0.1.0"  # Default version
try:
    with open("miniten/__init__.py") as f:
        content = f.read()
        match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content, re.MULTILINE)
        if match:
            version = match.group(1)
except FileNotFoundError:
    pass

# Read long description safely
long_description = ""
try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "A lightweight deep learning framework optimized for edge platforms"

setup(
    name="miniten",
    version=version,
    description="A lightweight deep learning framework optimized for edge platforms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MiniTen Contributors",
    author_email="",
    url="https://github.com/sumeshthakr/MiniTen",
    license="MIT",
    packages=find_packages(exclude=["tests", "examples", "benchmarks", "docs"]),
    ext_modules=cythonize(
        get_extensions(),
        compiler_directives={
            "language_level": "3",
            "embedsignature": True,
            "boundscheck": False,
            "wraparound": False,
        }
    ),
    include_dirs=[numpy.get_include()],
    install_requires=read_requirements(),
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="deep-learning edge-computing neural-networks machine-learning optimization",
    project_urls={
        "Bug Reports": "https://github.com/sumeshthakr/MiniTen/issues",
        "Source": "https://github.com/sumeshthakr/MiniTen",
        "Documentation": "https://github.com/sumeshthakr/MiniTen/tree/main/docs",
    },
)
