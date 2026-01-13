# MiniTen Installation Guide

This guide walks you through installing MiniTen on various platforms.

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 512 MB RAM (for basic operations)
- 50 MB disk space

### Recommended Requirements
- Python 3.9 or higher
- 2 GB RAM
- 200 MB disk space
- C compiler (GCC, Clang, or MSVC)

## Installation Methods

### Method 1: Install from Source (Current)

This is currently the only available method as MiniTen is in active development.

#### Step 1: Clone the Repository
```bash
git clone https://github.com/sumeshthakr/MiniTen.git
cd MiniTen
```

#### Step 2: Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Build Cython Extensions
```bash
python setup.py build_ext --inplace
```

#### Step 5: Install in Development Mode
```bash
pip install -e .
```

#### Step 6: Verify Installation
```bash
# Test imports
python -c "from miniten.core import operations, backprop; print('MiniTen imported successfully!')"

# Run tests
python tests/test_vector_operations.py

# Run examples
python examples/vector_operations.py
```

### Method 2: Install from PyPI (Coming Soon)

Once published to PyPI:
```bash
pip install miniten
```

## Platform-Specific Instructions

### Ubuntu/Debian
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev python3-pip gcc

# Install MiniTen
git clone https://github.com/sumeshthakr/MiniTen.git
cd MiniTen
pip install -r requirements.txt
python setup.py build_ext --inplace
pip install -e .
```

### macOS
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Python (if not already installed)
brew install python3

# Install MiniTen
git clone https://github.com/sumeshthakr/MiniTen.git
cd MiniTen
pip3 install -r requirements.txt
python3 setup.py build_ext --inplace
pip3 install -e .
```

### Windows
```powershell
# Install Visual Studio Build Tools (if not already installed)
# Download from: https://visualstudio.microsoft.com/downloads/

# Install MiniTen
git clone https://github.com/sumeshthakr/MiniTen.git
cd MiniTen
pip install -r requirements.txt
python setup.py build_ext --inplace
pip install -e .
```

### Raspberry Pi / ARM Devices
```bash
# Update system
sudo apt-get update
sudo apt-get upgrade

# Install dependencies
sudo apt-get install python3-dev python3-pip gcc

# Install MiniTen
git clone https://github.com/sumeshthakr/MiniTen.git
cd MiniTen
pip3 install -r requirements.txt
python3 setup.py build_ext --inplace
pip3 install -e .
```

### NVIDIA Jetson (for GPU support)
```bash
# Install CUDA (usually pre-installed)
# Verify: nvcc --version

# Install MiniTen
git clone https://github.com/sumeshthakr/MiniTen.git
cd MiniTen
pip3 install -r requirements.txt
python3 setup.py build_ext --inplace
pip3 install -e .

# GPU support will be added in future releases
```

## Optional Dependencies

### For Development
```bash
pip install pytest pytest-cov black flake8 mypy
```

### For Documentation
```bash
pip install sphinx sphinx-rtd-theme
```

### For Benchmarking
```bash
pip install matplotlib pandas
```

## GPU Support (Coming Soon)

### CUDA (NVIDIA)
```bash
# Install CUDA Toolkit from NVIDIA website
# https://developer.nvidia.com/cuda-downloads

# Verify installation
nvcc --version
```

### OpenCL (Cross-platform)
```bash
# Ubuntu/Debian
sudo apt-get install opencl-headers ocl-icd-opencl-dev

# macOS (built-in)
# No additional installation needed

# Windows
# Install from GPU vendor (NVIDIA, AMD, Intel)
```

### Metal (Apple Silicon)
```bash
# Built-in on macOS
# No additional installation needed
```

## Troubleshooting

### Build Errors

**Problem**: `error: command 'gcc' failed`
```bash
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# macOS
xcode-select --install

# Windows
# Install Visual Studio Build Tools
```

**Problem**: `ModuleNotFoundError: No module named 'numpy'`
```bash
pip install numpy
```

**Problem**: Cython compilation errors
```bash
pip install --upgrade cython
python setup.py clean --all
python setup.py build_ext --inplace
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'miniten'`
```bash
# Ensure you're in the right directory
cd /path/to/MiniTen

# Install in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH=/path/to/MiniTen:$PYTHONPATH
```

### Runtime Errors

**Problem**: Slow performance
```bash
# Ensure Cython modules are compiled
ls miniten/core/*.so  # Should show compiled modules

# If not, rebuild
python setup.py build_ext --inplace
```

## Verifying Installation

### Quick Test
```python
import miniten as mt
from miniten.core import operations, backprop
import numpy as np

# Test vector operations
a = np.array([1, 2, 3], dtype=np.float64)
b = np.array([4, 5, 6], dtype=np.float64)
result = operations.vector_addition(a, b)
print(f"Vector addition: {result}")

# Test backpropagation
bp = backprop.BackPropagation(2, 3, 1)
print("BackPropagation initialized successfully!")

print("\nMiniTen is ready to use! 🚀")
```

### Run Full Test Suite
```bash
# Run all tests
PYTHONPATH=. python tests/test_vector_operations.py
PYTHONPATH=. python tests/test_backprop.py

# Run examples
python examples/vector_operations.py
python examples/basic_network.py
```

## Next Steps

After installation:
1. Read the [Quick Start Guide](QUICKSTART.md)
2. Try the [Examples](../examples/)
3. Read the [Documentation](README.md)
4. Join the [Community](https://github.com/sumeshthakr/MiniTen/discussions)

## Getting Help

If you encounter issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Search [existing issues](https://github.com/sumeshthakr/MiniTen/issues)
3. Open a [new issue](https://github.com/sumeshthakr/MiniTen/issues/new)
4. Join the [discussion](https://github.com/sumeshthakr/MiniTen/discussions)

## Keeping MiniTen Updated

```bash
cd MiniTen
git pull origin main
python setup.py build_ext --inplace
pip install -e .
```

---

**Ready to start?** Head to the [Quick Start Guide](QUICKSTART.md)!
