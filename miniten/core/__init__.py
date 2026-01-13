"""
Core Module - Fundamental Operations and Data Structures

This module contains the foundational building blocks for MiniTen:
- Tensor operations
- Automatic differentiation
- Computational graph
- Memory management
- Optimized mathematical operations

All operations are implemented in Cython for maximum performance.
"""

# Import working Cython modules
try:
    from . import operations
    from . import backprop
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False
    import warnings
    warnings.warn("Cython modules not built. Run 'python setup.py build_ext --inplace'")

# Future imports (will be uncommented as implemented)
# from .tensor import Tensor
# from .autograd import AutogradEngine

__all__ = [
    'operations',
    'backprop',
    # 'Tensor',
    # 'AutogradEngine',
]
