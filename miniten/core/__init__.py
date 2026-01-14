"""
Core Module - Fundamental Operations and Data Structures

This module contains the foundational building blocks for MiniTen:
- Tensor operations with automatic differentiation
- Automatic differentiation engine
- Computational graph
- Memory management
- Optimized mathematical operations

All operations are implemented in Cython for maximum performance.
"""

# Import Tensor and factory functions
from .tensor import (
    Tensor, Context,
    zeros, ones, randn, rand, arange, eye,
    from_numpy, cat, stack
)

# Import autograd components
from .autograd import (
    AutogradEngine, GradientTape, Function, no_grad, grad,
    Add, Mul, MatMul, ReLU, Sigmoid, Tanh, Softmax, Log, Exp,
    CrossEntropyLoss, MSELoss
)

# Import working Cython modules
try:
    from . import operations
    from . import backprop
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False
    operations = None
    backprop = None
    import warnings
    warnings.warn("Cython modules not built. Run 'python setup.py build_ext --inplace'")

__all__ = [
    # Tensor
    'Tensor', 'Context',
    'zeros', 'ones', 'randn', 'rand', 'arange', 'eye',
    'from_numpy', 'cat', 'stack',
    # Autograd
    'AutogradEngine', 'GradientTape', 'Function', 'no_grad', 'grad',
    'Add', 'Mul', 'MatMul', 'ReLU', 'Sigmoid', 'Tanh', 'Softmax', 'Log', 'Exp',
    'CrossEntropyLoss', 'MSELoss',
    # Cython modules
    'operations',
    'backprop',
    'HAS_CYTHON',
]
