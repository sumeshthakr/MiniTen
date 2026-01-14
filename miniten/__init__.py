"""
MiniTen - A Lightweight Deep Learning Framework for Edge Platforms

MiniTen is a high-performance deep learning library optimized for edge computing,
built entirely in Python and Cython with minimal external dependencies.

Key Features:
- Optimized for edge platforms with minimal footprint
- Pure Python/Cython implementation
- Support for CNNs, RNNs, LSTMs, GNNs, and Transformers
- Built-in visualization engine (fast plotting)
- Training monitoring system (like TensorBoard)
- Edge deployment tools (quantization, pruning, ONNX)
- Reinforcement learning basics
- Comprehensive benchmarking suite
- Audio, video, signal, and NLP processing

Modules:
- core: Core tensor operations and computational graph
- nn: Neural network layers and building blocks
- optim: Optimization algorithms
- utils: Utility functions and data processing
- viz: Visualization engine (fast plotting)
- monitor: Training monitoring and callbacks
- edge: Edge deployment (quantization, pruning, export)
- rl: Reinforcement learning algorithms
- gpu: GPU acceleration and kernel optimization

Example:
    >>> import miniten as mt
    >>> from miniten.core import Tensor, zeros, randn
    >>> from miniten.rl import DQN, CartPoleEnv
    >>> 
    >>> # Create tensor with autograd
    >>> x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    >>> y = x * 2
    >>> y.sum().backward()
    >>> print(x.grad)
    
Version: 0.3.0
Author: MiniTen Contributors
License: MIT
"""

__version__ = "0.3.0"
__author__ = "MiniTen Contributors"
__license__ = "MIT"

# Import submodules for easy access
from . import core
from . import nn
from . import optim
from . import utils
from . import gpu

# Try to import optional modules
try:
    from . import viz
except ImportError:
    viz = None

try:
    from . import monitor
except ImportError:
    monitor = None

try:
    from . import edge
except ImportError:
    edge = None

try:
    from . import rl
except ImportError:
    rl = None

# Expose commonly used items at top level
from .core import Tensor, zeros, ones, randn, rand, arange, eye

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    # Modules
    "core",
    "nn",
    "optim",
    "utils",
    "gpu",
    "viz",
    "monitor",
    "edge",
    "rl",
    # Core tensor functions
    "Tensor",
    "zeros",
    "ones",
    "randn",
    "rand",
    "arange",
    "eye",
]
