"""
MiniTen - A Lightweight Deep Learning Framework for Edge Platforms

MiniTen is a high-performance deep learning library optimized for edge computing,
built entirely in Python and Cython with minimal external dependencies.

Key Features:
- Optimized for edge platforms with minimal footprint
- Pure Python/Cython implementation
- Support for CNNs, RNNs, LSTMs, GNNs, and Reinforcement Learning
- GPU acceleration for edge devices
- Processing for images, audio, video, language, and signals
- Comprehensive documentation and examples

Modules:
- core: Core tensor operations and computational graph
- nn: Neural network layers and building blocks
- optim: Optimization algorithms
- utils: Utility functions and data processing
- gpu: GPU acceleration backends

Example:
    >>> import miniten as mt
    >>> # Create a simple neural network
    >>> # model = mt.nn.Sequential([...])
    
Version: 0.1.0
Author: MiniTen Contributors
License: MIT
"""

__version__ = "0.1.0"
__author__ = "MiniTen Contributors"
__license__ = "MIT"

# Core imports will be added as modules are implemented
# from .core import *
# from .nn import *
# from .optim import *
# from .utils import *

__all__ = [
    "__version__",
    "__author__",
    "__license__",
]
