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
- Comprehensive documentation and examples

Modules:
- core: Core tensor operations and computational graph
- nn: Neural network layers and building blocks
- optim: Optimization algorithms
- utils: Utility functions and data processing
- viz: Visualization engine (fast plotting)
- monitor: Training monitoring and callbacks
- edge: Edge deployment (quantization, pruning, export)

Example:
    >>> import miniten as mt
    >>> from miniten.monitor import MetricsLogger
    >>> from miniten.viz import Figure
    >>> 
    >>> # Log training metrics
    >>> logger = MetricsLogger("./runs")
    >>> logger.log("loss", 0.5, step=1)
    >>> 
    >>> # Create visualizations
    >>> fig = Figure()
    >>> ax = fig.add_subplot()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> fig.save("plot.png")
    
Version: 0.2.0
Author: MiniTen Contributors
License: MIT
"""

__version__ = "0.2.0"
__author__ = "MiniTen Contributors"
__license__ = "MIT"

# Import submodules for easy access
from . import nn
from . import optim
from . import utils

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

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "nn",
    "optim",
    "utils",
    "viz",
    "monitor",
    "edge",
]
