"""
MiniTen Training Monitor

A built-in, optimized training monitoring system similar to
TensorBoard and Weights & Biases, but optimized for edge devices.

Features:
- Metrics logging (loss, accuracy, custom metrics)
- Training progress tracking with callbacks
- Model checkpointing
- Experiment tracking and comparison
- HTML dashboard generation
- Real-time visualization
- Memory-efficient logging

Example:
    >>> from miniten.monitor import MetricsLogger, Callback
    >>> logger = MetricsLogger(log_dir="./runs")
    >>> logger.log("loss", 0.5, step=1)
    >>> logger.log("accuracy", 0.85, step=1)
    >>> 
    >>> # Use callbacks
    >>> callback = Callback(logger)
    >>> callback.on_epoch_end(epoch=1, logs={"loss": 0.5})
"""

from .logger import MetricsLogger, Experiment
from .callbacks import Callback, EarlyStopping, ModelCheckpoint, ProgressBar
from .dashboard import Dashboard
from .profiler import MemoryProfiler, PerformanceProfiler

__all__ = [
    "MetricsLogger",
    "Experiment",
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "ProgressBar",
    "Dashboard",
    "MemoryProfiler",
    "PerformanceProfiler",
]
