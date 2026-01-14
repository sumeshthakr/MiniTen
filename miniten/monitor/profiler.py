"""
Profiling utilities for MiniTen.

Provides memory and performance profiling for edge device optimization.
"""

import time
import sys
import gc
from functools import wraps


class MemoryProfiler:
    """
    Memory profiler for tracking memory usage during training.
    
    Optimized for edge devices where memory is constrained.
    
    Args:
        logger: Optional MetricsLogger for logging
        
    Example:
        >>> profiler = MemoryProfiler()
        >>> profiler.start()
        >>> # ... training code ...
        >>> report = profiler.stop()
        >>> print(report)
    """
    
    def __init__(self, logger=None):
        """Initialize memory profiler."""
        self.logger = logger
        self._start_memory = None
        self._peak_memory = 0
        self._samples = []
    
    def start(self):
        """Start memory profiling."""
        gc.collect()
        self._start_memory = self._get_memory_usage()
        self._peak_memory = self._start_memory
        self._samples = [(time.time(), self._start_memory)]
    
    def sample(self, label=None):
        """Take a memory sample."""
        current = self._get_memory_usage()
        self._samples.append((time.time(), current))
        
        if current > self._peak_memory:
            self._peak_memory = current
        
        if self.logger and label:
            self.logger.log(f"memory_{label}", current)
        
        return current
    
    def stop(self):
        """Stop profiling and return report."""
        gc.collect()
        end_memory = self._get_memory_usage()
        self._samples.append((time.time(), end_memory))
        
        return {
            "start_memory_mb": self._start_memory,
            "end_memory_mb": end_memory,
            "peak_memory_mb": self._peak_memory,
            "memory_increase_mb": end_memory - self._start_memory,
            "samples": len(self._samples),
        }
    
    def _get_memory_usage(self):
        """Get current memory usage in MB."""
        try:
            import resource
            # Get memory usage from resource module (Unix)
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_maxrss / 1024  # Convert KB to MB
        except ImportError:
            try:
                import psutil
                process = psutil.Process()
                return process.memory_info().rss / (1024 * 1024)
            except ImportError:
                # Fallback: estimate from Python objects
                return sys.getsizeof(gc.get_objects()) / (1024 * 1024)
    
    def report(self):
        """Generate a detailed memory report."""
        if not self._samples:
            return "No profiling data available."
        
        lines = ["Memory Profile Report", "=" * 40]
        
        start_time = self._samples[0][0]
        for timestamp, memory in self._samples:
            elapsed = timestamp - start_time
            lines.append(f"  {elapsed:8.2f}s: {memory:8.2f} MB")
        
        lines.extend([
            "=" * 40,
            f"Peak Memory: {self._peak_memory:.2f} MB",
            f"Memory Increase: {self._samples[-1][1] - self._start_memory:.2f} MB",
        ])
        
        return "\n".join(lines)


class PerformanceProfiler:
    """
    Performance profiler for timing operations.
    
    Tracks time spent in different parts of training.
    
    Args:
        logger: Optional MetricsLogger for logging
        
    Example:
        >>> profiler = PerformanceProfiler()
        >>> with profiler.timer("forward_pass"):
        ...     output = model(input)
        >>> print(profiler.summary())
    """
    
    def __init__(self, logger=None):
        """Initialize performance profiler."""
        self.logger = logger
        self._timings = {}
        self._active_timers = {}
        self._start_time = None
    
    def start(self):
        """Start profiling session."""
        self._start_time = time.time()
        self._timings = {}
    
    def timer(self, name):
        """
        Context manager for timing a block of code.
        
        Args:
            name: Timer name
            
        Returns:
            Context manager
        """
        return _TimerContext(self, name)
    
    def start_timer(self, name):
        """Start a named timer."""
        self._active_timers[name] = time.time()
    
    def stop_timer(self, name):
        """Stop a named timer and record the elapsed time."""
        if name in self._active_timers:
            elapsed = time.time() - self._active_timers[name]
            if name not in self._timings:
                self._timings[name] = {"count": 0, "total": 0, "min": float('inf'), "max": 0}
            
            self._timings[name]["count"] += 1
            self._timings[name]["total"] += elapsed
            self._timings[name]["min"] = min(self._timings[name]["min"], elapsed)
            self._timings[name]["max"] = max(self._timings[name]["max"], elapsed)
            
            del self._active_timers[name]
            
            if self.logger:
                self.logger.log(f"time_{name}", elapsed)
            
            return elapsed
        return 0
    
    def summary(self):
        """Get a summary of all timings."""
        if not self._timings:
            return "No timing data available."
        
        total_time = time.time() - self._start_time if self._start_time else 0
        
        lines = ["Performance Profile Summary", "=" * 60]
        lines.append(f"{'Operation':<25} {'Count':>8} {'Total':>10} {'Mean':>10} {'%':>8}")
        lines.append("-" * 60)
        
        for name, stats in sorted(self._timings.items(), key=lambda x: -x[1]["total"]):
            count = stats["count"]
            total = stats["total"]
            mean = total / count if count > 0 else 0
            percent = (total / total_time * 100) if total_time > 0 else 0
            
            lines.append(f"{name:<25} {count:>8} {total:>9.3f}s {mean:>9.4f}s {percent:>7.1f}%")
        
        lines.append("-" * 60)
        lines.append(f"Total elapsed time: {total_time:.2f}s")
        
        return "\n".join(lines)
    
    def get_timing(self, name):
        """Get timing stats for a specific operation."""
        return self._timings.get(name, {})
    
    def reset(self):
        """Reset all timings."""
        self._timings = {}
        self._active_timers = {}
        self._start_time = None


class _TimerContext:
    """Context manager for timing."""
    
    def __init__(self, profiler, name):
        self.profiler = profiler
        self.name = name
    
    def __enter__(self):
        self.profiler.start_timer(self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.stop_timer(self.name)
        return False


def profile_function(func):
    """
    Decorator to profile a function's execution time.
    
    Example:
        >>> @profile_function
        ... def my_function():
        ...     # code
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__}: {elapsed:.4f}s")
        return result
    return wrapper


class FLOPsCounter:
    """
    Count FLOPs (floating point operations) in a model.
    
    Args:
        model: Model to analyze
        
    Example:
        >>> counter = FLOPsCounter(model)
        >>> flops = counter.count(input_shape=(1, 3, 224, 224))
        >>> print(f"FLOPs: {flops:,}")
    """
    
    def __init__(self, model=None):
        """Initialize FLOPs counter."""
        self.model = model
        self._flops = 0
    
    def count(self, input_shape):
        """
        Count FLOPs for a given input shape.
        
        Args:
            input_shape: Shape of input tensor
            
        Returns:
            Total FLOPs
        """
        self._flops = 0
        
        if self.model is None:
            return 0
        
        # Analyze model structure (placeholder for actual implementation)
        # Would traverse model layers and count operations
        
        return self._flops
    
    @staticmethod
    def count_linear_flops(in_features, out_features, bias=True):
        """Count FLOPs for a linear layer."""
        flops = in_features * out_features  # Multiplications
        flops += in_features * out_features  # Additions
        if bias:
            flops += out_features
        return flops
    
    @staticmethod
    def count_conv2d_flops(in_channels, out_channels, kernel_size, 
                           output_height, output_width, bias=True):
        """Count FLOPs for a Conv2d layer."""
        # Each output element: kernel_size^2 * in_channels multiplications + additions
        kernel_flops = kernel_size * kernel_size * in_channels * 2
        output_elements = output_height * output_width * out_channels
        flops = kernel_flops * output_elements
        if bias:
            flops += output_elements
        return flops


def estimate_model_size(model):
    """
    Estimate model size in bytes.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with size information
    """
    import numpy as np
    
    total_params = 0
    total_bytes = 0
    
    # Count parameters (placeholder for actual implementation)
    if hasattr(model, 'parameters'):
        for param in model.parameters():
            if hasattr(param, 'size'):
                total_params += np.prod(param.size)
                total_bytes += np.prod(param.size) * 4  # Assume float32
    
    return {
        "total_parameters": total_params,
        "size_bytes": total_bytes,
        "size_kb": total_bytes / 1024,
        "size_mb": total_bytes / (1024 * 1024),
    }
