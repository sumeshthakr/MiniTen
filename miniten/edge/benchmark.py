"""
Model benchmarking utilities for edge deployment.

Provides comprehensive benchmarking for evaluating model
performance on edge devices.
"""

import time
import numpy as np


def benchmark_model(model, input_shape, num_runs=100, warmup_runs=10):
    """
    Benchmark a model's inference performance.
    
    Args:
        model: Model to benchmark
        input_shape: Shape of input tensor
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Dictionary with benchmark results
    """
    # Generate random input
    x = np.random.randn(*input_shape).astype(np.float64)
    
    # Warmup runs
    for _ in range(warmup_runs):
        if hasattr(model, 'forward'):
            _ = model.forward(x)
        elif callable(model):
            _ = model(x)
    
    # Benchmark runs
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        if hasattr(model, 'forward'):
            _ = model.forward(x)
        elif callable(model):
            _ = model(x)
        end = time.perf_counter()
        latencies.append(end - start)
    
    latencies = np.array(latencies) * 1000  # Convert to ms
    
    return {
        "mean_latency_ms": np.mean(latencies),
        "std_latency_ms": np.std(latencies),
        "min_latency_ms": np.min(latencies),
        "max_latency_ms": np.max(latencies),
        "median_latency_ms": np.median(latencies),
        "p95_latency_ms": np.percentile(latencies, 95),
        "p99_latency_ms": np.percentile(latencies, 99),
        "throughput_fps": 1000 / np.mean(latencies),
        "num_runs": num_runs,
    }


def benchmark_layer(layer, input_shape, num_runs=100):
    """
    Benchmark a single layer's performance.
    
    Args:
        layer: Layer to benchmark
        input_shape: Input shape
        num_runs: Number of runs
        
    Returns:
        Benchmark results
    """
    return benchmark_model(layer, input_shape, num_runs=num_runs)


def latency_test(model, input_shape, target_latency_ms=10):
    """
    Test if model meets latency requirements.
    
    Args:
        model: Model to test
        input_shape: Input shape
        target_latency_ms: Target latency in milliseconds
        
    Returns:
        Dictionary with test results
    """
    results = benchmark_model(model, input_shape)
    
    return {
        "passes": results["mean_latency_ms"] <= target_latency_ms,
        "actual_latency_ms": results["mean_latency_ms"],
        "target_latency_ms": target_latency_ms,
        "margin_ms": target_latency_ms - results["mean_latency_ms"],
    }


def throughput_test(model, input_shape, duration_seconds=5):
    """
    Measure model throughput over a duration.
    
    Args:
        model: Model to test
        input_shape: Input shape
        duration_seconds: Test duration
        
    Returns:
        Throughput results
    """
    x = np.random.randn(*input_shape).astype(np.float64)
    
    count = 0
    start = time.time()
    
    while time.time() - start < duration_seconds:
        if hasattr(model, 'forward'):
            _ = model.forward(x)
        elif callable(model):
            _ = model(x)
        count += 1
    
    elapsed = time.time() - start
    
    return {
        "total_inferences": count,
        "duration_seconds": elapsed,
        "throughput_fps": count / elapsed,
        "latency_ms": (elapsed / count) * 1000,
    }


def memory_benchmark(model, input_shape):
    """
    Benchmark memory usage during inference.
    
    Args:
        model: Model to benchmark
        input_shape: Input shape
        
    Returns:
        Memory usage results
    """
    import gc
    
    # Force garbage collection
    gc.collect()
    
    try:
        import tracemalloc
        tracemalloc.start()
        
        # Run inference
        x = np.random.randn(*input_shape).astype(np.float64)
        if hasattr(model, 'forward'):
            _ = model.forward(x)
        elif callable(model):
            _ = model(x)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            "current_memory_mb": current / (1024 * 1024),
            "peak_memory_mb": peak / (1024 * 1024),
        }
    except ImportError:
        return {
            "error": "tracemalloc not available",
            "current_memory_mb": 0,
            "peak_memory_mb": 0,
        }


def compare_models(models, input_shape, labels=None):
    """
    Compare multiple models.
    
    Args:
        models: List of models to compare
        input_shape: Input shape
        labels: Optional labels for models
        
    Returns:
        Comparison table
    """
    if labels is None:
        labels = [f"model_{i}" for i in range(len(models))]
    
    results = []
    for model, label in zip(models, labels):
        benchmark = benchmark_model(model, input_shape)
        memory = memory_benchmark(model, input_shape)
        
        results.append({
            "label": label,
            "latency_ms": benchmark["mean_latency_ms"],
            "throughput_fps": benchmark["throughput_fps"],
            "peak_memory_mb": memory["peak_memory_mb"],
        })
    
    return results


def profile_by_layer(model, input_shape):
    """
    Profile each layer of a model.
    
    Args:
        model: Model to profile
        input_shape: Input shape
        
    Returns:
        Layer-by-layer profiling results
    """
    results = []
    
    if not hasattr(model, 'layers') and not hasattr(model, 'modules'):
        return {"error": "Model does not expose layers"}
    
    layers = getattr(model, 'layers', getattr(model, 'modules', []))
    
    x = np.random.randn(*input_shape).astype(np.float64)
    current_input = x
    
    for i, layer in enumerate(layers):
        start = time.perf_counter()
        if hasattr(layer, 'forward'):
            output = layer.forward(current_input)
        elif callable(layer):
            output = layer(current_input)
        else:
            continue
        end = time.perf_counter()
        
        results.append({
            "layer_index": i,
            "layer_type": type(layer).__name__,
            "input_shape": current_input.shape,
            "output_shape": output.shape if hasattr(output, 'shape') else 'unknown',
            "latency_ms": (end - start) * 1000,
        })
        
        current_input = output
    
    return results


class BenchmarkReport:
    """
    Generate comprehensive benchmark reports.
    """
    
    def __init__(self, model, input_shape):
        """
        Initialize benchmark report.
        
        Args:
            model: Model to benchmark
            input_shape: Input shape
        """
        self.model = model
        self.input_shape = input_shape
        self.results = {}
    
    def run_all(self):
        """Run all benchmarks."""
        self.results["latency"] = benchmark_model(self.model, self.input_shape)
        self.results["memory"] = memory_benchmark(self.model, self.input_shape)
        self.results["throughput"] = throughput_test(self.model, self.input_shape, 2)
        return self.results
    
    def summary(self):
        """Generate summary string."""
        if not self.results:
            self.run_all()
        
        lines = [
            "=" * 50,
            "BENCHMARK REPORT",
            "=" * 50,
            f"Input shape: {self.input_shape}",
            "",
            "LATENCY",
            f"  Mean: {self.results['latency']['mean_latency_ms']:.2f} ms",
            f"  Std:  {self.results['latency']['std_latency_ms']:.2f} ms",
            f"  P95:  {self.results['latency']['p95_latency_ms']:.2f} ms",
            "",
            "THROUGHPUT",
            f"  FPS: {self.results['throughput']['throughput_fps']:.1f}",
            "",
            "MEMORY",
            f"  Peak: {self.results['memory']['peak_memory_mb']:.2f} MB",
            "=" * 50,
        ]
        
        return "\n".join(lines)
    
    def to_dict(self):
        """Export results as dictionary."""
        if not self.results:
            self.run_all()
        return self.results
    
    def save(self, path):
        """Save results to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
