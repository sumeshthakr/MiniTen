"""
Comprehensive Benchmarking Suite

Performance benchmarking tools for MiniTen.
Measures latency, throughput, memory usage, and power efficiency.

Features:
- Operation benchmarking
- Model benchmarking
- Memory profiling
- Comparison with baselines
- Report generation
"""

import time
import math
from typing import List, Dict, Optional, Callable, Any, Tuple
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    mean_time: float  # seconds
    std_time: float
    min_time: float
    max_time: float
    iterations: int
    throughput: Optional[float] = None  # ops/sec
    memory_peak: Optional[int] = None  # bytes
    
    def __repr__(self):
        return (f"BenchmarkResult(name='{self.name}', "
                f"mean={self.mean_time*1000:.3f}ms, "
                f"std={self.std_time*1000:.3f}ms)")


class Timer:
    """High-resolution timer for benchmarking."""
    
    def __init__(self):
        self.start_time = 0
        self.elapsed = 0
    
    def start(self):
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        self.elapsed = time.perf_counter() - self.start_time
        return self.elapsed
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


def benchmark_function(func: Callable, 
                       args: tuple = (),
                       kwargs: dict = None,
                       warmup: int = 5,
                       iterations: int = 100,
                       name: str = None) -> BenchmarkResult:
    """
    Benchmark a function.
    
    Args:
        func: Function to benchmark
        args: Positional arguments for function
        kwargs: Keyword arguments for function
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
        name: Name for the benchmark
        
    Returns:
        BenchmarkResult with timing statistics
    """
    if kwargs is None:
        kwargs = {}
    
    name = name or func.__name__
    
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Timed runs
    times = []
    for _ in range(iterations):
        with Timer() as t:
            func(*args, **kwargs)
        times.append(t.elapsed)
    
    mean_time = sum(times) / len(times)
    variance = sum((t - mean_time) ** 2 for t in times) / len(times)
    std_time = math.sqrt(variance)
    
    return BenchmarkResult(
        name=name,
        mean_time=mean_time,
        std_time=std_time,
        min_time=min(times),
        max_time=max(times),
        iterations=iterations,
        throughput=1.0 / mean_time if mean_time > 0 else 0
    )


def compare_functions(funcs: Dict[str, Callable],
                      args: tuple = (),
                      kwargs: dict = None,
                      warmup: int = 5,
                      iterations: int = 100) -> Dict[str, BenchmarkResult]:
    """
    Compare performance of multiple functions.
    
    Args:
        funcs: Dictionary of name -> function
        args: Positional arguments for functions
        kwargs: Keyword arguments
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
        
    Returns:
        Dictionary of name -> BenchmarkResult
    """
    results = {}
    for name, func in funcs.items():
        results[name] = benchmark_function(
            func, args, kwargs, warmup, iterations, name
        )
    return results


class OperationBenchmark:
    """Benchmark suite for tensor operations."""
    
    def __init__(self, sizes: List[int] = None):
        self.sizes = sizes or [100, 1000, 10000, 100000]
        self.results: Dict[str, List[BenchmarkResult]] = {}
    
    def run_all(self, iterations: int = 100) -> Dict[str, List[BenchmarkResult]]:
        """Run all operation benchmarks."""
        self.results = {}
        
        # Vector operations
        self.results['vector_add'] = self._benchmark_vector_add(iterations)
        self.results['vector_mul'] = self._benchmark_vector_mul(iterations)
        self.results['dot_product'] = self._benchmark_dot_product(iterations)
        
        # Matrix operations
        self.results['matmul'] = self._benchmark_matmul(iterations)
        self.results['transpose'] = self._benchmark_transpose(iterations)
        
        # Activation functions
        self.results['relu'] = self._benchmark_relu(iterations)
        self.results['sigmoid'] = self._benchmark_sigmoid(iterations)
        self.results['softmax'] = self._benchmark_softmax(iterations)
        
        return self.results
    
    def _benchmark_vector_add(self, iterations: int) -> List[BenchmarkResult]:
        """Benchmark vector addition."""
        results = []
        import random
        
        for size in self.sizes:
            a = [random.random() for _ in range(size)]
            b = [random.random() for _ in range(size)]
            
            def vector_add():
                return [x + y for x, y in zip(a, b)]
            
            result = benchmark_function(
                vector_add, 
                iterations=iterations,
                name=f'vector_add_{size}'
            )
            results.append(result)
        
        return results
    
    def _benchmark_vector_mul(self, iterations: int) -> List[BenchmarkResult]:
        """Benchmark element-wise multiplication."""
        results = []
        import random
        
        for size in self.sizes:
            a = [random.random() for _ in range(size)]
            b = [random.random() for _ in range(size)]
            
            def vector_mul():
                return [x * y for x, y in zip(a, b)]
            
            result = benchmark_function(
                vector_mul,
                iterations=iterations,
                name=f'vector_mul_{size}'
            )
            results.append(result)
        
        return results
    
    def _benchmark_dot_product(self, iterations: int) -> List[BenchmarkResult]:
        """Benchmark dot product."""
        results = []
        import random
        
        for size in self.sizes:
            a = [random.random() for _ in range(size)]
            b = [random.random() for _ in range(size)]
            
            def dot_product():
                return sum(x * y for x, y in zip(a, b))
            
            result = benchmark_function(
                dot_product,
                iterations=iterations,
                name=f'dot_product_{size}'
            )
            results.append(result)
        
        return results
    
    def _benchmark_matmul(self, iterations: int) -> List[BenchmarkResult]:
        """Benchmark matrix multiplication."""
        results = []
        import random
        
        matrix_sizes = [10, 50, 100, 200]
        
        for size in matrix_sizes:
            a = [[random.random() for _ in range(size)] for _ in range(size)]
            b = [[random.random() for _ in range(size)] for _ in range(size)]
            
            def matmul():
                result = [[0.0] * size for _ in range(size)]
                for i in range(size):
                    for j in range(size):
                        for k in range(size):
                            result[i][j] += a[i][k] * b[k][j]
                return result
            
            result = benchmark_function(
                matmul,
                iterations=min(iterations, 10) if size >= 100 else iterations,
                name=f'matmul_{size}x{size}'
            )
            results.append(result)
        
        return results
    
    def _benchmark_transpose(self, iterations: int) -> List[BenchmarkResult]:
        """Benchmark matrix transpose."""
        results = []
        import random
        
        matrix_sizes = [10, 50, 100, 500]
        
        for size in matrix_sizes:
            a = [[random.random() for _ in range(size)] for _ in range(size)]
            
            def transpose():
                return [[a[j][i] for j in range(size)] for i in range(size)]
            
            result = benchmark_function(
                transpose,
                iterations=iterations,
                name=f'transpose_{size}x{size}'
            )
            results.append(result)
        
        return results
    
    def _benchmark_relu(self, iterations: int) -> List[BenchmarkResult]:
        """Benchmark ReLU activation."""
        results = []
        import random
        
        for size in self.sizes:
            a = [random.uniform(-1, 1) for _ in range(size)]
            
            def relu():
                return [max(0, x) for x in a]
            
            result = benchmark_function(
                relu,
                iterations=iterations,
                name=f'relu_{size}'
            )
            results.append(result)
        
        return results
    
    def _benchmark_sigmoid(self, iterations: int) -> List[BenchmarkResult]:
        """Benchmark sigmoid activation."""
        results = []
        import random
        
        for size in self.sizes:
            a = [random.uniform(-5, 5) for _ in range(size)]
            
            def sigmoid():
                return [1.0 / (1.0 + math.exp(-x)) for x in a]
            
            result = benchmark_function(
                sigmoid,
                iterations=iterations,
                name=f'sigmoid_{size}'
            )
            results.append(result)
        
        return results
    
    def _benchmark_softmax(self, iterations: int) -> List[BenchmarkResult]:
        """Benchmark softmax."""
        results = []
        import random
        
        for size in [10, 100, 1000]:
            a = [random.uniform(-5, 5) for _ in range(size)]
            
            def softmax():
                max_val = max(a)
                exp_vals = [math.exp(x - max_val) for x in a]
                sum_exp = sum(exp_vals)
                return [e / sum_exp for e in exp_vals]
            
            result = benchmark_function(
                softmax,
                iterations=iterations,
                name=f'softmax_{size}'
            )
            results.append(result)
        
        return results
    
    def generate_report(self) -> str:
        """Generate a text report of benchmark results."""
        lines = ["=" * 60, "Operation Benchmark Report", "=" * 60, ""]
        
        for op_name, results in self.results.items():
            lines.append(f"\n{op_name.upper()}")
            lines.append("-" * 40)
            lines.append(f"{'Size':<20} {'Mean (ms)':<15} {'Std (ms)':<15}")
            lines.append("-" * 40)
            
            for result in results:
                size = result.name.split('_')[-1]
                lines.append(
                    f"{size:<20} {result.mean_time*1000:>12.4f}   "
                    f"{result.std_time*1000:>12.4f}"
                )
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


class ModelBenchmark:
    """Benchmark suite for neural network models."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def benchmark_forward_pass(self, model_fn: Callable,
                               input_shape: Tuple[int, ...],
                               batch_sizes: List[int] = None,
                               iterations: int = 100) -> List[BenchmarkResult]:
        """
        Benchmark model forward pass.
        
        Args:
            model_fn: Function that creates and runs model
            input_shape: Shape of single input (excluding batch)
            batch_sizes: List of batch sizes to test
            iterations: Number of iterations
            
        Returns:
            List of BenchmarkResult
        """
        import random
        
        if batch_sizes is None:
            batch_sizes = [1, 8, 16, 32]
        
        results = []
        
        for batch_size in batch_sizes:
            # Create input
            total_elements = batch_size
            for dim in input_shape:
                total_elements *= dim
            
            input_data = [random.random() for _ in range(total_elements)]
            
            def forward():
                return model_fn(input_data)
            
            result = benchmark_function(
                forward,
                iterations=iterations,
                name=f'forward_batch_{batch_size}'
            )
            results.append(result)
        
        self.results.extend(results)
        return results
    
    def benchmark_inference_latency(self, model_fn: Callable,
                                    input_data: List[float],
                                    percentiles: List[float] = None,
                                    iterations: int = 1000) -> Dict[str, float]:
        """
        Measure inference latency with percentiles.
        
        Args:
            model_fn: Function to run
            input_data: Input data
            percentiles: Percentiles to compute (default: [50, 90, 95, 99])
            iterations: Number of iterations
            
        Returns:
            Dictionary with latency statistics
        """
        if percentiles is None:
            percentiles = [50, 90, 95, 99]
        
        times = []
        
        # Warmup
        for _ in range(10):
            model_fn(input_data)
        
        # Timed runs
        for _ in range(iterations):
            with Timer() as t:
                model_fn(input_data)
            times.append(t.elapsed * 1000)  # Convert to ms
        
        times.sort()
        
        result = {
            'mean_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times)
        }
        
        for p in percentiles:
            idx = int(len(times) * p / 100)
            result[f'p{int(p)}_ms'] = times[min(idx, len(times) - 1)]
        
        return result


class MemoryProfiler:
    """Profile memory usage."""
    
    def __init__(self):
        self.samples: List[int] = []
        self.peak_memory = 0
    
    def get_current_memory(self) -> int:
        """Get current memory usage in bytes."""
        import os
        import sys
        
        try:
            if sys.platform == 'linux':
                try:
                    with open('/proc/self/statm', 'r') as f:
                        pages = int(f.read().split()[1])
                        return pages * os.sysconf('SC_PAGE_SIZE')
                except (OSError, IOError, ValueError, IndexError):
                    # Fallback if /proc is not available
                    pass
            
            # Fallback for non-Linux or if /proc access failed
            try:
                import gc
                gc.collect()
                return sum(sys.getsizeof(obj) for obj in gc.get_objects())
            except Exception:
                return 0
        except Exception:
            return 0
    
    def sample(self):
        """Take a memory sample."""
        current = self.get_current_memory()
        self.samples.append(current)
        self.peak_memory = max(self.peak_memory, current)
    
    def reset(self):
        """Reset profiler."""
        self.samples = []
        self.peak_memory = 0
    
    def get_stats(self) -> Dict[str, int]:
        """Get memory statistics."""
        if not self.samples:
            return {'current': 0, 'peak': 0, 'average': 0}
        
        return {
            'current': self.samples[-1] if self.samples else 0,
            'peak': self.peak_memory,
            'average': sum(self.samples) // len(self.samples),
            'samples': len(self.samples)
        }


class ComprehensiveBenchmark:
    """
    Complete benchmarking suite combining all benchmarks.
    """
    
    def __init__(self):
        self.operation_benchmark = OperationBenchmark()
        self.model_benchmark = ModelBenchmark()
        self.memory_profiler = MemoryProfiler()
    
    def run_full_suite(self, iterations: int = 100) -> Dict[str, Any]:
        """
        Run complete benchmark suite.
        
        Args:
            iterations: Number of iterations per benchmark
            
        Returns:
            Complete benchmark results
        """
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'operations': {},
            'memory': {},
            'summary': {}
        }
        
        # Memory before
        self.memory_profiler.reset()
        self.memory_profiler.sample()
        mem_before = self.memory_profiler.get_current_memory()
        
        # Run operation benchmarks
        op_results = self.operation_benchmark.run_all(iterations)
        results['operations'] = {
            name: [
                {
                    'name': r.name,
                    'mean_ms': r.mean_time * 1000,
                    'std_ms': r.std_time * 1000,
                    'throughput': r.throughput
                }
                for r in benchmarks
            ]
            for name, benchmarks in op_results.items()
        }
        
        # Memory after
        self.memory_profiler.sample()
        mem_after = self.memory_profiler.get_current_memory()
        
        results['memory'] = {
            'before_bytes': mem_before,
            'after_bytes': mem_after,
            'delta_bytes': mem_after - mem_before,
            'peak_bytes': self.memory_profiler.peak_memory
        }
        
        # Summary
        total_ops = sum(len(v) for v in op_results.values())
        all_times = []
        for benchmarks in op_results.values():
            all_times.extend([r.mean_time for r in benchmarks])
        
        results['summary'] = {
            'total_benchmarks': total_ops,
            'total_time_seconds': sum(all_times),
            'average_time_ms': (sum(all_times) / len(all_times) * 1000) if all_times else 0
        }
        
        return results
    
    def generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML benchmark report."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>MiniTen Benchmark Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .section { margin: 30px 0; }
        .summary { background: #f9f9f9; padding: 15px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>MiniTen Benchmark Report</h1>
    <p>Generated: """ + results['timestamp'] + """</p>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Total Benchmarks: """ + str(results['summary']['total_benchmarks']) + """</p>
        <p>Total Time: """ + f"{results['summary']['total_time_seconds']:.2f}" + """ seconds</p>
        <p>Average Time: """ + f"{results['summary']['average_time_ms']:.3f}" + """ ms</p>
    </div>
    
    <div class="section">
        <h2>Operation Benchmarks</h2>
"""
        
        for op_name, benchmarks in results['operations'].items():
            html += f"<h3>{op_name}</h3>\n"
            html += """<table>
            <tr><th>Configuration</th><th>Mean (ms)</th><th>Std (ms)</th><th>Throughput (ops/s)</th></tr>
"""
            for b in benchmarks:
                html += f"<tr><td>{b['name']}</td><td>{b['mean_ms']:.4f}</td>"
                html += f"<td>{b['std_ms']:.4f}</td><td>{b['throughput']:.2f}</td></tr>\n"
            html += "</table>\n"
        
        html += """
    </div>
    
    <div class="section">
        <h2>Memory Usage</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Memory Before</td><td>""" + f"{results['memory']['before_bytes'] / 1024:.2f} KB" + """</td></tr>
            <tr><td>Memory After</td><td>""" + f"{results['memory']['after_bytes'] / 1024:.2f} KB" + """</td></tr>
            <tr><td>Peak Memory</td><td>""" + f"{results['memory']['peak_bytes'] / 1024:.2f} KB" + """</td></tr>
        </table>
    </div>
</body>
</html>"""
        
        return html
    
    def save_report(self, filepath: str, results: Dict[str, Any] = None):
        """Save benchmark report to file."""
        if results is None:
            results = self.run_full_suite()
        
        if filepath.endswith('.html'):
            content = self.generate_html_report(results)
        else:
            # JSON format
            import json
            content = json.dumps(results, indent=2)
        
        with open(filepath, 'w') as f:
            f.write(content)


def quick_benchmark(func: Callable, *args, **kwargs) -> float:
    """
    Quick benchmark - returns time in milliseconds.
    
    Usage:
        time_ms = quick_benchmark(my_function, arg1, arg2)
    """
    # Warmup
    func(*args, **kwargs)
    
    # Timed run
    start = time.perf_counter()
    func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    
    return elapsed * 1000
