#!/usr/bin/env python3
"""
MiniTen Benchmark Suite: Vector Operations Comparison

This benchmark compares MiniTen's Cython-optimized vector operations
against NumPy and optionally PyTorch for honest performance comparison.

Results are saved to benchmarks/results/ and included in the README.
"""

import time
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import MiniTen operations
try:
    import vector_operations as miniten_ops
    MINITEN_AVAILABLE = True
except ImportError:
    try:
        from miniten.core import operations as miniten_ops
        MINITEN_AVAILABLE = True
    except ImportError:
        MINITEN_AVAILABLE = False
        print("Warning: MiniTen operations not available. Run `python setup.py build_ext --inplace` first.")

# Try to import PyTorch for comparison
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Note: PyTorch not installed. Skipping PyTorch benchmarks.")


def benchmark_function(func, *args, warmup=3, iterations=1000):
    """
    Benchmark a function with warmup and multiple iterations.
    
    Returns:
        dict: Mean time, min time, max time, and std deviation in microseconds
    """
    # Warmup runs
    for _ in range(warmup):
        func(*args)
    
    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append((end - start) * 1_000_000)  # Convert to microseconds
    
    return {
        'mean_us': np.mean(times),
        'min_us': np.min(times),
        'max_us': np.max(times),
        'std_us': np.std(times),
        'iterations': iterations
    }


def print_comparison(operation_name, results):
    """Print a formatted comparison table."""
    print(f"\n{'='*60}")
    print(f"  {operation_name}")
    print(f"{'='*60}")
    print(f"{'Library':<15} {'Mean (μs)':<12} {'Min (μs)':<12} {'Std (μs)':<12}")
    print(f"{'-'*60}")
    
    baseline = None
    for lib_name, result in results.items():
        if baseline is None:
            baseline = result['mean_us']
        speedup = baseline / result['mean_us'] if result['mean_us'] > 0 else 0
        speedup_str = f"({speedup:.2f}x)" if lib_name != 'numpy' else "(baseline)"
        print(f"{lib_name:<15} {result['mean_us']:<12.2f} {result['min_us']:<12.2f} {result['std_us']:<12.2f} {speedup_str}")


def run_vector_addition_benchmark(size):
    """Benchmark vector addition operations."""
    a = np.random.rand(size).astype(np.float64)
    b = np.random.rand(size).astype(np.float64)
    
    results = {}
    
    # NumPy benchmark
    results['numpy'] = benchmark_function(lambda: a + b)
    
    # MiniTen benchmark
    if MINITEN_AVAILABLE:
        results['miniten'] = benchmark_function(lambda: miniten_ops.vector_addition(a, b))
    
    # PyTorch benchmark
    if PYTORCH_AVAILABLE:
        a_torch = torch.from_numpy(a)
        b_torch = torch.from_numpy(b)
        results['pytorch'] = benchmark_function(lambda: a_torch + b_torch)
    
    return results


def run_dot_product_benchmark(size):
    """Benchmark dot product operations."""
    a = np.random.rand(size).astype(np.float64)
    b = np.random.rand(size).astype(np.float64)
    
    results = {}
    
    # NumPy benchmark
    results['numpy'] = benchmark_function(lambda: np.dot(a, b))
    
    # MiniTen benchmark
    if MINITEN_AVAILABLE:
        results['miniten'] = benchmark_function(lambda: miniten_ops.dot_product(a, b))
    
    # PyTorch benchmark
    if PYTORCH_AVAILABLE:
        a_torch = torch.from_numpy(a)
        b_torch = torch.from_numpy(b)
        results['pytorch'] = benchmark_function(lambda: torch.dot(a_torch, b_torch))
    
    return results


def run_element_wise_multiply_benchmark(size):
    """Benchmark element-wise multiplication."""
    a = np.random.rand(size).astype(np.float64)
    b = np.random.rand(size).astype(np.float64)
    
    results = {}
    
    # NumPy benchmark
    results['numpy'] = benchmark_function(lambda: a * b)
    
    # MiniTen benchmark
    if MINITEN_AVAILABLE:
        results['miniten'] = benchmark_function(lambda: miniten_ops.vector_multiplication(a, b))
    
    # PyTorch benchmark
    if PYTORCH_AVAILABLE:
        a_torch = torch.from_numpy(a)
        b_torch = torch.from_numpy(b)
        results['pytorch'] = benchmark_function(lambda: a_torch * b_torch)
    
    return results


def run_scalar_multiply_benchmark(size):
    """Benchmark scalar multiplication."""
    a = np.random.rand(size).astype(np.float64)
    scalar = 2.5
    
    results = {}
    
    # NumPy benchmark
    results['numpy'] = benchmark_function(lambda: a * scalar)
    
    # MiniTen benchmark
    if MINITEN_AVAILABLE:
        results['miniten'] = benchmark_function(lambda: miniten_ops.scalar_multiplication(a, scalar))
    
    # PyTorch benchmark
    if PYTORCH_AVAILABLE:
        a_torch = torch.from_numpy(a)
        results['pytorch'] = benchmark_function(lambda: a_torch * scalar)
    
    return results


def generate_markdown_results(all_results, vector_sizes):
    """Generate markdown formatted benchmark results."""
    md = []
    md.append("## Benchmark Results\n")
    md.append(f"*Benchmark run on: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")
    md.append(f"*System: Python {sys.version.split()[0]}*\n")
    md.append("")
    
    # Get list of available libraries
    libs = ['numpy', 'miniten']
    if PYTORCH_AVAILABLE:
        libs.append('pytorch')
    
    for operation_name, results_by_size in all_results.items():
        md.append(f"### {operation_name}\n")
        
        # Table header
        header = "| Vector Size |"
        separator = "|------------|"
        for lib in libs:
            header += f" {lib.capitalize()} (μs) |"
            separator += "------------:|"
        header += " MiniTen vs NumPy |"
        separator += "----------------:|"
        
        md.append(header)
        md.append(separator)
        
        for size, results in results_by_size.items():
            row = f"| {size:,} |"
            numpy_time = results.get('numpy', {}).get('mean_us', 0)
            miniten_time = results.get('miniten', {}).get('mean_us', 0)
            
            for lib in libs:
                if lib in results:
                    row += f" {results[lib]['mean_us']:.2f} |"
                else:
                    row += " N/A |"
            
            # Calculate speedup (positive = MiniTen faster, negative = slower)
            if miniten_time > 0 and numpy_time > 0:
                ratio = numpy_time / miniten_time
                if ratio >= 1:
                    speedup_str = f"{ratio:.2f}x faster"
                else:
                    speedup_str = f"{1/ratio:.2f}x slower"
            else:
                speedup_str = "N/A"
            row += f" {speedup_str} |"
            
            md.append(row)
        
        md.append("")
    
    return "\n".join(md)


def main():
    """Run all benchmarks and generate results."""
    print("="*60)
    print("  MiniTen Benchmark Suite")
    print("  Vector Operations Performance Comparison")
    print("="*60)
    
    if not MINITEN_AVAILABLE:
        print("\nError: MiniTen not built. Please run:")
        print("  python setup.py build_ext --inplace")
        return 1
    
    # Vector sizes to test
    vector_sizes = [100, 1000, 10000, 100000]
    
    all_results = {
        'Vector Addition': {},
        'Dot Product': {},
        'Element-wise Multiply': {},
        'Scalar Multiply': {}
    }
    
    for size in vector_sizes:
        print(f"\n{'='*60}")
        print(f"  Vector Size: {size:,}")
        print(f"{'='*60}")
        
        # Vector Addition
        results = run_vector_addition_benchmark(size)
        all_results['Vector Addition'][size] = results
        print_comparison(f"Vector Addition (size={size})", results)
        
        # Dot Product
        results = run_dot_product_benchmark(size)
        all_results['Dot Product'][size] = results
        print_comparison(f"Dot Product (size={size})", results)
        
        # Element-wise Multiply
        results = run_element_wise_multiply_benchmark(size)
        all_results['Element-wise Multiply'][size] = results
        print_comparison(f"Element-wise Multiply (size={size})", results)
        
        # Scalar Multiply
        results = run_scalar_multiply_benchmark(size)
        all_results['Scalar Multiply'][size] = results
        print_comparison(f"Scalar Multiply (size={size})", results)
    
    # Generate markdown results
    markdown_results = generate_markdown_results(all_results, vector_sizes)
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results
    results_file = os.path.join(results_dir, 'vector_operations.md')
    with open(results_file, 'w') as f:
        f.write("# Vector Operations Benchmark Results\n\n")
        f.write(markdown_results)
    
    print(f"\n\nResults saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    print("\nKey findings:")
    print("- NumPy uses highly optimized BLAS/LAPACK libraries")
    print("- MiniTen's Cython implementation provides good performance")
    print("- For small vectors, function call overhead dominates")
    print("- For large vectors, NumPy's SIMD optimizations excel")
    print("\nMiniTen's advantages:")
    print("- Smaller memory footprint than full PyTorch/TensorFlow")
    print("- Educational codebase for learning ML internals")
    print("- Optimized for edge computing scenarios")
    print("- Easy to extend and customize")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
