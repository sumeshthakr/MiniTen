"""
Vector Operations Example

Demonstrates the optimized Cython vector operations in MiniTen.
"""

import numpy as np
import sys
import time
sys.path.insert(0, '..')

from miniten.core import operations


def benchmark_operation(name, func, *args, iterations=1000):
    """Benchmark a vector operation."""
    start = time.time()
    for _ in range(iterations):
        result = func(*args)
    end = time.time()
    elapsed = (end - start) * 1000  # Convert to milliseconds
    avg_time = elapsed / iterations
    return result, avg_time


def main():
    print("=" * 60)
    print("MiniTen - Optimized Vector Operations Example")
    print("=" * 60)
    
    # Create test vectors
    size = 1000
    a = np.random.randn(size).astype(np.float64)
    b = np.random.randn(size).astype(np.float64)
    scalar = 2.5
    
    print(f"\nVector size: {size}")
    print(f"Running {1000} iterations for benchmarking...\n")
    
    # Test and benchmark each operation
    operations_to_test = [
        ("Vector Addition", operations.vector_addition, a, b),
        ("Vector Subtraction", operations.vector_subtraction, a, b),
        ("Vector Multiplication", operations.vector_multiplication, a, b),
        ("Vector Division", operations.vector_division, a, b),
        ("Scalar Multiplication", operations.scalar_multiplication, a, scalar),
        ("Dot Product", operations.dot_product, a, b),
    ]
    
    print("-" * 60)
    print(f"{'Operation':<25} {'Result (first 5)':<25} {'Avg Time (ms)'}")
    print("-" * 60)
    
    for name, func, *args in operations_to_test:
        result, avg_time = benchmark_operation(name, func, *args)
        
        # Format result
        if isinstance(result, (int, float)):
            result_str = f"{result:.4f}"
        else:
            result_str = str(result[:5])
        
        print(f"{name:<25} {result_str:<25} {avg_time:.6f}")
    
    # Test cross product (3D vectors only)
    a_3d = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    b_3d = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    result, avg_time = benchmark_operation(
        "Cross Product (3D)",
        operations.cross_product,
        a_3d, b_3d
    )
    print(f"{'Cross Product (3D)':<25} {str(result):<25} {avg_time:.6f}")
    
    print("-" * 60)
    
    # Detailed example
    print("\n" + "=" * 60)
    print("Detailed Example:")
    print("=" * 60)
    
    a_small = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b_small = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    
    print(f"\nVector a: {a_small}")
    print(f"Vector b: {b_small}")
    print(f"Scalar:   {scalar}")
    
    print(f"\na + b = {operations.vector_addition(a_small, b_small)}")
    print(f"a - b = {operations.vector_subtraction(a_small, b_small)}")
    print(f"a * b (element-wise) = {operations.vector_multiplication(a_small, b_small)}")
    print(f"a / b (element-wise) = {operations.vector_division(a_small, b_small)}")
    print(f"a * {scalar} = {operations.scalar_multiplication(a_small, scalar)}")
    print(f"a · b (dot product) = {operations.dot_product(a_small, b_small)}")
    
    print("\n" + "=" * 60)
    print("All operations completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
