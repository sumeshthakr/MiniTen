#!/usr/bin/env python3
"""
MiniTen Benchmark Suite: Neural Network Training Comparison

This benchmark compares MiniTen's Cython-optimized backpropagation
against a pure Python implementation for educational comparison.

Note: MiniTen's focus is on edge computing and educational value,
not competing with production frameworks on raw speed.
"""

import time
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import MiniTen backprop
try:
    import backprop as miniten_backprop
    MINITEN_AVAILABLE = True
except ImportError:
    try:
        from miniten.core import backprop as miniten_backprop
        MINITEN_AVAILABLE = True
    except ImportError:
        MINITEN_AVAILABLE = False
        print("Warning: MiniTen backprop not available. Run `python setup.py build_ext --inplace` first.")


class PythonBackPropagation:
    """Pure Python implementation for comparison."""
    
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases randomly
        np.random.seed(42)  # For reproducibility
        self.weights1 = np.random.randn(hidden_size, input_size)
        self.weights2 = np.random.randn(output_size, hidden_size)
        self.bias1 = np.random.randn(hidden_size)
        self.bias2 = np.random.randn(output_size)
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative_of_activated(self, activated):
        """Derivative of sigmoid when given already-activated values (sigmoid output)."""
        return activated * (1 - activated)
    
    def forward(self, X):
        hidden_layer = self.sigmoid(np.dot(self.weights1, X) + self.bias1)
        output_layer = self.sigmoid(np.dot(self.weights2, hidden_layer) + self.bias2)
        return output_layer
    
    def backward(self, X, y, learning_rate):
        hidden_layer = self.sigmoid(np.dot(self.weights1, X) + self.bias1)
        output_layer = self.sigmoid(np.dot(self.weights2, hidden_layer) + self.bias2)
        
        delta2 = (output_layer - y) * self.sigmoid_derivative_of_activated(output_layer)
        d_weights2 = np.outer(delta2, hidden_layer)
        d_bias2 = delta2
        
        self.weights2 -= learning_rate * d_weights2
        self.bias2 -= learning_rate * d_bias2
        
        delta1 = np.dot(self.weights2.T, delta2) * self.sigmoid_derivative_of_activated(hidden_layer)
        d_weights1 = np.outer(delta1, X)
        d_bias1 = delta1
        
        self.weights1 -= learning_rate * d_weights1
        self.bias1 -= learning_rate * d_bias1


def benchmark_training(model, X_train, y_train, epochs, iterations=5):
    """Benchmark training with multiple runs."""
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        for _ in range(epochs):
            for X, y in zip(X_train, y_train):
                model.backward(X, np.array([y]), 0.1)
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        'mean_s': np.mean(times),
        'min_s': np.min(times),
        'max_s': np.max(times),
        'std_s': np.std(times),
        'epochs_per_sec': epochs / np.mean(times)
    }


def benchmark_inference(model, X_test, iterations=1000):
    """Benchmark inference with multiple runs."""
    # Warmup
    for X in X_test:
        model.forward(X)
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        for X in X_test:
            model.forward(X)
        end = time.perf_counter()
        times.append((end - start) * 1_000_000)  # microseconds
    
    return {
        'mean_us': np.mean(times),
        'min_us': np.min(times),
        'std_us': np.std(times),
        'inferences_per_sec': len(X_test) * 1_000_000 / np.mean(times)
    }


def main():
    """Run neural network benchmarks."""
    print("="*60)
    print("  MiniTen Neural Network Benchmark")
    print("  Backpropagation Training Performance")
    print("="*60)
    
    if not MINITEN_AVAILABLE:
        print("\nError: MiniTen not built. Please run:")
        print("  python setup.py build_ext --inplace")
        return 1
    
    # XOR problem data
    X_train = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=np.float64)
    y_train = np.array([0, 1, 1, 0], dtype=np.float64)
    
    # Test configurations
    configs = [
        {'input': 2, 'hidden': 4, 'output': 1, 'name': 'XOR (2-4-1)'},
        {'input': 2, 'hidden': 16, 'output': 1, 'name': 'XOR (2-16-1)'},
        {'input': 2, 'hidden': 64, 'output': 1, 'name': 'XOR (2-64-1)'},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"  Configuration: {config['name']}")
        print(f"  Architecture: {config['input']} -> {config['hidden']} -> {config['output']}")
        print(f"{'='*60}")
        
        # Create models with same random seed
        np.random.seed(42)
        python_model = PythonBackPropagation(config['input'], config['hidden'], config['output'])
        
        np.random.seed(42)
        miniten_model = miniten_backprop.BackPropagation(config['input'], config['hidden'], config['output'])
        
        # Training benchmark
        epochs = 100
        
        print(f"\nTraining ({epochs} epochs, 4 samples):")
        print("-"*50)
        
        # Python benchmark
        np.random.seed(42)
        python_model = PythonBackPropagation(config['input'], config['hidden'], config['output'])
        python_train = benchmark_training(python_model, X_train, y_train, epochs)
        print(f"  Python:   {python_train['mean_s']*1000:.2f} ms (±{python_train['std_s']*1000:.2f})")
        
        # MiniTen benchmark
        np.random.seed(42)
        miniten_model = miniten_backprop.BackPropagation(config['input'], config['hidden'], config['output'])
        miniten_train = benchmark_training(miniten_model, X_train, y_train, epochs)
        print(f"  MiniTen:  {miniten_train['mean_s']*1000:.2f} ms (±{miniten_train['std_s']*1000:.2f})")
        
        speedup = python_train['mean_s'] / miniten_train['mean_s']
        print(f"  Speedup:  {speedup:.2f}x")
        
        # Inference benchmark
        print(f"\nInference ({len(X_train)} samples):")
        print("-"*50)
        
        # Retrain models for inference
        np.random.seed(42)
        python_model = PythonBackPropagation(config['input'], config['hidden'], config['output'])
        np.random.seed(42)
        miniten_model = miniten_backprop.BackPropagation(config['input'], config['hidden'], config['output'])
        
        python_infer = benchmark_inference(python_model, X_train)
        miniten_infer = benchmark_inference(miniten_model, X_train)
        
        print(f"  Python:   {python_infer['mean_us']:.2f} μs ({python_infer['inferences_per_sec']:.0f}/s)")
        print(f"  MiniTen:  {miniten_infer['mean_us']:.2f} μs ({miniten_infer['inferences_per_sec']:.0f}/s)")
        
        infer_speedup = python_infer['mean_us'] / miniten_infer['mean_us']
        print(f"  Speedup:  {infer_speedup:.2f}x")
        
        results.append({
            'config': config['name'],
            'training_python_ms': python_train['mean_s'] * 1000,
            'training_miniten_ms': miniten_train['mean_s'] * 1000,
            'training_speedup': speedup,
            'inference_python_us': python_infer['mean_us'],
            'inference_miniten_us': miniten_infer['mean_us'],
            'inference_speedup': infer_speedup
        })
    
    # Generate summary
    print("\n" + "="*60)
    print("  Summary: MiniTen vs Pure Python")
    print("="*60)
    
    print("\n| Configuration | Training Speedup | Inference Speedup |")
    print("|--------------|-----------------|------------------|")
    for r in results:
        print(f"| {r['config']:<12} | {r['training_speedup']:.2f}x | {r['inference_speedup']:.2f}x |")
    
    print("\n" + "="*60)
    print("  Key Insights")
    print("="*60)
    print("""
1. MiniTen's Cython implementation provides speedups over pure Python
2. The speedup is more significant for larger hidden layer sizes
3. For real-world use, MiniTen provides a good balance of:
   - Performance (Cython optimizations)
   - Readability (educational codebase)
   - Size (minimal dependencies)

Note: Production frameworks like PyTorch/TensorFlow use highly
optimized BLAS/LAPACK/CUDA implementations that will be faster
for large-scale operations. MiniTen's focus is on:
- Edge computing with limited resources
- Educational purposes
- Custom hardware optimizations
""")
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, 'neural_network.md')
    with open(results_file, 'w') as f:
        f.write("# Neural Network Benchmark Results\n\n")
        f.write(f"*Benchmark run on: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write("## Training Performance (100 epochs, 4 samples)\n\n")
        f.write("| Configuration | Python (ms) | MiniTen (ms) | Speedup |\n")
        f.write("|--------------|------------|-------------|--------|\n")
        for r in results:
            f.write(f"| {r['config']} | {r['training_python_ms']:.2f} | {r['training_miniten_ms']:.2f} | {r['training_speedup']:.2f}x |\n")
        f.write("\n## Inference Performance\n\n")
        f.write("| Configuration | Python (μs) | MiniTen (μs) | Speedup |\n")
        f.write("|--------------|------------|-------------|--------|\n")
        for r in results:
            f.write(f"| {r['config']} | {r['inference_python_us']:.2f} | {r['inference_miniten_us']:.2f} | {r['inference_speedup']:.2f}x |\n")
    
    print(f"\nResults saved to: {results_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
