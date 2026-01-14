#!/usr/bin/env python3
"""
MiniTen Demo Script

This script demonstrates the key features of MiniTen for the README.
"""

import sys
sys.path.insert(0, '/home/runner/work/MiniTen/MiniTen')

def print_section(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60 + "\n")


def demo_tensor_autograd():
    """Demonstrate tensor operations with automatic differentiation."""
    print_section("1. TENSOR WITH AUTOGRAD")
    
    from miniten.core import Tensor, zeros, randn
    
    # Create tensors
    print("Creating tensors with autograd support:")
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    w = Tensor([[0.5, 0.3], [0.2, 0.4]], requires_grad=True)
    
    print(f"  x = {x.tolist()}")
    print(f"  w = {w.tolist()}")
    
    # Forward pass
    print("\nForward pass (y = x @ w):")
    y = x @ w
    print(f"  y = {y.tolist()}")
    
    # Sum and backward
    loss = y.sum()
    print(f"\nLoss (sum of y) = {loss.item()}")
    
    loss.backward()
    print("\nGradients after backward():")
    print(f"  x.grad = {x.grad.tolist()}")
    print(f"  w.grad = {w.grad.tolist()}")
    
    return True


def demo_signal_processing():
    """Demonstrate signal processing capabilities."""
    print_section("2. SIGNAL PROCESSING (Custom FFT)")
    
    from miniten.utils import signal
    import math
    
    # Create a simple sine wave
    sample_rate = 100
    frequency = 5  # 5 Hz
    duration = 1.0
    n_samples = int(sample_rate * duration)
    
    sine_wave = [math.sin(2 * math.pi * frequency * t / sample_rate) 
                 for t in range(n_samples)]
    
    print(f"Generated sine wave: {frequency} Hz, {sample_rate} Hz sample rate")
    print(f"  First 10 samples: {[round(s, 3) for s in sine_wave[:10]]}")
    
    # FFT
    real, imag = signal.fft(sine_wave[:64])  # Power of 2
    magnitudes = [math.sqrt(r**2 + i**2) for r, i in zip(real, imag)]
    
    print("\nFFT Results (first 10 frequency bins):")
    print(f"  Magnitudes: {[round(m, 2) for m in magnitudes[:10]]}")
    
    # Peak detection
    peaks = signal.find_peaks(sine_wave, threshold=0.5, distance=5)
    print(f"\nPeak detection found {len(peaks)} peaks at indices: {peaks[:5]}...")
    
    # Filter
    filtered = signal.lowpass_filter(sine_wave, cutoff=10, sample_rate=sample_rate)
    print(f"\nLow-pass filtered (cutoff=10Hz) - same length: {len(filtered)} samples")
    
    return True


def demo_audio_processing():
    """Demonstrate audio processing features."""
    print_section("3. AUDIO PROCESSING (No External Dependencies)")
    
    from miniten.utils import audio
    import math
    
    # Generate a simple audio signal
    sample_rate = 16000
    duration = 0.1  # 100ms
    frequency = 440  # A4 note
    
    n_samples = int(sample_rate * duration)
    audio_data = [math.sin(2 * math.pi * frequency * t / sample_rate) 
                  for t in range(n_samples)]
    
    audio_tensor = audio.AudioTensor(audio_data, sample_rate)
    
    print(f"Created audio signal: {frequency} Hz (A4 note)")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {audio_tensor.duration:.3f} seconds")
    print(f"  Samples: {audio_tensor.num_samples}")
    
    # Compute spectrogram
    spec = audio.spectrogram(audio_data, n_fft=256, hop_length=64)
    print(f"\nSpectrogram computed:")
    print(f"  Frequency bins: {len(spec)}")
    print(f"  Time frames: {len(spec[0])}")
    
    # Compute MFCC
    mfccs = audio.mfcc(audio_data, sample_rate=sample_rate, n_mfcc=13)
    print(f"\nMFCC features computed:")
    print(f"  Coefficients: {len(mfccs)}")
    print(f"  Time frames: {len(mfccs[0])}")
    
    # Add noise
    noisy = audio.add_noise(audio_tensor, noise_level=0.01)
    print(f"\nAdded noise (level=0.01)")
    
    return True


def demo_nlp():
    """Demonstrate NLP utilities."""
    print_section("4. NLP UTILITIES")
    
    from miniten.utils import text
    
    # Tokenization
    sample_text = "MiniTen is a lightweight deep learning framework for edge devices!"
    
    print(f"Input text: \"{sample_text}\"")
    
    # Word tokenizer
    tokenizer = text.WordTokenizer(lowercase=True)
    tokens = tokenizer.tokenize(sample_text)
    print(f"\nWord tokens: {tokens}")
    
    # Character tokenizer
    char_tokenizer = text.CharTokenizer()
    chars = char_tokenizer.tokenize(sample_text[:20])
    print(f"Character tokens: {chars}")
    
    # Vocabulary
    texts = [
        "deep learning on edge",
        "machine learning inference",
        "neural network optimization",
        "edge computing devices"
    ]
    
    vocab = text.Vocabulary(max_size=50)
    vocab.build(texts)
    print(f"\nVocabulary built from {len(texts)} texts")
    print(f"  Size: {len(vocab)} tokens")
    print(f"  Contains 'deep': {'deep' in vocab}")
    
    # TF-IDF
    tfidf_vectors, _ = text.tfidf(texts, vocab)
    print(f"\nTF-IDF computed:")
    print(f"  Vector length: {len(tfidf_vectors[0])}")
    
    # Text similarity
    sim = text.jaccard_similarity("deep learning", "machine learning")
    print(f"\nJaccard similarity ('deep learning', 'machine learning'): {sim:.3f}")
    
    return True


def demo_reinforcement_learning():
    """Demonstrate reinforcement learning."""
    print_section("5. REINFORCEMENT LEARNING")
    
    from miniten import rl
    
    # Create environment
    env = rl.GridWorld(size=5)
    print("GridWorld Environment (5x5)")
    print(f"  Action space: {env.action_space} (up, down, left, right)")
    print(f"  Goal: reach position (4, 4)")
    
    # Create Q-learning agent
    agent = rl.TabularQLearning(
        n_states=25, 
        n_actions=4,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0
    )
    print("\nQ-Learning Agent created")
    
    # Train for a few episodes
    print("\nTraining for 100 episodes...")
    rewards = []
    for episode in range(100):
        reward = agent.train_episode(env)
        rewards.append(reward)
    
    avg_reward = sum(rewards[-10:]) / 10
    print(f"  Average reward (last 10 episodes): {avg_reward:.2f}")
    print(f"  Epsilon after training: {agent.epsilon:.4f}")
    
    # DQN example
    print("\nDQN Agent for CartPole:")
    cartpole = rl.CartPoleEnv()
    dqn = rl.DQN(
        state_dim=4,
        n_actions=2,
        hidden_sizes=[32, 32],
        learning_rate=0.001
    )
    print("  State dimension: 4")
    print("  Hidden layers: [32, 32]")
    print("  Created successfully!")
    
    return True


def demo_gpu_optimization():
    """Demonstrate GPU kernel optimization and SIMD."""
    print_section("6. GPU KERNEL & SIMD OPTIMIZATION")
    
    from miniten.gpu import GPUKernel, SIMDOps, EdgeOptimizer, KernelConfig
    import time
    
    # Create test data
    size = 10000
    import random
    a = [random.random() for _ in range(size)]
    b = [random.random() for _ in range(size)]
    
    print(f"Benchmarking with {size} elements")
    
    # Regular Python
    start = time.perf_counter()
    _ = [x + y for x, y in zip(a, b)]
    regular_time = time.perf_counter() - start
    
    # SIMD-optimized
    start = time.perf_counter()
    _ = SIMDOps.vector_add_simd(a, b)
    simd_time = time.perf_counter() - start
    
    print(f"\nVector Addition:")
    print(f"  Regular Python: {regular_time*1000:.3f} ms")
    print(f"  SIMD-optimized: {simd_time*1000:.3f} ms")
    print(f"  Speedup: {regular_time/simd_time:.2f}x")
    
    # Dot product
    start = time.perf_counter()
    _ = sum(x * y for x, y in zip(a, b))
    regular_time = time.perf_counter() - start
    
    start = time.perf_counter()
    _ = SIMDOps.dot_product_simd(a, b)
    simd_time = time.perf_counter() - start
    
    print(f"\nDot Product:")
    print(f"  Regular Python: {regular_time*1000:.3f} ms")
    print(f"  SIMD-optimized: {simd_time*1000:.3f} ms")
    print(f"  Speedup: {regular_time/simd_time:.2f}x")
    
    # Edge optimization recommendations
    print("\nEdge Device Optimization:")
    print(f"  Optimal dtype for 100k elements (medium): {EdgeOptimizer.select_optimal_dtype(100000, 'medium')}")
    print(f"  Memory estimate for (1000, 1000) float32: {EdgeOptimizer.estimate_memory((1000, 1000), 'float32') / 1024:.0f} KB")
    
    return True


def demo_benchmarking():
    """Demonstrate benchmarking suite."""
    print_section("7. BENCHMARKING SUITE")
    
    from benchmarks.benchmark_suite import (
        benchmark_function, OperationBenchmark, quick_benchmark
    )
    
    # Quick benchmark
    def simple_matmul(size):
        a = [[1.0] * size for _ in range(size)]
        b = [[1.0] * size for _ in range(size)]
        result = [[0.0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    result[i][j] += a[i][k] * b[k][j]
        return result
    
    time_ms = quick_benchmark(simple_matmul, 50)
    print(f"Quick benchmark (50x50 matmul): {time_ms:.2f} ms")
    
    # Detailed benchmark
    result = benchmark_function(
        simple_matmul,
        args=(30,),
        warmup=2,
        iterations=5,
        name="matmul_30x30"
    )
    print(f"\nDetailed benchmark (30x30 matmul):")
    print(f"  Mean: {result.mean_time*1000:.3f} ms")
    print(f"  Std: {result.std_time*1000:.3f} ms")
    print(f"  Min: {result.min_time*1000:.3f} ms")
    print(f"  Max: {result.max_time*1000:.3f} ms")
    
    return True


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("         MiniTen Feature Demonstration")
    print("    Lightweight Deep Learning for Edge Devices")
    print("=" * 60)
    
    demos = [
        ("Tensor + Autograd", demo_tensor_autograd),
        ("Signal Processing", demo_signal_processing),
        ("Audio Processing", demo_audio_processing),
        ("NLP Utilities", demo_nlp),
        ("Reinforcement Learning", demo_reinforcement_learning),
        ("GPU/SIMD Optimization", demo_gpu_optimization),
        ("Benchmarking Suite", demo_benchmarking),
    ]
    
    results = []
    for name, demo_fn in demos:
        try:
            success = demo_fn()
            results.append((name, "✓ PASS" if success else "✗ FAIL"))
        except Exception as e:
            results.append((name, f"✗ ERROR: {e}"))
    
    print_section("SUMMARY")
    print("Demo Results:")
    for name, status in results:
        print(f"  {name}: {status}")
    
    all_passed = all("PASS" in r[1] for r in results)
    print("\n" + ("All demos passed! ✓" if all_passed else "Some demos failed! ✗"))
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
