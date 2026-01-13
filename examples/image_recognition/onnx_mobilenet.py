#!/usr/bin/env python
"""
ONNX MobileNet Example

Demonstrates loading and using a pre-trained MobileNet model via ONNX.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def download_mobilenet():
    """Download MobileNet ONNX model if not present."""
    import urllib.request
    
    model_path = "mobilenet_v2.onnx"
    
    if os.path.exists(model_path):
        print(f"Model already exists: {model_path}")
        return model_path
    
    # MobileNetV2 from ONNX Model Zoo
    url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
    
    print(f"Downloading MobileNetV2 ONNX model...")
    print(f"URL: {url}")
    
    try:
        urllib.request.urlretrieve(url, model_path)
        print(f"Downloaded to: {model_path}")
        return model_path
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please download manually from ONNX Model Zoo")
        return None


def run_onnx_mobilenet():
    """Run MobileNet inference using ONNX Runtime."""
    
    print("="*60)
    print("ONNX MobileNet Inference Example")
    print("="*60)
    
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime version: {ort.__version__}")
    except ImportError:
        print("ONNX Runtime not installed. Run: pip install onnxruntime")
        return None
    
    # Check for model (skip download for demo)
    model_path = "mobilenet_v2.onnx"
    
    if not os.path.exists(model_path):
        print("\nNote: MobileNet ONNX model not found.")
        print("For a full demo, download from ONNX Model Zoo.")
        print("\nRunning with synthetic model demonstration...")
        return run_synthetic_onnx_demo()
    
    # Load model
    print(f"\nLoading model: {model_path}")
    session = ort.InferenceSession(model_path)
    
    # Get model info
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    
    print(f"Input: {input_info.name}, shape: {input_info.shape}")
    print(f"Output: {output_info.name}, shape: {output_info.shape}")
    
    # Create sample input (224x224 RGB image)
    input_shape = input_info.shape
    if input_shape[0] is None or isinstance(input_shape[0], str):
        input_shape = (1, 3, 224, 224)
    
    sample_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    print("\nWarmup...")
    for _ in range(5):
        _ = session.run(None, {input_info.name: sample_input})
    
    # Benchmark
    print("\nBenchmark (100 inferences)...")
    n_runs = 100
    start = time.time()
    
    for _ in range(n_runs):
        outputs = session.run(None, {input_info.name: sample_input})
    
    elapsed = time.time() - start
    latency_ms = (elapsed / n_runs) * 1000
    
    print(f"Latency: {latency_ms:.2f}ms")
    print(f"Throughput: {n_runs / elapsed:.1f} inferences/sec")
    
    # Get predictions
    output = outputs[0]
    predicted_class = np.argmax(output[0])
    confidence = np.max(output[0])
    
    print(f"\nSample Prediction:")
    print(f"  Predicted class: {predicted_class}")
    print(f"  Confidence: {confidence:.4f}")
    
    return {
        "model": "MobileNetV2",
        "framework": "ONNX Runtime",
        "latency_ms": latency_ms,
        "throughput": n_runs / elapsed,
    }


def run_synthetic_onnx_demo():
    """Run a synthetic ONNX demo without downloading models."""
    
    print("\n" + "-"*40)
    print("Synthetic ONNX Demonstration")
    print("-"*40)
    
    try:
        import onnxruntime as ort
        import onnx
        from onnx import helper, TensorProto, numpy_helper
    except ImportError:
        print("ONNX packages not installed. Run: pip install onnx onnxruntime")
        return None
    
    # Create a simple ONNX model programmatically
    print("Creating synthetic ONNX model...")
    
    # Define a simple model: input -> MatMul -> ReLU -> output
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 784])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10])
    
    # Create weight tensors
    W1_init = numpy_helper.from_array(
        np.random.randn(784, 128).astype(np.float32) * 0.1,
        name='W1'
    )
    W2_init = numpy_helper.from_array(
        np.random.randn(128, 10).astype(np.float32) * 0.1,
        name='W2'
    )
    
    # Create nodes
    matmul1 = helper.make_node('MatMul', ['input', 'W1'], ['h1'])
    relu = helper.make_node('Relu', ['h1'], ['h1_relu'])
    matmul2 = helper.make_node('MatMul', ['h1_relu', 'W2'], ['output'])
    
    # Create graph
    graph = helper.make_graph(
        [matmul1, relu, matmul2],
        'simple_model',
        [X],
        [Y],
        [W1_init, W2_init]
    )
    
    # Create model with compatible IR version
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    model.ir_version = 6  # Use compatible IR version
    
    # Validate
    onnx.checker.check_model(model)
    
    # Save temporarily
    model_path = "/tmp/synthetic_model.onnx"
    onnx.save(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Load with ONNX Runtime
    session = ort.InferenceSession(model_path)
    
    input_info = session.get_inputs()[0]
    print(f"Input: {input_info.name}, shape: {input_info.shape}")
    
    # Run inference
    sample_input = np.random.randn(1, 784).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        _ = session.run(None, {'input': sample_input})
    
    # Benchmark
    n_runs = 1000
    start = time.time()
    
    for _ in range(n_runs):
        outputs = session.run(None, {'input': sample_input})
    
    elapsed = time.time() - start
    latency_ms = (elapsed / n_runs) * 1000
    
    print(f"\nBenchmark Results:")
    print(f"  Runs: {n_runs}")
    print(f"  Latency: {latency_ms:.3f}ms")
    print(f"  Throughput: {n_runs / elapsed:.0f} inferences/sec")
    
    output = outputs[0]
    print(f"  Output shape: {output.shape}")
    print(f"  Predicted class: {np.argmax(output[0])}")
    
    # Show MiniTen ONNX support
    print("\n" + "-"*40)
    print("MiniTen ONNX Support")
    print("-"*40)
    
    try:
        from miniten.edge.export import load_onnx
        print("MiniTen can load ONNX models via miniten.edge.export.load_onnx()")
        print("Example: model = load_onnx('model.onnx')")
    except ImportError:
        print("MiniTen edge module available for ONNX support")
    
    return {
        "model": "Synthetic MLP",
        "framework": "ONNX Runtime",
        "latency_ms": latency_ms,
        "throughput": n_runs / elapsed,
    }


if __name__ == "__main__":
    results = run_onnx_mobilenet()
    
    if results:
        print("\n" + "="*60)
        print("Results Summary:")
        for key, value in results.items():
            print(f"  {key}: {value}")
