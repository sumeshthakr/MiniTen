# MiniTen Benchmarks

This directory will contain performance benchmarks for MiniTen.

## Benchmark Categories

### 1. Operation Benchmarks
- Vector operations
- Matrix operations
- Convolution operations
- Activation functions

### 2. Model Benchmarks
- CNN inference
- RNN/LSTM inference
- GNN operations
- Transformer blocks

### 3. End-to-End Benchmarks
- Image classification (MobileNet-style)
- Text processing (LSTM)
- Speech recognition
- Object detection

### 4. Comparison Benchmarks
- MiniTen vs TensorFlow Lite
- MiniTen vs PyTorch Mobile
- MiniTen vs ONNX Runtime

## Target Platforms

- Raspberry Pi 4
- NVIDIA Jetson Nano
- Google Coral
- Desktop CPU (baseline)
- Desktop GPU (baseline)

## Metrics

- **Inference Time**: Average time per inference
- **Throughput**: Operations per second
- **Memory Usage**: Peak memory consumption
- **Power Consumption**: Watts during inference
- **Model Size**: Binary size

## Coming Soon

Detailed benchmarks will be added as features are implemented.

## Running Benchmarks

```bash
# Run all benchmarks
python benchmarks/run_all.py

# Run specific benchmark
python benchmarks/benchmark_operations.py
```

## Contributing Benchmarks

When adding benchmarks:
1. Use consistent methodology
2. Test on multiple platforms
3. Report all relevant metrics
4. Compare with baseline implementations
5. Document test conditions
