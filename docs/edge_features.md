# MiniTen Edge Device Features

This document lists all features included in MiniTen that make development on edge devices easy.

## Core Features for Edge Devices

### 1. Minimal Dependencies
- Only requires NumPy and Cython
- No heavy framework dependencies (PyTorch/TensorFlow optional)
- Small installation footprint (<10MB vs GB for full frameworks)

### 2. Optimized Cython Implementations
All performance-critical operations are implemented in Cython:
- Matrix operations with OpenMP parallelization
- Convolution and pooling layers
- RNN/LSTM/GRU cells
- Attention mechanisms
- Optimizer updates

### 3. Model Quantization (`miniten.edge.quantization`)
- **INT8 quantization**: 4x memory reduction
- **INT16 quantization**: 2x memory reduction
- Symmetric and asymmetric quantization
- Per-tensor and per-channel quantization
- Quantization-aware training support

```python
from miniten.edge import quantize_model
quantized = quantize_model(model, dtype='int8')
```

### 4. Weight Pruning (`miniten.edge.pruning`)
- **Magnitude pruning**: Remove smallest weights
- **Structured pruning**: Remove entire channels/neurons
- **Gradual pruning**: Progressive sparsity during training
- Pruning masks for efficient sparse computation

```python
from miniten.edge import prune_model
pruned = prune_model(model, sparsity=0.5)
```

### 5. Model Compression (`miniten.edge.compression`)
- **Knowledge distillation**: Train small student from large teacher
- **Weight sharing/clustering**: Reduce unique weight values
- **Low-rank approximation**: SVD-based compression

### 6. Model Export (`miniten.edge.export`)
- Export to **ONNX** format
- Export to **TFLite-compatible** format
- Native **MiniTen format** (.npz)
- ONNX model loading

```python
from miniten.edge import export_onnx, load_onnx
export_onnx(model, input_shape=(1, 3, 224, 224), output_path="model.onnx")
```

### 7. Benchmarking Tools (`miniten.edge.benchmark`)
- Latency measurement (mean, std, p95, p99)
- Throughput testing (FPS)
- Memory profiling
- Layer-by-layer profiling
- Model comparison utilities

```python
from miniten.edge import benchmark_model
results = benchmark_model(model, input_shape=(1, 3, 224, 224))
print(f"Latency: {results['mean_latency_ms']:.2f}ms")
```

### 8. Resource Estimation (`miniten.edge.utils`)
- **FLOP counting**: Estimate computation cost
- **Memory estimation**: Parameter and activation memory
- **Power estimation**: Energy per inference
- **Edge compatibility check**: Verify deployment constraints

```python
from miniten.edge import count_flops, estimate_memory, estimate_power
flops = count_flops(model, input_shape)
memory = estimate_memory(model, input_shape)
```

## Training Features

### 9. Training Monitor (`miniten.monitor`)
- **Metrics logging**: Loss, accuracy, custom metrics
- **Callbacks**: EarlyStopping, ModelCheckpoint, ProgressBar
- **Experiment tracking**: Compare runs
- **HTML Dashboard**: Visual reports
- Memory-efficient JSONL logging

### 10. Profilers (`miniten.monitor.profiler`)
- Memory profiler
- Performance profiler
- Function timing decorator
- FLOP counter

## Visualization Features

### 11. Visualization Engine (`miniten.viz`)
- Fast Cython-based rendering
- Line plots, scatter plots, bar charts, histograms
- Image display (imshow, heatmap)
- Export to PNG/SVG
- No matplotlib dependency required

## Neural Network Layers

### 12. Efficient Layer Implementations
All implemented in Cython for maximum performance:
- Linear layers with optimized backprop
- Conv2d with im2col optimization
- MaxPool2d, AvgPool2d
- RNN, LSTM, GRU cells
- Multi-head attention
- Layer normalization
- Graph convolution (GCN)

### 13. Model Components
- Sequential container
- Activation functions (ReLU, Sigmoid, Tanh, GELU)
- Dropout with deterministic mode
- Batch normalization (optimized)

## Optimizers

### 14. Efficient Optimizers (`miniten.optim`)
All in Cython:
- SGD with momentum
- Adam with bias correction
- AdamW (decoupled weight decay)
- RMSprop

### 15. Learning Rate Schedulers
- Step LR
- Exponential LR
- Cosine annealing
- Warmup + cosine

### 16. Gradient Utilities
- Gradient clipping by norm
- Gradient clipping by value

## Data Processing

### 17. Data Loading (`miniten.utils.data`)
- Efficient DataLoader with batching
- Random/sequential samplers
- Train/test split utilities
- Dataset concatenation

### 18. Vision Utilities (`miniten.utils.vision`)
- Image loading/saving
- Resize, crop, normalize
- Data augmentation (flip, rotate, brightness)
- Edge detection (Sobel, Canny)
- Transform composition

## Summary Table

| Feature Category | Features Included |
|-----------------|-------------------|
| Quantization | INT8, INT16, symmetric/asymmetric, per-channel |
| Pruning | Magnitude, structured, gradual |
| Compression | Distillation, weight sharing, low-rank |
| Export | ONNX, TFLite-compatible, native |
| Benchmarking | Latency, throughput, memory, power |
| Monitoring | Logging, callbacks, dashboard, profilers |
| Visualization | Plots, images, heatmaps, SVG export |
| Layers | Linear, Conv, Pool, RNN, LSTM, Attention |
| Optimizers | SGD, Adam, AdamW, RMSprop |
| Data | DataLoader, transforms, augmentation |

## Comparison with Other Frameworks

| Feature | MiniTen | PyTorch | TensorFlow |
|---------|---------|---------|------------|
| Install Size | ~10MB | ~2GB | ~2GB |
| Dependencies | 2 | 10+ | 20+ |
| Quantization | ✓ | ✓ | ✓ |
| Pruning | ✓ | ✓ | ✓ |
| ONNX Export | ✓ | ✓ | ✓ |
| Edge Focus | ✓✓✓ | ✓ | ✓ |
| Customizable | ✓✓✓ | ✓✓ | ✓ |
| Learning Curve | Low | Medium | High |

## Recommended Deployment Pipeline

1. **Train** model using MiniTen or import from PyTorch/TensorFlow
2. **Quantize** to INT8 for 4x size reduction
3. **Prune** to 50%+ sparsity for additional compression
4. **Benchmark** on target device
5. **Export** to ONNX for cross-platform deployment
6. **Deploy** to edge device

```python
from miniten.edge import quantize_model, prune_model, export_onnx, benchmark_model

# 1. Quantize
quantized = quantize_model(model, dtype='int8')

# 2. Prune
pruned = prune_model(quantized, sparsity=0.5)

# 3. Benchmark
results = benchmark_model(pruned, input_shape)
print(f"Latency: {results['mean_latency_ms']:.2f}ms")

# 4. Export
export_onnx(pruned, input_shape, "model_edge.onnx")
```
