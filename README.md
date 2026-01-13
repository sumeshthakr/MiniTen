# MiniTen

**A Lightweight Deep Learning Framework Optimized for Edge Platforms**

[![CI](https://github.com/sumeshthakr/MiniTen/actions/workflows/ci.yml/badge.svg)](https://github.com/sumeshthakr/MiniTen/actions/workflows/ci.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Vision

MiniTen is a high-performance deep learning library designed from the ground up for edge computing. Built purely in Python and Cython with minimal external dependencies, MiniTen delivers exceptional performance in a fraction of the size of traditional frameworks like TensorFlow and PyTorch.

### Why MiniTen?

- **🚀 Optimized for Edge**: Designed specifically for edge platforms (IoT devices, mobile, embedded systems)
- **📦 Minimal Footprint**: Fraction of the size compared to TensorFlow/PyTorch
- **⚡ High Performance**: Highly optimized Cython implementations for critical operations
- **🎓 Educational**: Clear, well-documented code showing how deep learning works internally
- **🔧 Modular**: Easy to understand, extend, and contribute to
- **🌐 GPU Support**: Supports CUDA, OpenCL, Metal, and Vulkan for edge GPUs
- **🔋 Power Efficient**: Optimized for low-power edge computing scenarios

## ✨ Features

### Neural Network Architectures
- **CNNs**: Convolutional Neural Networks for computer vision
- **RNNs**: Recurrent Neural Networks for sequential data
- **LSTMs**: Long Short-Term Memory networks
- **GRUs**: Gated Recurrent Units
- **GNNs**: Graph Neural Networks
- **Transformers**: Attention-based models (coming soon)
- **Reinforcement Learning**: RL algorithms (coming soon)

### Data Processing
- **Vision**: Image processing and augmentation
- **Audio**: Speech and sound processing
- **Video**: Video analysis and processing
- **Text/NLP**: Natural language processing
- **Signal**: Time-series and sensor data processing

### Core Features
- Automatic differentiation
- GPU acceleration (CUDA, OpenCL, Metal, Vulkan)
- Optimized tensor operations in Cython
- Memory-efficient computation
- Model serialization and deployment
- Comprehensive documentation and examples

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sumeshthakr/MiniTen.git
cd MiniTen

# Install dependencies
pip install -r requirements.txt

# Build and install
python setup.py build_ext --inplace
pip install -e .
```

### Basic Example

```python
import miniten as mt
import numpy as np

# Create a simple neural network (coming soon)
# model = mt.nn.Sequential([
#     mt.nn.Linear(784, 128),
#     mt.nn.ReLU(),
#     mt.nn.Linear(128, 10),
#     mt.nn.Softmax()
# ])

# Current working example with backpropagation
from miniten.core import backprop
bp = backprop.BackPropagation(2, 3, 1)

# Training data (XOR problem)
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 1, 1, 0])

# Train the network
for epoch in range(1000):
    for X, y in zip(X_train, y_train):
        bp.backward(X, y, learning_rate=0.1)

# Test
for X in X_train:
    output = bp.forward(X)
    print(f"Input: {X}, Output: {output}")
```

## 📖 Documentation

### Project Structure

```
MiniTen/
├── miniten/              # Main package
│   ├── core/            # Core tensor operations and autograd
│   ├── nn/              # Neural network modules
│   │   ├── layers.py    # Common layers (Linear, Conv2d, etc.)
│   │   ├── activations.py  # Activation functions
│   │   ├── rnn.py       # RNN, LSTM, GRU
│   │   ├── cnn.py       # CNN-specific layers
│   │   └── gnn.py       # Graph neural networks
│   ├── optim/           # Optimizers (SGD, Adam, etc.)
│   ├── utils/           # Utilities
│   │   ├── data.py      # Data loading
│   │   ├── vision.py    # Image processing
│   │   ├── audio.py     # Audio processing
│   │   ├── text.py      # Text/NLP utilities
│   │   └── signal.py    # Signal processing
│   └── gpu/             # GPU backends
├── docs/                # Documentation
├── examples/            # Example scripts
├── tests/               # Test suite
└── benchmarks/          # Performance benchmarks
```

### Key Modules

#### Core (`miniten.core`)
- **Tensor**: Multi-dimensional arrays with autograd
- **Autograd**: Automatic differentiation engine
- **Operations**: Optimized mathematical operations (Cython)

#### Neural Networks (`miniten.nn`)
- **Layers**: Linear, Conv2d, MaxPool2d, Dropout, BatchNorm
- **Activations**: ReLU, Sigmoid, Tanh, Softmax, GELU
- **RNN**: RNN, LSTM, GRU with bidirectional support
- **CNN**: Depthwise separable convolutions, dilated convolutions
- **GNN**: GraphConv, GraphAttention, SAGEConv

#### Optimizers (`miniten.optim`)
- SGD with momentum
- Adam, AdamW, Adamax
- RMSprop, Adagrad
- Learning rate schedulers

#### GPU Support (`miniten.gpu`)
- CUDA for NVIDIA GPUs (including Jetson)
- OpenCL for cross-platform support
- Metal for Apple Silicon
- Vulkan for cross-platform compute

## 🎯 Roadmap

### Phase 1: Foundation (Current)
- [x] Project structure and architecture
- [x] Core module stubs (Tensor, Autograd)
- [x] Basic backpropagation (working)
- [x] Vector operations (working)
- [ ] Complete Tensor implementation
- [ ] Automatic differentiation engine
- [ ] GPU backend infrastructure

### Phase 2: Neural Networks
- [ ] Common layers (Linear, Conv2d, Pooling)
- [ ] Activation functions
- [ ] Loss functions
- [ ] Model containers (Sequential, ModuleList)
- [ ] RNN/LSTM/GRU implementations
- [ ] CNN optimizations for edge

### Phase 3: Advanced Features
- [ ] Graph Neural Networks
- [ ] Attention mechanisms
- [ ] Transformer architecture
- [ ] Reinforcement Learning basics
- [ ] Model quantization
- [ ] Pruning and compression

### Phase 4: Optimization & Deployment
- [ ] GPU kernel optimization
- [ ] SIMD optimizations
- [ ] Memory pooling
- [ ] Model serialization
- [ ] Edge deployment tools
- [ ] Benchmarking suite

### Phase 5: Data Processing
- [ ] Image processing pipeline
- [ ] Audio processing pipeline
- [ ] Video processing
- [ ] NLP utilities
- [ ] Signal processing
- [ ] Data augmentation

## 🤝 Contributing

We welcome contributions! MiniTen is designed to be an educational and collaborative project.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Add tests** for your changes
5. **Ensure tests pass**: `python -m pytest tests/`
6. **Commit your changes**: `git commit -m 'Add amazing feature'`
7. **Push to branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### Development Guidelines

- Write clear, documented code
- Follow existing code style
- Add comprehensive tests
- Update documentation
- Optimize for edge devices
- Minimize external dependencies

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📊 Performance

### Benchmark Results

MiniTen provides Cython-optimized implementations compared to pure Python.

**Neural Network Training/Inference (vs Pure Python):**

| Configuration | Training Speedup | Inference Speedup |
|--------------|-----------------|------------------|
| XOR (2-4-1)  | 1.58x | 1.88x |
| XOR (2-16-1) | 1.45x | 1.87x |
| XOR (2-64-1) | 1.43x | 1.86x |

**Vector Operations (vs NumPy):**

| Operation | Small Vectors (100) | Large Vectors (100K) |
|-----------|---------------------|----------------------|
| Vector Addition | 1.75x slower | 1.17x slower |
| Dot Product | 1.15x faster | 28x slower |
| Element-wise Multiply | 1.90x slower | 1.08x slower |
| Scalar Multiply | 1.35x faster | 1.81x slower |

> **Note**: NumPy uses highly optimized BLAS/LAPACK libraries with SIMD instructions.
> MiniTen's focus is on educational value, minimal footprint, and edge computing—not
> competing with production frameworks on raw speed.

### MiniTen's Advantages

- **Minimal footprint**: Pure Python/Cython with minimal dependencies
- **Educational**: Clear, readable code for learning ML internals
- **Customizable**: Easy to extend for custom hardware or specialized use cases
- **Edge-optimized**: Designed for resource-constrained environments

See detailed benchmarks in the [`benchmarks/`](benchmarks/) directory.

## 🔧 Requirements

- Python 3.8+
- Cython 0.29+
- NumPy (minimal usage, only where necessary)

Optional:
- CUDA toolkit (for NVIDIA GPU support)
- OpenCL (for cross-platform GPU support)
- Metal (automatically available on macOS/iOS)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

MiniTen is inspired by:
- PyTorch's design philosophy
- TensorFlow Lite's edge optimization
- Tinygrad's minimalism
- Educational resources from Fast.ai and Andrej Karpathy

## 📬 Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/sumeshthakr/MiniTen/issues)
- **Discussions**: [Join the discussion](https://github.com/sumeshthakr/MiniTen/discussions)

## 🌟 Show Your Support

If you find MiniTen useful, please consider:
- ⭐ Starring the repository
- 🐛 Reporting bugs
- 💡 Suggesting new features
- 📝 Contributing code or documentation
- 📢 Spreading the word

---

**Built with ❤️ for the edge computing community**
