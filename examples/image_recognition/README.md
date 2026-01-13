# Image Classification Example

This folder contains image classification examples comparing MiniTen with PyTorch and TensorFlow.

## Contents

1. `train_miniten.py` - Train a simple CNN using MiniTen
2. `train_pytorch.py` - Train using PyTorch
3. `train_tensorflow.py` - Train using TensorFlow
4. `compare_performance.py` - Compare all three frameworks
5. `onnx_mobilenet.py` - Load and use MobileNet via ONNX

## Running the Examples

```bash
# Install dependencies
pip install torch torchvision tensorflow pillow onnx onnxruntime

# Run individual frameworks
python train_miniten.py
python train_pytorch.py  
python train_tensorflow.py

# Run comparison
python compare_performance.py
```

## Results

See `comparison_report.md` for performance comparison results.
