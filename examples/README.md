# MiniTen Examples

This directory contains examples demonstrating how to use MiniTen.

## Available Examples

### Working Examples (Current Implementation)

1. **basic_network.py** - XOR problem with backpropagation
   ```bash
   cd examples
   python basic_network.py
   ```
   Demonstrates training a simple neural network to learn the XOR function.

2. **vector_operations.py** - Optimized vector operations
   ```bash
   cd examples
   python vector_operations.py
   ```
   Shows the performance of Cython-optimized vector operations with benchmarks.

### Future Examples (Coming Soon)

3. **mnist_classifier.py** - MNIST digit classification
4. **image_classification.py** - CNN for image classification
5. **text_classification.py** - RNN for sentiment analysis
6. **graph_neural_net.py** - GNN for node classification
7. **reinforcement_learning.py** - Simple RL agent
8. **audio_processing.py** - Audio feature extraction
9. **video_classification.py** - Video understanding
10. **custom_layer.py** - Creating custom layers
11. **gpu_acceleration.py** - Using GPU backends
12. **model_deployment.py** - Deploying to edge devices

## Running Examples

### Prerequisites
```bash
# Build the Cython extensions first
python setup.py build_ext --inplace
```

### Run an example
```bash
cd examples
python <example_name>.py
```

## Example Structure

Each example includes:
- Clear documentation
- Step-by-step explanations
- Performance metrics (where applicable)
- Expected output

## Contributing Examples

We welcome new examples! When contributing:
1. Follow the existing example structure
2. Add clear documentation
3. Include expected output
4. Test on different platforms if possible
5. Update this README

## Learning Path

Recommended order for learning:
1. vector_operations.py - Understand basic operations
2. basic_network.py - See backpropagation in action
3. (More examples coming soon...)

## Support

If you have questions about the examples:
- Check the main [README](../README.md)
- See the [documentation](../docs/)
- Open an issue on GitHub
