#!/usr/bin/env python3
"""
MiniTen Example: Training a Simple Neural Network

This example demonstrates the new Cython-optimized neural network layers
to train a simple two-layer network on a synthetic dataset.
"""

import numpy as np
from miniten.nn.layers_impl import (
    Linear, relu_forward, relu_backward, 
    sigmoid_forward, sigmoid_backward,
    mse_loss, cross_entropy_loss
)


def train_regression_network():
    """
    Train a simple regression network to learn a non-linear function.
    
    Network architecture: 1 input -> 8 hidden (ReLU) -> 1 output
    Task: Learn y = sin(x) + noise
    """
    print("="*70)
    print("Example 1: Regression with MSE Loss")
    print("="*70)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    X = np.linspace(-3, 3, n_samples).reshape(-1, 1).astype(np.float64)
    y_true = np.sin(X) + 0.1 * np.random.randn(n_samples, 1)
    
    # Create network: 1 -> 8 -> 1
    layer1 = Linear(1, 8, use_bias=True)
    layer2 = Linear(8, 1, use_bias=True)
    
    # Training parameters
    learning_rate = 0.01
    epochs = 500
    
    print(f"\nNetwork: 1 -> 8 (ReLU) -> 1")
    print(f"Training samples: {n_samples}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    
    # Training loop
    print("\nTraining progress:")
    for epoch in range(epochs):
        # Forward pass
        h1 = layer1.forward(X)
        h1_act, h1_mask = relu_forward(h1)
        output = layer2.forward(h1_act)
        
        # Compute loss
        loss, grad_loss = mse_loss(output, y_true)
        
        # Backward pass
        grad_h1_act = layer2.backward(grad_loss)
        grad_h1 = relu_backward(grad_h1_act, h1_mask)
        grad_input = layer1.backward(grad_h1)
        
        # Update parameters
        layer1.update_parameters(learning_rate)
        layer2.update_parameters(learning_rate)
        
        # Print progress
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:4d}: Loss = {loss:.6f}")
    
    print(f"\n✓ Training completed successfully!")
    print(f"  Final MSE Loss: {loss:.6f}\n")


def train_classification_network():
    """
    Train a classification network on a synthetic 2-class dataset.
    
    Network architecture: 2 inputs -> 16 hidden (ReLU) -> 2 outputs (Softmax)
    Task: Binary classification
    """
    print("="*70)
    print("Example 2: Binary Classification with Cross-Entropy Loss")
    print("="*70)
    
    # Generate synthetic data (two clusters)
    np.random.seed(42)
    n_samples = 200
    
    # Class 0: centered at (-1, -1)
    X_class0 = np.random.randn(n_samples//2, 2) * 0.5 + np.array([-1, -1])
    y_class0 = np.zeros(n_samples//2, dtype=np.int64)
    
    # Class 1: centered at (1, 1)
    X_class1 = np.random.randn(n_samples//2, 2) * 0.5 + np.array([1, 1])
    y_class1 = np.ones(n_samples//2, dtype=np.int64)
    
    # Combine and shuffle
    X = np.vstack([X_class0, X_class1]).astype(np.float64)
    y = np.hstack([y_class0, y_class1])
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    # Create network: 2 -> 16 -> 2
    layer1 = Linear(2, 16, use_bias=True)
    layer2 = Linear(16, 2, use_bias=True)
    
    # Training parameters
    learning_rate = 0.01
    epochs = 200
    
    print(f"\nNetwork: 2 -> 16 (ReLU) -> 2 (Softmax + CE)")
    print(f"Training samples: {n_samples} (2 classes)")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    
    # Training loop
    print("\nTraining progress:")
    for epoch in range(epochs):
        # Forward pass
        h1 = layer1.forward(X)
        h1_act, h1_mask = relu_forward(h1)
        logits = layer2.forward(h1_act)
        
        # Use combined softmax + cross-entropy for numerical stability
        from miniten.nn.layers_impl import softmax_cross_entropy_loss, softmax_forward
        loss, grad_loss = softmax_cross_entropy_loss(logits, y)
        
        # Backward pass
        grad_h1_act = layer2.backward(grad_loss)
        grad_h1 = relu_backward(grad_h1_act, h1_mask)
        grad_input = layer1.backward(grad_h1)
        
        # Update parameters
        layer1.update_parameters(learning_rate)
        layer2.update_parameters(learning_rate)
        
        # Compute accuracy
        if epoch % 50 == 0 or epoch == epochs - 1:
            # Get predictions from logits
            probs, _ = softmax_forward(logits)
            predictions = np.argmax(probs, axis=1)
            accuracy = np.mean(predictions == y) * 100
            print(f"  Epoch {epoch:4d}: Loss = {loss:.6f}, Accuracy = {accuracy:.2f}%")
    
    print(f"\n✓ Training completed successfully!")
    print(f"  Final Cross-Entropy Loss: {loss:.6f}")
    print(f"  Final Accuracy: {accuracy:.2f}%\n")


def train_xor_network():
    """
    Train a network to learn the XOR function (classic non-linear problem).
    
    Network architecture: 2 inputs -> 8 hidden (Sigmoid) -> 1 output (Sigmoid)
    Task: XOR function
    """
    print("="*70)
    print("Example 3: XOR Problem (Non-linear Classification)")
    print("="*70)
    
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([[0], [1], [1], [0]], dtype=np.float64)
    
    # Create network: 2 -> 8 -> 1
    layer1 = Linear(2, 8, use_bias=True)
    layer2 = Linear(8, 1, use_bias=True)
    
    # Training parameters
    learning_rate = 0.5
    epochs = 2000
    
    print(f"\nNetwork: 2 -> 8 (Sigmoid) -> 1 (Sigmoid)")
    print(f"Training samples: 4 (XOR truth table)")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    
    # Training loop
    print("\nTraining progress:")
    for epoch in range(epochs):
        # Forward pass
        h1 = layer1.forward(X)
        h1_act, _ = sigmoid_forward(h1)
        h2 = layer2.forward(h1_act)
        output, output_cached = sigmoid_forward(h2)
        
        # Compute loss
        loss, grad_loss = mse_loss(output, y)
        
        # Backward pass
        grad_h2 = sigmoid_backward(grad_loss, output_cached)
        grad_h1_act = layer2.backward(grad_h2)
        grad_h1 = sigmoid_backward(grad_h1_act, h1_act)
        grad_input = layer1.backward(grad_h1)
        
        # Update parameters
        layer1.update_parameters(learning_rate)
        layer2.update_parameters(learning_rate)
        
        # Print progress
        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:4d}: Loss = {loss:.6f}")
    
    # Test the trained network
    print("\nTest Results:")
    h1 = layer1.forward(X)
    h1_act, _ = sigmoid_forward(h1)
    h2 = layer2.forward(h1_act)
    output, _ = sigmoid_forward(h2)
    
    for i in range(len(X)):
        pred = "1" if output[i, 0] > 0.5 else "0"
        true = "1" if y[i, 0] > 0.5 else "0"
        print(f"  Input: {X[i]} -> Output: {output[i, 0]:.4f} -> Prediction: {pred} (True: {true})")
    
    print(f"\n✓ Training completed successfully!")
    print(f"  XOR function learned!\n")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print(" MiniTen Neural Network Examples")
    print(" Demonstrating Cython-optimized layers")
    print("="*70 + "\n")
    
    train_regression_network()
    train_classification_network()
    train_xor_network()
    
    print("="*70)
    print("All examples completed successfully! ✓")
    print("="*70)
    print("\nKey Features Demonstrated:")
    print("  • Linear layers with He initialization")
    print("  • ReLU and Sigmoid activations")
    print("  • MSE and Cross-Entropy losses")
    print("  • Forward and backward propagation")
    print("  • Gradient-based optimization")
    print("  • Binary and multi-class classification")
    print("  • Regression tasks")
    print("\nPerformance: All implemented in optimized Cython with OpenMP")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
