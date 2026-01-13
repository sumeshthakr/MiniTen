"""
Basic Neural Network Example

This example demonstrates how to use MiniTen's current working features
to train a simple neural network on the XOR problem.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

# Import working backpropagation implementation
from miniten.core import backprop


def xor_example():
    """
    Train a neural network to learn the XOR function.
    
    XOR truth table:
    0 XOR 0 = 0
    0 XOR 1 = 1
    1 XOR 0 = 1
    1 XOR 1 = 0
    """
    print("=" * 50)
    print("MiniTen - XOR Problem Example")
    print("=" * 50)
    
    # Create network: 2 inputs, 3 hidden neurons, 1 output
    print("\nCreating neural network (2-3-1 architecture)...")
    network = backprop.BackPropagation(
        input_size=2,
        hidden_size=3,
        output_size=1
    )
    
    # Training data
    X_train = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=np.float64)
    y_train = np.array([[0], [1], [1], [0]], dtype=np.float64)
    
    print("\nTraining data:")
    for i, (x, y) in enumerate(zip(X_train, y_train)):
        print(f"  Input: {x} → Target: {y[0]}")
    
    # Training parameters
    epochs = 10000
    learning_rate = 0.1
    
    print(f"\nTraining for {epochs} epochs with learning rate {learning_rate}...")
    
    # Training loop
    for epoch in range(epochs):
        for X, y in zip(X_train, y_train):
            network.backward(X, y, learning_rate=learning_rate)
        
        # Print progress every 2000 epochs
        if (epoch + 1) % 2000 == 0:
            # Calculate average error
            errors = []
            for X, y in zip(X_train, y_train):
                output = network.forward(X)
                error = (output[0] - y[0]) ** 2
                errors.append(error)
            avg_error = np.mean(errors)
            print(f"  Epoch {epoch + 1:5d} - Average Error: {avg_error:.6f}")
    
    print("\n" + "=" * 50)
    print("Testing trained network:")
    print("=" * 50)
    
    # Test the network
    for X, y in zip(X_train, y_train):
        output = network.forward(X)
        print(f"Input: {X} → Output: {output[0]:.4f} (Target: {y[0]})")
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)


if __name__ == "__main__":
    xor_example()
