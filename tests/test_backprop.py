# test_backpropagation.py

import backprop
import numpy as np

if __name__ == '__main__':
    # Create a backpropagation object with 2 input nodes, 3 hidden nodes, and 1 output node
    bp = backprop.BackPropagation(2, 3, 1)
    
    # Training data
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 1, 1, 0])
    
    # Training loop
    epochs = 10000
    for _ in range(epochs):
        # Perform forward and backward passes
        for X, y in zip(X_train, y_train):
            bp.backward(X, y, learning_rate=0.1)
        
    # Test the network
    for X in X_train:
        output = bp.forward(X)
        print(f"Input: {X}, Output: {output}")