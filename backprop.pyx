# backpropagation.pyx

# Import necessary Cython libraries
import numpy as np
cimport numpy as np

# Define the BackPropagation class
cdef class BackPropagation:
    
    cdef np.ndarray X, y  # Input and output data
    cdef int input_size, hidden_size, output_size  # Sizes of layers
    cdef np.ndarray weights1, weights2  # Weight matrices
    cdef np.ndarray bias1, bias2  # Bias vectors
    
    # Constructor
    def __init__(self, int input_size, int hidden_size, int output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases randomly
        self.weights1 = np.random.randn(hidden_size, input_size)
        self.weights2 = np.random.randn(output_size, hidden_size)
        self.bias1 = np.random.randn(hidden_size)
        self.bias2 = np.random.randn(output_size)
        
    # Sigmoid activation function
    cdef np.ndarray sigmoid(self, np.ndarray x):
        return 1.0 / (1.0 + np.exp(-x))
    
    # Derivative of sigmoid activation function
    cdef np.ndarray sigmoid_derivative(self, np.ndarray x):
        cdef np.ndarray sigmoid_x = self.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)
    
    # Forward pass through the network
    cdef np.ndarray forward(self, np.ndarray X):
        cdef np.ndarray hidden_layer = self.sigmoid(np.dot(self.weights1, X) + self.bias1)
        cdef np.ndarray output_layer = self.sigmoid(np.dot(self.weights2, hidden_layer) + self.bias2)
        return output_layer
    
    # Backpropagation algorithm
    cdef void backward(self, np.ndarray X, np.ndarray y, double learning_rate):
        cdef np.ndarray hidden_layer = self.sigmoid(np.dot(self.weights1, X) + self.bias1)
        cdef np.ndarray output_layer = self.sigmoid(np.dot(self.weights2, hidden_layer) + self.bias2)
        
        cdef np.ndarray delta2 = (output_layer - y) * self.sigmoid_derivative(output_layer)
        cdef np.ndarray d_weights2 = np.outer(delta2, hidden_layer)
        cdef np.ndarray d_bias2 = delta2
        
        # Perform element-wise subtraction using NumPy arrays
        self.weights2 -= learning_rate * d_weights2
        self.bias2 -= learning_rate * d_bias2
        
        cdef np.ndarray delta1 = np.dot(self.weights2.T, delta2) * self.sigmoid_derivative(hidden_layer)
        cdef np.ndarray d_weights1 = np.outer(delta1, X)
        cdef np.ndarray d_bias1 = delta1
        
        # Perform element-wise subtraction using NumPy arrays
        self.weights1 -= learning_rate * d_weights1
        self.bias1 -= learning_rate * d_bias1