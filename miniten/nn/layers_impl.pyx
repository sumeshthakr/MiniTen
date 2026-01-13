# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Optimized neural network layers for MiniTen.

This module provides high-performance implementations of common neural network layers:
- Linear (fully connected) layer
- Activation functions (ReLU, Sigmoid, Tanh, Softmax)
- Loss functions (MSE, Cross Entropy)
"""

import cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport exp, log, fmax, sqrt

# Initialize NumPy C API
np.import_array()


cdef class Linear:
    """
    Optimized fully connected (linear) layer: y = xW^T + b
    
    This implementation uses Cython for performance and supports
    forward and backward passes for training.
    
    Attributes:
        weight: Weight matrix of shape (out_features, in_features)
        bias: Bias vector of shape (out_features,)
        grad_weight: Gradient of weight
        grad_bias: Gradient of bias
        cached_input: Cached input for backward pass
    """
    
    cdef public np.ndarray weight
    cdef public np.ndarray bias
    cdef public np.ndarray grad_weight
    cdef public np.ndarray grad_bias
    cdef public np.ndarray cached_input
    cdef public int in_features
    cdef public int out_features
    cdef public bint use_bias
    
    def __init__(self, int in_features, int out_features, bint use_bias=True):
        """
        Initialize Linear layer.
        
        Args:
            in_features: Size of input features
            out_features: Size of output features
            use_bias: Whether to use bias term
        """
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        
        # Initialize weights using He initialization
        cdef double std = sqrt(2.0 / in_features)
        self.weight = np.random.randn(out_features, in_features).astype(np.float64) * std
        
        if use_bias:
            self.bias = np.zeros(out_features, dtype=np.float64)
        else:
            self.bias = None
        
        self.grad_weight = np.zeros((out_features, in_features), dtype=np.float64)
        if use_bias:
            self.grad_bias = np.zeros(out_features, dtype=np.float64)
        self.cached_input = None
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def forward(self, np.ndarray[np.float64_t, ndim=2] x):
        """
        Forward pass: y = xW^T + b
        
        Args:
            x: Input of shape (batch_size, in_features)
            
        Returns:
            Output of shape (batch_size, out_features)
        """
        cdef Py_ssize_t batch_size = x.shape[0]
        cdef Py_ssize_t i, j, k
        cdef double temp
        cdef double[::1] bias_view
        
        # Cache input for backward pass
        self.cached_input = x.copy()
        
        # Output = xW^T
        cdef np.ndarray[np.float64_t, ndim=2] output = np.zeros((batch_size, self.out_features), dtype=np.float64)
        cdef double[:, ::1] x_view = x
        cdef double[:, ::1] w_view = self.weight
        cdef double[:, ::1] out_view = output
        
        # Matrix multiplication: output[i, j] = sum_k(x[i, k] * weight[j, k])
        for i in range(batch_size):
            for j in range(self.out_features):
                temp = 0.0
                for k in range(self.in_features):
                    temp = temp + x_view[i, k] * w_view[j, k]
                out_view[i, j] = temp
        
        # Add bias if enabled
        if self.use_bias:
            bias_view = self.bias
            for i in range(batch_size):
                for j in range(self.out_features):
                    out_view[i, j] = out_view[i, j] + bias_view[j]
        
        return output
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def backward(self, np.ndarray[np.float64_t, ndim=2] grad_output):
        """
        Backward pass to compute gradients.
        
        Args:
            grad_output: Gradient from next layer of shape (batch_size, out_features)
            
        Returns:
            grad_input: Gradient w.r.t. input of shape (batch_size, in_features)
        """
        cdef Py_ssize_t batch_size = grad_output.shape[0]
        cdef Py_ssize_t i, j, k
        cdef double temp
        cdef double[::1] grad_b_view
        
        cdef np.ndarray[np.float64_t, ndim=2] grad_input = np.zeros((batch_size, self.in_features), dtype=np.float64)
        cdef double[:, ::1] grad_out_view = grad_output
        cdef double[:, ::1] grad_in_view = grad_input
        cdef double[:, ::1] x_view = self.cached_input
        cdef double[:, ::1] w_view = self.weight
        cdef double[:, ::1] grad_w_view = self.grad_weight
        
        # Compute grad_input = grad_output @ W
        for i in range(batch_size):
            for k in range(self.in_features):
                temp = 0.0
                for j in range(self.out_features):
                    temp = temp + grad_out_view[i, j] * w_view[j, k]
                grad_in_view[i, k] = temp
        
        # Compute grad_weight = grad_output^T @ input
        # grad_weight[j, k] = sum_i(grad_output[i, j] * input[i, k])
        grad_w_view[:, :] = 0.0
        for i in range(batch_size):
            for j in range(self.out_features):
                for k in range(self.in_features):
                    grad_w_view[j, k] = grad_w_view[j, k] + grad_out_view[i, j] * x_view[i, k]
        
        # Compute grad_bias = sum(grad_output, axis=0)
        if self.use_bias:
            grad_b_view = self.grad_bias
            grad_b_view[:] = 0.0
            for i in range(batch_size):
                for j in range(self.out_features):
                    grad_b_view[j] = grad_b_view[j] + grad_out_view[i, j]
        
        return grad_input
    
    def update_parameters(self, double learning_rate):
        """
        Update parameters using gradients.
        
        Args:
            learning_rate: Learning rate for gradient descent
        """
        self.weight -= learning_rate * self.grad_weight
        if self.use_bias:
            self.bias -= learning_rate * self.grad_bias


# Activation functions

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple relu_forward(np.ndarray[np.float64_t, ndim=2] x):
    """
    ReLU activation: f(x) = max(0, x)
    
    Returns:
        (output, mask) for backward pass
    """
    cdef Py_ssize_t m = x.shape[0]
    cdef Py_ssize_t n = x.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] output = np.empty((m, n), dtype=np.float64)
    cdef np.ndarray[np.uint8_t, ndim=2] mask = np.empty((m, n), dtype=np.uint8)
    cdef double[:, ::1] x_view = x
    cdef double[:, ::1] out_view = output
    cdef unsigned char[:, ::1] mask_view = mask
    cdef Py_ssize_t i, j
    
    for i in range(m):
        for j in range(n):
            if x_view[i, j] > 0.0:
                out_view[i, j] = x_view[i, j]
                mask_view[i, j] = 1
            else:
                out_view[i, j] = 0.0
                mask_view[i, j] = 0
    
    return output, mask


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] relu_backward(np.ndarray[np.float64_t, ndim=2] grad_output, 
                                                      np.ndarray[np.uint8_t, ndim=2] mask):
    """
    ReLU backward pass.
    
    Args:
        grad_output: Gradient from next layer
        mask: Mask from forward pass
        
    Returns:
        Gradient w.r.t. input
    """
    cdef Py_ssize_t m = grad_output.shape[0]
    cdef Py_ssize_t n = grad_output.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] grad_input = np.empty((m, n), dtype=np.float64)
    cdef double[:, ::1] grad_out_view = grad_output
    cdef double[:, ::1] grad_in_view = grad_input
    cdef unsigned char[:, ::1] mask_view = mask
    cdef Py_ssize_t i, j
    
    for i in range(m):
        for j in range(n):
            if mask_view[i, j] == 1:
                grad_in_view[i, j] = grad_out_view[i, j]
            else:
                grad_in_view[i, j] = 0.0
    
    return grad_input


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple sigmoid_forward(np.ndarray[np.float64_t, ndim=2] x):
    """
    Sigmoid activation: f(x) = 1 / (1 + exp(-x))
    
    Returns:
        (output, output) for backward pass (we cache output)
    """
    cdef Py_ssize_t m = x.shape[0]
    cdef Py_ssize_t n = x.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] output = np.empty((m, n), dtype=np.float64)
    cdef double[:, ::1] x_view = x
    cdef double[:, ::1] out_view = output
    cdef Py_ssize_t i, j
    
    for i in range(m):
        for j in range(n):
            out_view[i, j] = 1.0 / (1.0 + exp(-x_view[i, j]))
    
    return output, output


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] sigmoid_backward(np.ndarray[np.float64_t, ndim=2] grad_output,
                                                         np.ndarray[np.float64_t, ndim=2] output):
    """
    Sigmoid backward: grad = grad_output * output * (1 - output)
    """
    cdef Py_ssize_t m = grad_output.shape[0]
    cdef Py_ssize_t n = grad_output.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] grad_input = np.empty((m, n), dtype=np.float64)
    cdef double[:, ::1] grad_out_view = grad_output
    cdef double[:, ::1] grad_in_view = grad_input
    cdef double[:, ::1] out_view = output
    cdef Py_ssize_t i, j
    
    for i in range(m):
        for j in range(n):
            grad_in_view[i, j] = grad_out_view[i, j] * out_view[i, j] * (1.0 - out_view[i, j])
    
    return grad_input


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple softmax_forward(np.ndarray[np.float64_t, ndim=2] x):
    """
    Softmax activation: f(x_i) = exp(x_i) / sum(exp(x_j))
    
    Numerically stable implementation.
    """
    cdef Py_ssize_t m = x.shape[0]
    cdef Py_ssize_t n = x.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] output = np.empty((m, n), dtype=np.float64)
    cdef double[:, ::1] x_view = x
    cdef double[:, ::1] out_view = output
    cdef Py_ssize_t i, j
    cdef double max_val, sum_exp, temp
    
    for i in range(m):
        # Find max for numerical stability
        max_val = x_view[i, 0]
        for j in range(1, n):
            if x_view[i, j] > max_val:
                max_val = x_view[i, j]
        
        # Compute exp and sum
        sum_exp = 0.0
        for j in range(n):
            temp = exp(x_view[i, j] - max_val)
            out_view[i, j] = temp
            sum_exp = sum_exp + temp
        
        # Normalize
        for j in range(n):
            out_view[i, j] = out_view[i, j] / sum_exp
    
    return output, output


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] softmax_backward(np.ndarray[np.float64_t, ndim=2] grad_output,
                                                         np.ndarray[np.float64_t, ndim=2] output):
    """
    Softmax backward pass.
    """
    cdef Py_ssize_t m = grad_output.shape[0]
    cdef Py_ssize_t n = grad_output.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] grad_input = np.zeros((m, n), dtype=np.float64)
    cdef double[:, ::1] grad_out_view = grad_output
    cdef double[:, ::1] grad_in_view = grad_input
    cdef double[:, ::1] out_view = output
    cdef Py_ssize_t i, j, k
    cdef double sum_val
    
    for i in range(m):
        for j in range(n):
            sum_val = 0.0
            for k in range(n):
                if j == k:
                    sum_val = sum_val + grad_out_view[i, k] * out_view[i, j] * (1.0 - out_view[i, j])
                else:
                    sum_val = sum_val - grad_out_view[i, k] * out_view[i, j] * out_view[i, k]
            grad_in_view[i, j] = sum_val
    
    return grad_input


# Loss functions

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple mse_loss(np.ndarray[np.float64_t, ndim=2] predictions, 
                     np.ndarray[np.float64_t, ndim=2] targets):
    """
    Mean Squared Error loss: MSE = mean((predictions - targets)^2)
    
    Returns:
        (loss, grad) where grad is gradient w.r.t. predictions
    """
    cdef Py_ssize_t m = predictions.shape[0]
    cdef Py_ssize_t n = predictions.shape[1]
    cdef double[:, ::1] pred_view = predictions
    cdef double[:, ::1] target_view = targets
    cdef double loss = 0.0
    cdef double diff
    cdef Py_ssize_t i, j
    
    cdef np.ndarray[np.float64_t, ndim=2] grad = np.empty((m, n), dtype=np.float64)
    cdef double[:, ::1] grad_view = grad
    
    for i in range(m):
        for j in range(n):
            diff = pred_view[i, j] - target_view[i, j]
            loss = loss + diff * diff
            grad_view[i, j] = 2.0 * diff / (m * n)
    
    loss = loss / (m * n)
    return loss, grad


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple cross_entropy_loss(np.ndarray[np.float64_t, ndim=2] predictions,
                                np.ndarray[np.int64_t, ndim=1] targets):
    """
    Cross Entropy loss: CE = -mean(log(predictions[target]))
    
    Args:
        predictions: Predicted probabilities of shape (batch_size, num_classes)
        targets: Target class indices of shape (batch_size,)
        
    Returns:
        (loss, grad) where grad is gradient w.r.t. predictions
    """
    cdef Py_ssize_t batch_size = predictions.shape[0]
    cdef Py_ssize_t num_classes = predictions.shape[1]
    cdef double[:, ::1] pred_view = predictions
    cdef long[::1] target_view = targets
    cdef double loss = 0.0
    cdef Py_ssize_t i, j
    cdef long target_idx
    cdef double eps = 1e-10  # Small epsilon for numerical stability
    
    cdef np.ndarray[np.float64_t, ndim=2] grad = np.zeros((batch_size, num_classes), dtype=np.float64)
    cdef double[:, ::1] grad_view = grad
    
    for i in range(batch_size):
        target_idx = target_view[i]
        # Compute loss
        loss = loss - log(pred_view[i, target_idx] + eps)
        # Compute gradient
        for j in range(num_classes):
            if j == target_idx:
                grad_view[i, j] = -1.0 / (pred_view[i, target_idx] + eps) / batch_size
            else:
                grad_view[i, j] = 0.0
    
    loss = loss / batch_size
    return loss, grad
