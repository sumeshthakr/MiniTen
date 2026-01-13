# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Optimized RNN layers for MiniTen.

This module provides high-performance implementations of:
- RNN (Vanilla Recurrent Neural Network)
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)

All implementations use Cython for maximum performance and minimal memory footprint,
optimized for edge devices.
"""

import cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport exp, tanh, sqrt

# Initialize NumPy C API
np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double sigmoid_scalar(double x) noexcept nogil:
    """Inline sigmoid function for a single value."""
    return 1.0 / (1.0 + exp(-x))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double tanh_scalar(double x) noexcept nogil:
    """Inline tanh function for a single value."""
    return tanh(x)


cdef class RNNCell:
    """
    Optimized vanilla RNN cell.
    
    Computes: h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
    
    Attributes:
        weight_ih: Input-hidden weights of shape (hidden_size, input_size)
        weight_hh: Hidden-hidden weights of shape (hidden_size, hidden_size)
        bias_ih: Input-hidden bias of shape (hidden_size,)
        bias_hh: Hidden-hidden bias of shape (hidden_size,)
    """
    
    cdef public int input_size
    cdef public int hidden_size
    cdef public np.ndarray weight_ih
    cdef public np.ndarray weight_hh
    cdef public np.ndarray bias_ih
    cdef public np.ndarray bias_hh
    cdef public np.ndarray grad_weight_ih
    cdef public np.ndarray grad_weight_hh
    cdef public np.ndarray grad_bias_ih
    cdef public np.ndarray grad_bias_hh
    # Cache for backward pass
    cdef public list cached_inputs
    cdef public list cached_hiddens
    cdef public list cached_pre_activations
    
    def __init__(self, int input_size, int hidden_size):
        """
        Initialize RNN cell.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights with Xavier/Glorot initialization
        cdef double std = sqrt(2.0 / (input_size + hidden_size))
        self.weight_ih = np.random.randn(hidden_size, input_size).astype(np.float64) * std
        self.weight_hh = np.random.randn(hidden_size, hidden_size).astype(np.float64) * std
        self.bias_ih = np.zeros(hidden_size, dtype=np.float64)
        self.bias_hh = np.zeros(hidden_size, dtype=np.float64)
        
        # Initialize gradients
        self.grad_weight_ih = np.zeros_like(self.weight_ih)
        self.grad_weight_hh = np.zeros_like(self.weight_hh)
        self.grad_bias_ih = np.zeros_like(self.bias_ih)
        self.grad_bias_hh = np.zeros_like(self.bias_hh)
        
        self.cached_inputs = []
        self.cached_hiddens = []
        self.cached_pre_activations = []
    
    def reset_cache(self):
        """Reset cache for new sequence."""
        self.cached_inputs = []
        self.cached_hiddens = []
        self.cached_pre_activations = []
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def forward(self, np.ndarray[np.float64_t, ndim=2] x, 
                np.ndarray[np.float64_t, ndim=2] h_prev=None):
        """
        Forward pass for single timestep.
        
        Args:
            x: Input of shape (batch_size, input_size)
            h_prev: Previous hidden state of shape (batch_size, hidden_size)
            
        Returns:
            h_next: Next hidden state of shape (batch_size, hidden_size)
        """
        cdef Py_ssize_t batch_size = x.shape[0]
        
        if h_prev is None:
            h_prev = np.zeros((batch_size, self.hidden_size), dtype=np.float64)
        
        # Cache for backward
        self.cached_inputs.append(x.copy())
        self.cached_hiddens.append(h_prev.copy())
        
        # Compute pre-activation: W_ih @ x + b_ih + W_hh @ h_prev + b_hh
        cdef np.ndarray[np.float64_t, ndim=2] pre_act = (
            np.dot(x, self.weight_ih.T) + self.bias_ih +
            np.dot(h_prev, self.weight_hh.T) + self.bias_hh
        )
        
        self.cached_pre_activations.append(pre_act.copy())
        
        # Apply tanh activation
        cdef np.ndarray[np.float64_t, ndim=2] h_next = np.tanh(pre_act)
        
        return h_next
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def backward(self, np.ndarray[np.float64_t, ndim=2] grad_h):
        """
        Backward pass for single timestep.
        
        Args:
            grad_h: Gradient w.r.t. hidden state of shape (batch_size, hidden_size)
            
        Returns:
            (grad_x, grad_h_prev): Gradients w.r.t. input and previous hidden state
        """
        if len(self.cached_inputs) == 0:
            raise ValueError("No cached values for backward pass")
        
        # Pop cached values (LIFO for backprop through time)
        cdef np.ndarray[np.float64_t, ndim=2] x = self.cached_inputs.pop()
        cdef np.ndarray[np.float64_t, ndim=2] h_prev = self.cached_hiddens.pop()
        cdef np.ndarray[np.float64_t, ndim=2] pre_act = self.cached_pre_activations.pop()
        
        # Gradient through tanh: dtanh/dx = 1 - tanh^2(x)
        cdef np.ndarray[np.float64_t, ndim=2] h_next = np.tanh(pre_act)
        cdef np.ndarray[np.float64_t, ndim=2] grad_pre = grad_h * (1.0 - h_next * h_next)
        
        # Accumulate weight gradients
        self.grad_weight_ih += np.dot(grad_pre.T, x)
        self.grad_weight_hh += np.dot(grad_pre.T, h_prev)
        self.grad_bias_ih += np.sum(grad_pre, axis=0)
        self.grad_bias_hh += np.sum(grad_pre, axis=0)
        
        # Compute gradients w.r.t. inputs
        cdef np.ndarray[np.float64_t, ndim=2] grad_x = np.dot(grad_pre, self.weight_ih)
        cdef np.ndarray[np.float64_t, ndim=2] grad_h_prev = np.dot(grad_pre, self.weight_hh)
        
        return grad_x, grad_h_prev
    
    def update_parameters(self, double learning_rate):
        """Update parameters using gradients."""
        self.weight_ih -= learning_rate * self.grad_weight_ih
        self.weight_hh -= learning_rate * self.grad_weight_hh
        self.bias_ih -= learning_rate * self.grad_bias_ih
        self.bias_hh -= learning_rate * self.grad_bias_hh
    
    def zero_grad(self):
        """Reset gradients to zero."""
        self.grad_weight_ih[:] = 0.0
        self.grad_weight_hh[:] = 0.0
        self.grad_bias_ih[:] = 0.0
        self.grad_bias_hh[:] = 0.0


cdef class LSTMCell:
    """
    Optimized LSTM cell.
    
    Implements the LSTM equations:
        i_t = sigmoid(W_ii @ x_t + b_ii + W_hi @ h_{t-1} + b_hi)  # Input gate
        f_t = sigmoid(W_if @ x_t + b_if + W_hf @ h_{t-1} + b_hf)  # Forget gate
        g_t = tanh(W_ig @ x_t + b_ig + W_hg @ h_{t-1} + b_hg)     # Cell gate
        o_t = sigmoid(W_io @ x_t + b_io + W_ho @ h_{t-1} + b_ho)  # Output gate
        c_t = f_t * c_{t-1} + i_t * g_t                           # Cell state
        h_t = o_t * tanh(c_t)                                      # Hidden state
    
    All gates are packed into single weight matrices for efficiency.
    """
    
    cdef public int input_size
    cdef public int hidden_size
    # Packed weights: [i, f, g, o] = 4 * hidden_size
    cdef public np.ndarray weight_ih  # (4 * hidden_size, input_size)
    cdef public np.ndarray weight_hh  # (4 * hidden_size, hidden_size)
    cdef public np.ndarray bias_ih    # (4 * hidden_size,)
    cdef public np.ndarray bias_hh    # (4 * hidden_size,)
    cdef public np.ndarray grad_weight_ih
    cdef public np.ndarray grad_weight_hh
    cdef public np.ndarray grad_bias_ih
    cdef public np.ndarray grad_bias_hh
    # Cache for backward pass
    cdef public list cached_inputs
    cdef public list cached_hiddens
    cdef public list cached_cells
    cdef public list cached_gates
    
    def __init__(self, int input_size, int hidden_size):
        """
        Initialize LSTM cell.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights with Xavier/Glorot initialization
        cdef double std = sqrt(2.0 / (input_size + hidden_size))
        self.weight_ih = np.random.randn(4 * hidden_size, input_size).astype(np.float64) * std
        self.weight_hh = np.random.randn(4 * hidden_size, hidden_size).astype(np.float64) * std
        self.bias_ih = np.zeros(4 * hidden_size, dtype=np.float64)
        self.bias_hh = np.zeros(4 * hidden_size, dtype=np.float64)
        
        # Forget gate bias initialization (set to 1.0 for better gradient flow)
        self.bias_ih[hidden_size:2*hidden_size] = 1.0
        
        # Initialize gradients
        self.grad_weight_ih = np.zeros_like(self.weight_ih)
        self.grad_weight_hh = np.zeros_like(self.weight_hh)
        self.grad_bias_ih = np.zeros_like(self.bias_ih)
        self.grad_bias_hh = np.zeros_like(self.bias_hh)
        
        self.cached_inputs = []
        self.cached_hiddens = []
        self.cached_cells = []
        self.cached_gates = []
    
    def reset_cache(self):
        """Reset cache for new sequence."""
        self.cached_inputs = []
        self.cached_hiddens = []
        self.cached_cells = []
        self.cached_gates = []
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def forward(self, np.ndarray[np.float64_t, ndim=2] x,
                tuple hc_prev=None):
        """
        Forward pass for single timestep.
        
        Args:
            x: Input of shape (batch_size, input_size)
            hc_prev: Tuple of (h_prev, c_prev) or None
            
        Returns:
            (h_next, c_next): Next hidden and cell states
        """
        cdef Py_ssize_t batch_size = x.shape[0]
        cdef np.ndarray[np.float64_t, ndim=2] h_prev, c_prev
        
        if hc_prev is None:
            h_prev = np.zeros((batch_size, self.hidden_size), dtype=np.float64)
            c_prev = np.zeros((batch_size, self.hidden_size), dtype=np.float64)
        else:
            h_prev, c_prev = hc_prev
        
        # Cache for backward
        self.cached_inputs.append(x.copy())
        self.cached_hiddens.append(h_prev.copy())
        self.cached_cells.append(c_prev.copy())
        
        # Compute all gates at once: (batch_size, 4 * hidden_size)
        cdef np.ndarray[np.float64_t, ndim=2] gates = (
            np.dot(x, self.weight_ih.T) + self.bias_ih +
            np.dot(h_prev, self.weight_hh.T) + self.bias_hh
        )
        
        cdef Py_ssize_t h = self.hidden_size
        
        # Split gates
        cdef np.ndarray[np.float64_t, ndim=2] i_gate = 1.0 / (1.0 + np.exp(-gates[:, :h]))
        cdef np.ndarray[np.float64_t, ndim=2] f_gate = 1.0 / (1.0 + np.exp(-gates[:, h:2*h]))
        cdef np.ndarray[np.float64_t, ndim=2] g_gate = np.tanh(gates[:, 2*h:3*h])
        cdef np.ndarray[np.float64_t, ndim=2] o_gate = 1.0 / (1.0 + np.exp(-gates[:, 3*h:]))
        
        # Cache gates for backward
        self.cached_gates.append((i_gate, f_gate, g_gate, o_gate))
        
        # Compute cell and hidden states
        cdef np.ndarray[np.float64_t, ndim=2] c_next = f_gate * c_prev + i_gate * g_gate
        cdef np.ndarray[np.float64_t, ndim=2] h_next = o_gate * np.tanh(c_next)
        
        return h_next, c_next
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def backward(self, np.ndarray[np.float64_t, ndim=2] grad_h,
                 np.ndarray[np.float64_t, ndim=2] grad_c):
        """
        Backward pass for single timestep.
        
        Args:
            grad_h: Gradient w.r.t. hidden state
            grad_c: Gradient w.r.t. cell state (from future)
            
        Returns:
            (grad_x, grad_h_prev, grad_c_prev)
        """
        if len(self.cached_inputs) == 0:
            raise ValueError("No cached values for backward pass")
        
        # Pop cached values
        cdef np.ndarray[np.float64_t, ndim=2] x = self.cached_inputs.pop()
        cdef np.ndarray[np.float64_t, ndim=2] h_prev = self.cached_hiddens.pop()
        cdef np.ndarray[np.float64_t, ndim=2] c_prev = self.cached_cells.pop()
        i_gate, f_gate, g_gate, o_gate = self.cached_gates.pop()
        
        cdef Py_ssize_t h = self.hidden_size
        
        # Recompute cell state
        cdef np.ndarray[np.float64_t, ndim=2] c_next = f_gate * c_prev + i_gate * g_gate
        cdef np.ndarray[np.float64_t, ndim=2] tanh_c = np.tanh(c_next)
        
        # Gradient through hidden state
        cdef np.ndarray[np.float64_t, ndim=2] grad_o = grad_h * tanh_c
        cdef np.ndarray[np.float64_t, ndim=2] grad_c_total = grad_c + grad_h * o_gate * (1.0 - tanh_c * tanh_c)
        
        # Gradient through gates
        cdef np.ndarray[np.float64_t, ndim=2] grad_i = grad_c_total * g_gate
        cdef np.ndarray[np.float64_t, ndim=2] grad_f = grad_c_total * c_prev
        cdef np.ndarray[np.float64_t, ndim=2] grad_g = grad_c_total * i_gate
        
        # Gradient through activations
        cdef np.ndarray[np.float64_t, ndim=2] grad_i_pre = grad_i * i_gate * (1.0 - i_gate)
        cdef np.ndarray[np.float64_t, ndim=2] grad_f_pre = grad_f * f_gate * (1.0 - f_gate)
        cdef np.ndarray[np.float64_t, ndim=2] grad_g_pre = grad_g * (1.0 - g_gate * g_gate)
        cdef np.ndarray[np.float64_t, ndim=2] grad_o_pre = grad_o * o_gate * (1.0 - o_gate)
        
        # Concatenate gate gradients
        cdef np.ndarray[np.float64_t, ndim=2] grad_gates = np.hstack([
            grad_i_pre, grad_f_pre, grad_g_pre, grad_o_pre
        ])
        
        # Accumulate weight gradients
        self.grad_weight_ih += np.dot(grad_gates.T, x)
        self.grad_weight_hh += np.dot(grad_gates.T, h_prev)
        self.grad_bias_ih += np.sum(grad_gates, axis=0)
        self.grad_bias_hh += np.sum(grad_gates, axis=0)
        
        # Compute gradients w.r.t. inputs
        cdef np.ndarray[np.float64_t, ndim=2] grad_x = np.dot(grad_gates, self.weight_ih)
        cdef np.ndarray[np.float64_t, ndim=2] grad_h_prev = np.dot(grad_gates, self.weight_hh)
        cdef np.ndarray[np.float64_t, ndim=2] grad_c_prev = grad_c_total * f_gate
        
        return grad_x, grad_h_prev, grad_c_prev
    
    def update_parameters(self, double learning_rate):
        """Update parameters using gradients."""
        self.weight_ih -= learning_rate * self.grad_weight_ih
        self.weight_hh -= learning_rate * self.grad_weight_hh
        self.bias_ih -= learning_rate * self.grad_bias_ih
        self.bias_hh -= learning_rate * self.grad_bias_hh
    
    def zero_grad(self):
        """Reset gradients to zero."""
        self.grad_weight_ih[:] = 0.0
        self.grad_weight_hh[:] = 0.0
        self.grad_bias_ih[:] = 0.0
        self.grad_bias_hh[:] = 0.0


cdef class GRUCell:
    """
    Optimized GRU cell.
    
    Implements the GRU equations:
        r_t = sigmoid(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)  # Reset gate
        z_t = sigmoid(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)  # Update gate
        n_t = tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_{t-1} + b_hn))  # New gate
        h_t = (1 - z_t) * n_t + z_t * h_{t-1}                      # Hidden state
    """
    
    cdef public int input_size
    cdef public int hidden_size
    # Packed weights: [r, z, n] = 3 * hidden_size
    cdef public np.ndarray weight_ih  # (3 * hidden_size, input_size)
    cdef public np.ndarray weight_hh  # (3 * hidden_size, hidden_size)
    cdef public np.ndarray bias_ih    # (3 * hidden_size,)
    cdef public np.ndarray bias_hh    # (3 * hidden_size,)
    cdef public np.ndarray grad_weight_ih
    cdef public np.ndarray grad_weight_hh
    cdef public np.ndarray grad_bias_ih
    cdef public np.ndarray grad_bias_hh
    # Cache for backward pass
    cdef public list cached_inputs
    cdef public list cached_hiddens
    cdef public list cached_gates
    
    def __init__(self, int input_size, int hidden_size):
        """
        Initialize GRU cell.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights with Xavier/Glorot initialization
        cdef double std = sqrt(2.0 / (input_size + hidden_size))
        self.weight_ih = np.random.randn(3 * hidden_size, input_size).astype(np.float64) * std
        self.weight_hh = np.random.randn(3 * hidden_size, hidden_size).astype(np.float64) * std
        self.bias_ih = np.zeros(3 * hidden_size, dtype=np.float64)
        self.bias_hh = np.zeros(3 * hidden_size, dtype=np.float64)
        
        # Initialize gradients
        self.grad_weight_ih = np.zeros_like(self.weight_ih)
        self.grad_weight_hh = np.zeros_like(self.weight_hh)
        self.grad_bias_ih = np.zeros_like(self.bias_ih)
        self.grad_bias_hh = np.zeros_like(self.bias_hh)
        
        self.cached_inputs = []
        self.cached_hiddens = []
        self.cached_gates = []
    
    def reset_cache(self):
        """Reset cache for new sequence."""
        self.cached_inputs = []
        self.cached_hiddens = []
        self.cached_gates = []
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def forward(self, np.ndarray[np.float64_t, ndim=2] x,
                np.ndarray[np.float64_t, ndim=2] h_prev=None):
        """
        Forward pass for single timestep.
        
        Args:
            x: Input of shape (batch_size, input_size)
            h_prev: Previous hidden state or None
            
        Returns:
            h_next: Next hidden state
        """
        cdef Py_ssize_t batch_size = x.shape[0]
        
        if h_prev is None:
            h_prev = np.zeros((batch_size, self.hidden_size), dtype=np.float64)
        
        # Cache for backward
        self.cached_inputs.append(x.copy())
        self.cached_hiddens.append(h_prev.copy())
        
        cdef Py_ssize_t h = self.hidden_size
        
        # Compute input transform
        cdef np.ndarray[np.float64_t, ndim=2] x_transform = (
            np.dot(x, self.weight_ih.T) + self.bias_ih
        )
        
        # Compute hidden transform
        cdef np.ndarray[np.float64_t, ndim=2] h_transform = (
            np.dot(h_prev, self.weight_hh.T) + self.bias_hh
        )
        
        # Reset and update gates
        cdef np.ndarray[np.float64_t, ndim=2] r_gate = 1.0 / (1.0 + np.exp(
            -(x_transform[:, :h] + h_transform[:, :h])
        ))
        cdef np.ndarray[np.float64_t, ndim=2] z_gate = 1.0 / (1.0 + np.exp(
            -(x_transform[:, h:2*h] + h_transform[:, h:2*h])
        ))
        
        # New gate (with reset)
        cdef np.ndarray[np.float64_t, ndim=2] n_gate = np.tanh(
            x_transform[:, 2*h:] + r_gate * h_transform[:, 2*h:]
        )
        
        # Cache gates for backward
        self.cached_gates.append((r_gate, z_gate, n_gate, x_transform, h_transform))
        
        # Compute hidden state
        cdef np.ndarray[np.float64_t, ndim=2] h_next = (1.0 - z_gate) * n_gate + z_gate * h_prev
        
        return h_next
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def backward(self, np.ndarray[np.float64_t, ndim=2] grad_h):
        """
        Backward pass for single timestep.
        
        Args:
            grad_h: Gradient w.r.t. hidden state
            
        Returns:
            (grad_x, grad_h_prev)
        """
        if len(self.cached_inputs) == 0:
            raise ValueError("No cached values for backward pass")
        
        # Pop cached values
        cdef np.ndarray[np.float64_t, ndim=2] x = self.cached_inputs.pop()
        cdef np.ndarray[np.float64_t, ndim=2] h_prev = self.cached_hiddens.pop()
        r_gate, z_gate, n_gate, x_transform, h_transform = self.cached_gates.pop()
        
        cdef Py_ssize_t h = self.hidden_size
        
        # Gradient through hidden state computation
        cdef np.ndarray[np.float64_t, ndim=2] grad_n = grad_h * (1.0 - z_gate)
        cdef np.ndarray[np.float64_t, ndim=2] grad_z = grad_h * (h_prev - n_gate)
        cdef np.ndarray[np.float64_t, ndim=2] grad_h_prev_direct = grad_h * z_gate
        
        # Gradient through new gate (tanh)
        cdef np.ndarray[np.float64_t, ndim=2] grad_n_pre = grad_n * (1.0 - n_gate * n_gate)
        
        # Gradient through reset gate
        cdef np.ndarray[np.float64_t, ndim=2] grad_r = grad_n_pre * h_transform[:, 2*h:]
        cdef np.ndarray[np.float64_t, ndim=2] grad_h_transform_n = grad_n_pre * r_gate
        
        # Gradient through gate activations (sigmoid)
        cdef np.ndarray[np.float64_t, ndim=2] grad_r_pre = grad_r * r_gate * (1.0 - r_gate)
        cdef np.ndarray[np.float64_t, ndim=2] grad_z_pre = grad_z * z_gate * (1.0 - z_gate)
        
        # Concatenate gate gradients for weights
        cdef np.ndarray[np.float64_t, ndim=2] grad_x_transform = np.hstack([
            grad_r_pre, grad_z_pre, grad_n_pre
        ])
        cdef np.ndarray[np.float64_t, ndim=2] grad_h_transform = np.hstack([
            grad_r_pre, grad_z_pre, grad_h_transform_n
        ])
        
        # Accumulate weight gradients
        self.grad_weight_ih += np.dot(grad_x_transform.T, x)
        self.grad_weight_hh += np.dot(grad_h_transform.T, h_prev)
        self.grad_bias_ih += np.sum(grad_x_transform, axis=0)
        self.grad_bias_hh += np.sum(grad_h_transform, axis=0)
        
        # Compute gradients w.r.t. inputs
        cdef np.ndarray[np.float64_t, ndim=2] grad_x = np.dot(grad_x_transform, self.weight_ih)
        cdef np.ndarray[np.float64_t, ndim=2] grad_h_prev = (
            grad_h_prev_direct +
            np.dot(grad_h_transform, self.weight_hh)
        )
        
        return grad_x, grad_h_prev
    
    def update_parameters(self, double learning_rate):
        """Update parameters using gradients."""
        self.weight_ih -= learning_rate * self.grad_weight_ih
        self.weight_hh -= learning_rate * self.grad_weight_hh
        self.bias_ih -= learning_rate * self.grad_bias_ih
        self.bias_hh -= learning_rate * self.grad_bias_hh
    
    def zero_grad(self):
        """Reset gradients to zero."""
        self.grad_weight_ih[:] = 0.0
        self.grad_weight_hh[:] = 0.0
        self.grad_bias_ih[:] = 0.0
        self.grad_bias_hh[:] = 0.0


# Sequence processing helpers

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple rnn_forward_sequence(object cell, np.ndarray x, object h_init=None):
    """
    Process a full sequence through an RNN cell.
    
    Args:
        cell: RNNCell instance
        x: Input sequence of shape (seq_len, batch_size, input_size)
        h_init: Initial hidden state or None
        
    Returns:
        (outputs, h_final) where outputs is (seq_len, batch_size, hidden_size)
    """
    cdef Py_ssize_t seq_len = x.shape[0]
    cdef Py_ssize_t batch_size = x.shape[1]
    cdef Py_ssize_t hidden_size = cell.hidden_size
    
    cdef np.ndarray[np.float64_t, ndim=3] outputs = np.empty(
        (seq_len, batch_size, hidden_size), dtype=np.float64
    )
    
    cdef np.ndarray[np.float64_t, ndim=2] h = h_init
    cdef Py_ssize_t t
    
    cell.reset_cache()
    
    for t in range(seq_len):
        h = cell.forward(x[t], h)
        outputs[t] = h
    
    return outputs, h


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple lstm_forward_sequence(object cell, np.ndarray x, tuple hc_init=None):
    """
    Process a full sequence through an LSTM cell.
    
    Args:
        cell: LSTMCell instance
        x: Input sequence of shape (seq_len, batch_size, input_size)
        hc_init: Tuple of (h_init, c_init) or None
        
    Returns:
        (outputs, (h_final, c_final))
    """
    cdef Py_ssize_t seq_len = x.shape[0]
    cdef Py_ssize_t batch_size = x.shape[1]
    cdef Py_ssize_t hidden_size = cell.hidden_size
    
    cdef np.ndarray[np.float64_t, ndim=3] outputs = np.empty(
        (seq_len, batch_size, hidden_size), dtype=np.float64
    )
    
    cdef tuple hc = hc_init
    cdef np.ndarray[np.float64_t, ndim=2] h, c
    cdef Py_ssize_t t
    
    cell.reset_cache()
    
    for t in range(seq_len):
        h, c = cell.forward(x[t], hc)
        hc = (h, c)
        outputs[t] = h
    
    return outputs, hc
