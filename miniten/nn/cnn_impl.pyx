# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Optimized CNN layers for MiniTen.

This module provides high-performance implementations of:
- Conv2d layer with im2col optimization
- MaxPool2d and AvgPool2d layers
- Dropout layer
- BatchNorm2d layer

All implementations use Cython for maximum performance and minimal memory footprint,
optimized for edge devices.
"""

import cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport sqrt, fmax
from libc.stdlib cimport malloc, free

# Initialize NumPy C API
np.import_array()

# Threshold for parallel operations
DEF PARALLEL_THRESHOLD = 10000


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] im2col(np.ndarray[np.float64_t, ndim=4] x,
                                               int kernel_h, int kernel_w,
                                               int stride_h, int stride_w,
                                               int pad_h, int pad_w):
    """
    Convert image to column matrix for efficient convolution.
    
    This is the core operation that converts convolution to matrix multiplication.
    
    Args:
        x: Input tensor of shape (N, C, H, W)
        kernel_h, kernel_w: Kernel dimensions
        stride_h, stride_w: Stride dimensions
        pad_h, pad_w: Padding dimensions
        
    Returns:
        Column matrix of shape (N * out_h * out_w, C * kernel_h * kernel_w)
    """
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t C = x.shape[1]
    cdef Py_ssize_t H = x.shape[2]
    cdef Py_ssize_t W = x.shape[3]
    
    cdef Py_ssize_t out_h = (H + 2 * pad_h - kernel_h) // stride_h + 1
    cdef Py_ssize_t out_w = (W + 2 * pad_w - kernel_w) // stride_w + 1
    
    cdef Py_ssize_t col_h = N * out_h * out_w
    cdef Py_ssize_t col_w = C * kernel_h * kernel_w
    
    cdef np.ndarray[np.float64_t, ndim=2] col = np.zeros((col_h, col_w), dtype=np.float64)
    cdef double[:, :, :, ::1] x_view = x
    cdef double[:, ::1] col_view = col
    
    cdef Py_ssize_t n, c, oh, ow, kh, kw
    cdef Py_ssize_t h_in, w_in
    cdef Py_ssize_t row_idx, col_idx
    
    for n in range(N):
        for oh in range(out_h):
            for ow in range(out_w):
                row_idx = n * out_h * out_w + oh * out_w + ow
                for c in range(C):
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            h_in = oh * stride_h - pad_h + kh
                            w_in = ow * stride_w - pad_w + kw
                            col_idx = c * kernel_h * kernel_w + kh * kernel_w + kw
                            if 0 <= h_in < H and 0 <= w_in < W:
                                col_view[row_idx, col_idx] = x_view[n, c, h_in, w_in]
                            else:
                                col_view[row_idx, col_idx] = 0.0
    
    return col


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=4] col2im(np.ndarray[np.float64_t, ndim=2] col,
                                               tuple input_shape,
                                               int kernel_h, int kernel_w,
                                               int stride_h, int stride_w,
                                               int pad_h, int pad_w):
    """
    Convert column matrix back to image (for backward pass).
    
    Args:
        col: Column matrix of shape (N * out_h * out_w, C * kernel_h * kernel_w)
        input_shape: Original input shape (N, C, H, W)
        kernel_h, kernel_w: Kernel dimensions
        stride_h, stride_w: Stride dimensions
        pad_h, pad_w: Padding dimensions
        
    Returns:
        Image tensor of shape (N, C, H, W)
    """
    cdef Py_ssize_t N = input_shape[0]
    cdef Py_ssize_t C = input_shape[1]
    cdef Py_ssize_t H = input_shape[2]
    cdef Py_ssize_t W = input_shape[3]
    
    cdef Py_ssize_t out_h = (H + 2 * pad_h - kernel_h) // stride_h + 1
    cdef Py_ssize_t out_w = (W + 2 * pad_w - kernel_w) // stride_w + 1
    
    cdef np.ndarray[np.float64_t, ndim=4] img = np.zeros((N, C, H, W), dtype=np.float64)
    cdef double[:, ::1] col_view = col
    cdef double[:, :, :, ::1] img_view = img
    
    cdef Py_ssize_t n, c, oh, ow, kh, kw
    cdef Py_ssize_t h_in, w_in
    cdef Py_ssize_t row_idx, col_idx
    
    for n in range(N):
        for oh in range(out_h):
            for ow in range(out_w):
                row_idx = n * out_h * out_w + oh * out_w + ow
                for c in range(C):
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            h_in = oh * stride_h - pad_h + kh
                            w_in = ow * stride_w - pad_w + kw
                            col_idx = c * kernel_h * kernel_w + kh * kernel_w + kw
                            if 0 <= h_in < H and 0 <= w_in < W:
                                img_view[n, c, h_in, w_in] += col_view[row_idx, col_idx]
    
    return img


cdef class Conv2d:
    """
    Optimized 2D Convolutional layer using im2col + GEMM.
    
    This implementation converts convolution to matrix multiplication for
    efficient computation, suitable for edge devices.
    
    Attributes:
        weight: Weight tensor of shape (out_channels, in_channels, kernel_h, kernel_w)
        bias: Bias vector of shape (out_channels,)
        grad_weight: Gradient of weight
        grad_bias: Gradient of bias
    """
    
    cdef public np.ndarray weight
    cdef public np.ndarray bias
    cdef public np.ndarray grad_weight
    cdef public np.ndarray grad_bias
    cdef public int in_channels
    cdef public int out_channels
    cdef public int kernel_h
    cdef public int kernel_w
    cdef public int stride_h
    cdef public int stride_w
    cdef public int pad_h
    cdef public int pad_w
    cdef public bint use_bias
    # Cache for backward pass
    cdef public np.ndarray cached_input
    cdef public np.ndarray cached_col
    cdef public tuple input_shape
    
    def __init__(self, int in_channels, int out_channels, 
                 int kernel_size, int stride=1, int padding=0, bint use_bias=True):
        """
        Initialize Conv2d layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolution kernel (assumed square)
            stride: Stride of the convolution
            padding: Zero-padding added to both sides
            use_bias: Whether to use bias
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h = kernel_size
        self.kernel_w = kernel_size
        self.stride_h = stride
        self.stride_w = stride
        self.pad_h = padding
        self.pad_w = padding
        self.use_bias = use_bias
        
        # He initialization for weights
        cdef double std = sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weight = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float64) * std
        
        if use_bias:
            self.bias = np.zeros(out_channels, dtype=np.float64)
        else:
            self.bias = None
        
        self.grad_weight = np.zeros_like(self.weight)
        if use_bias:
            self.grad_bias = np.zeros(out_channels, dtype=np.float64)
        
        self.cached_input = None
        self.cached_col = None
        self.input_shape = None
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def forward(self, np.ndarray[np.float64_t, ndim=4] x):
        """
        Forward pass using im2col + GEMM.
        
        Args:
            x: Input of shape (N, C, H, W)
            
        Returns:
            Output of shape (N, out_channels, out_h, out_w)
        """
        cdef Py_ssize_t N = x.shape[0]
        cdef Py_ssize_t C = x.shape[1]
        cdef Py_ssize_t H = x.shape[2]
        cdef Py_ssize_t W = x.shape[3]
        
        cdef Py_ssize_t out_h = (H + 2 * self.pad_h - self.kernel_h) // self.stride_h + 1
        cdef Py_ssize_t out_w = (W + 2 * self.pad_w - self.kernel_w) // self.stride_w + 1
        
        # Cache for backward pass
        self.cached_input = x.copy()
        self.input_shape = (N, C, H, W)
        
        # im2col: (N * out_h * out_w, C * kernel_h * kernel_w)
        cdef np.ndarray[np.float64_t, ndim=2] col = im2col(x, self.kernel_h, self.kernel_w,
                                                            self.stride_h, self.stride_w,
                                                            self.pad_h, self.pad_w)
        self.cached_col = col
        
        # Reshape weight: (out_channels, C * kernel_h * kernel_w)
        cdef np.ndarray[np.float64_t, ndim=2] weight_col = self.weight.reshape(self.out_channels, -1)
        
        # Matrix multiplication: (N * out_h * out_w, out_channels)
        cdef np.ndarray[np.float64_t, ndim=2] out_col = np.dot(col, weight_col.T)
        
        # Add bias if enabled
        if self.use_bias:
            out_col = out_col + self.bias
        
        # Reshape to output: (N, out_channels, out_h, out_w)
        cdef np.ndarray[np.float64_t, ndim=4] output = out_col.reshape(N, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2).copy()
        
        return output
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def backward(self, np.ndarray[np.float64_t, ndim=4] grad_output):
        """
        Backward pass to compute gradients.
        
        Args:
            grad_output: Gradient from next layer of shape (N, out_channels, out_h, out_w)
            
        Returns:
            grad_input: Gradient w.r.t. input of shape (N, C, H, W)
        """
        cdef Py_ssize_t N = grad_output.shape[0]
        cdef Py_ssize_t out_h = grad_output.shape[2]
        cdef Py_ssize_t out_w = grad_output.shape[3]
        
        # Reshape grad_output: (N * out_h * out_w, out_channels)
        cdef np.ndarray[np.float64_t, ndim=2] grad_out_col = grad_output.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        
        # Gradient w.r.t. weight
        # grad_weight = grad_out_col.T @ col -> (out_channels, C * kernel_h * kernel_w)
        cdef tuple weight_shape = (self.out_channels, self.in_channels, self.kernel_h, self.kernel_w)
        self.grad_weight = np.dot(grad_out_col.T, self.cached_col).reshape(weight_shape)
        
        # Gradient w.r.t. bias
        if self.use_bias:
            self.grad_bias = np.sum(grad_out_col, axis=0)
        
        # Gradient w.r.t. input
        # grad_col = grad_out_col @ weight_col -> (N * out_h * out_w, C * kernel_h * kernel_w)
        cdef np.ndarray[np.float64_t, ndim=2] weight_col = self.weight.reshape(self.out_channels, -1)
        cdef np.ndarray[np.float64_t, ndim=2] grad_col = np.dot(grad_out_col, weight_col)
        
        # col2im to get grad_input
        cdef np.ndarray[np.float64_t, ndim=4] grad_input = col2im(grad_col, self.input_shape,
                                                                   self.kernel_h, self.kernel_w,
                                                                   self.stride_h, self.stride_w,
                                                                   self.pad_h, self.pad_w)
        
        return grad_input
    
    def update_parameters(self, double learning_rate):
        """Update parameters using gradients."""
        self.weight -= learning_rate * self.grad_weight
        if self.use_bias:
            self.bias -= learning_rate * self.grad_bias


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple maxpool2d_forward(np.ndarray[np.float64_t, ndim=4] x, 
                               int kernel_size, int stride=0):
    """
    Max pooling 2D forward pass.
    
    Args:
        x: Input of shape (N, C, H, W)
        kernel_size: Size of pooling window
        stride: Stride of pooling (default: kernel_size)
        
    Returns:
        (output, indices) where indices are for backward pass
    """
    if stride == 0:
        stride = kernel_size
    
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t C = x.shape[1]
    cdef Py_ssize_t H = x.shape[2]
    cdef Py_ssize_t W = x.shape[3]
    
    cdef Py_ssize_t out_h = (H - kernel_size) // stride + 1
    cdef Py_ssize_t out_w = (W - kernel_size) // stride + 1
    
    cdef np.ndarray[np.float64_t, ndim=4] output = np.empty((N, C, out_h, out_w), dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim=4] indices = np.empty((N, C, out_h, out_w), dtype=np.int64)
    
    cdef double[:, :, :, ::1] x_view = x
    cdef double[:, :, :, ::1] out_view = output
    cdef long[:, :, :, ::1] idx_view = indices
    
    cdef Py_ssize_t n, c, oh, ow, kh, kw
    cdef Py_ssize_t h_start, w_start
    cdef double max_val, val
    cdef Py_ssize_t max_idx
    
    for n in range(N):
        for c in range(C):
            for oh in range(out_h):
                for ow in range(out_w):
                    h_start = oh * stride
                    w_start = ow * stride
                    max_val = x_view[n, c, h_start, w_start]
                    max_idx = 0
                    
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            val = x_view[n, c, h_start + kh, w_start + kw]
                            if val > max_val:
                                max_val = val
                                max_idx = kh * kernel_size + kw
                    
                    out_view[n, c, oh, ow] = max_val
                    idx_view[n, c, oh, ow] = max_idx
    
    return output, indices


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=4] maxpool2d_backward(np.ndarray[np.float64_t, ndim=4] grad_output,
                                                           np.ndarray[np.int64_t, ndim=4] indices,
                                                           tuple input_shape,
                                                           int kernel_size, int stride=0):
    """
    Max pooling 2D backward pass.
    
    Args:
        grad_output: Gradient from next layer
        indices: Indices from forward pass
        input_shape: Original input shape
        kernel_size: Size of pooling window
        stride: Stride of pooling
        
    Returns:
        Gradient w.r.t. input
    """
    if stride == 0:
        stride = kernel_size
    
    cdef Py_ssize_t N = input_shape[0]
    cdef Py_ssize_t C = input_shape[1]
    cdef Py_ssize_t H = input_shape[2]
    cdef Py_ssize_t W = input_shape[3]
    
    cdef Py_ssize_t out_h = grad_output.shape[2]
    cdef Py_ssize_t out_w = grad_output.shape[3]
    
    cdef np.ndarray[np.float64_t, ndim=4] grad_input = np.zeros((N, C, H, W), dtype=np.float64)
    
    cdef double[:, :, :, ::1] grad_out_view = grad_output
    cdef double[:, :, :, ::1] grad_in_view = grad_input
    cdef long[:, :, :, ::1] idx_view = indices
    
    cdef Py_ssize_t n, c, oh, ow
    cdef Py_ssize_t h_start, w_start, kh, kw
    cdef long max_idx
    
    for n in range(N):
        for c in range(C):
            for oh in range(out_h):
                for ow in range(out_w):
                    h_start = oh * stride
                    w_start = ow * stride
                    max_idx = idx_view[n, c, oh, ow]
                    kh = max_idx // kernel_size
                    kw = max_idx % kernel_size
                    grad_in_view[n, c, h_start + kh, w_start + kw] += grad_out_view[n, c, oh, ow]
    
    return grad_input


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple avgpool2d_forward(np.ndarray[np.float64_t, ndim=4] x, 
                               int kernel_size, int stride=0):
    """
    Average pooling 2D forward pass.
    
    Args:
        x: Input of shape (N, C, H, W)
        kernel_size: Size of pooling window
        stride: Stride of pooling (default: kernel_size)
        
    Returns:
        output of shape (N, C, out_h, out_w)
    """
    if stride == 0:
        stride = kernel_size
    
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t C = x.shape[1]
    cdef Py_ssize_t H = x.shape[2]
    cdef Py_ssize_t W = x.shape[3]
    
    cdef Py_ssize_t out_h = (H - kernel_size) // stride + 1
    cdef Py_ssize_t out_w = (W - kernel_size) // stride + 1
    
    cdef np.ndarray[np.float64_t, ndim=4] output = np.empty((N, C, out_h, out_w), dtype=np.float64)
    
    cdef double[:, :, :, ::1] x_view = x
    cdef double[:, :, :, ::1] out_view = output
    
    cdef Py_ssize_t n, c, oh, ow, kh, kw
    cdef Py_ssize_t h_start, w_start
    cdef double sum_val
    cdef double pool_size = kernel_size * kernel_size
    
    for n in range(N):
        for c in range(C):
            for oh in range(out_h):
                for ow in range(out_w):
                    h_start = oh * stride
                    w_start = ow * stride
                    sum_val = 0.0
                    
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            sum_val += x_view[n, c, h_start + kh, w_start + kw]
                    
                    out_view[n, c, oh, ow] = sum_val / pool_size
    
    return output, None


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=4] avgpool2d_backward(np.ndarray[np.float64_t, ndim=4] grad_output,
                                                           tuple input_shape,
                                                           int kernel_size, int stride=0):
    """
    Average pooling 2D backward pass.
    
    Args:
        grad_output: Gradient from next layer
        input_shape: Original input shape
        kernel_size: Size of pooling window
        stride: Stride of pooling
        
    Returns:
        Gradient w.r.t. input
    """
    if stride == 0:
        stride = kernel_size
    
    cdef Py_ssize_t N = input_shape[0]
    cdef Py_ssize_t C = input_shape[1]
    cdef Py_ssize_t H = input_shape[2]
    cdef Py_ssize_t W = input_shape[3]
    
    cdef Py_ssize_t out_h = grad_output.shape[2]
    cdef Py_ssize_t out_w = grad_output.shape[3]
    
    cdef np.ndarray[np.float64_t, ndim=4] grad_input = np.zeros((N, C, H, W), dtype=np.float64)
    
    cdef double[:, :, :, ::1] grad_out_view = grad_output
    cdef double[:, :, :, ::1] grad_in_view = grad_input
    
    cdef Py_ssize_t n, c, oh, ow, kh, kw
    cdef Py_ssize_t h_start, w_start
    cdef double grad_val
    cdef double pool_size = kernel_size * kernel_size
    
    for n in range(N):
        for c in range(C):
            for oh in range(out_h):
                for ow in range(out_w):
                    h_start = oh * stride
                    w_start = ow * stride
                    grad_val = grad_out_view[n, c, oh, ow] / pool_size
                    
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            grad_in_view[n, c, h_start + kh, w_start + kw] += grad_val
    
    return grad_input


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple dropout_forward(np.ndarray[np.float64_t, ndim=2] x, double p, bint training):
    """
    Dropout forward pass.
    
    Args:
        x: Input tensor
        p: Dropout probability
        training: Whether in training mode
        
    Returns:
        (output, mask) for backward pass
    """
    if not training or p == 0.0:
        return x, None
    
    cdef Py_ssize_t m = x.shape[0]
    cdef Py_ssize_t n = x.shape[1]
    
    # Generate mask
    cdef np.ndarray[np.uint8_t, ndim=2] mask = (np.random.rand(m, n) > p).astype(np.uint8)
    cdef double scale = 1.0 / (1.0 - p)
    
    cdef np.ndarray[np.float64_t, ndim=2] output = np.empty((m, n), dtype=np.float64)
    cdef double[:, ::1] x_view = x
    cdef double[:, ::1] out_view = output
    cdef unsigned char[:, ::1] mask_view = mask
    cdef Py_ssize_t i, j
    
    for i in range(m):
        for j in range(n):
            if mask_view[i, j] == 1:
                out_view[i, j] = x_view[i, j] * scale
            else:
                out_view[i, j] = 0.0
    
    return output, mask


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] dropout_backward(np.ndarray[np.float64_t, ndim=2] grad_output,
                                                         np.ndarray[np.uint8_t, ndim=2] mask,
                                                         double p):
    """
    Dropout backward pass.
    
    Args:
        grad_output: Gradient from next layer
        mask: Mask from forward pass
        p: Dropout probability
        
    Returns:
        Gradient w.r.t. input
    """
    if mask is None:
        return grad_output
    
    cdef Py_ssize_t m = grad_output.shape[0]
    cdef Py_ssize_t n = grad_output.shape[1]
    cdef double scale = 1.0 / (1.0 - p)
    
    cdef np.ndarray[np.float64_t, ndim=2] grad_input = np.empty((m, n), dtype=np.float64)
    cdef double[:, ::1] grad_out_view = grad_output
    cdef double[:, ::1] grad_in_view = grad_input
    cdef unsigned char[:, ::1] mask_view = mask
    cdef Py_ssize_t i, j
    
    for i in range(m):
        for j in range(n):
            if mask_view[i, j] == 1:
                grad_in_view[i, j] = grad_out_view[i, j] * scale
            else:
                grad_in_view[i, j] = 0.0
    
    return grad_input


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple batchnorm2d_forward(np.ndarray[np.float64_t, ndim=4] x,
                                 np.ndarray[np.float64_t, ndim=1] gamma,
                                 np.ndarray[np.float64_t, ndim=1] beta,
                                 np.ndarray[np.float64_t, ndim=1] running_mean,
                                 np.ndarray[np.float64_t, ndim=1] running_var,
                                 double momentum, double eps, bint training):
    """
    Batch normalization 2D forward pass.
    
    Args:
        x: Input of shape (N, C, H, W)
        gamma: Scale parameter of shape (C,)
        beta: Shift parameter of shape (C,)
        running_mean: Running mean of shape (C,)
        running_var: Running variance of shape (C,)
        momentum: Momentum for running stats
        eps: Numerical stability constant
        training: Whether in training mode
        
    Returns:
        (output, cache) where cache is for backward pass
    """
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t C = x.shape[1]
    cdef Py_ssize_t H = x.shape[2]
    cdef Py_ssize_t W = x.shape[3]
    
    cdef np.ndarray[np.float64_t, ndim=1] mean, var
    cdef np.ndarray[np.float64_t, ndim=4] x_norm, output
    
    if training:
        # Compute mean and variance per channel
        mean = np.mean(x, axis=(0, 2, 3))
        var = np.var(x, axis=(0, 2, 3))
        
        # Update running stats
        running_mean[:] = (1 - momentum) * running_mean + momentum * mean
        running_var[:] = (1 - momentum) * running_var + momentum * var
    else:
        mean = running_mean
        var = running_var
    
    # Normalize
    x_norm = np.empty_like(x)
    cdef double[:, :, :, ::1] x_view = x
    cdef double[:, :, :, ::1] xn_view = x_norm
    cdef double[::1] mean_view = mean
    cdef double[::1] var_view = var
    cdef double[::1] gamma_view = gamma
    cdef double[::1] beta_view = beta
    
    cdef Py_ssize_t n, c, h, w
    cdef double std
    
    for c in range(C):
        std = sqrt(var_view[c] + eps)
        for n in range(N):
            for h in range(H):
                for w in range(W):
                    xn_view[n, c, h, w] = (x_view[n, c, h, w] - mean_view[c]) / std
    
    # Scale and shift
    output = np.empty_like(x)
    cdef double[:, :, :, ::1] out_view = output
    
    for c in range(C):
        for n in range(N):
            for h in range(H):
                for w in range(W):
                    out_view[n, c, h, w] = gamma_view[c] * xn_view[n, c, h, w] + beta_view[c]
    
    # Cache for backward
    cache = (x_norm, gamma, mean, var, eps)
    
    return output, cache


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=4] global_avgpool2d_forward(np.ndarray[np.float64_t, ndim=4] x):
    """
    Global average pooling forward pass.
    
    Reduces spatial dimensions to 1x1 by averaging.
    
    Args:
        x: Input of shape (N, C, H, W)
        
    Returns:
        Output of shape (N, C, 1, 1)
    """
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t C = x.shape[1]
    cdef Py_ssize_t H = x.shape[2]
    cdef Py_ssize_t W = x.shape[3]
    
    cdef np.ndarray[np.float64_t, ndim=4] output = np.empty((N, C, 1, 1), dtype=np.float64)
    cdef double[:, :, :, ::1] x_view = x
    cdef double[:, :, :, ::1] out_view = output
    
    cdef Py_ssize_t n, c, h, w
    cdef double sum_val
    cdef double spatial_size = H * W
    
    for n in range(N):
        for c in range(C):
            sum_val = 0.0
            for h in range(H):
                for w in range(W):
                    sum_val += x_view[n, c, h, w]
            out_view[n, c, 0, 0] = sum_val / spatial_size
    
    return output
