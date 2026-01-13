# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Advanced neural network features for MiniTen.

This module provides high-performance implementations of:
- Graph Neural Networks (GCN, GAT)
- Attention mechanisms (scaled dot-product, multi-head)
- Transformer encoder layer
- Model quantization utilities
- Weight pruning utilities

All implementations use Cython for maximum performance and minimal memory footprint,
optimized for edge devices.
"""

import cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport exp, sqrt, fabs, floor

# Initialize NumPy C API
np.import_array()

# Threshold for parallel operations
DEF PARALLEL_THRESHOLD = 10000


# ============================================================================
# Graph Neural Networks
# ============================================================================

cdef class GraphConv:
    """
    Optimized Graph Convolutional Layer.
    
    Implements: H' = sigma(D^{-1/2} A D^{-1/2} H W)
    
    For edge computing, we use a simplified aggregation approach.
    """
    
    cdef public int in_features
    cdef public int out_features
    cdef public np.ndarray weight
    cdef public np.ndarray bias
    cdef public np.ndarray grad_weight
    cdef public np.ndarray grad_bias
    cdef public bint use_bias
    cdef public np.ndarray cached_input
    cdef public np.ndarray cached_adj_norm
    
    def __init__(self, int in_features, int out_features, bint use_bias=True):
        """
        Initialize GraphConv layer.
        
        Args:
            in_features: Size of input node features
            out_features: Size of output node features
            use_bias: Whether to use bias
        """
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        
        # Xavier/Glorot initialization
        cdef double std = sqrt(2.0 / (in_features + out_features))
        self.weight = np.random.randn(in_features, out_features).astype(np.float64) * std
        
        if use_bias:
            self.bias = np.zeros(out_features, dtype=np.float64)
        else:
            self.bias = None
        
        self.grad_weight = np.zeros_like(self.weight)
        if use_bias:
            self.grad_bias = np.zeros(out_features, dtype=np.float64)
        else:
            self.grad_bias = None
        
        self.cached_input = None
        self.cached_adj_norm = None
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def forward(self, np.ndarray[np.float64_t, ndim=2] x, 
                np.ndarray[np.float64_t, ndim=2] adj):
        """
        Forward pass.
        
        Args:
            x: Node features of shape (num_nodes, in_features)
            adj: Adjacency matrix of shape (num_nodes, num_nodes)
            
        Returns:
            Output features of shape (num_nodes, out_features)
        """
        # Cache for backward
        self.cached_input = x.copy()
        
        # Normalize adjacency matrix: D^{-1/2} A D^{-1/2}
        cdef np.ndarray[np.float64_t, ndim=1] degree = np.sum(adj, axis=1)
        cdef np.ndarray[np.float64_t, ndim=1] degree_inv_sqrt = np.power(degree + 1e-10, -0.5)
        cdef np.ndarray[np.float64_t, ndim=2] adj_norm = (
            adj * degree_inv_sqrt[:, np.newaxis] * degree_inv_sqrt[np.newaxis, :]
        )
        self.cached_adj_norm = adj_norm
        
        # Aggregate: A_norm @ X
        cdef np.ndarray[np.float64_t, ndim=2] aggregated = np.dot(adj_norm, x)
        
        # Transform: aggregated @ W + b
        cdef np.ndarray[np.float64_t, ndim=2] output = np.dot(aggregated, self.weight)
        
        if self.use_bias:
            output = output + self.bias
        
        return output
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def backward(self, np.ndarray[np.float64_t, ndim=2] grad_output):
        """
        Backward pass.
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient w.r.t. input features
        """
        # Gradient w.r.t. weight
        cdef np.ndarray[np.float64_t, ndim=2] aggregated = np.dot(self.cached_adj_norm, self.cached_input)
        self.grad_weight = np.dot(aggregated.T, grad_output)
        
        # Gradient w.r.t. bias
        if self.use_bias:
            self.grad_bias = np.sum(grad_output, axis=0)
        
        # Gradient w.r.t. input
        cdef np.ndarray[np.float64_t, ndim=2] grad_aggregated = np.dot(grad_output, self.weight.T)
        cdef np.ndarray[np.float64_t, ndim=2] grad_input = np.dot(self.cached_adj_norm.T, grad_aggregated)
        
        return grad_input
    
    def update_parameters(self, double learning_rate):
        """Update parameters using gradients."""
        self.weight -= learning_rate * self.grad_weight
        if self.use_bias:
            self.bias -= learning_rate * self.grad_bias


# ============================================================================
# Attention Mechanisms
# ============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple scaled_dot_product_attention(np.ndarray query,
                                          np.ndarray key,
                                          np.ndarray value,
                                          object mask=None):
    """
    Scaled dot-product attention.
    
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Args:
        query: Query tensor of shape (batch_size, seq_len, d_k)
        key: Key tensor of shape (batch_size, seq_len, d_k)
        value: Value tensor of shape (batch_size, seq_len, d_v)
        mask: Optional mask of shape (seq_len, seq_len)
        
    Returns:
        (output, attention_weights)
    """
    cdef Py_ssize_t batch_size = query.shape[0]
    cdef Py_ssize_t seq_len = query.shape[1]
    cdef Py_ssize_t d_k = query.shape[2]
    
    cdef double scale = 1.0 / sqrt(<double>d_k)
    
    # Compute attention scores: Q @ K^T / sqrt(d_k)
    # Shape: (batch_size, seq_len, seq_len)
    cdef np.ndarray[np.float64_t, ndim=3] scores = np.empty((batch_size, seq_len, seq_len), dtype=np.float64)
    cdef Py_ssize_t b, i, j, k
    cdef double temp
    
    for b in range(batch_size):
        for i in range(seq_len):
            for j in range(seq_len):
                temp = 0.0
                for k in range(d_k):
                    temp += query[b, i, k] * key[b, j, k]
                scores[b, i, j] = temp * scale
    
    # Apply mask if provided (for causal attention)
    if mask is not None:
        for b in range(batch_size):
            for i in range(seq_len):
                for j in range(seq_len):
                    if mask[i, j] == 0:
                        scores[b, i, j] = -1e9  # Large negative value
    
    # Softmax over last dimension
    cdef np.ndarray[np.float64_t, ndim=3] attn_weights = np.empty_like(scores)
    cdef double max_val, sum_exp
    
    for b in range(batch_size):
        for i in range(seq_len):
            # Find max for numerical stability
            max_val = scores[b, i, 0]
            for j in range(1, seq_len):
                if scores[b, i, j] > max_val:
                    max_val = scores[b, i, j]
            
            # Compute exp and sum
            sum_exp = 0.0
            for j in range(seq_len):
                attn_weights[b, i, j] = exp(scores[b, i, j] - max_val)
                sum_exp += attn_weights[b, i, j]
            
            # Normalize
            for j in range(seq_len):
                attn_weights[b, i, j] /= sum_exp
    
    # Compute output: attention_weights @ V
    cdef Py_ssize_t d_v = value.shape[2]
    cdef np.ndarray[np.float64_t, ndim=3] output = np.zeros((batch_size, seq_len, d_v), dtype=np.float64)
    
    for b in range(batch_size):
        for i in range(seq_len):
            for k in range(d_v):
                temp = 0.0
                for j in range(seq_len):
                    temp += attn_weights[b, i, j] * value[b, j, k]
                output[b, i, k] = temp
    
    return output, attn_weights


cdef class MultiHeadAttention:
    """
    Multi-Head Attention mechanism.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W^O
    where head_i = Attention(Q @ W^Q_i, K @ W^K_i, V @ W^V_i)
    """
    
    cdef public int embed_dim
    cdef public int num_heads
    cdef public int head_dim
    cdef public np.ndarray W_q
    cdef public np.ndarray W_k
    cdef public np.ndarray W_v
    cdef public np.ndarray W_o
    cdef public np.ndarray grad_W_q
    cdef public np.ndarray grad_W_k
    cdef public np.ndarray grad_W_v
    cdef public np.ndarray grad_W_o
    # Cache for backward
    cdef public np.ndarray cached_query
    cdef public np.ndarray cached_key
    cdef public np.ndarray cached_value
    cdef public np.ndarray cached_attn_weights
    cdef public np.ndarray cached_attn_output
    
    def __init__(self, int embed_dim, int num_heads):
        """
        Initialize Multi-Head Attention.
        
        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of attention heads
        """
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Initialize projection weights
        cdef double std = sqrt(2.0 / (embed_dim + embed_dim))
        self.W_q = np.random.randn(embed_dim, embed_dim).astype(np.float64) * std
        self.W_k = np.random.randn(embed_dim, embed_dim).astype(np.float64) * std
        self.W_v = np.random.randn(embed_dim, embed_dim).astype(np.float64) * std
        self.W_o = np.random.randn(embed_dim, embed_dim).astype(np.float64) * std
        
        # Initialize gradients
        self.grad_W_q = np.zeros_like(self.W_q)
        self.grad_W_k = np.zeros_like(self.W_k)
        self.grad_W_v = np.zeros_like(self.W_v)
        self.grad_W_o = np.zeros_like(self.W_o)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def forward(self, np.ndarray query,
                np.ndarray key,
                np.ndarray value,
                object mask=None):
        """
        Forward pass.
        
        Args:
            query: Query of shape (batch_size, seq_len, embed_dim)
            key: Key of shape (batch_size, seq_len, embed_dim)
            value: Value of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Output of shape (batch_size, seq_len, embed_dim)
        """
        cdef Py_ssize_t batch_size = query.shape[0]
        cdef Py_ssize_t seq_len = query.shape[1]
        
        # Cache inputs
        self.cached_query = np.asarray(query).copy()
        self.cached_key = np.asarray(key).copy()
        self.cached_value = np.asarray(value).copy()
        
        # Project Q, K, V: (batch, seq, embed) @ (embed, embed) -> (batch, seq, embed)
        # Use tensordot or einsum for 3D x 2D matmul
        cdef np.ndarray Q = np.tensordot(query, self.W_q, axes=([2], [0]))
        cdef np.ndarray K = np.tensordot(key, self.W_k, axes=([2], [0]))
        cdef np.ndarray V = np.tensordot(value, self.W_v, axes=([2], [0]))
        
        # Reshape for multi-head: (batch, seq, heads, head_dim) -> (batch * heads, seq, head_dim)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        Q = Q.reshape(batch_size * self.num_heads, seq_len, self.head_dim).copy()
        K = K.reshape(batch_size * self.num_heads, seq_len, self.head_dim).copy()
        V = V.reshape(batch_size * self.num_heads, seq_len, self.head_dim).copy()
        
        # Apply scaled dot-product attention
        cdef np.ndarray attn_output
        cdef np.ndarray attn_weights
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        self.cached_attn_weights = attn_weights
        self.cached_attn_output = np.asarray(attn_output).copy()
        
        # Reshape back: (batch * heads, seq, head_dim) -> (batch, seq, embed)
        attn_output = attn_output.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim).copy()
        
        # Final projection
        cdef np.ndarray output = np.tensordot(attn_output, self.W_o, axes=([2], [0]))
        
        return output
    
    def update_parameters(self, double learning_rate):
        """Update parameters using gradients."""
        self.W_q -= learning_rate * self.grad_W_q
        self.W_k -= learning_rate * self.grad_W_k
        self.W_v -= learning_rate * self.grad_W_v
        self.W_o -= learning_rate * self.grad_W_o
    
    def zero_grad(self):
        """Reset gradients to zero."""
        self.grad_W_q[:] = 0.0
        self.grad_W_k[:] = 0.0
        self.grad_W_v[:] = 0.0
        self.grad_W_o[:] = 0.0


# ============================================================================
# Transformer Components
# ============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=3] layer_norm(np.ndarray[np.float64_t, ndim=3] x,
                                                    np.ndarray[np.float64_t, ndim=1] gamma,
                                                    np.ndarray[np.float64_t, ndim=1] beta,
                                                    double eps=1e-5):
    """
    Layer normalization.
    
    Args:
        x: Input of shape (batch_size, seq_len, embed_dim)
        gamma: Scale parameter of shape (embed_dim,)
        beta: Shift parameter of shape (embed_dim,)
        eps: Numerical stability constant
        
    Returns:
        Normalized output
    """
    cdef Py_ssize_t batch_size = x.shape[0]
    cdef Py_ssize_t seq_len = x.shape[1]
    cdef Py_ssize_t embed_dim = x.shape[2]
    
    cdef np.ndarray[np.float64_t, ndim=3] output = np.empty_like(x)
    cdef double[:, :, ::1] x_view = x
    cdef double[:, :, ::1] out_view = output
    cdef double[::1] gamma_view = gamma
    cdef double[::1] beta_view = beta
    
    cdef Py_ssize_t b, s, d
    cdef double mean, var, std, val
    
    for b in range(batch_size):
        for s in range(seq_len):
            # Compute mean
            mean = 0.0
            for d in range(embed_dim):
                mean += x_view[b, s, d]
            mean /= embed_dim
            
            # Compute variance
            var = 0.0
            for d in range(embed_dim):
                val = x_view[b, s, d] - mean
                var += val * val
            var /= embed_dim
            
            # Normalize
            std = sqrt(var + eps)
            for d in range(embed_dim):
                out_view[b, s, d] = (x_view[b, s, d] - mean) / std * gamma_view[d] + beta_view[d]
    
    return output


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=3] gelu(np.ndarray[np.float64_t, ndim=3] x):
    """
    GELU activation function (approximation).
    
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    cdef double sqrt_2_over_pi = 0.7978845608028654
    cdef double coef = 0.044715
    
    return 0.5 * x * (1.0 + np.tanh(sqrt_2_over_pi * (x + coef * x * x * x)))


cdef class FeedForward:
    """
    Position-wise Feed-Forward Network for Transformer.
    
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2  (ReLU version)
    or
    FFN(x) = GELU(xW_1 + b_1)W_2 + b_2  (GELU version)
    """
    
    cdef public int embed_dim
    cdef public int ff_dim
    cdef public np.ndarray W_1
    cdef public np.ndarray b_1
    cdef public np.ndarray W_2
    cdef public np.ndarray b_2
    cdef public np.ndarray grad_W_1
    cdef public np.ndarray grad_b_1
    cdef public np.ndarray grad_W_2
    cdef public np.ndarray grad_b_2
    cdef public np.ndarray cached_input
    cdef public np.ndarray cached_hidden
    
    def __init__(self, int embed_dim, int ff_dim):
        """
        Initialize Feed-Forward Network.
        
        Args:
            embed_dim: Input/output dimension
            ff_dim: Hidden dimension (typically 4 * embed_dim)
        """
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        
        # Initialize weights
        cdef double std1 = sqrt(2.0 / (embed_dim + ff_dim))
        cdef double std2 = sqrt(2.0 / (ff_dim + embed_dim))
        
        self.W_1 = np.random.randn(embed_dim, ff_dim).astype(np.float64) * std1
        self.b_1 = np.zeros(ff_dim, dtype=np.float64)
        self.W_2 = np.random.randn(ff_dim, embed_dim).astype(np.float64) * std2
        self.b_2 = np.zeros(embed_dim, dtype=np.float64)
        
        # Initialize gradients
        self.grad_W_1 = np.zeros_like(self.W_1)
        self.grad_b_1 = np.zeros_like(self.b_1)
        self.grad_W_2 = np.zeros_like(self.W_2)
        self.grad_b_2 = np.zeros_like(self.b_2)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def forward(self, np.ndarray x):
        """
        Forward pass with GELU activation.
        
        Args:
            x: Input of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Output of shape (batch_size, seq_len, embed_dim)
        """
        self.cached_input = np.asarray(x).copy()
        
        # First linear + GELU: use tensordot for 3D x 2D
        cdef np.ndarray hidden = np.tensordot(x, self.W_1, axes=([2], [0])) + self.b_1
        hidden = gelu(hidden)
        self.cached_hidden = np.asarray(hidden).copy()
        
        # Second linear
        cdef np.ndarray output = np.tensordot(hidden, self.W_2, axes=([2], [0])) + self.b_2
        
        return output
    
    def update_parameters(self, double learning_rate):
        """Update parameters using gradients."""
        self.W_1 -= learning_rate * self.grad_W_1
        self.b_1 -= learning_rate * self.grad_b_1
        self.W_2 -= learning_rate * self.grad_W_2
        self.b_2 -= learning_rate * self.grad_b_2
    
    def zero_grad(self):
        """Reset gradients to zero."""
        self.grad_W_1[:] = 0.0
        self.grad_b_1[:] = 0.0
        self.grad_W_2[:] = 0.0
        self.grad_b_2[:] = 0.0


cdef class TransformerEncoderLayer:
    """
    Transformer Encoder Layer.
    
    Consists of:
    1. Multi-Head Self-Attention + Add & Norm
    2. Feed-Forward Network + Add & Norm
    """
    
    cdef public int embed_dim
    cdef public int num_heads
    cdef public int ff_dim
    cdef public object attention
    cdef public object ffn
    cdef public np.ndarray ln1_gamma
    cdef public np.ndarray ln1_beta
    cdef public np.ndarray ln2_gamma
    cdef public np.ndarray ln2_beta
    
    def __init__(self, int embed_dim, int num_heads, int ff_dim=0):
        """
        Initialize Transformer Encoder Layer.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension (default: 4 * embed_dim)
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim if ff_dim > 0 else 4 * embed_dim
        
        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        
        # Feed-forward network
        self.ffn = FeedForward(embed_dim, self.ff_dim)
        
        # Layer norm parameters
        self.ln1_gamma = np.ones(embed_dim, dtype=np.float64)
        self.ln1_beta = np.zeros(embed_dim, dtype=np.float64)
        self.ln2_gamma = np.ones(embed_dim, dtype=np.float64)
        self.ln2_beta = np.zeros(embed_dim, dtype=np.float64)
    
    def forward(self, np.ndarray x, object mask=None):
        """
        Forward pass.
        
        Args:
            x: Input of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Output of shape (batch_size, seq_len, embed_dim)
        """
        # Self-attention + residual + layer norm
        cdef np.ndarray[np.float64_t, ndim=3] attn_output = self.attention.forward(x, x, x, mask)
        x = layer_norm(x + attn_output, self.ln1_gamma, self.ln1_beta)
        
        # Feed-forward + residual + layer norm
        cdef np.ndarray[np.float64_t, ndim=3] ffn_output = self.ffn.forward(x)
        x = layer_norm(x + ffn_output, self.ln2_gamma, self.ln2_beta)
        
        return x
    
    def update_parameters(self, double learning_rate):
        """Update all parameters."""
        self.attention.update_parameters(learning_rate)
        self.ffn.update_parameters(learning_rate)


# ============================================================================
# Model Quantization
# ============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple quantize_tensor_int8(np.ndarray[np.float64_t, ndim=2] tensor):
    """
    Quantize a tensor to int8.
    
    Uses symmetric quantization: q = round(x / scale)
    
    Args:
        tensor: Float tensor to quantize
        
    Returns:
        (quantized_tensor, scale) where scale is for dequantization
    """
    cdef double max_val = np.max(np.abs(tensor))
    cdef double scale = max_val / 127.0 if max_val > 0 else 1.0
    
    cdef np.ndarray[np.int8_t, ndim=2] quantized = np.clip(
        np.round(tensor / scale), -127, 127
    ).astype(np.int8)
    
    return quantized, scale


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] dequantize_tensor_int8(np.ndarray[np.int8_t, ndim=2] tensor,
                                                                double scale):
    """
    Dequantize an int8 tensor back to float64.
    
    Args:
        tensor: Quantized tensor
        scale: Scale factor from quantization
        
    Returns:
        Dequantized float tensor
    """
    return tensor.astype(np.float64) * scale


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.int8_t, ndim=2] quantized_matmul_int8(np.ndarray[np.int8_t, ndim=2] a,
                                                           np.ndarray[np.int8_t, ndim=2] b,
                                                           double scale_a,
                                                           double scale_b):
    """
    Perform matrix multiplication on quantized tensors.
    
    This is a simulated int8 matmul - in practice, would use hardware int8 ops.
    
    Args:
        a: First quantized matrix
        b: Second quantized matrix
        scale_a: Scale of first matrix
        scale_b: Scale of second matrix
        
    Returns:
        Quantized result
    """
    # For edge devices, this would use optimized int8 GEMM
    # Here we simulate the quantized computation
    cdef np.ndarray[np.int32_t, ndim=2] result_int32 = np.dot(
        a.astype(np.int32), b.astype(np.int32)
    )
    
    # Rescale and requantize
    cdef double combined_scale = scale_a * scale_b
    cdef double output_scale = np.max(np.abs(result_int32)) * combined_scale / 127.0
    
    if output_scale < 1e-10:
        output_scale = 1.0
    
    cdef np.ndarray[np.int8_t, ndim=2] result = np.clip(
        np.round(result_int32.astype(np.float64) * combined_scale / output_scale),
        -127, 127
    ).astype(np.int8)
    
    return result


# ============================================================================
# Weight Pruning
# ============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple magnitude_prune(np.ndarray[np.float64_t, ndim=2] weights, double sparsity):
    """
    Prune weights by magnitude (set smallest weights to zero).
    
    Args:
        weights: Weight matrix to prune
        sparsity: Target sparsity (fraction of weights to prune, 0-1)
        
    Returns:
        (pruned_weights, mask) where mask indicates non-zero weights
    """
    if sparsity <= 0.0:
        return weights.copy(), np.ones_like(weights, dtype=np.uint8)
    if sparsity >= 1.0:
        return np.zeros_like(weights), np.zeros_like(weights, dtype=np.uint8)
    
    cdef np.ndarray[np.float64_t, ndim=1] flat_weights = np.abs(weights).flatten()
    cdef Py_ssize_t num_weights = flat_weights.shape[0]
    cdef Py_ssize_t num_prune = <Py_ssize_t>(sparsity * num_weights)
    
    # Find threshold
    cdef np.ndarray[np.float64_t, ndim=1] sorted_weights = np.sort(flat_weights)
    cdef double threshold
    if num_prune < num_weights:
        threshold = sorted_weights[num_prune]
    else:
        threshold = sorted_weights[num_weights - 1]
    
    # Create mask
    cdef np.ndarray[np.uint8_t, ndim=2] mask = (np.abs(weights) >= threshold).astype(np.uint8)
    
    # Apply mask
    cdef np.ndarray[np.float64_t, ndim=2] pruned = weights * mask.astype(np.float64)
    
    return pruned, mask


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] apply_pruning_mask(np.ndarray[np.float64_t, ndim=2] weights,
                                                           np.ndarray[np.uint8_t, ndim=2] mask):
    """
    Apply a pruning mask to weights.
    
    Args:
        weights: Weight matrix
        mask: Binary mask
        
    Returns:
        Pruned weights
    """
    return weights * mask.astype(np.float64)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double compute_sparsity(np.ndarray[np.float64_t, ndim=2] weights):
    """
    Compute the sparsity of a weight matrix.
    
    Args:
        weights: Weight matrix
        
    Returns:
        Sparsity ratio (fraction of zeros)
    """
    cdef Py_ssize_t total = weights.size
    cdef Py_ssize_t zeros = np.sum(weights == 0.0)
    return <double>zeros / <double>total


# ============================================================================
# Positional Encoding
# ============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] sinusoidal_position_encoding(int max_len, int embed_dim):
    """
    Generate sinusoidal positional encodings.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        max_len: Maximum sequence length
        embed_dim: Embedding dimension
        
    Returns:
        Positional encoding matrix of shape (max_len, embed_dim)
    """
    cdef np.ndarray[np.float64_t, ndim=2] pe = np.zeros((max_len, embed_dim), dtype=np.float64)
    cdef double[:, ::1] pe_view = pe
    
    cdef Py_ssize_t pos, i
    cdef double div_term
    cdef double log_10000 = 9.210340371976184  # log(10000)
    
    for pos in range(max_len):
        for i in range(0, embed_dim, 2):
            div_term = exp(-log_10000 * i / embed_dim)
            pe_view[pos, i] = np.sin(pos * div_term)
            if i + 1 < embed_dim:
                pe_view[pos, i + 1] = np.cos(pos * div_term)
    
    return pe
