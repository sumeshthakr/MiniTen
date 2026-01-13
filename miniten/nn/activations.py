"""
Activation Functions

Common non-linear activation functions used in neural networks.
All implemented in optimized Cython for performance.
"""

from .module import Module


class ReLU(Module):
    """
    Rectified Linear Unit: f(x) = max(0, x)
    
    Example:
        >>> relu = ReLU()
        >>> x = Tensor([-1, 0, 1, 2])
        >>> y = relu(x)  # [0, 0, 1, 2]
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("ReLU to be implemented in Cython")


class Sigmoid(Module):
    """
    Sigmoid activation: f(x) = 1 / (1 + exp(-x))
    
    Example:
        >>> sigmoid = Sigmoid()
        >>> x = Tensor([0])
        >>> y = sigmoid(x)  # [0.5]
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("Sigmoid to be implemented")


class Tanh(Module):
    """
    Hyperbolic tangent: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    Example:
        >>> tanh = Tanh()
        >>> x = Tensor([0])
        >>> y = tanh(x)  # [0]
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("Tanh to be implemented")


class Softmax(Module):
    """
    Softmax activation: softmax(x_i) = exp(x_i) / sum(exp(x_j))
    
    Args:
        dim: Dimension to apply softmax (default: -1)
    
    Example:
        >>> softmax = Softmax(dim=-1)
        >>> x = Tensor([[1, 2, 3]])
        >>> y = softmax(x)  # Probabilities sum to 1
    """
    
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("Softmax to be implemented")


class LeakyReLU(Module):
    """
    Leaky ReLU: f(x) = max(negative_slope * x, x)
    
    Args:
        negative_slope: Slope for negative values (default: 0.01)
    """
    
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("LeakyReLU to be implemented")


class GELU(Module):
    """
    Gaussian Error Linear Unit.
    Used in transformers and modern architectures.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("GELU to be implemented")
