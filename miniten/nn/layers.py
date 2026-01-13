"""
Neural Network Layers

Common layers used in neural networks:
- Linear (Fully Connected)
- Convolutional layers
- Pooling layers
- Normalization layers
- Dropout
- Embedding
"""

from .module import Module


class Linear(Module):
    """
    Fully connected linear layer: y = xW^T + b
    
    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to include bias term (default: True)
    
    Shape:
        - Input: (*, in_features)
        - Output: (*, out_features)
    
    Attributes:
        weight: Learnable weights of shape (out_features, in_features)
        bias: Learnable bias of shape (out_features)
    
    Example:
        >>> layer = Linear(20, 10)
        >>> input = Tensor(np.random.randn(128, 20))
        >>> output = layer(input)
        >>> print(output.shape)  # (128, 10)
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        # Weight initialization will be implemented
        raise NotImplementedError("Linear layer to be implemented")
    
    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("To be implemented")


class Conv2d(Module):
    """
    2D Convolutional layer.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution (default: 1)
        padding: Padding added to input (default: 0)
        bias: Whether to include bias (default: True)
    
    Example:
        >>> conv = Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        >>> input = Tensor(np.random.randn(1, 3, 224, 224))
        >>> output = conv(input)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        raise NotImplementedError("Conv2d to be implemented")
    
    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("To be implemented")


class MaxPool2d(Module):
    """
    2D Max pooling layer.
    
    Args:
        kernel_size: Size of pooling window
        stride: Stride of pooling (default: kernel_size)
        padding: Padding to add (default: 0)
    """
    
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        raise NotImplementedError("MaxPool2d to be implemented")
    
    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("To be implemented")


class Dropout(Module):
    """
    Dropout layer for regularization.
    
    Args:
        p: Probability of dropping a unit (default: 0.5)
    """
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        raise NotImplementedError("Dropout to be implemented")
    
    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("To be implemented")


class BatchNorm2d(Module):
    """
    Batch normalization for 2D inputs.
    
    Args:
        num_features: Number of features/channels
        eps: Small value for numerical stability (default: 1e-5)
        momentum: Momentum for running statistics (default: 0.1)
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        raise NotImplementedError("BatchNorm2d to be implemented")
    
    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError("To be implemented")
