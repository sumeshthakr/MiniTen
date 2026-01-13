"""
Base Module Classes

Defines the Module class which is the base for all neural network components.
All layers and models should inherit from nn.Module.
"""


class Module:
    """
    Base class for all neural network modules.
    
    Your models should subclass this class. Modules can contain other modules,
    allowing for nested architectures.
    
    Features:
    - Automatic parameter management
    - Easy model serialization
    - GPU/CPU device management
    - Training/evaluation mode switching
    
    Example:
        >>> class MyModel(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linear = Linear(10, 5)
        ...     
        ...     def forward(self, x):
        ...         return self.linear(x)
    """
    
    def __init__(self):
        """Initialize module."""
        self._parameters = {}
        self._modules = {}
        self.training = True
    
    def forward(self, *args, **kwargs):
        """
        Define the forward pass computation.
        
        Should be overridden by all subclasses.
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def __call__(self, *args, **kwargs):
        """Make module callable, forwarding to forward()."""
        return self.forward(*args, **kwargs)
    
    def parameters(self):
        """
        Return an iterator over module parameters.
        
        Yields:
            Parameter tensors of the module
        """
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            for param in module.parameters():
                yield param
    
    def train(self, mode=True):
        """Set module in training mode."""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self):
        """Set module in evaluation mode."""
        return self.train(False)
    
    def to(self, device):
        """Move all parameters to specified device."""
        raise NotImplementedError("To be implemented")
    
    def save(self, path):
        """Save model parameters to file."""
        raise NotImplementedError("To be implemented")
    
    def load(self, path):
        """Load model parameters from file."""
        raise NotImplementedError("To be implemented")
