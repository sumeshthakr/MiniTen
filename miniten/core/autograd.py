"""
Automatic Differentiation Engine

This module implements reverse-mode automatic differentiation (backpropagation)
for computing gradients efficiently.

Features:
- Computational graph construction
- Reverse-mode differentiation
- Gradient accumulation
- Memory-efficient gradient computation
- Support for higher-order derivatives

The engine tracks operations on tensors and builds a directed acyclic graph (DAG)
that can be traversed backward to compute gradients.

Future Implementation:
- Dynamic computation graph
- Gradient checkpointing for memory efficiency
- JIT compilation of gradient functions
- Forward-mode AD for certain use cases
"""

class AutogradEngine:
    """
    Core engine for automatic differentiation.
    
    This class manages the computational graph and gradient computation.
    It uses a tape-based approach where operations are recorded and
    played back in reverse for gradient computation.
    """
    
    def __init__(self):
        """Initialize the autograd engine."""
        raise NotImplementedError("AutogradEngine will be implemented")
    
    def backward(self, tensor, grad_output=None):
        """
        Compute gradients for all tensors in the computational graph.
        
        Args:
            tensor: The tensor to compute gradients for
            grad_output: Gradient of the loss with respect to this tensor
        """
        raise NotImplementedError("To be implemented")


class Function:
    """
    Base class for all differentiable functions.
    
    Custom operations should inherit from this class and implement
    the forward and backward methods.
    """
    
    @staticmethod
    def forward(ctx, *args, **kwargs):
        """
        Perform the forward pass computation.
        
        Args:
            ctx: Context object to save information for backward
            *args: Input tensors
            **kwargs: Additional arguments
            
        Returns:
            Output tensor(s)
        """
        raise NotImplementedError("Must implement forward")
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Compute gradients with respect to inputs.
        
        Args:
            ctx: Context object with saved tensors from forward
            *grad_outputs: Gradients of outputs
            
        Returns:
            Gradients with respect to inputs
        """
        raise NotImplementedError("Must implement backward")
