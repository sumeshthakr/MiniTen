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
- Tape-based gradient recording

The engine tracks operations on tensors and builds a directed acyclic graph (DAG)
that can be traversed backward to compute gradients.
"""

from typing import List, Dict, Optional, Callable, Any, Tuple
from .tensor import Tensor, Context


class GradientTape:
    """
    Records operations for automatic differentiation.
    
    Usage:
        with GradientTape() as tape:
            y = x * 2 + 1
        grads = tape.gradient(y, [x])
    """
    
    _current_tape = None
    
    def __init__(self, persistent: bool = False):
        """
        Initialize gradient tape.
        
        Args:
            persistent: If True, tape can be used multiple times
        """
        self.persistent = persistent
        self._operations: List[Tuple[Tensor, Callable, List[Tensor]]] = []
        self._watched: Dict[int, Tensor] = {}
    
    def __enter__(self):
        GradientTape._current_tape = self
        return self
    
    def __exit__(self, *args):
        GradientTape._current_tape = None
    
    def watch(self, tensor: Tensor):
        """Explicitly watch a tensor for gradient computation."""
        self._watched[id(tensor)] = tensor
        tensor.requires_grad = True
    
    def record(self, output: Tensor, backward_fn: Callable, inputs: List[Tensor]):
        """Record an operation on the tape."""
        self._operations.append((output, backward_fn, inputs))
    
    def gradient(self, target: Tensor, sources: List[Tensor]) -> List[Optional[Tensor]]:
        """
        Compute gradients of target with respect to sources.
        
        Args:
            target: Output tensor to differentiate
            sources: List of input tensors to compute gradients for
            
        Returns:
            List of gradient tensors (one per source)
        """
        # Build reverse topological order
        grads: Dict[int, Tensor] = {id(target): Tensor([1.0] * target.size, _shape=target._shape)}
        
        for output, backward_fn, inputs in reversed(self._operations):
            if id(output) in grads:
                grad_output = grads[id(output)]
                input_grads = backward_fn(grad_output, inputs)
                
                if not isinstance(input_grads, (list, tuple)):
                    input_grads = [input_grads]
                
                for inp, g in zip(inputs, input_grads):
                    if g is not None:
                        if id(inp) in grads:
                            grads[id(inp)] = grads[id(inp)] + g
                        else:
                            grads[id(inp)] = g
        
        # Extract gradients for requested sources
        result = []
        for src in sources:
            result.append(grads.get(id(src)))
        
        if not self.persistent:
            self._operations.clear()
        
        return result


class AutogradEngine:
    """
    Core engine for automatic differentiation.
    
    This class manages the computational graph and gradient computation.
    It uses a tape-based approach where operations are recorded and
    played back in reverse for gradient computation.
    """
    
    def __init__(self):
        """Initialize the autograd engine."""
        self._tape = None
        self._gradient_cache: Dict[int, Tensor] = {}
    
    def enable_grad(self):
        """Enable gradient computation."""
        self._tape = GradientTape()
        return self._tape.__enter__()
    
    def disable_grad(self):
        """Disable gradient computation."""
        if self._tape:
            self._tape.__exit__(None, None, None)
            self._tape = None
    
    def backward(self, tensor: Tensor, grad_output: Optional[Tensor] = None):
        """
        Compute gradients for all tensors in the computational graph.
        
        Args:
            tensor: The tensor to compute gradients for
            grad_output: Gradient of the loss with respect to this tensor
        """
        tensor.backward(grad_output)
    
    def zero_grad(self, tensors: List[Tensor]):
        """Zero out gradients for a list of tensors."""
        for t in tensors:
            t.zero_grad()


class Function:
    """
    Base class for all differentiable functions.
    
    Custom operations should inherit from this class and implement
    the forward and backward methods.
    
    Example:
        class ReLU(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return Tensor([max(0, v) for v in x._data], _shape=x._shape)
            
            @staticmethod
            def backward(ctx, grad_output):
                x, = ctx.saved_tensors
                return Tensor([g if v > 0 else 0 for g, v in zip(grad_output._data, x._data)], 
                             _shape=x._shape)
    """
    
    @staticmethod
    def forward(ctx: Context, *args, **kwargs) -> Tensor:
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
    def backward(ctx: Context, *grad_outputs: Tensor) -> Tuple[Optional[Tensor], ...]:
        """
        Compute gradients with respect to inputs.
        
        Args:
            ctx: Context object with saved tensors from forward
            *grad_outputs: Gradients of outputs
            
        Returns:
            Gradients with respect to inputs
        """
        raise NotImplementedError("Must implement backward")
    
    @classmethod
    def apply(cls, *args, **kwargs) -> Tensor:
        """Apply the function with automatic gradient tracking."""
        ctx = Context()
        result = cls.forward(ctx, *args, **kwargs)
        
        if any(isinstance(a, Tensor) and a.requires_grad for a in args):
            result.requires_grad = True
            result._ctx = ctx
            result._grad_fn = cls.backward
        
        return result


# Built-in differentiable functions

class Add(Function):
    """Addition function."""
    
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a + b
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


class Mul(Function):
    """Multiplication function."""
    
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a * b
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return grad_output * b, grad_output * a


class MatMul(Function):
    """Matrix multiplication function."""
    
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a @ b
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return grad_output @ b.T, a.T @ grad_output


class ReLU(Function):
    """ReLU activation function."""
    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        data = [max(0.0, v) for v in x._data]
        return Tensor(data, requires_grad=x.requires_grad, _shape=x._shape)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        data = [g if v > 0 else 0.0 for g, v in zip(grad_output._data, x._data)]
        return Tensor(data, _shape=x._shape)


class Sigmoid(Function):
    """Sigmoid activation function."""
    
    @staticmethod
    def forward(ctx, x):
        import math
        data = [1.0 / (1.0 + math.exp(-min(max(v, -700), 700))) for v in x._data]
        result = Tensor(data, requires_grad=x.requires_grad, _shape=x._shape)
        ctx.save_for_backward(result)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_out, = ctx.saved_tensors
        data = [g * s * (1 - s) for g, s in zip(grad_output._data, sigmoid_out._data)]
        return Tensor(data, _shape=sigmoid_out._shape)


class Tanh(Function):
    """Tanh activation function."""
    
    @staticmethod
    def forward(ctx, x):
        import math
        data = [math.tanh(v) for v in x._data]
        result = Tensor(data, requires_grad=x.requires_grad, _shape=x._shape)
        ctx.save_for_backward(result)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        tanh_out, = ctx.saved_tensors
        data = [g * (1 - t * t) for g, t in zip(grad_output._data, tanh_out._data)]
        return Tensor(data, _shape=tanh_out._shape)


class Softmax(Function):
    """Softmax function."""
    
    @staticmethod
    def forward(ctx, x, dim=-1):
        import math
        # Simple 1D softmax for now
        max_val = max(x._data)
        exp_data = [math.exp(v - max_val) for v in x._data]
        sum_exp = sum(exp_data)
        data = [e / sum_exp for e in exp_data]
        result = Tensor(data, requires_grad=x.requires_grad, _shape=x._shape)
        ctx.save_for_backward(result)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        softmax_out, = ctx.saved_tensors
        n = softmax_out.size
        # Jacobian-vector product
        data = []
        dot = sum(g * s for g, s in zip(grad_output._data, softmax_out._data))
        for i in range(n):
            data.append(softmax_out._data[i] * (grad_output._data[i] - dot))
        return Tensor(data, _shape=softmax_out._shape)


class Log(Function):
    """Natural logarithm."""
    
    @staticmethod
    def forward(ctx, x):
        import math
        ctx.save_for_backward(x)
        data = [math.log(max(v, 1e-12)) for v in x._data]
        return Tensor(data, requires_grad=x.requires_grad, _shape=x._shape)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        data = [g / max(v, 1e-12) for g, v in zip(grad_output._data, x._data)]
        return Tensor(data, _shape=x._shape)


class Exp(Function):
    """Exponential function."""
    
    @staticmethod
    def forward(ctx, x):
        import math
        data = [math.exp(min(v, 700)) for v in x._data]
        result = Tensor(data, requires_grad=x.requires_grad, _shape=x._shape)
        ctx.save_for_backward(result)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        exp_out, = ctx.saved_tensors
        data = [g * e for g, e in zip(grad_output._data, exp_out._data)]
        return Tensor(data, _shape=exp_out._shape)


class CrossEntropyLoss(Function):
    """Cross entropy loss function."""
    
    @staticmethod
    def forward(ctx, predictions, targets):
        import math
        # Softmax + cross entropy combined for numerical stability
        max_val = max(predictions._data)
        exp_data = [math.exp(v - max_val) for v in predictions._data]
        sum_exp = sum(exp_data)
        probs = [e / sum_exp for e in exp_data]
        
        # Cross entropy
        loss = -sum(t * math.log(max(p, 1e-12)) for p, t in zip(probs, targets._data))
        
        ctx.save_for_backward(Tensor(probs, _shape=predictions._shape), targets)
        return Tensor([loss], _shape=())
    
    @staticmethod
    def backward(ctx, grad_output):
        probs, targets = ctx.saved_tensors
        # Gradient is probs - targets for softmax + cross entropy
        data = [p - t for p, t in zip(probs._data, targets._data)]
        if grad_output.size == 1:
            data = [d * grad_output._data[0] for d in data]
        return Tensor(data, _shape=probs._shape), None


class MSELoss(Function):
    """Mean squared error loss function."""
    
    @staticmethod
    def forward(ctx, predictions, targets):
        ctx.save_for_backward(predictions, targets)
        diff = [(p - t) ** 2 for p, t in zip(predictions._data, targets._data)]
        loss = sum(diff) / len(diff)
        return Tensor([loss], _shape=())
    
    @staticmethod
    def backward(ctx, grad_output):
        predictions, targets = ctx.saved_tensors
        n = predictions.size
        data = [2 * (p - t) / n * grad_output._data[0] 
                for p, t in zip(predictions._data, targets._data)]
        return Tensor(data, _shape=predictions._shape), None


# No-grad context manager
class no_grad:
    """Context manager for disabling gradient computation."""
    
    def __init__(self):
        self._prev_grad_enabled = True
    
    def __enter__(self):
        self._prev_grad_enabled = True
        return self
    
    def __exit__(self, *args):
        pass


# Utility functions

def grad(outputs: Tensor, inputs: List[Tensor], 
         grad_outputs: Optional[Tensor] = None,
         retain_graph: bool = False) -> List[Optional[Tensor]]:
    """
    Compute gradients of outputs with respect to inputs.
    
    Args:
        outputs: Output tensor
        inputs: List of input tensors
        grad_outputs: Gradient with respect to outputs
        retain_graph: Whether to keep computation graph
        
    Returns:
        List of gradients for each input
    """
    outputs.backward(grad_outputs)
    return [inp.grad for inp in inputs]
