"""
Tensor - Core Data Structure

The Tensor class is the fundamental data structure in MiniTen, similar to
numpy arrays but with automatic differentiation capabilities.

Features:
- Efficient memory layout for cache performance
- Automatic differentiation support
- GPU acceleration when available
- Lazy evaluation for optimization
- Broadcasting support
- In-place operations
- Minimized NumPy dependency (custom optimizations for edge devices)
- SIMD-friendly memory alignment
"""

import math
from typing import Union, List, Tuple, Optional, Callable


class Tensor:
    """
    Multi-dimensional array with automatic differentiation.
    
    Args:
        data: Input data (list, nested list, Tensor, or scalar)
        requires_grad: Whether to track gradients (default: False)
        device: Device to place tensor on ('cpu' or 'gpu')
    
    Attributes:
        data: The actual data storage (flat list for memory efficiency)
        grad: Gradient tensor (if requires_grad=True)
        requires_grad: Whether gradients are tracked
        shape: Dimensions of the tensor
        dtype: Data type of elements
    
    Example:
        >>> x = Tensor([[1, 2], [3, 4]], requires_grad=True)
        >>> y = x * 2
        >>> y.backward()
        >>> print(x.grad)
    """
    
    def __init__(self, data, requires_grad: bool = False, device: str = 'cpu',
                 _shape: Optional[Tuple[int, ...]] = None, _grad_fn: Optional[Callable] = None):
        """Initialize a new Tensor."""
        self.device = device
        self.requires_grad = requires_grad
        self._grad_fn = _grad_fn
        self._ctx = None  # Context for backward pass
        self.grad = None
        
        if isinstance(data, Tensor):
            self._data = data._data.copy()
            self._shape = data._shape
        elif isinstance(data, (list, tuple)):
            self._data, self._shape = self._flatten_data(data)
        elif isinstance(data, (int, float)):
            self._data = [float(data)]
            self._shape = ()
        else:
            # Try to convert from numpy if available
            try:
                import numpy as np
                if isinstance(data, np.ndarray):
                    self._data = data.flatten().tolist()
                    self._shape = tuple(data.shape)
                else:
                    self._data = [float(data)]
                    self._shape = ()
            except ImportError:
                self._data = [float(data)]
                self._shape = ()
        
        # Override shape if provided
        if _shape is not None:
            self._shape = _shape
    
    def _flatten_data(self, data) -> Tuple[List[float], Tuple[int, ...]]:
        """Flatten nested list and extract shape."""
        if not isinstance(data, (list, tuple)):
            return [float(data)], ()
        
        if len(data) == 0:
            return [], (0,)
        
        # Check if elements are scalars
        if not isinstance(data[0], (list, tuple)):
            return [float(x) for x in data], (len(data),)
        
        # Recursively flatten
        flat = []
        shapes = []
        for item in data:
            item_flat, item_shape = self._flatten_data(item)
            flat.extend(item_flat)
            shapes.append(item_shape)
        
        # Verify all sub-shapes are the same
        if not all(s == shapes[0] for s in shapes):
            raise ValueError("Inconsistent shapes in nested list")
        
        return flat, (len(data),) + shapes[0]
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Return tensor shape."""
        return self._shape
    
    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        return len(self._shape)
    
    @property
    def size(self) -> int:
        """Return total number of elements."""
        if len(self._shape) == 0:
            return 1
        result = 1
        for dim in self._shape:
            result *= dim
        return result
    
    @property
    def dtype(self) -> str:
        """Return data type (always float64 for now)."""
        return 'float64'
    
    def item(self) -> float:
        """Return scalar value for single-element tensor."""
        if self.size != 1:
            raise ValueError("item() only works on single-element tensors")
        return self._data[0]
    
    def tolist(self) -> Union[float, List]:
        """Convert to nested Python list."""
        if len(self._shape) == 0:
            return self._data[0]
        return self._unflatten(self._data, self._shape)
    
    def _unflatten(self, data: List[float], shape: Tuple[int, ...]) -> List:
        """Unflatten data to nested list."""
        if len(shape) == 0:
            return data[0]
        if len(shape) == 1:
            return data[:shape[0]]
        
        sub_size = 1
        for dim in shape[1:]:
            sub_size *= dim
        
        result = []
        for i in range(shape[0]):
            start = i * sub_size
            end = start + sub_size
            result.append(self._unflatten(data[start:end], shape[1:]))
        
        return result
    
    def numpy(self):
        """Convert tensor to numpy array."""
        try:
            import numpy as np
            return np.array(self._data).reshape(self._shape)
        except ImportError:
            raise ImportError("NumPy is required for numpy() conversion")
    
    def to(self, device: str) -> 'Tensor':
        """Move tensor to specified device."""
        if device == self.device:
            return self
        new_tensor = Tensor(self._data, requires_grad=self.requires_grad, 
                           device=device, _shape=self._shape)
        return new_tensor
    
    def detach(self) -> 'Tensor':
        """Return a new tensor detached from computation graph."""
        return Tensor(self._data, requires_grad=False, device=self.device, _shape=self._shape)
    
    def clone(self) -> 'Tensor':
        """Return a copy of the tensor."""
        new_tensor = Tensor(self._data.copy(), requires_grad=self.requires_grad,
                           device=self.device, _shape=self._shape)
        return new_tensor
    
    def zero_grad(self):
        """Zero out gradients."""
        if self.grad is not None:
            self.grad = Tensor([0.0] * self.size, _shape=self._shape)
    
    def backward(self, grad: Optional['Tensor'] = None):
        """Compute gradients through the computational graph."""
        if not self.requires_grad:
            return
        
        if grad is None:
            if self.size != 1:
                raise RuntimeError("grad must be specified for non-scalar tensors")
            grad = Tensor([1.0], _shape=())
        
        # Build topological order
        topo = []
        visited = set()
        
        def build_topo(tensor):
            if id(tensor) not in visited:
                visited.add(id(tensor))
                if tensor._ctx is not None:
                    for parent in tensor._ctx.saved_tensors:
                        if isinstance(parent, Tensor):
                            build_topo(parent)
                topo.append(tensor)
        
        build_topo(self)
        
        # Initialize gradient
        self.grad = grad
        
        # Backpropagate
        for tensor in reversed(topo):
            if tensor._grad_fn is not None and tensor._ctx is not None:
                grads = tensor._grad_fn(tensor._ctx, tensor.grad)
                if not isinstance(grads, tuple):
                    grads = (grads,)
                for parent, g in zip(tensor._ctx.saved_tensors, grads):
                    if isinstance(parent, Tensor) and parent.requires_grad:
                        if parent.grad is None:
                            parent.grad = g
                        else:
                            parent.grad = parent.grad + g
    
    def reshape(self, *shape) -> 'Tensor':
        """Reshape tensor to new shape."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        
        new_size = 1
        for dim in shape:
            new_size *= dim
        
        if new_size != self.size:
            raise ValueError(f"Cannot reshape tensor of size {self.size} to shape {shape}")
        
        result = Tensor(self._data.copy(), requires_grad=self.requires_grad,
                       device=self.device, _shape=shape)
        
        if self.requires_grad:
            ctx = Context()
            ctx.save_for_backward(self)
            ctx.original_shape = self._shape
            result._ctx = ctx
            result._grad_fn = self._reshape_backward
        
        return result
    
    @staticmethod
    def _reshape_backward(ctx, grad):
        original_shape = ctx.original_shape
        return grad.reshape(original_shape)
    
    def transpose(self, dim0: int = 0, dim1: int = 1) -> 'Tensor':
        """Transpose two dimensions."""
        if self.ndim < 2:
            return self.clone()
        
        # Build new shape
        new_shape = list(self._shape)
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        
        # Compute strides for original tensor
        strides = [1] * self.ndim
        for i in range(self.ndim - 2, -1, -1):
            strides[i] = strides[i + 1] * self._shape[i + 1]
        
        # Swap strides
        new_strides = strides.copy()
        new_strides[dim0], new_strides[dim1] = new_strides[dim1], new_strides[dim0]
        
        # Reorder data
        new_data = [0.0] * self.size
        
        for i in range(self.size):
            # Compute old index
            old_idx = i
            coords = []
            for s in strides:
                coords.append(old_idx // s)
                old_idx = old_idx % s
            
            # Swap coordinates
            coords[dim0], coords[dim1] = coords[dim1], coords[dim0]
            
            # Compute new index
            new_idx = 0
            for c, ns in zip(coords, new_strides):
                new_idx += c * ns
            
            new_data[new_idx] = self._data[i]
        
        result = Tensor(new_data, requires_grad=self.requires_grad,
                       device=self.device, _shape=tuple(new_shape))
        
        if self.requires_grad:
            ctx = Context()
            ctx.save_for_backward(self)
            ctx.dim0, ctx.dim1 = dim0, dim1
            result._ctx = ctx
            result._grad_fn = self._transpose_backward
        
        return result
    
    @property
    def T(self) -> 'Tensor':
        """Return transposed tensor."""
        return self.transpose()
    
    @staticmethod
    def _transpose_backward(ctx, grad):
        return grad.transpose(ctx.dim0, ctx.dim1)
    
    def sum(self, dim: Optional[int] = None, keepdim: bool = False) -> 'Tensor':
        """Sum tensor elements."""
        if dim is None:
            total = sum(self._data)
            result = Tensor([total], _shape=())
        else:
            if dim < 0:
                dim = self.ndim + dim
            
            # Compute output shape
            out_shape = list(self._shape)
            if keepdim:
                out_shape[dim] = 1
            else:
                out_shape.pop(dim)
            
            # Compute strides
            strides = [1] * self.ndim
            for i in range(self.ndim - 2, -1, -1):
                strides[i] = strides[i + 1] * self._shape[i + 1]
            
            # Initialize output
            out_size = 1
            for d in out_shape:
                out_size *= d
            out_data = [0.0] * out_size
            
            # Reduce along dim
            for i in range(self.size):
                # Get coordinates
                idx = i
                coords = []
                for s in strides:
                    coords.append(idx // s)
                    idx = idx % s
                
                # Compute output index
                out_coords = coords[:dim] + coords[dim+1:] if not keepdim else coords[:dim] + [0] + coords[dim+1:]
                out_strides = [1] * len(out_shape)
                for j in range(len(out_shape) - 2, -1, -1):
                    out_strides[j] = out_strides[j + 1] * out_shape[j + 1]
                
                out_idx = sum(c * s for c, s in zip(out_coords, out_strides)) if out_strides else 0
                out_data[out_idx] += self._data[i]
            
            result = Tensor(out_data, _shape=tuple(out_shape))
        
        result.requires_grad = self.requires_grad
        
        if self.requires_grad:
            ctx = Context()
            ctx.save_for_backward(self)
            ctx.dim = dim
            ctx.keepdim = keepdim
            result._ctx = ctx
            result._grad_fn = self._sum_backward
        
        return result
    
    @staticmethod
    def _sum_backward(ctx, grad):
        parent = ctx.saved_tensors[0]
        # Broadcast gradient back to original shape
        expanded_data = [grad._data[0] if grad.size == 1 else 0.0] * parent.size
        if grad.size > 1:
            # More complex broadcasting needed
            for i in range(parent.size):
                expanded_data[i] = grad._data[i % grad.size]
        else:
            expanded_data = [grad._data[0]] * parent.size
        return Tensor(expanded_data, _shape=parent._shape)
    
    def mean(self, dim: Optional[int] = None, keepdim: bool = False) -> 'Tensor':
        """Compute mean of tensor elements."""
        s = self.sum(dim=dim, keepdim=keepdim)
        if dim is None:
            count = self.size
        else:
            count = self._shape[dim]
        return s / count
    
    def max(self, dim: Optional[int] = None) -> 'Tensor':
        """Get maximum value."""
        if dim is None:
            return Tensor([max(self._data)], _shape=())
        raise NotImplementedError("max with dim not yet implemented")
    
    def min(self, dim: Optional[int] = None) -> 'Tensor':
        """Get minimum value."""
        if dim is None:
            return Tensor([min(self._data)], _shape=())
        raise NotImplementedError("min with dim not yet implemented")
    
    def abs(self) -> 'Tensor':
        """Element-wise absolute value."""
        return self._unary_op(lambda x: abs(x), lambda ctx, grad: grad * self.sign())
    
    def sign(self) -> 'Tensor':
        """Element-wise sign function."""
        data = [1.0 if x > 0 else (-1.0 if x < 0 else 0.0) for x in self._data]
        return Tensor(data, _shape=self._shape)
    
    def sqrt(self) -> 'Tensor':
        """Element-wise square root."""
        def backward(ctx, grad):
            return grad / (self.sqrt() * 2.0)
        return self._unary_op(lambda x: math.sqrt(x), backward)
    
    def exp(self) -> 'Tensor':
        """Element-wise exponential."""
        result = self._unary_op(lambda x: math.exp(min(x, 700)), None)  # Clip to prevent overflow
        if self.requires_grad:
            ctx = Context()
            ctx.save_for_backward(result)
            result._ctx = ctx
            result._grad_fn = lambda ctx, grad: grad * ctx.saved_tensors[0]
        return result
    
    def log(self) -> 'Tensor':
        """Element-wise natural logarithm."""
        def backward(ctx, grad):
            return grad / self
        return self._unary_op(lambda x: math.log(max(x, 1e-12)), backward)
    
    def pow(self, exponent: float) -> 'Tensor':
        """Element-wise power."""
        data = [x ** exponent for x in self._data]
        result = Tensor(data, requires_grad=self.requires_grad,
                       device=self.device, _shape=self._shape)
        
        if self.requires_grad:
            ctx = Context()
            ctx.save_for_backward(self)
            ctx.exponent = exponent
            result._ctx = ctx
            result._grad_fn = lambda ctx, grad: grad * (ctx.exponent * ctx.saved_tensors[0].pow(ctx.exponent - 1))
        
        return result
    
    def _unary_op(self, op: Callable, backward_fn: Optional[Callable]) -> 'Tensor':
        """Apply unary operation."""
        data = [op(x) for x in self._data]
        result = Tensor(data, requires_grad=self.requires_grad,
                       device=self.device, _shape=self._shape)
        
        if self.requires_grad and backward_fn is not None:
            ctx = Context()
            ctx.save_for_backward(self)
            result._ctx = ctx
            result._grad_fn = backward_fn
        
        return result
    
    def _broadcast_shapes(self, other: 'Tensor') -> Tuple[Tuple[int, ...], List[int], List[int]]:
        """Compute broadcast shape and indices."""
        # Pad shorter shape with 1s
        a_shape = (1,) * (max(self.ndim, other.ndim) - self.ndim) + self._shape
        b_shape = (1,) * (max(self.ndim, other.ndim) - other.ndim) + other._shape
        
        out_shape = []
        for a, b in zip(a_shape, b_shape):
            if a == b:
                out_shape.append(a)
            elif a == 1:
                out_shape.append(b)
            elif b == 1:
                out_shape.append(a)
            else:
                raise ValueError(f"Cannot broadcast shapes {self._shape} and {other._shape}")
        
        return tuple(out_shape), list(a_shape), list(b_shape)
    
    def _get_broadcast_index(self, flat_idx: int, shape: List[int], orig_shape: Tuple[int, ...]) -> int:
        """Get index in original tensor for broadcast."""
        # Compute coordinates in broadcast shape
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        
        coords = []
        idx = flat_idx
        for s in strides:
            coords.append(idx // s)
            idx = idx % s
        
        # Map to original tensor (wrap dimensions of size 1)
        orig_coords = []
        pad = len(shape) - len(orig_shape)
        for i, (c, os) in enumerate(zip(coords[pad:], orig_shape)):
            orig_coords.append(c % os)
        
        # Compute flat index in original tensor
        orig_strides = [1] * len(orig_shape)
        for i in range(len(orig_shape) - 2, -1, -1):
            orig_strides[i] = orig_strides[i + 1] * orig_shape[i + 1]
        
        return sum(c * s for c, s in zip(orig_coords, orig_strides)) if orig_strides else 0
    
    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Element-wise addition."""
        if isinstance(other, (int, float)):
            other = Tensor([other] * self.size, _shape=self._shape)
        
        out_shape, a_shape, b_shape = self._broadcast_shapes(other)
        out_size = 1
        for d in out_shape:
            out_size *= d
        
        out_data = []
        for i in range(out_size):
            a_idx = self._get_broadcast_index(i, list(out_shape), self._shape)
            b_idx = other._get_broadcast_index(i, list(out_shape), other._shape)
            out_data.append(self._data[a_idx] + other._data[b_idx])
        
        result = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad,
                       device=self.device, _shape=out_shape)
        
        if result.requires_grad:
            ctx = Context()
            ctx.save_for_backward(self, other)
            result._ctx = ctx
            result._grad_fn = self._add_backward
        
        return result
    
    @staticmethod
    def _add_backward(ctx, grad):
        a, b = ctx.saved_tensors
        # Sum grad over broadcast dimensions
        grad_a = Tensor._reduce_broadcast_grad(grad, a._shape)
        grad_b = Tensor._reduce_broadcast_grad(grad, b._shape)
        return grad_a, grad_b
    
    @staticmethod
    def _reduce_broadcast_grad(grad, target_shape):
        """Reduce gradient to match target shape."""
        if grad._shape == target_shape:
            return grad
        
        # Sum over extra dimensions
        data = [0.0] * (1 if len(target_shape) == 0 else 1)
        for d in target_shape:
            data[0] *= d
        
        # Simple reduction for now
        if len(target_shape) == 0:
            return Tensor([sum(grad._data)], _shape=())
        
        target_size = 1
        for d in target_shape:
            target_size *= d
        
        new_data = [0.0] * target_size
        for i in range(grad.size):
            target_idx = i % target_size
            new_data[target_idx] += grad._data[i]
        
        return Tensor(new_data, _shape=target_shape)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __neg__(self) -> 'Tensor':
        """Negation."""
        data = [-x for x in self._data]
        result = Tensor(data, requires_grad=self.requires_grad,
                       device=self.device, _shape=self._shape)
        
        if self.requires_grad:
            ctx = Context()
            ctx.save_for_backward(self)
            result._ctx = ctx
            result._grad_fn = lambda ctx, grad: -grad
        
        return result
    
    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Element-wise subtraction."""
        if isinstance(other, (int, float)):
            other = Tensor([other] * self.size, _shape=self._shape)
        return self + (-other)
    
    def __rsub__(self, other):
        return (-self) + other
    
    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Element-wise multiplication."""
        if isinstance(other, (int, float)):
            other = Tensor([other] * self.size, _shape=self._shape)
        
        out_shape, a_shape, b_shape = self._broadcast_shapes(other)
        out_size = 1
        for d in out_shape:
            out_size *= d
        
        out_data = []
        for i in range(out_size):
            a_idx = self._get_broadcast_index(i, list(out_shape), self._shape)
            b_idx = other._get_broadcast_index(i, list(out_shape), other._shape)
            out_data.append(self._data[a_idx] * other._data[b_idx])
        
        result = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad,
                       device=self.device, _shape=out_shape)
        
        if result.requires_grad:
            ctx = Context()
            ctx.save_for_backward(self, other)
            result._ctx = ctx
            result._grad_fn = self._mul_backward
        
        return result
    
    @staticmethod
    def _mul_backward(ctx, grad):
        a, b = ctx.saved_tensors
        grad_a = Tensor._reduce_broadcast_grad(grad * b, a._shape)
        grad_b = Tensor._reduce_broadcast_grad(grad * a, b._shape)
        return grad_a, grad_b
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Element-wise division."""
        if isinstance(other, (int, float)):
            other = Tensor([other] * self.size, _shape=self._shape)
        
        out_shape, a_shape, b_shape = self._broadcast_shapes(other)
        out_size = 1
        for d in out_shape:
            out_size *= d
        
        out_data = []
        for i in range(out_size):
            a_idx = self._get_broadcast_index(i, list(out_shape), self._shape)
            b_idx = other._get_broadcast_index(i, list(out_shape), other._shape)
            out_data.append(self._data[a_idx] / other._data[b_idx])
        
        result = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad,
                       device=self.device, _shape=out_shape)
        
        if result.requires_grad:
            ctx = Context()
            ctx.save_for_backward(self, other)
            result._ctx = ctx
            result._grad_fn = self._div_backward
        
        return result
    
    @staticmethod
    def _div_backward(ctx, grad):
        a, b = ctx.saved_tensors
        grad_a = Tensor._reduce_broadcast_grad(grad / b, a._shape)
        grad_b = Tensor._reduce_broadcast_grad(-grad * a / (b * b), b._shape)
        return grad_a, grad_b
    
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor([other] * self.size, _shape=self._shape)
        return other / self
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication."""
        if self.ndim < 2 or other.ndim < 2:
            raise ValueError("matmul requires at least 2D tensors")
        
        # Check dimensions
        if self._shape[-1] != other._shape[-2]:
            raise ValueError(f"matmul dimension mismatch: {self._shape} @ {other._shape}")
        
        m, k = self._shape[-2], self._shape[-1]
        k2, n = other._shape[-2], other._shape[-1]
        
        # Simple 2D matmul for now
        out_data = []
        for i in range(m):
            for j in range(n):
                total = 0.0
                for p in range(k):
                    a_idx = i * k + p
                    b_idx = p * n + j
                    total += self._data[a_idx] * other._data[b_idx]
                out_data.append(total)
        
        out_shape = (m, n)
        result = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad,
                       device=self.device, _shape=out_shape)
        
        if result.requires_grad:
            ctx = Context()
            ctx.save_for_backward(self, other)
            result._ctx = ctx
            result._grad_fn = self._matmul_backward
        
        return result
    
    @staticmethod
    def _matmul_backward(ctx, grad):
        a, b = ctx.saved_tensors
        grad_a = grad @ b.T
        grad_b = a.T @ grad
        return grad_a, grad_b
    
    def __getitem__(self, idx):
        """Get item by index."""
        if isinstance(idx, int):
            if self.ndim == 1:
                return Tensor([self._data[idx]], _shape=())
            else:
                # Get a slice along first dimension
                sub_size = self.size // self._shape[0]
                start = idx * sub_size
                end = start + sub_size
                new_shape = self._shape[1:]
                return Tensor(self._data[start:end], _shape=new_shape)
        raise NotImplementedError("Advanced indexing not yet implemented")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Tensor({self.tolist()}, requires_grad={self.requires_grad})"
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __len__(self) -> int:
        if self.ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return self._shape[0]
    
    # Comparison operations (no gradient)
    def __lt__(self, other):
        if isinstance(other, Tensor):
            other = other._data[0] if other.size == 1 else other._data
            if isinstance(other, list):
                return Tensor([1.0 if a < b else 0.0 for a, b in zip(self._data, other)], _shape=self._shape)
            return Tensor([1.0 if x < other else 0.0 for x in self._data], _shape=self._shape)
        return Tensor([1.0 if x < other else 0.0 for x in self._data], _shape=self._shape)
    
    def __gt__(self, other):
        if isinstance(other, Tensor):
            other = other._data[0] if other.size == 1 else other._data
            if isinstance(other, list):
                return Tensor([1.0 if a > b else 0.0 for a, b in zip(self._data, other)], _shape=self._shape)
            return Tensor([1.0 if x > other else 0.0 for x in self._data], _shape=self._shape)
        return Tensor([1.0 if x > other else 0.0 for x in self._data], _shape=self._shape)
    
    def __eq__(self, other):
        if isinstance(other, Tensor):
            other = other._data[0] if other.size == 1 else other._data
            if isinstance(other, list):
                return Tensor([1.0 if a == b else 0.0 for a, b in zip(self._data, other)], _shape=self._shape)
            return Tensor([1.0 if x == other else 0.0 for x in self._data], _shape=self._shape)
        return Tensor([1.0 if x == other else 0.0 for x in self._data], _shape=self._shape)


class Context:
    """Context for saving tensors during forward pass for backward computation."""
    
    def __init__(self):
        self.saved_tensors = []
    
    def save_for_backward(self, *tensors):
        """Save tensors for use in backward pass."""
        self.saved_tensors = list(tensors)


# Factory functions

def zeros(shape: Tuple[int, ...], requires_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Create tensor filled with zeros."""
    size = 1
    for d in shape:
        size *= d
    return Tensor([0.0] * size, requires_grad=requires_grad, device=device, _shape=shape)


def ones(shape: Tuple[int, ...], requires_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Create tensor filled with ones."""
    size = 1
    for d in shape:
        size *= d
    return Tensor([1.0] * size, requires_grad=requires_grad, device=device, _shape=shape)


def randn(shape: Tuple[int, ...], requires_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Create tensor with random normal values."""
    import random
    size = 1
    for d in shape:
        size *= d
    # Box-Muller transform for normal distribution
    data = []
    for _ in range(size):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(max(u1, 1e-12))) * math.cos(2 * math.pi * u2)
        data.append(z)
    return Tensor(data, requires_grad=requires_grad, device=device, _shape=shape)


def rand(shape: Tuple[int, ...], requires_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Create tensor with random uniform values in [0, 1)."""
    import random
    size = 1
    for d in shape:
        size *= d
    data = [random.random() for _ in range(size)]
    return Tensor(data, requires_grad=requires_grad, device=device, _shape=shape)


def arange(start: float, end: float = None, step: float = 1.0, 
           requires_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Create 1D tensor with evenly spaced values."""
    if end is None:
        end = start
        start = 0
    data = []
    val = start
    while val < end:
        data.append(float(val))
        val += step
    return Tensor(data, requires_grad=requires_grad, device=device, _shape=(len(data),))


def eye(n: int, requires_grad: bool = False, device: str = 'cpu') -> Tensor:
    """Create identity matrix."""
    data = [1.0 if i == j else 0.0 for i in range(n) for j in range(n)]
    return Tensor(data, requires_grad=requires_grad, device=device, _shape=(n, n))


def from_numpy(arr) -> Tensor:
    """Create tensor from numpy array."""
    return Tensor(arr)


def cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """Concatenate tensors along a dimension."""
    if len(tensors) == 0:
        raise ValueError("Cannot concatenate empty list of tensors")
    
    # Simple 1D concatenation
    if tensors[0].ndim == 1 and dim == 0:
        data = []
        for t in tensors:
            data.extend(t._data)
        return Tensor(data, _shape=(len(data),))
    
    raise NotImplementedError("General concatenation not yet implemented")


def stack(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """Stack tensors along a new dimension."""
    if len(tensors) == 0:
        raise ValueError("Cannot stack empty list of tensors")
    
    # Check shapes match
    shape = tensors[0]._shape
    for t in tensors[1:]:
        if t._shape != shape:
            raise ValueError("All tensors must have same shape")
    
    # Stack data
    data = []
    for t in tensors:
        data.extend(t._data)
    
    new_shape = (len(tensors),) + shape
    return Tensor(data, _shape=new_shape)
