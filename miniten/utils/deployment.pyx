# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Model serialization and deployment utilities for MiniTen.

This module provides:
- Model save/load functionality
- Memory pooling for efficient tensor allocation
- Edge deployment utilities
- Model compression

All implementations are optimized for edge devices with minimal memory footprint.
"""

import cython
import numpy as np
cimport numpy as np
import pickle
import json
import os
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

# Initialize NumPy C API
np.import_array()


# ============================================================================
# Model Serialization
# ============================================================================

def save_model(model, str filepath, bint save_optimizer=True, dict metadata=None):
    """
    Save a model to disk.
    
    Supports saving:
    - Model parameters (weights, biases)
    - Optimizer state (optional)
    - Model architecture info
    - Custom metadata
    
    Args:
        model: Model object with parameters() method
        filepath: Path to save the model
        save_optimizer: Whether to save optimizer state
        metadata: Optional metadata dictionary
    """
    state_dict = {
        'model_state': {},
        'metadata': metadata or {},
        'version': '0.1.0'
    }
    
    # Extract model parameters
    if hasattr(model, 'state_dict'):
        state_dict['model_state'] = model.state_dict()
    elif hasattr(model, 'parameters'):
        params = model.parameters()
        for i, p in enumerate(params):
            if hasattr(p, 'weight'):
                state_dict['model_state'][f'layer_{i}_weight'] = np.array(p.weight)
            if hasattr(p, 'bias') and p.bias is not None:
                state_dict['model_state'][f'layer_{i}_bias'] = np.array(p.bias)
    else:
        raise ValueError("Model must have state_dict() or parameters() method")
    
    # Save optimizer state if requested
    if save_optimizer and hasattr(model, 'optimizer') and model.optimizer is not None:
        if hasattr(model.optimizer, 'state_dict'):
            state_dict['optimizer_state'] = model.optimizer.state_dict()
    
    # Save to file
    with open(filepath, 'wb') as f:
        pickle.dump(state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(model, str filepath, bint load_optimizer=True):
    """
    Load a model from disk.
    
    Args:
        model: Model object to load into
        filepath: Path to the saved model
        load_optimizer: Whether to load optimizer state
        
    Returns:
        Loaded metadata dictionary
    """
    with open(filepath, 'rb') as f:
        state_dict = pickle.load(f)
    
    # Load model parameters
    if hasattr(model, 'load_state_dict'):
        model.load_state_dict(state_dict['model_state'])
    elif hasattr(model, 'parameters'):
        params = model.parameters()
        for i, p in enumerate(params):
            weight_key = f'layer_{i}_weight'
            bias_key = f'layer_{i}_bias'
            if weight_key in state_dict['model_state']:
                if hasattr(p, 'weight'):
                    p.weight[:] = state_dict['model_state'][weight_key]
            if bias_key in state_dict['model_state']:
                if hasattr(p, 'bias') and p.bias is not None:
                    p.bias[:] = state_dict['model_state'][bias_key]
    
    # Load optimizer state if requested
    if load_optimizer and 'optimizer_state' in state_dict:
        if hasattr(model, 'optimizer') and model.optimizer is not None:
            if hasattr(model.optimizer, 'load_state_dict'):
                model.optimizer.load_state_dict(state_dict['optimizer_state'])
    
    return state_dict.get('metadata', {})


def save_weights_only(params, str filepath):
    """
    Save only the weights (lightweight format).
    
    Args:
        params: Dictionary of parameter name -> numpy array
        filepath: Path to save
    """
    np.savez_compressed(filepath, **params)


def load_weights_only(str filepath):
    """
    Load weights from lightweight format.
    
    Args:
        filepath: Path to load from
        
    Returns:
        Dictionary of parameter name -> numpy array
    """
    data = np.load(filepath)
    return {key: data[key] for key in data.files}


def export_to_json(model, str filepath):
    """
    Export model architecture to JSON (for deployment info).
    
    Args:
        model: Model to export
        filepath: Output path
    """
    arch_info = {
        'layers': [],
        'total_params': 0
    }
    
    if hasattr(model, 'layers'):
        for i, layer in enumerate(model.layers):
            layer_info = {
                'index': i,
                'type': type(layer).__name__,
            }
            
            if hasattr(layer, 'in_features'):
                layer_info['in_features'] = layer.in_features
            if hasattr(layer, 'out_features'):
                layer_info['out_features'] = layer.out_features
            if hasattr(layer, 'kernel_size'):
                layer_info['kernel_size'] = layer.kernel_size
            
            # Count parameters
            param_count = 0
            if hasattr(layer, 'weight'):
                param_count += np.prod(layer.weight.shape)
            if hasattr(layer, 'bias') and layer.bias is not None:
                param_count += np.prod(layer.bias.shape)
            
            layer_info['params'] = int(param_count)
            arch_info['total_params'] += param_count
            arch_info['layers'].append(layer_info)
    
    with open(filepath, 'w') as f:
        json.dump(arch_info, f, indent=2)


# ============================================================================
# Memory Pool
# ============================================================================

cdef class MemoryPool:
    """
    Memory pool for efficient tensor allocation.
    
    Pre-allocates memory blocks to reduce allocation overhead during
    forward/backward passes. Optimized for edge devices with limited memory.
    """
    
    cdef dict _pools
    cdef Py_ssize_t _max_size
    cdef Py_ssize_t _current_size
    
    def __init__(self, Py_ssize_t max_size_mb=100):
        """
        Initialize memory pool.
        
        Args:
            max_size_mb: Maximum pool size in megabytes
        """
        self._pools = {}  # shape -> list of available arrays
        self._max_size = max_size_mb * 1024 * 1024  # Convert to bytes
        self._current_size = 0
    
    cpdef np.ndarray allocate(self, tuple shape, dtype=np.float64):
        """
        Allocate an array from the pool.
        
        Args:
            shape: Shape of the array
            dtype: Data type
            
        Returns:
            Numpy array (may be reused from pool)
        """
        cdef str key = str(shape) + str(dtype)
        
        # Check if we have a pooled array of this shape
        if key in self._pools and len(self._pools[key]) > 0:
            return self._pools[key].pop()
        
        # Allocate new array
        cdef np.ndarray arr = np.empty(shape, dtype=dtype)
        cdef Py_ssize_t arr_size = arr.nbytes
        
        # Track memory usage
        self._current_size += arr_size
        
        return arr
    
    cpdef void release(self, np.ndarray arr):
        """
        Return an array to the pool.
        
        Args:
            arr: Array to return to pool
        """
        # Create key from shape and dtype - need to convert to Python objects
        cdef object arr_shape = (<object>arr).shape
        cdef str key = str(arr_shape) + str(arr.dtype)
        
        if key not in self._pools:
            self._pools[key] = []
        
        # Only pool if we're under the size limit
        if self._current_size < self._max_size:
            self._pools[key].append(arr)
    
    cpdef void clear(self):
        """Clear all pooled arrays."""
        self._pools.clear()
        self._current_size = 0
    
    cpdef Py_ssize_t get_pool_size(self):
        """Get current pool size in bytes."""
        return self._current_size
    
    cpdef dict get_pool_stats(self):
        """Get statistics about the pool."""
        stats = {
            'current_size_mb': self._current_size / (1024 * 1024),
            'max_size_mb': self._max_size / (1024 * 1024),
            'num_shapes': len(self._pools),
            'shapes': {}
        }
        
        for key, arrays in self._pools.items():
            stats['shapes'][key] = len(arrays)
        
        return stats


# Global memory pool instance
_global_pool = None


def get_memory_pool(max_size_mb=100):
    """Get or create the global memory pool."""
    global _global_pool
    if _global_pool is None:
        _global_pool = MemoryPool(max_size_mb)
    return _global_pool


def reset_memory_pool():
    """Reset the global memory pool."""
    global _global_pool
    if _global_pool is not None:
        _global_pool.clear()
    _global_pool = None


# ============================================================================
# Model Containers
# ============================================================================

class Sequential:
    """
    Sequential container for layers.
    
    Layers are executed in the order they are added.
    
    Example:
        model = Sequential([
            Linear(784, 128),
            ReLU(),
            Linear(128, 10)
        ])
        output = model(input)
    """
    
    def __init__(self, layers=None):
        """
        Initialize Sequential container.
        
        Args:
            layers: List of layers or None
        """
        self.layers = layers or []
        self._training = True
    
    def add(self, layer):
        """Add a layer to the sequence."""
        self.layers.append(layer)
        return self
    
    def forward(self, x):
        """
        Forward pass through all layers.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        for layer in self.layers:
            if hasattr(layer, 'forward'):
                x = layer.forward(x)
            elif callable(layer):
                x = layer(x)
        return x
    
    def __call__(self, x):
        """Forward pass (callable interface)."""
        return self.forward(x)
    
    def backward(self, grad):
        """
        Backward pass through all layers.
        
        Args:
            grad: Gradient from loss
            
        Returns:
            Gradient w.r.t. input
        """
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                grad = layer.backward(grad)
        return grad
    
    def parameters(self):
        """Get all parameters."""
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
            elif hasattr(layer, 'weight'):
                params.append(layer)
        return params
    
    def update_parameters(self, learning_rate):
        """Update all parameters."""
        for layer in self.layers:
            if hasattr(layer, 'update_parameters'):
                layer.update_parameters(learning_rate)
    
    def zero_grad(self):
        """Reset all gradients."""
        for layer in self.layers:
            if hasattr(layer, 'zero_grad'):
                layer.zero_grad()
    
    def train(self, mode=True):
        """Set training mode."""
        self._training = mode
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train(mode)
    
    def eval(self):
        """Set evaluation mode."""
        self.train(False)
    
    def state_dict(self):
        """Get state dictionary."""
        state = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weight'):
                state[f'layer_{i}_weight'] = np.array(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                state[f'layer_{i}_bias'] = np.array(layer.bias)
        return state
    
    def load_state_dict(self, state_dict):
        """Load state dictionary."""
        for i, layer in enumerate(self.layers):
            weight_key = f'layer_{i}_weight'
            bias_key = f'layer_{i}_bias'
            if weight_key in state_dict and hasattr(layer, 'weight'):
                layer.weight[:] = state_dict[weight_key]
            if bias_key in state_dict and hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias[:] = state_dict[bias_key]


class ModuleList:
    """
    List of modules.
    
    Like a Python list but registers all modules for parameter tracking.
    
    Example:
        layers = ModuleList([Linear(10, 10) for _ in range(5)])
    """
    
    def __init__(self, modules=None):
        """
        Initialize ModuleList.
        
        Args:
            modules: List of modules or None
        """
        self._modules = list(modules) if modules else []
    
    def append(self, module):
        """Add a module."""
        self._modules.append(module)
        return self
    
    def extend(self, modules):
        """Add multiple modules."""
        self._modules.extend(modules)
        return self
    
    def __getitem__(self, idx):
        """Get module by index."""
        return self._modules[idx]
    
    def __setitem__(self, idx, module):
        """Set module by index."""
        self._modules[idx] = module
    
    def __len__(self):
        """Get number of modules."""
        return len(self._modules)
    
    def __iter__(self):
        """Iterate over modules."""
        return iter(self._modules)
    
    def parameters(self):
        """Get all parameters from all modules."""
        params = []
        for module in self._modules:
            if hasattr(module, 'parameters'):
                params.extend(module.parameters())
            elif hasattr(module, 'weight'):
                params.append(module)
        return params
    
    def update_parameters(self, learning_rate):
        """Update all parameters."""
        for module in self._modules:
            if hasattr(module, 'update_parameters'):
                module.update_parameters(learning_rate)


# ============================================================================
# Edge Deployment Utilities
# ============================================================================

def estimate_model_size(model):
    """
    Estimate model size in bytes.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with size breakdown
    """
    size_info = {
        'total_bytes': 0,
        'total_mb': 0.0,
        'layers': [],
        'total_params': 0
    }
    
    if hasattr(model, 'parameters'):
        for param in model.parameters():
            if hasattr(param, 'weight'):
                weight_bytes = param.weight.nbytes
                size_info['total_bytes'] += weight_bytes
                size_info['total_params'] += param.weight.size
                
                layer_info = {
                    'type': type(param).__name__,
                    'weight_shape': tuple(param.weight.shape),
                    'weight_bytes': weight_bytes
                }
                
                if hasattr(param, 'bias') and param.bias is not None:
                    bias_bytes = param.bias.nbytes
                    size_info['total_bytes'] += bias_bytes
                    size_info['total_params'] += param.bias.size
                    layer_info['bias_shape'] = tuple(param.bias.shape)
                    layer_info['bias_bytes'] = bias_bytes
                
                size_info['layers'].append(layer_info)
    
    size_info['total_mb'] = size_info['total_bytes'] / (1024 * 1024)
    
    return size_info


def count_flops(model, input_shape):
    """
    Estimate FLOPs for a forward pass.
    
    Args:
        model: Model to analyze
        input_shape: Shape of input tensor
        
    Returns:
        Dictionary with FLOP count
    """
    flops_info = {
        'total_flops': 0,
        'layers': []
    }
    
    # This is a simplified estimation
    if hasattr(model, 'layers'):
        batch_size = input_shape[0]
        current_features = input_shape[len(input_shape) - 1] if len(input_shape) > 1 else 1
        
        for layer in model.layers:
            layer_flops = 0
            
            if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
                # Linear layer: 2 * in * out (multiply-add)
                layer_flops = 2 * layer.in_features * layer.out_features * batch_size
                current_features = layer.out_features
            
            elif hasattr(layer, 'kernel_size') and hasattr(layer, 'out_channels'):
                # Conv2d: rough estimate
                if hasattr(layer, 'in_channels'):
                    k = layer.kernel_size
                    layer_flops = 2 * layer.in_channels * layer.out_channels * k * k * batch_size
            
            flops_info['total_flops'] += layer_flops
            flops_info['layers'].append({
                'type': type(layer).__name__,
                'flops': layer_flops
            })
    
    flops_info['total_gflops'] = flops_info['total_flops'] / 1e9
    
    return flops_info


def benchmark_inference(model, input_data, num_runs=100, warmup=10):
    """
    Benchmark model inference time.
    
    Args:
        model: Model to benchmark
        input_data: Sample input data
        num_runs: Number of runs for timing
        warmup: Number of warmup runs
        
    Returns:
        Dictionary with timing statistics
    """
    import time
    
    # Warmup runs
    for _ in range(warmup):
        if hasattr(model, 'forward'):
            model.forward(input_data)
        elif callable(model):
            model(input_data)
    
    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        if hasattr(model, 'forward'):
            model.forward(input_data)
        elif callable(model):
            model(input_data)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'num_runs': num_runs
    }
