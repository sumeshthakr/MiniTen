"""
Utility functions for edge deployment.

Provides helper functions for analyzing models for edge deployment.
"""

import numpy as np


def count_parameters(model):
    """
    Count total number of parameters in a model.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with parameter counts
    """
    total = 0
    trainable = 0
    
    if hasattr(model, 'parameters'):
        for param in model.parameters():
            if hasattr(param, 'size'):
                size = np.prod(param.shape if hasattr(param, 'shape') else param.size)
            elif hasattr(param, 'shape'):
                size = np.prod(param.shape)
            else:
                continue
            
            total += size
            trainable += size  # Assume all trainable for now
    
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "non_trainable_parameters": int(total - trainable),
    }


def count_flops(model, input_shape):
    """
    Estimate FLOPs (floating point operations) for a model.
    
    Args:
        model: Model to analyze
        input_shape: Shape of input tensor
        
    Returns:
        Dictionary with FLOP counts
    """
    total_flops = 0
    flops_by_layer = []
    
    if hasattr(model, 'layers'):
        current_shape = list(input_shape)
        
        for i, layer in enumerate(model.layers):
            layer_flops = _estimate_layer_flops(layer, current_shape)
            flops_by_layer.append({
                "layer": i,
                "type": type(layer).__name__,
                "flops": layer_flops,
            })
            total_flops += layer_flops
            
            # Update shape for next layer
            if hasattr(layer, 'out_features'):
                current_shape[-1] = layer.out_features
    
    return {
        "total_flops": total_flops,
        "total_mflops": total_flops / 1e6,
        "total_gflops": total_flops / 1e9,
        "by_layer": flops_by_layer,
    }


def _estimate_layer_flops(layer, input_shape):
    """Estimate FLOPs for a single layer."""
    layer_type = type(layer).__name__
    
    if layer_type == 'Linear':
        # Linear: 2 * in_features * out_features (multiply-add)
        in_features = getattr(layer, 'in_features', input_shape[-1])
        out_features = getattr(layer, 'out_features', in_features)
        return 2 * in_features * out_features
    
    elif layer_type == 'Conv2d':
        # Conv: 2 * in_channels * out_channels * kernel_h * kernel_w * out_h * out_w
        in_channels = getattr(layer, 'in_channels', 1)
        out_channels = getattr(layer, 'out_channels', 1)
        kernel_size = getattr(layer, 'kernel_size', 3)
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        
        # Estimate output size (simplified)
        out_h = input_shape[-2] if len(input_shape) > 2 else 1
        out_w = input_shape[-1] if len(input_shape) > 1 else 1
        
        return 2 * in_channels * out_channels * kernel_size * kernel_size * out_h * out_w
    
    elif layer_type in ['ReLU', 'Sigmoid', 'Tanh']:
        # Activation: 1 FLOP per element
        return np.prod(input_shape)
    
    elif layer_type in ['MaxPool2d', 'AvgPool2d']:
        # Pooling: comparisons or additions
        return np.prod(input_shape)
    
    else:
        # Unknown layer type
        return 0


def estimate_memory(model, input_shape, dtype='float32'):
    """
    Estimate memory usage for inference.
    
    Args:
        model: Model to analyze
        input_shape: Input shape
        dtype: Data type
        
    Returns:
        Dictionary with memory estimates
    """
    dtype_sizes = {
        'float32': 4,
        'float64': 8,
        'float16': 2,
        'int8': 1,
        'int16': 2,
        'int32': 4,
    }
    
    bytes_per_element = dtype_sizes.get(dtype, 4)
    
    # Model parameters
    param_count = count_parameters(model)
    param_memory = param_count["total_parameters"] * bytes_per_element
    
    # Input memory
    input_memory = np.prod(input_shape) * bytes_per_element
    
    # Estimate activation memory (rough estimate)
    # Assume activations similar size to input
    activation_memory = input_memory * 2  # Conservative estimate
    
    total_memory = param_memory + input_memory + activation_memory
    
    return {
        "parameter_memory_bytes": param_memory,
        "parameter_memory_mb": param_memory / (1024 * 1024),
        "input_memory_bytes": input_memory,
        "input_memory_mb": input_memory / (1024 * 1024),
        "activation_memory_bytes": activation_memory,
        "activation_memory_mb": activation_memory / (1024 * 1024),
        "total_memory_bytes": total_memory,
        "total_memory_mb": total_memory / (1024 * 1024),
    }


def estimate_power(model, input_shape, device='cpu', ops_per_watt=1e9):
    """
    Estimate power consumption for inference.
    
    This is a rough estimate based on FLOP count and device efficiency.
    
    Args:
        model: Model to analyze
        input_shape: Input shape
        device: Target device type
        ops_per_watt: Operations per watt for the device
        
    Returns:
        Dictionary with power estimates
    """
    # Device efficiency estimates (ops/watt)
    device_efficiency = {
        'cpu': 1e9,           # ~1 GFLOP/W for typical CPU
        'gpu': 10e9,          # ~10 GFLOP/W for typical GPU
        'edge_cpu': 5e9,      # ~5 GFLOP/W for efficient edge CPU
        'npu': 20e9,          # ~20 GFLOP/W for neural accelerator
        'dsp': 15e9,          # ~15 GFLOP/W for DSP
    }
    
    efficiency = device_efficiency.get(device, ops_per_watt)
    
    # Get FLOP count
    flops = count_flops(model, input_shape)
    total_flops = flops["total_flops"]
    
    # Estimate power per inference
    power_per_inference = total_flops / efficiency  # Joules
    
    return {
        "flops": total_flops,
        "device": device,
        "ops_per_watt": efficiency,
        "energy_per_inference_j": power_per_inference,
        "energy_per_inference_mj": power_per_inference * 1000,
        "inferences_per_joule": 1 / power_per_inference if power_per_inference > 0 else float('inf'),
    }


def model_summary(model, input_shape):
    """
    Generate a comprehensive model summary.
    
    Args:
        model: Model to analyze
        input_shape: Input shape
        
    Returns:
        Summary string
    """
    params = count_parameters(model)
    flops = count_flops(model, input_shape)
    memory = estimate_memory(model, input_shape)
    
    lines = [
        "=" * 60,
        "MODEL SUMMARY",
        "=" * 60,
        f"Input shape: {input_shape}",
        "",
        "PARAMETERS",
        f"  Total: {params['total_parameters']:,}",
        f"  Trainable: {params['trainable_parameters']:,}",
        "",
        "COMPUTATION",
        f"  FLOPs: {flops['total_mflops']:.2f}M",
        f"  GFLOPs: {flops['total_gflops']:.4f}",
        "",
        "MEMORY",
        f"  Parameters: {memory['parameter_memory_mb']:.2f} MB",
        f"  Total (estimate): {memory['total_memory_mb']:.2f} MB",
        "",
    ]
    
    if hasattr(model, 'layers'):
        lines.append("LAYERS")
        for i, layer in enumerate(model.layers):
            layer_type = type(layer).__name__
            lines.append(f"  [{i}] {layer_type}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def check_edge_compatibility(model, input_shape, constraints=None):
    """
    Check if a model meets edge deployment constraints.
    
    Args:
        model: Model to check
        input_shape: Input shape
        constraints: Dictionary of constraints
        
    Returns:
        Dictionary with compatibility results
    """
    if constraints is None:
        constraints = {
            "max_parameters": 10_000_000,  # 10M params
            "max_memory_mb": 100,          # 100 MB
            "max_flops": 1e9,              # 1 GFLOP
        }
    
    params = count_parameters(model)
    flops = count_flops(model, input_shape)
    memory = estimate_memory(model, input_shape)
    
    results = {
        "passes_all": True,
        "checks": [],
    }
    
    # Check parameters
    param_check = {
        "name": "parameters",
        "actual": params["total_parameters"],
        "limit": constraints["max_parameters"],
        "passes": params["total_parameters"] <= constraints["max_parameters"],
    }
    results["checks"].append(param_check)
    results["passes_all"] &= param_check["passes"]
    
    # Check memory
    memory_check = {
        "name": "memory",
        "actual": memory["total_memory_mb"],
        "limit": constraints["max_memory_mb"],
        "passes": memory["total_memory_mb"] <= constraints["max_memory_mb"],
    }
    results["checks"].append(memory_check)
    results["passes_all"] &= memory_check["passes"]
    
    # Check FLOPs
    flops_check = {
        "name": "flops",
        "actual": flops["total_flops"],
        "limit": constraints["max_flops"],
        "passes": flops["total_flops"] <= constraints["max_flops"],
    }
    results["checks"].append(flops_check)
    results["passes_all"] &= flops_check["passes"]
    
    return results
