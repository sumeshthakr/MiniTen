"""
MiniTen Edge Device Features

Comprehensive tools and utilities for deploying models on edge devices:
- Model quantization (int8, int16)
- Weight pruning and sparsity
- Model compression
- Memory profiling
- Power consumption estimation
- Model benchmarking
- Export to edge formats

Example:
    >>> from miniten.edge import quantize, prune, benchmark
    >>> quantized_model = quantize(model, dtype='int8')
    >>> pruned_model = prune(model, sparsity=0.5)
    >>> benchmark_results = benchmark(model, input_shape=(1, 3, 224, 224))
"""

from .quantization import (
    quantize_model,
    quantize_tensor,
    dequantize_tensor,
    QuantizedLinear,
    QuantizedConv2d,
)
from .pruning import (
    prune_model,
    magnitude_prune,
    structured_prune,
    PrunedLinear,
)
from .compression import (
    compress_model,
    knowledge_distillation,
    weight_sharing,
)
from .benchmark import (
    benchmark_model,
    benchmark_layer,
    latency_test,
    throughput_test,
)
from .export import (
    export_onnx,
    export_tflite,
    export_miniten,
    load_onnx,
)
from .utils import (
    estimate_power,
    estimate_memory,
    count_flops,
    count_parameters,
)

__all__ = [
    # Quantization
    "quantize_model",
    "quantize_tensor",
    "dequantize_tensor",
    "QuantizedLinear",
    "QuantizedConv2d",
    # Pruning
    "prune_model",
    "magnitude_prune",
    "structured_prune",
    "PrunedLinear",
    # Compression
    "compress_model",
    "knowledge_distillation",
    "weight_sharing",
    # Benchmarking
    "benchmark_model",
    "benchmark_layer",
    "latency_test",
    "throughput_test",
    # Export
    "export_onnx",
    "export_tflite",
    "export_miniten",
    "load_onnx",
    # Utilities
    "estimate_power",
    "estimate_memory",
    "count_flops",
    "count_parameters",
]
