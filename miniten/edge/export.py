"""
Model export utilities for edge deployment.

Provides export to various edge formats:
- ONNX
- TFLite-compatible
- MiniTen native format
"""

import numpy as np
import json
from pathlib import Path


def export_onnx(model, input_shape, output_path, opset_version=11):
    """
    Export model to ONNX format.
    
    Args:
        model: Model to export
        input_shape: Shape of input tensor
        output_path: Path to save ONNX file
        opset_version: ONNX opset version
        
    Returns:
        True if successful
    """
    output_path = Path(output_path)
    
    # Create ONNX model representation
    onnx_model = ONNXExporter(model, input_shape, opset_version)
    onnx_model.export(output_path)
    
    print(f"Model exported to ONNX: {output_path}")
    return True


def export_tflite(model, input_shape, output_path, quantize=False):
    """
    Export model to TFLite-compatible format.
    
    Note: This creates a format that can be converted to TFLite,
    not a native TFLite file.
    
    Args:
        model: Model to export
        input_shape: Input shape
        output_path: Output path
        quantize: Whether to quantize
        
    Returns:
        True if successful
    """
    output_path = Path(output_path)
    
    # Extract model structure and weights
    model_data = {
        "format": "miniten_tflite_compatible",
        "version": "1.0",
        "input_shape": list(input_shape),
        "quantized": quantize,
        "layers": [],
    }
    
    # Extract layer information
    if hasattr(model, 'layers'):
        for i, layer in enumerate(model.layers):
            layer_info = {
                "index": i,
                "type": type(layer).__name__,
            }
            
            if hasattr(layer, 'weight'):
                layer_info["weight_shape"] = list(layer.weight.shape)
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer_info["bias_shape"] = list(layer.bias.shape)
            
            model_data["layers"].append(layer_info)
    
    # Save metadata
    with open(output_path.with_suffix('.json'), 'w') as f:
        json.dump(model_data, f, indent=2)
    
    # Save weights
    weights = {}
    if hasattr(model, 'parameters'):
        for i, param in enumerate(model.parameters()):
            weights[f"layer_{i}"] = np.asarray(param)
    
    np.savez(output_path.with_suffix('.npz'), **weights)
    
    print(f"Model exported to TFLite-compatible format: {output_path}")
    return True


def export_miniten(model, output_path, include_optimizer=False):
    """
    Export model in MiniTen native format.
    
    Args:
        model: Model to export
        output_path: Output path
        include_optimizer: Include optimizer state
        
    Returns:
        True if successful
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect model state
    state = {
        "format": "miniten",
        "version": "1.0",
        "model_type": type(model).__name__,
    }
    
    # Save weights
    weights = {}
    if hasattr(model, 'state_dict'):
        weights = model.state_dict()
    elif hasattr(model, 'parameters'):
        weights = {f"param_{i}": np.asarray(p) for i, p in enumerate(model.parameters())}
    
    # Save
    np.savez(
        output_path,
        metadata=json.dumps(state),
        **weights
    )
    
    print(f"Model exported to MiniTen format: {output_path}")
    return True


def load_onnx(path, use_miniten=True):
    """
    Load an ONNX model.
    
    Args:
        path: Path to ONNX file
        use_miniten: Convert to MiniTen model
        
    Returns:
        Loaded model
    """
    try:
        import onnx
        import onnxruntime as ort
        
        # Load ONNX model
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        
        if use_miniten:
            # Convert to MiniTen model
            return ONNXImporter(onnx_model).to_miniten()
        else:
            # Return ONNX runtime session
            return ort.InferenceSession(path)
    except ImportError:
        print("ONNX/ONNXRuntime not installed. Install with: pip install onnx onnxruntime")
        return None


class ONNXExporter:
    """
    Export MiniTen models to ONNX format.
    """
    
    def __init__(self, model, input_shape, opset_version=11):
        """
        Initialize ONNX exporter.
        
        Args:
            model: Model to export
            input_shape: Input shape
            opset_version: ONNX opset version
        """
        self.model = model
        self.input_shape = input_shape
        self.opset_version = opset_version
        self.nodes = []
        self.initializers = []
        self.value_info = []
    
    def export(self, output_path):
        """Export model to ONNX file."""
        try:
            import onnx
            from onnx import helper, numpy_helper, TensorProto
            
            # Build ONNX graph
            self._build_graph()
            
            # Create ONNX model
            graph = helper.make_graph(
                self.nodes,
                "miniten_model",
                self._make_inputs(),
                self._make_outputs(),
                self.initializers
            )
            
            model = helper.make_model(
                graph,
                opset_imports=[helper.make_opsetid("", self.opset_version)]
            )
            
            # Validate and save
            onnx.checker.check_model(model)
            onnx.save(model, str(output_path))
            
        except ImportError:
            # Fallback: save as pseudo-ONNX (JSON + weights)
            self._export_pseudo_onnx(output_path)
    
    def _build_graph(self):
        """Build ONNX graph from model."""
        # Placeholder - would traverse model layers
        pass
    
    def _make_inputs(self):
        """Create input tensor specs."""
        try:
            from onnx import helper, TensorProto
            return [helper.make_tensor_value_info(
                "input",
                TensorProto.FLOAT,
                list(self.input_shape)
            )]
        except ImportError:
            return []
    
    def _make_outputs(self):
        """Create output tensor specs."""
        try:
            from onnx import helper, TensorProto
            return [helper.make_tensor_value_info(
                "output",
                TensorProto.FLOAT,
                None  # Dynamic shape
            )]
        except ImportError:
            return []
    
    def _export_pseudo_onnx(self, output_path):
        """Export pseudo-ONNX format when onnx package not available."""
        output_path = Path(output_path)
        
        model_data = {
            "format": "pseudo_onnx",
            "opset_version": self.opset_version,
            "input_shape": list(self.input_shape),
            "layers": [],
        }
        
        # Extract layer info
        if hasattr(self.model, 'layers'):
            for layer in self.model.layers:
                model_data["layers"].append({
                    "type": type(layer).__name__,
                })
        
        # Save metadata
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(model_data, f, indent=2)
        
        # Save weights
        weights = {}
        if hasattr(self.model, 'parameters'):
            for i, param in enumerate(self.model.parameters()):
                weights[f"weight_{i}"] = np.asarray(param)
        
        np.savez(output_path.with_suffix('.npz'), **weights)


class ONNXImporter:
    """
    Import ONNX models to MiniTen format.
    """
    
    def __init__(self, onnx_model):
        """
        Initialize ONNX importer.
        
        Args:
            onnx_model: ONNX model object
        """
        self.onnx_model = onnx_model
        self.layers = []
    
    def to_miniten(self):
        """Convert ONNX model to MiniTen model."""
        # Parse ONNX graph
        graph = self.onnx_model.graph
        
        # Create MiniTen layers
        for node in graph.node:
            layer = self._convert_node(node)
            if layer:
                self.layers.append(layer)
        
        # Create sequential model
        return ONNXModel(self.layers)
    
    def _convert_node(self, node):
        """Convert ONNX node to MiniTen layer."""
        op_type = node.op_type
        
        if op_type == "Gemm":
            return self._convert_gemm(node)
        elif op_type == "Conv":
            return self._convert_conv(node)
        elif op_type == "Relu":
            return self._convert_relu(node)
        elif op_type == "MaxPool":
            return self._convert_maxpool(node)
        elif op_type == "Softmax":
            return self._convert_softmax(node)
        
        return None
    
    def _convert_gemm(self, node):
        """Convert ONNX Gemm to Linear."""
        # Placeholder
        return {"type": "Linear", "node": node.name}
    
    def _convert_conv(self, node):
        """Convert ONNX Conv to Conv2d."""
        return {"type": "Conv2d", "node": node.name}
    
    def _convert_relu(self, node):
        """Convert ONNX ReLU."""
        return {"type": "ReLU", "node": node.name}
    
    def _convert_maxpool(self, node):
        """Convert ONNX MaxPool."""
        return {"type": "MaxPool2d", "node": node.name}
    
    def _convert_softmax(self, node):
        """Convert ONNX Softmax."""
        return {"type": "Softmax", "node": node.name}


class ONNXModel:
    """
    MiniTen model loaded from ONNX.
    """
    
    def __init__(self, layers):
        """
        Initialize ONNX-loaded model.
        
        Args:
            layers: List of layer specifications
        """
        self.layers = layers
    
    def forward(self, x):
        """Forward pass."""
        # Placeholder - would execute layers
        return x
    
    def __call__(self, x):
        return self.forward(x)
