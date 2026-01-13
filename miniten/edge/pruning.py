"""
Model pruning utilities for edge deployment.

Provides weight pruning for reducing model size and improving
inference efficiency on edge devices.
"""

import numpy as np


def magnitude_prune(weights, sparsity):
    """
    Prune weights by magnitude (set smallest weights to zero).
    
    Args:
        weights: Weight array to prune
        sparsity: Target sparsity (fraction of weights to prune, 0-1)
        
    Returns:
        (pruned_weights, mask)
    """
    weights = np.asarray(weights)
    
    if sparsity <= 0:
        return weights.copy(), np.ones_like(weights, dtype=bool)
    if sparsity >= 1:
        return np.zeros_like(weights), np.zeros_like(weights, dtype=bool)
    
    # Flatten for threshold computation
    flat_weights = np.abs(weights).flatten()
    num_weights = len(flat_weights)
    num_prune = int(sparsity * num_weights)
    
    # Find threshold
    sorted_weights = np.sort(flat_weights)
    threshold = sorted_weights[num_prune] if num_prune < num_weights else float('inf')
    
    # Create mask
    mask = np.abs(weights) >= threshold
    
    # Apply mask
    pruned = weights * mask
    
    return pruned, mask


def structured_prune(weights, sparsity, dim=0):
    """
    Structured pruning (prune entire rows/columns/filters).
    
    Args:
        weights: Weight array to prune
        sparsity: Target sparsity
        dim: Dimension along which to prune
        
    Returns:
        (pruned_weights, mask, indices_to_keep)
    """
    weights = np.asarray(weights)
    
    # Calculate importance along the dimension
    importance = np.sum(np.abs(weights), axis=tuple(
        i for i in range(len(weights.shape)) if i != dim
    ))
    
    num_elements = len(importance)
    num_prune = int(sparsity * num_elements)
    
    # Find indices to keep
    indices = np.argsort(importance)
    indices_to_keep = indices[num_prune:]
    
    # Create mask
    mask = np.zeros(num_elements, dtype=bool)
    mask[indices_to_keep] = True
    
    # Apply pruning
    if dim == 0:
        pruned = weights[indices_to_keep]
    else:
        pruned = np.take(weights, indices_to_keep, axis=dim)
    
    return pruned, mask, indices_to_keep


def prune_model(model, sparsity, method='magnitude'):
    """
    Prune an entire model.
    
    Args:
        model: Model to prune
        sparsity: Target sparsity
        method: Pruning method ('magnitude', 'structured')
        
    Returns:
        Pruned model
    """
    pruned_model = PrunedModel(model, sparsity, method)
    return pruned_model


class PrunedModel:
    """
    Wrapper for pruned models.
    
    Maintains pruning masks and handles sparse computation.
    """
    
    def __init__(self, model, sparsity, method='magnitude'):
        """
        Initialize pruned model.
        
        Args:
            model: Original model
            sparsity: Target sparsity
            method: Pruning method
        """
        self.original_model = model
        self.sparsity = sparsity
        self.method = method
        self.pruned_weights = {}
        self.masks = {}
        
        self._prune_weights()
    
    def _prune_weights(self):
        """Prune all model weights."""
        if hasattr(self.original_model, 'parameters'):
            for name, param in enumerate(self.original_model.parameters()):
                if hasattr(param, 'shape') and len(param.shape) >= 2:
                    if self.method == 'magnitude':
                        pruned, mask = magnitude_prune(param, self.sparsity)
                    else:
                        pruned, mask, _ = structured_prune(param, self.sparsity)
                    
                    self.pruned_weights[name] = pruned
                    self.masks[name] = mask
    
    def forward(self, x):
        """Forward pass with pruned weights."""
        # Placeholder - would use sparse computation
        return self.original_model.forward(x)
    
    def actual_sparsity(self):
        """Calculate actual sparsity achieved."""
        total_weights = 0
        zero_weights = 0
        
        for weights in self.pruned_weights.values():
            total_weights += weights.size
            zero_weights += np.sum(weights == 0)
        
        return zero_weights / total_weights if total_weights > 0 else 0
    
    def compression_ratio(self):
        """Calculate compression ratio."""
        sparsity = self.actual_sparsity()
        return 1 / (1 - sparsity) if sparsity < 1 else float('inf')
    
    def save(self, path):
        """Save pruned model."""
        np.savez(
            path,
            weights=self.pruned_weights,
            masks=self.masks,
            sparsity=self.sparsity,
            method=self.method,
        )


class PrunedLinear:
    """
    Linear layer with weight pruning.
    
    Maintains a mask for pruned weights and supports
    gradual pruning during training.
    """
    
    def __init__(self, in_features, out_features, sparsity=0.0):
        """
        Initialize pruned linear layer.
        
        Args:
            in_features: Input features
            out_features: Output features
            sparsity: Initial sparsity
        """
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        
        # Initialize weights
        std = np.sqrt(2.0 / in_features)
        self.weight = np.random.randn(out_features, in_features).astype(np.float64) * std
        self.bias = np.zeros(out_features, dtype=np.float64)
        
        # Pruning mask
        self.mask = np.ones_like(self.weight, dtype=bool)
        
        if sparsity > 0:
            self.apply_pruning(sparsity)
    
    def apply_pruning(self, sparsity):
        """
        Apply pruning to the layer.
        
        Args:
            sparsity: Target sparsity
        """
        self.sparsity = sparsity
        self.weight, self.mask = magnitude_prune(self.weight, sparsity)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Apply mask to ensure pruned weights stay zero
        effective_weight = self.weight * self.mask
        output = np.dot(x, effective_weight.T) + self.bias
        return output
    
    def backward(self, grad_output, cached_input):
        """
        Backward pass with masked gradients.
        
        Args:
            grad_output: Gradient from next layer
            cached_input: Cached input from forward pass
            
        Returns:
            grad_input
        """
        # Compute gradients
        grad_weight = np.dot(grad_output.T, cached_input)
        grad_bias = np.sum(grad_output, axis=0)
        grad_input = np.dot(grad_output, self.weight * self.mask)
        
        # Mask gradient for pruned weights
        grad_weight = grad_weight * self.mask
        
        return grad_input, grad_weight, grad_bias
    
    def update_parameters(self, grad_weight, grad_bias, learning_rate):
        """Update parameters while maintaining sparsity."""
        self.weight -= learning_rate * grad_weight
        self.bias -= learning_rate * grad_bias
        
        # Re-apply mask to maintain sparsity
        self.weight = self.weight * self.mask


class GradualPruning:
    """
    Gradual magnitude pruning scheduler.
    
    Increases sparsity over training according to a schedule.
    """
    
    def __init__(self, initial_sparsity=0.0, final_sparsity=0.9,
                 start_step=0, end_step=1000, frequency=100):
        """
        Initialize gradual pruning.
        
        Args:
            initial_sparsity: Starting sparsity
            final_sparsity: Final sparsity
            start_step: Step to start pruning
            end_step: Step to reach final sparsity
            frequency: How often to update pruning
        """
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.start_step = start_step
        self.end_step = end_step
        self.frequency = frequency
    
    def get_sparsity(self, step):
        """
        Get target sparsity for current step.
        
        Uses a cubic schedule for smooth transition.
        """
        if step < self.start_step:
            return self.initial_sparsity
        if step >= self.end_step:
            return self.final_sparsity
        
        # Cubic interpolation
        progress = (step - self.start_step) / (self.end_step - self.start_step)
        sparsity = self.final_sparsity + (self.initial_sparsity - self.final_sparsity) * \
                   (1 - progress) ** 3
        
        return sparsity
    
    def should_update(self, step):
        """Check if pruning should be updated at this step."""
        return step >= self.start_step and step % self.frequency == 0
