"""
Model compression utilities for edge deployment.

Provides various compression techniques for reducing model size
while maintaining accuracy.
"""

import numpy as np


def compress_model(model, method='quantize', **kwargs):
    """
    Compress a model using specified method.
    
    Args:
        model: Model to compress
        method: Compression method ('quantize', 'prune', 'distill', 'share')
        **kwargs: Method-specific arguments
        
    Returns:
        Compressed model
    """
    if method == 'quantize':
        from .quantization import quantize_model
        return quantize_model(model, **kwargs)
    elif method == 'prune':
        from .pruning import prune_model
        return prune_model(model, **kwargs)
    elif method == 'distill':
        return knowledge_distillation(model, **kwargs)
    elif method == 'share':
        return weight_sharing(model, **kwargs)
    else:
        raise ValueError(f"Unknown compression method: {method}")


def knowledge_distillation(student_model, teacher_model=None, 
                          temperature=3.0, alpha=0.5):
    """
    Knowledge distillation from teacher to student model.
    
    Args:
        student_model: Student model to train
        teacher_model: Teacher model (larger, pre-trained)
        temperature: Softmax temperature for soft targets
        alpha: Weight for distillation loss vs hard labels
        
    Returns:
        DistillationTrainer object
    """
    return DistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        temperature=temperature,
        alpha=alpha
    )


class DistillationTrainer:
    """
    Trainer for knowledge distillation.
    
    Trains a smaller student model to mimic a larger teacher model.
    """
    
    def __init__(self, student_model, teacher_model=None,
                 temperature=3.0, alpha=0.5):
        """
        Initialize distillation trainer.
        
        Args:
            student_model: Student model
            teacher_model: Teacher model
            temperature: Temperature for soft targets
            alpha: Weight for distillation loss
        """
        self.student = student_model
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha = alpha
    
    def distillation_loss(self, student_logits, teacher_logits, labels):
        """
        Compute distillation loss.
        
        Combines soft target loss and hard label loss.
        
        Args:
            student_logits: Student model output
            teacher_logits: Teacher model output
            labels: Ground truth labels
            
        Returns:
            Combined loss
        """
        # Soft target loss (KL divergence)
        soft_targets = self._softmax_with_temperature(teacher_logits, self.temperature)
        soft_predictions = self._log_softmax_with_temperature(student_logits, self.temperature)
        soft_loss = -np.mean(np.sum(soft_targets * soft_predictions, axis=-1))
        
        # Hard label loss (cross entropy)
        hard_loss = self._cross_entropy(student_logits, labels)
        
        # Combined loss
        loss = self.alpha * soft_loss * (self.temperature ** 2) + (1 - self.alpha) * hard_loss
        
        return loss
    
    def _softmax_with_temperature(self, logits, temperature):
        """Softmax with temperature."""
        scaled = logits / temperature
        exp_scaled = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=-1, keepdims=True)
    
    def _log_softmax_with_temperature(self, logits, temperature):
        """Log softmax with temperature."""
        scaled = logits / temperature
        return scaled - np.log(np.sum(np.exp(scaled - np.max(scaled, axis=-1, keepdims=True)), axis=-1, keepdims=True))
    
    def _cross_entropy(self, logits, labels):
        """Cross entropy loss."""
        probs = self._softmax_with_temperature(logits, 1.0)
        n = len(labels)
        log_probs = -np.log(probs[np.arange(n), labels] + 1e-10)
        return np.mean(log_probs)
    
    def train_step(self, x, labels):
        """
        Single training step.
        
        Args:
            x: Input data
            labels: Ground truth labels
            
        Returns:
            Loss value
        """
        # Get teacher predictions (no gradient)
        if self.teacher is not None:
            teacher_logits = self.teacher.forward(x)
        else:
            teacher_logits = None
        
        # Get student predictions
        student_logits = self.student.forward(x)
        
        # Compute loss
        if teacher_logits is not None:
            loss = self.distillation_loss(student_logits, teacher_logits, labels)
        else:
            loss = self._cross_entropy(student_logits, labels)
        
        return loss


def weight_sharing(model, num_clusters=16):
    """
    Apply weight sharing (weight clustering) to a model.
    
    Groups weights into clusters and uses cluster centroids
    to reduce storage requirements.
    
    Args:
        model: Model to compress
        num_clusters: Number of weight clusters
        
    Returns:
        WeightSharedModel
    """
    return WeightSharedModel(model, num_clusters)


class WeightSharedModel:
    """
    Model with weight sharing for compression.
    
    Weights are quantized to cluster centroids, reducing
    the number of unique weight values.
    """
    
    def __init__(self, model, num_clusters=16):
        """
        Initialize weight-shared model.
        
        Args:
            model: Original model
            num_clusters: Number of clusters
        """
        self.original_model = model
        self.num_clusters = num_clusters
        self.cluster_indices = {}
        self.centroids = {}
        
        self._cluster_weights()
    
    def _cluster_weights(self):
        """Cluster weights using k-means."""
        if not hasattr(self.original_model, 'parameters'):
            return
        
        for name, param in enumerate(self.original_model.parameters()):
            if hasattr(param, 'shape') and len(param.shape) >= 1:
                weights = np.asarray(param).flatten()
                
                # Simple k-means clustering
                centroids, indices = self._kmeans(weights, self.num_clusters)
                
                self.centroids[name] = centroids
                self.cluster_indices[name] = indices.reshape(param.shape)
    
    def _kmeans(self, data, k, max_iters=100):
        """
        Simple k-means clustering.
        
        Args:
            data: 1D array of values
            k: Number of clusters
            max_iters: Maximum iterations
            
        Returns:
            (centroids, assignments)
        """
        # Initialize centroids with quantiles
        percentiles = np.linspace(0, 100, k)
        centroids = np.percentile(data, percentiles)
        
        for _ in range(max_iters):
            # Assign points to nearest centroid
            distances = np.abs(data[:, np.newaxis] - centroids[np.newaxis, :])
            assignments = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                cluster_points = data[assignments == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = np.mean(cluster_points)
                else:
                    new_centroids[i] = centroids[i]
            
            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids
        
        return centroids, assignments
    
    def get_compressed_weights(self, name):
        """Get compressed weight representation."""
        if name not in self.centroids:
            return None
        
        centroids = self.centroids[name]
        indices = self.cluster_indices[name]
        
        # Reconstruct weights from indices
        return centroids[indices]
    
    def compression_ratio(self):
        """Calculate compression ratio."""
        # Original: 32 bits per weight
        # Compressed: log2(num_clusters) bits per weight + centroid storage
        bits_per_weight = np.log2(self.num_clusters)
        return 32 / bits_per_weight
    
    def forward(self, x):
        """Forward pass with shared weights."""
        # Placeholder - would use shared weights
        return self.original_model.forward(x)


class LowRankApproximation:
    """
    Low-rank matrix approximation for weight compression.
    
    Decomposes weight matrices into products of smaller matrices
    using SVD.
    """
    
    def __init__(self, rank_ratio=0.5):
        """
        Initialize low-rank approximation.
        
        Args:
            rank_ratio: Fraction of singular values to keep
        """
        self.rank_ratio = rank_ratio
    
    def compress_matrix(self, weight_matrix):
        """
        Compress a weight matrix using SVD.
        
        Args:
            weight_matrix: 2D weight matrix
            
        Returns:
            (U, S, V) decomposition with reduced rank
        """
        U, S, Vt = np.linalg.svd(weight_matrix, full_matrices=False)
        
        # Determine rank to keep
        rank = max(1, int(len(S) * self.rank_ratio))
        
        # Truncate
        U_reduced = U[:, :rank]
        S_reduced = S[:rank]
        Vt_reduced = Vt[:rank, :]
        
        return U_reduced, S_reduced, Vt_reduced
    
    def reconstruct(self, U, S, Vt):
        """Reconstruct matrix from SVD components."""
        return np.dot(U * S, Vt)
    
    def compression_ratio(self, original_shape, rank):
        """Calculate compression ratio."""
        m, n = original_shape
        original_params = m * n
        compressed_params = m * rank + rank + rank * n
        return original_params / compressed_params
