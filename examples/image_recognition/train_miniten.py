#!/usr/bin/env python
"""
MiniTen Image Classification Example

Train a simple image classifier using MiniTen's optimized Cython layers.
"""

import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def create_synthetic_data(n_samples=1000, image_size=28, n_classes=10):
    """
    Create synthetic image classification data.
    
    Similar to MNIST but simplified for demonstration.
    """
    np.random.seed(42)
    
    # Create simple patterns for each class
    X = np.zeros((n_samples, 1, image_size, image_size), dtype=np.float64)
    y = np.zeros(n_samples, dtype=np.int64)
    
    for i in range(n_samples):
        class_idx = i % n_classes
        y[i] = class_idx
        
        # Create a simple pattern based on class
        # Each class has a different pattern
        if class_idx == 0:
            # Horizontal line
            X[i, 0, image_size//2, :] = 1.0
        elif class_idx == 1:
            # Vertical line
            X[i, 0, :, image_size//2] = 1.0
        elif class_idx == 2:
            # Diagonal
            for j in range(image_size):
                X[i, 0, j, j] = 1.0
        elif class_idx == 3:
            # Cross
            X[i, 0, image_size//2, :] = 1.0
            X[i, 0, :, image_size//2] = 1.0
        elif class_idx == 4:
            # Box outline
            X[i, 0, 5, 5:23] = 1.0
            X[i, 0, 22, 5:23] = 1.0
            X[i, 0, 5:23, 5] = 1.0
            X[i, 0, 5:23, 22] = 1.0
        elif class_idx == 5:
            # Filled box
            X[i, 0, 10:18, 10:18] = 1.0
        elif class_idx == 6:
            # Circle (approximate)
            center = image_size // 2
            for h in range(image_size):
                for w in range(image_size):
                    if abs((h-center)**2 + (w-center)**2 - 64) < 20:
                        X[i, 0, h, w] = 1.0
        elif class_idx == 7:
            # X pattern
            for j in range(image_size):
                X[i, 0, j, j] = 1.0
                X[i, 0, j, image_size-1-j] = 1.0
        elif class_idx == 8:
            # Top half
            X[i, 0, :image_size//2, :] = 1.0
        else:
            # Bottom half
            X[i, 0, image_size//2:, :] = 1.0
        
        # Add noise
        X[i] += np.random.randn(1, image_size, image_size) * 0.1
    
    # Normalize
    X = np.clip(X, 0, 1)
    
    return X, y


def train_miniten():
    """Train a model using MiniTen's Cython layers."""
    
    print("="*60)
    print("MiniTen Image Classification")
    print("="*60)
    
    # Try to import MiniTen layers
    try:
        from miniten.nn.layers_impl import Linear, relu_forward, relu_backward
        from miniten.nn.layers_impl import softmax_cross_entropy_loss
        from miniten.nn.cnn_impl import Conv2d, maxpool2d_forward, global_avgpool2d_forward
        use_cython = True
        print("Using Cython-optimized layers")
    except ImportError:
        print("Cython layers not built. Using NumPy fallback.")
        use_cython = False
    
    # Create data
    print("\nCreating synthetic dataset...")
    X_train, y_train = create_synthetic_data(n_samples=800)
    X_test, y_test = create_synthetic_data(n_samples=200)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Image shape: {X_train.shape[1:]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Model parameters
    image_size = 28
    n_classes = 10
    learning_rate = 0.01
    epochs = 10
    batch_size = 32
    
    if use_cython:
        # Build model using Cython layers
        conv1 = Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        conv2 = Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        fc = Linear(16 * 7 * 7, n_classes)
        
        def forward(x, training=True):
            # Conv1 -> ReLU -> MaxPool
            out = conv1.forward(x)
            out = np.maximum(out, 0)  # ReLU
            out, pool_idx1 = maxpool2d_forward(out, 2)  # 28->14
            
            # Conv2 -> ReLU -> MaxPool
            out = conv2.forward(out)
            out = np.maximum(out, 0)  # ReLU
            out, pool_idx2 = maxpool2d_forward(out, 2)  # 14->7
            
            # Flatten
            batch_size = out.shape[0]
            out = out.reshape(batch_size, -1)
            
            # FC
            out = fc.forward(out)
            
            return out
        
        def train_step(x_batch, y_batch):
            # Forward
            logits = forward(x_batch)
            
            # Loss
            loss, grad = softmax_cross_entropy_loss(logits, y_batch)
            
            # Backward through FC
            grad_fc = fc.backward(grad)
            fc.update_parameters(learning_rate)
            
            return loss
        
    else:
        # NumPy fallback implementation
        np.random.seed(42)
        W1 = np.random.randn(784, 128) * 0.1
        b1 = np.zeros(128)
        W2 = np.random.randn(128, n_classes) * 0.1
        b2 = np.zeros(n_classes)
        
        def forward(x, training=True):
            batch_size = x.shape[0]
            x_flat = x.reshape(batch_size, -1)
            
            # FC1 -> ReLU
            h1 = np.dot(x_flat, W1) + b1
            h1 = np.maximum(h1, 0)
            
            # FC2
            logits = np.dot(h1, W2) + b2
            
            return logits, x_flat, h1
        
        def softmax(x):
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        def train_step(x_batch, y_batch):
            nonlocal W1, b1, W2, b2
            
            logits, x_flat, h1 = forward(x_batch)
            
            # Softmax + Cross entropy
            probs = softmax(logits)
            batch_size = x_batch.shape[0]
            
            loss = -np.mean(np.log(probs[np.arange(batch_size), y_batch] + 1e-10))
            
            # Backward
            grad_logits = probs.copy()
            grad_logits[np.arange(batch_size), y_batch] -= 1
            grad_logits /= batch_size
            
            # Grad W2, b2
            grad_W2 = np.dot(h1.T, grad_logits)
            grad_b2 = np.sum(grad_logits, axis=0)
            
            # Grad h1
            grad_h1 = np.dot(grad_logits, W2.T)
            grad_h1[h1 <= 0] = 0  # ReLU backward
            
            # Grad W1, b1
            grad_W1 = np.dot(x_flat.T, grad_h1)
            grad_b1 = np.sum(grad_h1, axis=0)
            
            # Update
            W2 -= learning_rate * grad_W2
            b2 -= learning_rate * grad_b2
            W1 -= learning_rate * grad_W1
            b1 -= learning_rate * grad_b1
            
            return loss
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    print("-"*60)
    
    train_start = time.time()
    history = {"loss": [], "accuracy": []}
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0
        n_batches = 0
        
        # Shuffle
        indices = np.random.permutation(len(X_train))
        
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            x_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            
            loss = train_step(x_batch, y_batch)
            epoch_loss += loss
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        epoch_time = time.time() - epoch_start
        
        # Evaluate
        if use_cython:
            test_logits = forward(X_test, training=False)
            predictions = np.argmax(test_logits, axis=1)
        else:
            test_logits, _, _ = forward(X_test, training=False)
            predictions = np.argmax(test_logits, axis=1)
        
        accuracy = np.mean(predictions == y_test)
        
        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)
        
        print(f"Epoch {epoch+1:2d}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.2%}, Time={epoch_time:.2f}s")
    
    train_time = time.time() - train_start
    
    print("-"*60)
    print(f"Training completed in {train_time:.2f}s")
    print(f"Final Accuracy: {accuracy:.2%}")
    
    # Inference benchmark
    print("\nInference Benchmark:")
    n_infer = 100
    infer_start = time.time()
    for _ in range(n_infer):
        if use_cython:
            _ = forward(X_test[:32], training=False)
        else:
            _, _, _ = forward(X_test[:32], training=False)
    infer_time = time.time() - infer_start
    
    latency_ms = (infer_time / n_infer) * 1000
    print(f"Latency (32 samples): {latency_ms:.2f}ms")
    print(f"Throughput: {(n_infer * 32) / infer_time:.0f} samples/sec")
    
    return {
        "framework": "MiniTen",
        "train_time": train_time,
        "final_loss": history["loss"][-1],
        "final_accuracy": accuracy,
        "latency_ms": latency_ms,
        "epochs": epochs,
        "use_cython": use_cython,
    }


if __name__ == "__main__":
    results = train_miniten()
    print("\n" + "="*60)
    print("Results Summary:")
    for key, value in results.items():
        print(f"  {key}: {value}")
