#!/usr/bin/env python
"""
TensorFlow Image Classification Example

Train the same image classifier using TensorFlow for comparison.
"""

import time
import numpy as np
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def create_synthetic_data(n_samples=1000, image_size=28, n_classes=10):
    """Create synthetic image classification data - same as other versions."""
    np.random.seed(42)
    
    X = np.zeros((n_samples, image_size, image_size, 1), dtype=np.float32)  # TF uses NHWC
    y = np.zeros(n_samples, dtype=np.int64)
    
    for i in range(n_samples):
        class_idx = i % n_classes
        y[i] = class_idx
        
        if class_idx == 0:
            X[i, image_size//2, :, 0] = 1.0
        elif class_idx == 1:
            X[i, :, image_size//2, 0] = 1.0
        elif class_idx == 2:
            for j in range(image_size):
                X[i, j, j, 0] = 1.0
        elif class_idx == 3:
            X[i, image_size//2, :, 0] = 1.0
            X[i, :, image_size//2, 0] = 1.0
        elif class_idx == 4:
            X[i, 5, 5:23, 0] = 1.0
            X[i, 22, 5:23, 0] = 1.0
            X[i, 5:23, 5, 0] = 1.0
            X[i, 5:23, 22, 0] = 1.0
        elif class_idx == 5:
            X[i, 10:18, 10:18, 0] = 1.0
        elif class_idx == 6:
            center = image_size // 2
            for h in range(image_size):
                for w in range(image_size):
                    if abs((h-center)**2 + (w-center)**2 - 64) < 20:
                        X[i, h, w, 0] = 1.0
        elif class_idx == 7:
            for j in range(image_size):
                X[i, j, j, 0] = 1.0
                X[i, j, image_size-1-j, 0] = 1.0
        elif class_idx == 8:
            X[i, :image_size//2, :, 0] = 1.0
        else:
            X[i, image_size//2:, :, 0] = 1.0
        
        X[i, :, :, 0] += np.random.randn(image_size, image_size).astype(np.float32) * 0.1
    
    X = np.clip(X, 0, 1)
    return X, y


def train_tensorflow():
    """Train a model using TensorFlow."""
    
    print("="*60)
    print("TensorFlow Image Classification")
    print("="*60)
    
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow not installed. Run: pip install tensorflow")
        return None
    
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs available: {len(gpus)}")
        device = "GPU"
    else:
        print("Using CPU")
        device = "CPU"
    
    # Create data
    print("\nCreating synthetic dataset...")
    X_train, y_train = create_synthetic_data(n_samples=800)
    X_test, y_test = create_synthetic_data(n_samples=200)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Define model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, 3, padding='same', activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Count parameters
    n_params = model.count_params()
    print(f"Model parameters: {n_params:,}")
    
    # Training
    epochs = 10
    batch_size = 32
    print(f"\nTraining for {epochs} epochs...")
    print("-"*60)
    
    # Custom training loop for timing
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(1000).batch(batch_size)
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    
    train_start = time.time()
    history = {"loss": [], "accuracy": []}
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0
        n_batches = 0
        
        for x_batch, y_batch in train_dataset:
            with tf.GradientTape() as tape:
                predictions = model(x_batch, training=True)
                loss = loss_fn(y_batch, predictions)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            epoch_loss += loss.numpy()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        epoch_time = time.time() - epoch_start
        
        # Evaluate
        test_predictions = model(X_test, training=False)
        test_pred_classes = tf.argmax(test_predictions, axis=1).numpy()
        accuracy = np.mean(test_pred_classes == y_test)
        
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
    test_batch = X_test[:32]
    
    # Warmup
    for _ in range(10):
        _ = model(test_batch, training=False)
    
    infer_start = time.time()
    for _ in range(n_infer):
        _ = model(test_batch, training=False)
    infer_time = time.time() - infer_start
    
    latency_ms = (infer_time / n_infer) * 1000
    print(f"Latency (32 samples): {latency_ms:.2f}ms")
    print(f"Throughput: {(n_infer * 32) / infer_time:.0f} samples/sec")
    
    return {
        "framework": "TensorFlow",
        "version": tf.__version__,
        "device": device,
        "train_time": train_time,
        "final_loss": history["loss"][-1],
        "final_accuracy": accuracy,
        "latency_ms": latency_ms,
        "epochs": epochs,
        "n_params": n_params,
    }


if __name__ == "__main__":
    results = train_tensorflow()
    if results:
        print("\n" + "="*60)
        print("Results Summary:")
        for key, value in results.items():
            print(f"  {key}: {value}")
