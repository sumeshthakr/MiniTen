#!/usr/bin/env python
"""
PyTorch Image Classification Example

Train the same image classifier using PyTorch for comparison.
"""

import time
import numpy as np


def create_synthetic_data(n_samples=1000, image_size=28, n_classes=10):
    """Create synthetic image classification data - same as MiniTen version."""
    np.random.seed(42)
    
    X = np.zeros((n_samples, 1, image_size, image_size), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    
    for i in range(n_samples):
        class_idx = i % n_classes
        y[i] = class_idx
        
        if class_idx == 0:
            X[i, 0, image_size//2, :] = 1.0
        elif class_idx == 1:
            X[i, 0, :, image_size//2] = 1.0
        elif class_idx == 2:
            for j in range(image_size):
                X[i, 0, j, j] = 1.0
        elif class_idx == 3:
            X[i, 0, image_size//2, :] = 1.0
            X[i, 0, :, image_size//2] = 1.0
        elif class_idx == 4:
            X[i, 0, 5, 5:23] = 1.0
            X[i, 0, 22, 5:23] = 1.0
            X[i, 0, 5:23, 5] = 1.0
            X[i, 0, 5:23, 22] = 1.0
        elif class_idx == 5:
            X[i, 0, 10:18, 10:18] = 1.0
        elif class_idx == 6:
            center = image_size // 2
            for h in range(image_size):
                for w in range(image_size):
                    if abs((h-center)**2 + (w-center)**2 - 64) < 20:
                        X[i, 0, h, w] = 1.0
        elif class_idx == 7:
            for j in range(image_size):
                X[i, 0, j, j] = 1.0
                X[i, 0, j, image_size-1-j] = 1.0
        elif class_idx == 8:
            X[i, 0, :image_size//2, :] = 1.0
        else:
            X[i, 0, image_size//2:, :] = 1.0
        
        X[i] += np.random.randn(1, image_size, image_size).astype(np.float32) * 0.1
    
    X = np.clip(X, 0, 1)
    return X, y


def train_pytorch():
    """Train a model using PyTorch."""
    
    print("="*60)
    print("PyTorch Image Classification")
    print("="*60)
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("PyTorch not installed. Run: pip install torch")
        return None
    
    print(f"PyTorch version: {torch.__version__}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create data
    print("\nCreating synthetic dataset...")
    X_train, y_train = create_synthetic_data(n_samples=800)
    X_test, y_test = create_synthetic_data(n_samples=200)
    
    # Convert to tensors
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Define model
    class SimpleCNN(nn.Module):
        def __init__(self, n_classes=10):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc = nn.Linear(16 * 7 * 7, n_classes)
            
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))  # 28->14
            x = self.pool(torch.relu(self.conv2(x)))  # 14->7
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = SimpleCNN(n_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Training
    epochs = 10
    print(f"\nTraining for {epochs} epochs...")
    print("-"*60)
    
    train_start = time.time()
    history = {"loss": [], "accuracy": []}
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        epoch_time = time.time() - epoch_start
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t.to(device))
            predictions = test_outputs.argmax(dim=1).cpu().numpy()
        
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
    model.eval()
    n_infer = 100
    
    with torch.no_grad():
        test_batch = X_test_t[:32].to(device)
        
        # Warmup
        for _ in range(10):
            _ = model(test_batch)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        infer_start = time.time()
        for _ in range(n_infer):
            _ = model(test_batch)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        infer_time = time.time() - infer_start
    
    latency_ms = (infer_time / n_infer) * 1000
    print(f"Latency (32 samples): {latency_ms:.2f}ms")
    print(f"Throughput: {(n_infer * 32) / infer_time:.0f} samples/sec")
    
    return {
        "framework": "PyTorch",
        "version": torch.__version__,
        "device": str(device),
        "train_time": train_time,
        "final_loss": history["loss"][-1],
        "final_accuracy": accuracy,
        "latency_ms": latency_ms,
        "epochs": epochs,
        "n_params": n_params,
    }


if __name__ == "__main__":
    results = train_pytorch()
    if results:
        print("\n" + "="*60)
        print("Results Summary:")
        for key, value in results.items():
            print(f"  {key}: {value}")
