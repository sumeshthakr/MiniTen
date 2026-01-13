# Neural Network Benchmark Results

*Benchmark run on: 2026-01-13 19:10:35*

## Training Performance (100 epochs, 4 samples)

| Configuration | Python (ms) | MiniTen (ms) | Speedup |
|--------------|------------|-------------|--------|
| XOR (2-4-1) | 16.24 | 10.28 | 1.58x |
| XOR (2-16-1) | 15.28 | 10.52 | 1.45x |
| XOR (2-64-1) | 15.52 | 10.82 | 1.43x |

## Inference Performance

| Configuration | Python (μs) | MiniTen (μs) | Speedup |
|--------------|------------|-------------|--------|
| XOR (2-4-1) | 51.61 | 27.44 | 1.88x |
| XOR (2-16-1) | 51.56 | 27.63 | 1.87x |
| XOR (2-64-1) | 52.23 | 28.06 | 1.86x |
