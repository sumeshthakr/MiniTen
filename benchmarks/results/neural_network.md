# Neural Network Benchmark Results

*Benchmark run on: 2026-01-13 19:15:36*

## Training Performance (100 epochs, 4 samples)

| Configuration | Python (ms) | MiniTen (ms) | Speedup |
|--------------|------------|-------------|--------|
| XOR (2-4-1) | 12.25 | 8.53 | 1.44x |
| XOR (2-16-1) | 11.25 | 8.65 | 1.30x |
| XOR (2-64-1) | 11.51 | 8.93 | 1.29x |

## Inference Performance

| Configuration | Python (μs) | MiniTen (μs) | Speedup |
|--------------|------------|-------------|--------|
| XOR (2-4-1) | 51.31 | 27.61 | 1.86x |
| XOR (2-16-1) | 51.92 | 27.67 | 1.88x |
| XOR (2-64-1) | 52.48 | 28.21 | 1.86x |
