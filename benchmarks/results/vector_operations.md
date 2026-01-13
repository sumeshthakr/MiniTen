# Vector Operations Benchmark Results

## Benchmark Results

*Benchmark run on: 2026-01-13 19:08:37*

*System: Python 3.12.3*


### Vector Addition

| Vector Size | Numpy (μs) | Miniten (μs) | MiniTen vs NumPy |
|------------|------------:|------------:|----------------:|
| 100 | 1.09 | 1.90 | 1.75x slower |
| 1,000 | 0.92 | 1.75 | 1.90x slower |
| 10,000 | 4.23 | 7.82 | 1.85x slower |
| 100,000 | 81.00 | 94.43 | 1.17x slower |

### Dot Product

| Vector Size | Numpy (μs) | Miniten (μs) | MiniTen vs NumPy |
|------------|------------:|------------:|----------------:|
| 100 | 1.59 | 1.39 | 1.15x faster |
| 1,000 | 0.92 | 4.51 | 4.92x slower |
| 10,000 | 3.40 | 40.83 | 12.01x slower |
| 100,000 | 14.36 | 405.15 | 28.21x slower |

### Element-wise Multiply

| Vector Size | Numpy (μs) | Miniten (μs) | MiniTen vs NumPy |
|------------|------------:|------------:|----------------:|
| 100 | 0.99 | 1.88 | 1.90x slower |
| 1,000 | 1.00 | 1.74 | 1.73x slower |
| 10,000 | 5.40 | 7.85 | 1.46x slower |
| 100,000 | 85.58 | 92.64 | 1.08x slower |

### Scalar Multiply

| Vector Size | Numpy (μs) | Miniten (μs) | MiniTen vs NumPy |
|------------|------------:|------------:|----------------:|
| 100 | 1.23 | 0.91 | 1.35x faster |
| 1,000 | 1.16 | 1.61 | 1.39x slower |
| 10,000 | 2.61 | 7.79 | 2.99x slower |
| 100,000 | 43.05 | 77.84 | 1.81x slower |
