# Vector Operations Benchmark Results

## Benchmark Results

*Benchmark run on: 2026-01-13 19:15:45*

*System: Python 3.12.3*


### Vector Addition

| Vector Size | Numpy (μs) | Miniten (μs) | MiniTen vs NumPy |
|------------|------------:|------------:|----------------:|
| 100 | 0.98 | 1.90 | 1.95x slower |
| 1,000 | 0.95 | 1.81 | 1.91x slower |
| 10,000 | 5.93 | 7.91 | 1.33x slower |
| 100,000 | 78.76 | 95.96 | 1.22x slower |

### Dot Product

| Vector Size | Numpy (μs) | Miniten (μs) | MiniTen vs NumPy |
|------------|------------:|------------:|----------------:|
| 100 | 1.54 | 1.35 | 1.14x faster |
| 1,000 | 0.96 | 4.49 | 4.67x slower |
| 10,000 | 3.42 | 40.83 | 11.94x slower |
| 100,000 | 10.56 | 405.54 | 38.39x slower |

### Element-wise Multiply

| Vector Size | Numpy (μs) | Miniten (μs) | MiniTen vs NumPy |
|------------|------------:|------------:|----------------:|
| 100 | 0.99 | 1.85 | 1.88x slower |
| 1,000 | 0.91 | 1.78 | 1.95x slower |
| 10,000 | 5.76 | 7.84 | 1.36x slower |
| 100,000 | 91.68 | 92.93 | 1.01x slower |

### Scalar Multiply

| Vector Size | Numpy (μs) | Miniten (μs) | MiniTen vs NumPy |
|------------|------------:|------------:|----------------:|
| 100 | 1.40 | 1.19 | 1.17x faster |
| 1,000 | 1.10 | 1.60 | 1.45x slower |
| 10,000 | 2.61 | 7.84 | 3.00x slower |
| 100,000 | 43.27 | 75.97 | 1.76x slower |
