# MiniTen Tests

This directory contains the test suite for MiniTen.

## Test Structure

```
tests/
├── test_backprop.py           # Backpropagation tests
├── test_vector_operations.py  # Vector operations tests
├── test_tensor.py             # Tensor tests (to be added)
├── test_autograd.py           # Autograd tests (to be added)
├── test_layers.py             # Neural network layer tests (to be added)
└── ...
```

## Running Tests

### All Tests
```bash
python -m pytest tests/ -v
```

### Specific Test File
```bash
python -m pytest tests/test_vector_operations.py -v
```

### With Coverage
```bash
python -m pytest tests/ -v --cov=miniten --cov-report=term-missing
```

### Individual Test Files (Legacy)
```bash
python tests/test_vector_operations.py
python tests/test_backprop.py
```

## Writing Tests

### Test Naming Convention
- Test files: `test_*.py`
- Test functions: `test_*`
- Test classes: `Test*`

### Example Test
```python
import numpy as np
from miniten.core import operations

def test_vector_addition():
    a = np.array([1, 2, 3], dtype=np.float64)
    b = np.array([4, 5, 6], dtype=np.float64)
    expected = np.array([5, 7, 9], dtype=np.float64)
    result = operations.vector_addition(a, b)
    assert np.array_equal(result, expected)
```

## Test Coverage Goals

- Core operations: > 90%
- Neural network layers: > 85%
- Optimizers: > 80%
- Utilities: > 75%

## Performance Tests

Performance benchmarks should be placed in the `benchmarks/` directory, not here.
