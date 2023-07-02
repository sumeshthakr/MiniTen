# test_vector_multiplication.py

import numpy as np
from vec_multi import vec_multi

def test_vector_multiplication():
    a = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    b = np.array([2, 4, 6, 8, 10], dtype=np.float64)
    expected_result = np.array([2, 8, 18, 32, 50], dtype=np.float64)
    result = vec_multi(a, b)
    assert np.array_equal(result, expected_result), "Test failed."

test_vector_multiplication()
print("All tests passed.")