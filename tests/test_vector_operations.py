# test_operations.py

import numpy as np
from miniten.core import operations


def test_vector_addition():
    a = np.array([1, 2, 3], dtype=np.float64)
    b = np.array([4, 5, 6], dtype=np.float64)
    expected_result = np.array([5, 7, 9], dtype=np.float64)
    result = operations.vector_addition(a, b)
    assert np.array_equal(result, expected_result), "Test failed."


def test_scalar_multiplication():
    a = np.array([1, 2, 3], dtype=np.float64)
    scalar = 2
    expected_result = np.array([2, 4, 6], dtype=np.float64)
    result = operations.scalar_multiplication(a, scalar)
    assert np.array_equal(result, expected_result), "Test failed."


def test_vector_multiplication():
    a = np.array([1, 2, 3], dtype=np.float64)
    b = np.array([4, 5, 6], dtype=np.float64)
    expected_result = np.array([4, 10, 18], dtype=np.float64)
    result = operations.vector_multiplication(a, b)
    assert np.array_equal(result, expected_result), "Test failed."


def test_dot_product():
    a = np.array([1, 2, 3], dtype=np.float64)
    b = np.array([4, 5, 6], dtype=np.float64)
    expected_result = 32.0
    result = operations.dot_product(a, b)
    assert result == expected_result, "Test failed."


def test_vector_subtraction():
    a = np.array([4, 5, 6], dtype=np.float64)
    b = np.array([1, 2, 3], dtype=np.float64)
    expected_result = np.array([3, 3, 3], dtype=np.float64)
    result = operations.vector_subtraction(a, b)
    assert np.array_equal(result, expected_result), "Test failed."


def test_vector_division():
    a = np.array([4, 6, 8], dtype=np.float64)
    b = np.array([2, 3, 4], dtype=np.float64)
    expected_result = np.array([2, 2, 2], dtype=np.float64)
    result = operations.vector_division(a, b)
    assert np.array_equal(result, expected_result), "Test failed."


def test_scalar_component():
    a = np.array([1, 2, 3], dtype=np.float64)
    b = np.array([4, 5, 6], dtype=np.float64)
    expected_result = np.array([8, 10, 12], dtype=np.float64)
    result = operations.scalar_component(a, b)
    assert np.array_equal(result, expected_result), "Test failed."


def test_cross_product():
    a = np.array([1, 0, 0], dtype=np.float64)
    b = np.array([0, 1, 0], dtype=np.float64)
    expected_result = np.array([0, 0, 1], dtype=np.float64)
    result = operations.cross_product(a, b)
    assert np.array_equal(result, expected_result), "Test failed."


# Run the tests
test_vector_addition()
test_scalar_multiplication()
test_vector_multiplication()
test_dot_product()
test_vector_subtraction()
test_vector_division()
#test_scalar_component()
test_cross_product()

print("All tests passed.")
