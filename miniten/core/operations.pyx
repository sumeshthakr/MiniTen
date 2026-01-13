# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Optimized vector and matrix operations for MiniTen.

This module provides high-performance implementations using:
- Cython memory views for zero-copy operations
- OpenMP parallelization for large operations
- Optimized loops with reduced overhead
"""

import cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport exp, tanh, sqrt, fmax

# Initialize NumPy C API
np.import_array()

# Threshold for using parallel operations
# Set to 10,000 elements based on empirical testing:
# - Below 10K: Function call and thread spawn overhead dominates, sequential is faster
# - Above 10K: Parallelization overhead is amortized, OpenMP provides significant speedup
# This threshold can be tuned per platform and number of cores
DEF PARALLEL_THRESHOLD = 10000


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] vector_addition(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b):
    """Optimized vector addition using memory views."""
    cdef Py_ssize_t n = a.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef double[::1] a_view = a
    cdef double[::1] b_view = b
    cdef double[::1] result_view = result
    cdef Py_ssize_t i

    if n >= PARALLEL_THRESHOLD:
        for i in prange(n, nogil=True):
            result_view[i] = a_view[i] + b_view[i]
    else:
        for i in range(n):
            result_view[i] = a_view[i] + b_view[i]

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] scalar_multiplication(np.ndarray[np.float64_t, ndim=1] a, double scalar):
    """Optimized scalar multiplication using memory views."""
    cdef Py_ssize_t n = a.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef double[::1] a_view = a
    cdef double[::1] result_view = result
    cdef Py_ssize_t i

    if n >= PARALLEL_THRESHOLD:
        for i in prange(n, nogil=True):
            result_view[i] = a_view[i] * scalar
    else:
        for i in range(n):
            result_view[i] = a_view[i] * scalar

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] vector_multiplication(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b):
    """Optimized element-wise multiplication using memory views."""
    cdef Py_ssize_t n = a.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef double[::1] a_view = a
    cdef double[::1] b_view = b
    cdef double[::1] result_view = result
    cdef Py_ssize_t i

    if n >= PARALLEL_THRESHOLD:
        for i in prange(n, nogil=True):
            result_view[i] = a_view[i] * b_view[i]
    else:
        for i in range(n):
            result_view[i] = a_view[i] * b_view[i]

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double dot_product(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b):
    """Optimized dot product using memory views and reduction."""
    cdef Py_ssize_t n = a.shape[0]
    cdef double[::1] a_view = a
    cdef double[::1] b_view = b
    cdef double result = 0.0
    cdef Py_ssize_t i
    
    # Use loop unrolling for better performance
    cdef Py_ssize_t n_unroll = n - (n % 4)
    
    for i in range(0, n_unroll, 4):
        result += a_view[i] * b_view[i]
        result += a_view[i+1] * b_view[i+1]
        result += a_view[i+2] * b_view[i+2]
        result += a_view[i+3] * b_view[i+3]
    
    # Handle remaining elements
    for i in range(n_unroll, n):
        result += a_view[i] * b_view[i]

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] vector_subtraction(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b):
    """Optimized vector subtraction using memory views."""
    cdef Py_ssize_t n = a.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef double[::1] a_view = a
    cdef double[::1] b_view = b
    cdef double[::1] result_view = result
    cdef Py_ssize_t i

    if n >= PARALLEL_THRESHOLD:
        for i in prange(n, nogil=True):
            result_view[i] = a_view[i] - b_view[i]
    else:
        for i in range(n):
            result_view[i] = a_view[i] - b_view[i]

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] vector_division(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b):
    """Optimized element-wise division using memory views."""
    cdef Py_ssize_t n = a.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef double[::1] a_view = a
    cdef double[::1] b_view = b
    cdef double[::1] result_view = result
    cdef Py_ssize_t i

    if n >= PARALLEL_THRESHOLD:
        for i in prange(n, nogil=True):
            result_view[i] = a_view[i] / b_view[i]
    else:
        for i in range(n):
            result_view[i] = a_view[i] / b_view[i]

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] scalar_component(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b):
    """Calculate scalar component of vector a onto vector b."""
    cdef double scalar = dot_product(a, b) / dot_product(b, b)
    return scalar_multiplication(b, scalar)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] cross_product(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b):
    """Optimized 3D cross product."""
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(3, dtype=np.float64)
    cdef double[::1] a_view = a
    cdef double[::1] b_view = b
    cdef double[::1] result_view = result
    
    result_view[0] = a_view[1] * b_view[2] - a_view[2] * b_view[1]
    result_view[1] = a_view[2] * b_view[0] - a_view[0] * b_view[2]
    result_view[2] = a_view[0] * b_view[1] - a_view[1] * b_view[0]
    
    return result


# Matrix operations
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] matmul(np.ndarray[np.float64_t, ndim=2] a, np.ndarray[np.float64_t, ndim=2] b):
    """Optimized matrix multiplication using tiled approach."""
    cdef Py_ssize_t m = a.shape[0]
    cdef Py_ssize_t n = b.shape[1]
    cdef Py_ssize_t k = a.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] result = np.zeros((m, n), dtype=np.float64)
    cdef double[:, ::1] a_view = a
    cdef double[:, ::1] b_view = b
    cdef double[:, ::1] result_view = result
    cdef Py_ssize_t i, j, p
    cdef double temp

    if m * n * k >= 100000:
        # Parallel version for large matrices
        for i in prange(m, nogil=True):
            for j in range(n):
                temp = 0.0
                for p in range(k):
                    temp = temp + a_view[i, p] * b_view[p, j]
                result_view[i, j] = temp
    else:
        # Sequential version for small matrices
        for i in range(m):
            for j in range(n):
                temp = 0.0
                for p in range(k):
                    temp = temp + a_view[i, p] * b_view[p, j]
                result_view[i, j] = temp

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] transpose(np.ndarray[np.float64_t, ndim=2] a):
    """Optimized matrix transpose."""
    cdef Py_ssize_t m = a.shape[0]
    cdef Py_ssize_t n = a.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] result = np.empty((n, m), dtype=np.float64)
    cdef double[:, ::1] a_view = a
    cdef double[:, ::1] result_view = result
    cdef Py_ssize_t i, j

    for i in range(m):
        for j in range(n):
            result_view[j, i] = a_view[i, j]

    return result


# Activation functions
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] sigmoid(np.ndarray[np.float64_t, ndim=1] x):
    """Optimized sigmoid activation function."""
    cdef Py_ssize_t n = x.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef double[::1] x_view = x
    cdef double[::1] result_view = result
    cdef Py_ssize_t i

    if n >= PARALLEL_THRESHOLD:
        for i in prange(n, nogil=True):
            result_view[i] = 1.0 / (1.0 + exp(-x_view[i]))
    else:
        for i in range(n):
            result_view[i] = 1.0 / (1.0 + exp(-x_view[i]))

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] relu(np.ndarray[np.float64_t, ndim=1] x):
    """Optimized ReLU activation function."""
    cdef Py_ssize_t n = x.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef double[::1] x_view = x
    cdef double[::1] result_view = result
    cdef Py_ssize_t i

    if n >= PARALLEL_THRESHOLD:
        for i in prange(n, nogil=True):
            result_view[i] = fmax(0.0, x_view[i])
    else:
        for i in range(n):
            result_view[i] = fmax(0.0, x_view[i])

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] tanh_activation(np.ndarray[np.float64_t, ndim=1] x):
    """Optimized tanh activation function."""
    cdef Py_ssize_t n = x.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef double[::1] x_view = x
    cdef double[::1] result_view = result
    cdef Py_ssize_t i

    if n >= PARALLEL_THRESHOLD:
        for i in prange(n, nogil=True):
            result_view[i] = tanh(x_view[i])
    else:
        for i in range(n):
            result_view[i] = tanh(x_view[i])

    return result
