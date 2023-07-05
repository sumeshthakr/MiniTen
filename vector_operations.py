# vector_operations.pyx

import numpy as np
#cimport numpy as np

# Vector addition
cpdef np.ndarray[np.float64_t, ndim=1] vector_addition(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b):
    cdef int n = len(a)
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef int i

    for i in range(n):
        result[i] = a[i] + b[i]

    return result


# Scalar multiplication
cpdef np.ndarray[np.float64_t, ndim=1] scalar_multiplication(np.ndarray[np.float64_t, ndim=1] a, float scalar):
    cdef int n = len(a)
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef int i

    for i in range(n):
        result[i] = a[i] * scalar

    return result


# Vector multiplication (element-wise)
cpdef np.ndarray[np.float64_t, ndim=1] vector_multiplication(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b):
    cdef int n = len(a)
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef int i

    for i in range(n):
        result[i] = a[i] * b[i]

    return result


# Dot product
cpdef float dot_product(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b):
    cdef int n = len(a)
    cdef float result = 0.0
    cdef int i

    for i in range(n):
        result += a[i] * b[i]

    return result


# Vector subtraction
cpdef np.ndarray[np.float64_t, ndim=1] vector_subtraction(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b):
    cdef int n = len(a)
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef int i

    for i in range(n):
        result[i] = a[i] - b[i]

    return result


# Vector division (element-wise)
cpdef np.ndarray[np.float64_t, ndim=1] vector_division(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b):
    cdef int n = len(a)
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef int i

    for i in range(n):
        result[i] = a[i] / b[i]

    return result


# Scalar component
cpdef np.ndarray[np.float64_t, ndim=1] scalar_component(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b):
    cdef float scalar = dot_product(a, b) / dot_product(b, b)
    return scalar * b


# Cross product
cpdef np.ndarray[np.float64_t, ndim=1] cross_product(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b):
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(3, dtype=np.float64)
    result[0] = a[1] * b[2] - a[2] * b[1]
    result[1] = a[2] * b[0] - a[0] * b[2]
    result[2] = a[0] * b[1] - a[1] * b[0]
    return result
