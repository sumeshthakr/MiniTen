#Vector Multiplication
"""
Author Sumesh Thakur
"""

import numpy as np
cimport numpy as np

cpdef np.ndarray[np.float64_t, ndim=1] vec_multi(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b):
    cdef int n = len(a)
    cdef np.ndarray[np.float64_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef int i

    for i in range(n):
        result[i] = a[i] * b[i]

    return result