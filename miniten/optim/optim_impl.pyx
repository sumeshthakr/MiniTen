# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Optimization utilities for MiniTen.

This module provides high-performance implementations of:
- SGD optimizer
- Adam optimizer
- AdamW optimizer
- RMSprop optimizer

All implementations use Cython for maximum performance and minimal memory footprint,
optimized for edge devices.
"""

import cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt

# Initialize NumPy C API
np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void sgd_update(np.ndarray[np.float64_t, ndim=2] param,
                       np.ndarray[np.float64_t, ndim=2] grad,
                       np.ndarray[np.float64_t, ndim=2] velocity,
                       double lr, double momentum, double weight_decay) noexcept:
    """
    SGD update with momentum and weight decay.
    
    v = momentum * v + grad + weight_decay * param
    param = param - lr * v
    
    Args:
        param: Parameter tensor (modified in-place)
        grad: Gradient tensor
        velocity: Momentum buffer (modified in-place)
        lr: Learning rate
        momentum: Momentum coefficient
        weight_decay: L2 regularization coefficient
    """
    cdef Py_ssize_t m = param.shape[0]
    cdef Py_ssize_t n = param.shape[1]
    cdef double[:, ::1] p_view = param
    cdef double[:, ::1] g_view = grad
    cdef double[:, ::1] v_view = velocity
    
    cdef Py_ssize_t i, j
    cdef double g
    
    for i in range(m):
        for j in range(n):
            g = g_view[i, j] + weight_decay * p_view[i, j]
            v_view[i, j] = momentum * v_view[i, j] + g
            p_view[i, j] = p_view[i, j] - lr * v_view[i, j]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void sgd_update_1d(np.ndarray[np.float64_t, ndim=1] param,
                          np.ndarray[np.float64_t, ndim=1] grad,
                          np.ndarray[np.float64_t, ndim=1] velocity,
                          double lr, double momentum, double weight_decay) noexcept:
    """
    SGD update for 1D tensors (biases).
    """
    cdef Py_ssize_t n = param.shape[0]
    cdef double[::1] p_view = param
    cdef double[::1] g_view = grad
    cdef double[::1] v_view = velocity
    
    cdef Py_ssize_t i
    cdef double g
    
    for i in range(n):
        g = g_view[i] + weight_decay * p_view[i]
        v_view[i] = momentum * v_view[i] + g
        p_view[i] = p_view[i] - lr * v_view[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void adam_update(np.ndarray[np.float64_t, ndim=2] param,
                        np.ndarray[np.float64_t, ndim=2] grad,
                        np.ndarray[np.float64_t, ndim=2] m,
                        np.ndarray[np.float64_t, ndim=2] v,
                        double lr, double beta1, double beta2,
                        double eps, double weight_decay, int t) noexcept:
    """
    Adam update with bias correction.
    
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad^2
    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)
    param = param - lr * m_hat / (sqrt(v_hat) + eps)
    
    Args:
        param: Parameter tensor (modified in-place)
        grad: Gradient tensor
        m: First moment estimate (modified in-place)
        v: Second moment estimate (modified in-place)
        lr: Learning rate
        beta1: First moment decay rate
        beta2: Second moment decay rate
        eps: Numerical stability constant
        weight_decay: L2 regularization (added to gradient)
        t: Current timestep (for bias correction)
    """
    cdef Py_ssize_t rows = param.shape[0]
    cdef Py_ssize_t cols = param.shape[1]
    cdef double[:, ::1] p_view = param
    cdef double[:, ::1] g_view = grad
    cdef double[:, ::1] m_view = m
    cdef double[:, ::1] v_view = v
    
    cdef double bias_correction1 = 1.0 - beta1 ** t
    cdef double bias_correction2 = 1.0 - beta2 ** t
    
    cdef Py_ssize_t i, j
    cdef double g, m_hat, v_hat
    
    for i in range(rows):
        for j in range(cols):
            g = g_view[i, j] + weight_decay * p_view[i, j]
            
            # Update biased first moment estimate
            m_view[i, j] = beta1 * m_view[i, j] + (1.0 - beta1) * g
            
            # Update biased second raw moment estimate
            v_view[i, j] = beta2 * v_view[i, j] + (1.0 - beta2) * g * g
            
            # Bias-corrected estimates
            m_hat = m_view[i, j] / bias_correction1
            v_hat = v_view[i, j] / bias_correction2
            
            # Update parameter
            p_view[i, j] = p_view[i, j] - lr * m_hat / (sqrt(v_hat) + eps)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void adam_update_1d(np.ndarray[np.float64_t, ndim=1] param,
                           np.ndarray[np.float64_t, ndim=1] grad,
                           np.ndarray[np.float64_t, ndim=1] m,
                           np.ndarray[np.float64_t, ndim=1] v,
                           double lr, double beta1, double beta2,
                           double eps, double weight_decay, int t) noexcept:
    """
    Adam update for 1D tensors (biases).
    """
    cdef Py_ssize_t n = param.shape[0]
    cdef double[::1] p_view = param
    cdef double[::1] g_view = grad
    cdef double[::1] m_view = m
    cdef double[::1] v_view = v
    
    cdef double bias_correction1 = 1.0 - beta1 ** t
    cdef double bias_correction2 = 1.0 - beta2 ** t
    
    cdef Py_ssize_t i
    cdef double g, m_hat, v_hat
    
    for i in range(n):
        g = g_view[i] + weight_decay * p_view[i]
        m_view[i] = beta1 * m_view[i] + (1.0 - beta1) * g
        v_view[i] = beta2 * v_view[i] + (1.0 - beta2) * g * g
        m_hat = m_view[i] / bias_correction1
        v_hat = v_view[i] / bias_correction2
        p_view[i] = p_view[i] - lr * m_hat / (sqrt(v_hat) + eps)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void adamw_update(np.ndarray[np.float64_t, ndim=2] param,
                         np.ndarray[np.float64_t, ndim=2] grad,
                         np.ndarray[np.float64_t, ndim=2] m,
                         np.ndarray[np.float64_t, ndim=2] v,
                         double lr, double beta1, double beta2,
                         double eps, double weight_decay, int t) noexcept:
    """
    AdamW update (decoupled weight decay).
    
    Unlike Adam, weight decay is applied directly to parameters,
    not to the gradient.
    
    Args:
        param: Parameter tensor (modified in-place)
        grad: Gradient tensor
        m: First moment estimate (modified in-place)
        v: Second moment estimate (modified in-place)
        lr: Learning rate
        beta1: First moment decay rate
        beta2: Second moment decay rate
        eps: Numerical stability constant
        weight_decay: Decoupled weight decay coefficient
        t: Current timestep (for bias correction)
    """
    cdef Py_ssize_t rows = param.shape[0]
    cdef Py_ssize_t cols = param.shape[1]
    cdef double[:, ::1] p_view = param
    cdef double[:, ::1] g_view = grad
    cdef double[:, ::1] m_view = m
    cdef double[:, ::1] v_view = v
    
    cdef double bias_correction1 = 1.0 - beta1 ** t
    cdef double bias_correction2 = 1.0 - beta2 ** t
    
    cdef Py_ssize_t i, j
    cdef double g, m_hat, v_hat
    
    for i in range(rows):
        for j in range(cols):
            g = g_view[i, j]
            
            # Update biased first moment estimate
            m_view[i, j] = beta1 * m_view[i, j] + (1.0 - beta1) * g
            
            # Update biased second raw moment estimate
            v_view[i, j] = beta2 * v_view[i, j] + (1.0 - beta2) * g * g
            
            # Bias-corrected estimates
            m_hat = m_view[i, j] / bias_correction1
            v_hat = v_view[i, j] / bias_correction2
            
            # Update parameter with decoupled weight decay
            p_view[i, j] = p_view[i, j] * (1.0 - lr * weight_decay) - lr * m_hat / (sqrt(v_hat) + eps)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void rmsprop_update(np.ndarray[np.float64_t, ndim=2] param,
                           np.ndarray[np.float64_t, ndim=2] grad,
                           np.ndarray[np.float64_t, ndim=2] v,
                           double lr, double alpha, double eps,
                           double weight_decay) noexcept:
    """
    RMSprop update.
    
    v = alpha * v + (1 - alpha) * grad^2
    param = param - lr * grad / (sqrt(v) + eps)
    
    Args:
        param: Parameter tensor (modified in-place)
        grad: Gradient tensor
        v: Running average of squared gradients (modified in-place)
        lr: Learning rate
        alpha: Smoothing constant
        eps: Numerical stability constant
        weight_decay: L2 regularization coefficient
    """
    cdef Py_ssize_t rows = param.shape[0]
    cdef Py_ssize_t cols = param.shape[1]
    cdef double[:, ::1] p_view = param
    cdef double[:, ::1] g_view = grad
    cdef double[:, ::1] v_view = v
    
    cdef Py_ssize_t i, j
    cdef double g
    
    for i in range(rows):
        for j in range(cols):
            g = g_view[i, j] + weight_decay * p_view[i, j]
            v_view[i, j] = alpha * v_view[i, j] + (1.0 - alpha) * g * g
            p_view[i, j] = p_view[i, j] - lr * g / (sqrt(v_view[i, j]) + eps)


# ============================================================================
# Learning Rate Schedulers
# ============================================================================

cpdef double step_lr(double initial_lr, int epoch, int step_size, double gamma):
    """
    Step learning rate decay.
    
    lr = initial_lr * gamma^(epoch // step_size)
    """
    return initial_lr * (gamma ** (epoch // step_size))


cpdef double exponential_lr(double initial_lr, int epoch, double gamma):
    """
    Exponential learning rate decay.
    
    lr = initial_lr * gamma^epoch
    """
    return initial_lr * (gamma ** epoch)


cpdef double cosine_annealing_lr(double initial_lr, int epoch, int T_max, double eta_min=0.0):
    """
    Cosine annealing learning rate.
    
    lr = eta_min + 0.5 * (initial_lr - eta_min) * (1 + cos(epoch / T_max * pi))
    """
    cdef double pi = 3.141592653589793
    cdef double ratio = <double>epoch / <double>T_max
    return eta_min + 0.5 * (initial_lr - eta_min) * (1.0 + np.cos(ratio * pi))


cpdef double warmup_lr(double target_lr, int current_step, int warmup_steps):
    """
    Linear warmup learning rate.
    
    lr = target_lr * current_step / warmup_steps
    """
    if current_step >= warmup_steps:
        return target_lr
    return target_lr * current_step / warmup_steps


cpdef double warmup_cosine_lr(double initial_lr, int current_step, int warmup_steps, 
                               int total_steps, double eta_min=0.0):
    """
    Warmup + cosine annealing learning rate.
    """
    if current_step < warmup_steps:
        return warmup_lr(initial_lr, current_step, warmup_steps)
    
    cdef int decay_steps = total_steps - warmup_steps
    cdef int current_decay = current_step - warmup_steps
    return cosine_annealing_lr(initial_lr, current_decay, decay_steps, eta_min)


# ============================================================================
# Gradient Clipping
# ============================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double clip_grad_norm(list grads, double max_norm):
    """
    Clip gradients by global norm.
    
    Args:
        grads: List of gradient arrays
        max_norm: Maximum allowed norm
        
    Returns:
        Total gradient norm before clipping
    """
    cdef double total_norm = 0.0
    cdef double clip_coef
    cdef np.ndarray grad
    
    # Compute total norm
    for grad in grads:
        total_norm += np.sum(grad * grad)
    total_norm = sqrt(total_norm)
    
    # Clip if necessary
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for grad in grads:
            grad *= clip_coef
    
    return total_norm


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void clip_grad_value(list grads, double clip_value):
    """
    Clip gradients by value.
    
    Args:
        grads: List of gradient arrays
        clip_value: Maximum absolute value
    """
    cdef np.ndarray grad
    
    for grad in grads:
        np.clip(grad, -clip_value, clip_value, out=grad)
