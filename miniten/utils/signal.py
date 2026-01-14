"""
Signal Processing Utilities

General signal processing for time-series and sensor data.
Optimized for edge devices with minimal dependencies.

Features:
- Filtering (low-pass, high-pass, band-pass)
- FFT and spectral analysis - custom implementation
- Wavelet transforms
- Signal smoothing
- Peak detection
- Optimized for real-time edge processing
- Minimal NumPy dependency - uses custom optimizations
"""

import math
from typing import List, Tuple, Optional, Union


# ============================================================================
# Core Signal Types
# ============================================================================

class Signal:
    """
    Lightweight signal container for edge devices.
    """
    
    def __init__(self, data: List[float], sample_rate: Optional[float] = None):
        self.data = data
        self.sample_rate = sample_rate
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) -> float:
        return self.data[idx]


# ============================================================================
# FFT (Cooley-Tukey) - Custom Implementation
# ============================================================================

def _bit_reverse(n: int, bits: int) -> int:
    """Bit reversal for FFT."""
    result = 0
    for _ in range(bits):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result


def _next_power_of_2(n: int) -> int:
    """Get next power of 2."""
    p = 1
    while p < n:
        p *= 2
    return p


def fft(signal: Union[List[float], Signal]) -> Tuple[List[float], List[float]]:
    """
    Compute Fast Fourier Transform using Cooley-Tukey algorithm.
    
    Args:
        signal: Input signal
        
    Returns:
        Tuple of (real, imaginary) components
    """
    if isinstance(signal, Signal):
        data = signal.data
    else:
        data = list(signal)
    
    n = len(data)
    if n == 0:
        return [], []
    
    # Pad to power of 2
    n_padded = _next_power_of_2(n)
    real = data + [0.0] * (n_padded - n)
    imag = [0.0] * n_padded
    
    # Bit reversal
    bits = 0
    temp = n_padded
    while temp > 1:
        temp //= 2
        bits += 1
    
    for i in range(n_padded):
        j = _bit_reverse(i, bits)
        if i < j:
            real[i], real[j] = real[j], real[i]
    
    # FFT butterfly
    size = 2
    while size <= n_padded:
        half_size = size // 2
        angle_step = -2 * math.pi / size
        
        for start in range(0, n_padded, size):
            for k in range(half_size):
                angle = angle_step * k
                cos_val = math.cos(angle)
                sin_val = math.sin(angle)
                
                i = start + k
                j = start + k + half_size
                
                t_real = cos_val * real[j] - sin_val * imag[j]
                t_imag = sin_val * real[j] + cos_val * imag[j]
                
                real[j] = real[i] - t_real
                imag[j] = imag[i] - t_imag
                real[i] = real[i] + t_real
                imag[i] = imag[i] + t_imag
        
        size *= 2
    
    return real, imag


def ifft(real: List[float], imag: List[float]) -> List[float]:
    """
    Compute Inverse Fast Fourier Transform.
    
    Uses the relationship: IFFT(x) = conj(FFT(conj(x))) / N
    
    Args:
        real: Real part of spectrum
        imag: Imaginary part of spectrum
        
    Returns:
        Time domain signal
    """
    n = len(real)
    if n == 0:
        return []
    
    # Conjugate the input
    imag_conj = [-x for x in imag]
    
    # Pad to power of 2 if necessary
    n_padded = _next_power_of_2(n)
    if n_padded > n:
        real = list(real) + [0.0] * (n_padded - n)
        imag_conj = list(imag_conj) + [0.0] * (n_padded - n)
    else:
        real = list(real)
        imag_conj = list(imag_conj)
    
    # Apply forward FFT to conjugated input
    result_real, result_imag = _fft_core(real, imag_conj)
    
    # Conjugate and scale the result
    return [r / n_padded for r in result_real]


def _fft_core(real: List[float], imag: List[float]) -> Tuple[List[float], List[float]]:
    """
    Core FFT computation (Cooley-Tukey radix-2).
    Assumes input is already padded to power of 2.
    """
    n = len(real)
    if n <= 1:
        return real, imag
    
    # Bit reversal
    bits = 0
    temp = n
    while temp > 1:
        temp //= 2
        bits += 1
    
    for i in range(n):
        j = _bit_reverse(i, bits)
        if i < j:
            real[i], real[j] = real[j], real[i]
            imag[i], imag[j] = imag[j], imag[i]
    
    # FFT butterfly
    size = 2
    while size <= n:
        half_size = size // 2
        angle_step = -2 * math.pi / size
        
        for start in range(0, n, size):
            for k in range(half_size):
                angle = angle_step * k
                cos_val = math.cos(angle)
                sin_val = math.sin(angle)
                
                i_idx = start + k
                j_idx = start + k + half_size
                
                t_real = cos_val * real[j_idx] - sin_val * imag[j_idx]
                t_imag = sin_val * real[j_idx] + cos_val * imag[j_idx]
                
                real[j_idx] = real[i_idx] - t_real
                imag[j_idx] = imag[i_idx] - t_imag
                real[i_idx] = real[i_idx] + t_real
                imag[i_idx] = imag[i_idx] + t_imag
        
        size *= 2
    
    return real, imag


def rfft(signal: List[float]) -> Tuple[List[float], List[float]]:
    """Real FFT - returns only positive frequencies."""
    real, imag = fft(signal)
    n = len(real) // 2 + 1
    return real[:n], imag[:n]


def irfft(real: List[float], imag: List[float], n: int) -> List[float]:
    """Inverse real FFT."""
    # Reconstruct full spectrum
    full_real = real + [0.0] * (n - len(real))
    full_imag = imag + [0.0] * (n - len(imag))
    
    # Mirror for negative frequencies
    for i in range(1, n // 2):
        full_real[n - i] = full_real[i]
        full_imag[n - i] = -full_imag[i]
    
    return ifft(full_real, full_imag)


# ============================================================================
# Filtering
# ============================================================================

def _convolve(signal: List[float], kernel: List[float]) -> List[float]:
    """1D convolution (valid mode)."""
    n = len(signal)
    k = len(kernel)
    
    if n < k:
        return []
    
    result = []
    for i in range(n - k + 1):
        val = 0.0
        for j in range(k):
            val += signal[i + j] * kernel[k - 1 - j]
        result.append(val)
    
    return result


def _convolve_same(signal: List[float], kernel: List[float]) -> List[float]:
    """1D convolution (same mode - output same length as input)."""
    n = len(signal)
    k = len(kernel)
    
    # Pad signal
    pad = k // 2
    padded = [0.0] * pad + signal + [0.0] * pad
    
    result = []
    for i in range(n):
        val = 0.0
        for j in range(k):
            val += padded[i + j] * kernel[k - 1 - j]
        result.append(val)
    
    return result


def _sinc(x: float) -> float:
    """Normalized sinc function."""
    if x == 0:
        return 1.0
    return math.sin(math.pi * x) / (math.pi * x)


def _design_lowpass_kernel(cutoff: float, sample_rate: float, 
                           num_taps: int = 31) -> List[float]:
    """Design FIR lowpass filter kernel using windowed sinc."""
    fc = cutoff / sample_rate  # Normalized cutoff
    
    kernel = []
    mid = num_taps // 2
    
    for i in range(num_taps):
        n = i - mid
        
        # Sinc
        if n == 0:
            h = 2 * fc
        else:
            h = math.sin(2 * math.pi * fc * n) / (math.pi * n)
        
        # Hamming window
        w = 0.54 - 0.46 * math.cos(2 * math.pi * i / (num_taps - 1))
        
        kernel.append(h * w)
    
    # Normalize
    total = sum(kernel)
    kernel = [k / total for k in kernel]
    
    return kernel


def lowpass_filter(signal: Union[List[float], Signal], 
                   cutoff: float, 
                   sample_rate: float, 
                   order: int = 31) -> List[float]:
    """
    Apply low-pass filter using FIR.
    
    Args:
        signal: Input signal
        cutoff: Cutoff frequency in Hz
        sample_rate: Sample rate in Hz
        order: Filter order (number of taps)
        
    Returns:
        Filtered signal
    """
    if isinstance(signal, Signal):
        data = signal.data
    else:
        data = list(signal)
    
    kernel = _design_lowpass_kernel(cutoff, sample_rate, order)
    return _convolve_same(data, kernel)


def highpass_filter(signal: Union[List[float], Signal], 
                    cutoff: float, 
                    sample_rate: float, 
                    order: int = 31) -> List[float]:
    """
    Apply high-pass filter using spectral inversion.
    
    Args:
        signal: Input signal
        cutoff: Cutoff frequency in Hz
        sample_rate: Sample rate in Hz
        order: Filter order
        
    Returns:
        Filtered signal
    """
    if isinstance(signal, Signal):
        data = signal.data
    else:
        data = list(signal)
    
    # Design lowpass kernel
    lp_kernel = _design_lowpass_kernel(cutoff, sample_rate, order)
    
    # Spectral inversion for highpass
    hp_kernel = [-k for k in lp_kernel]
    hp_kernel[order // 2] += 1.0
    
    return _convolve_same(data, hp_kernel)


def bandpass_filter(signal: Union[List[float], Signal], 
                    low_cutoff: float, 
                    high_cutoff: float, 
                    sample_rate: float, 
                    order: int = 31) -> List[float]:
    """
    Apply band-pass filter.
    
    Args:
        signal: Input signal
        low_cutoff: Low cutoff frequency in Hz
        high_cutoff: High cutoff frequency in Hz
        sample_rate: Sample rate in Hz
        order: Filter order
        
    Returns:
        Filtered signal
    """
    if isinstance(signal, Signal):
        data = signal.data
    else:
        data = list(signal)
    
    # Apply highpass then lowpass
    hp_filtered = highpass_filter(data, low_cutoff, sample_rate, order)
    return lowpass_filter(hp_filtered, high_cutoff, sample_rate, order)


def bandstop_filter(signal: Union[List[float], Signal],
                    low_cutoff: float,
                    high_cutoff: float,
                    sample_rate: float,
                    order: int = 31) -> List[float]:
    """Apply band-stop (notch) filter."""
    if isinstance(signal, Signal):
        data = signal.data
    else:
        data = list(signal)
    
    # Low pass + high pass combined
    lp = lowpass_filter(data, low_cutoff, sample_rate, order)
    hp = highpass_filter(data, high_cutoff, sample_rate, order)
    
    return [l + h for l, h in zip(lp, hp)]


# ============================================================================
# Smoothing and Moving Average
# ============================================================================

def moving_average(signal: Union[List[float], Signal], 
                   window_size: int) -> List[float]:
    """
    Compute moving average for smoothing.
    
    Args:
        signal: Input signal
        window_size: Size of averaging window
        
    Returns:
        Smoothed signal
    """
    if isinstance(signal, Signal):
        data = signal.data
    else:
        data = list(signal)
    
    if window_size <= 0 or len(data) < window_size:
        return data
    
    result = []
    window_sum = sum(data[:window_size])
    
    for i in range(len(data)):
        if i < window_size:
            result.append(sum(data[:i+1]) / (i + 1))
        else:
            window_sum = window_sum - data[i - window_size] + data[i]
            result.append(window_sum / window_size)
    
    return result


def exponential_moving_average(signal: Union[List[float], Signal],
                               alpha: float = 0.3) -> List[float]:
    """
    Exponential moving average.
    
    Args:
        signal: Input signal
        alpha: Smoothing factor (0 < alpha <= 1)
        
    Returns:
        Smoothed signal
    """
    if isinstance(signal, Signal):
        data = signal.data
    else:
        data = list(signal)
    
    if not data:
        return []
    
    result = [data[0]]
    for i in range(1, len(data)):
        result.append(alpha * data[i] + (1 - alpha) * result[-1])
    
    return result


def gaussian_smooth(signal: Union[List[float], Signal],
                    sigma: float = 1.0,
                    kernel_size: Optional[int] = None) -> List[float]:
    """Apply Gaussian smoothing."""
    if isinstance(signal, Signal):
        data = signal.data
    else:
        data = list(signal)
    
    if kernel_size is None:
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
    
    # Gaussian kernel
    mid = kernel_size // 2
    kernel = []
    for i in range(kernel_size):
        x = i - mid
        kernel.append(math.exp(-x**2 / (2 * sigma**2)))
    
    # Normalize
    total = sum(kernel)
    kernel = [k / total for k in kernel]
    
    return _convolve_same(data, kernel)


# ============================================================================
# Peak Detection
# ============================================================================

def find_peaks(signal: Union[List[float], Signal], 
               threshold: Optional[float] = None,
               distance: int = 1,
               prominence: Optional[float] = None) -> List[int]:
    """
    Find peaks in signal.
    
    Args:
        signal: Input signal
        threshold: Minimum peak height
        distance: Minimum distance between peaks
        prominence: Minimum prominence of peaks
        
    Returns:
        Indices of peaks
    """
    if isinstance(signal, Signal):
        data = signal.data
    else:
        data = list(signal)
    
    if len(data) < 3:
        return []
    
    peaks = []
    
    for i in range(1, len(data) - 1):
        # Check if local maximum
        if data[i] > data[i-1] and data[i] > data[i+1]:
            # Check threshold
            if threshold is not None and data[i] < threshold:
                continue
            
            # Check prominence
            if prominence is not None:
                left_min = min(data[max(0, i-distance):i])
                right_min = min(data[i+1:min(len(data), i+distance+1)])
                prom = data[i] - max(left_min, right_min)
                if prom < prominence:
                    continue
            
            peaks.append(i)
    
    # Filter by distance
    if distance > 1 and len(peaks) > 1:
        filtered_peaks = [peaks[0]]
        for peak in peaks[1:]:
            if peak - filtered_peaks[-1] >= distance:
                filtered_peaks.append(peak)
        peaks = filtered_peaks
    
    return peaks


def find_valleys(signal: Union[List[float], Signal],
                 threshold: Optional[float] = None,
                 distance: int = 1) -> List[int]:
    """Find valleys (local minima) in signal."""
    if isinstance(signal, Signal):
        data = signal.data
    else:
        data = list(signal)
    
    # Invert signal and find peaks
    inverted = [-x for x in data]
    peaks = find_peaks(inverted, 
                       threshold=-threshold if threshold is not None else None,
                       distance=distance)
    return peaks


# ============================================================================
# Wavelet Transform
# ============================================================================

def _haar_wavelet_step(data: List[float]) -> Tuple[List[float], List[float]]:
    """Single step of Haar wavelet transform."""
    n = len(data) // 2
    approx = []
    detail = []
    
    for i in range(n):
        avg = (data[2*i] + data[2*i + 1]) / math.sqrt(2)
        diff = (data[2*i] - data[2*i + 1]) / math.sqrt(2)
        approx.append(avg)
        detail.append(diff)
    
    return approx, detail


def wavelet_transform(signal: Union[List[float], Signal],
                      wavelet: str = 'haar',
                      level: int = 1) -> Tuple[List[float], List[List[float]]]:
    """
    Compute discrete wavelet transform.
    
    Args:
        signal: Input signal
        wavelet: Wavelet type ('haar' supported)
        level: Decomposition level
        
    Returns:
        Tuple of (approximation coefficients, list of detail coefficients)
    """
    if isinstance(signal, Signal):
        data = signal.data
    else:
        data = list(signal)
    
    if wavelet != 'haar':
        raise NotImplementedError(f"Wavelet '{wavelet}' not implemented. Use 'haar'.")
    
    # Pad to power of 2
    n = len(data)
    n_padded = _next_power_of_2(n)
    data = data + [0.0] * (n_padded - n)
    
    details = []
    approx = data
    
    for _ in range(level):
        if len(approx) < 2:
            break
        approx, detail = _haar_wavelet_step(approx)
        details.append(detail)
    
    return approx, details


def inverse_wavelet_transform(approx: List[float],
                              details: List[List[float]],
                              wavelet: str = 'haar') -> List[float]:
    """
    Compute inverse discrete wavelet transform.
    
    Args:
        approx: Approximation coefficients
        details: List of detail coefficients per level
        wavelet: Wavelet type
        
    Returns:
        Reconstructed signal
    """
    if wavelet != 'haar':
        raise NotImplementedError(f"Wavelet '{wavelet}' not implemented. Use 'haar'.")
    
    data = approx
    
    for detail in reversed(details):
        # Inverse Haar step
        n = len(data)
        reconstructed = [0.0] * (2 * n)
        
        for i in range(n):
            reconstructed[2*i] = (data[i] + detail[i]) / math.sqrt(2)
            reconstructed[2*i + 1] = (data[i] - detail[i]) / math.sqrt(2)
        
        data = reconstructed
    
    return data


# ============================================================================
# Spectral Analysis
# ============================================================================

def power_spectrum(signal: Union[List[float], Signal], 
                   sample_rate: float) -> Tuple[List[float], List[float]]:
    """
    Compute power spectral density.
    
    Args:
        signal: Input signal
        sample_rate: Sample rate
        
    Returns:
        Tuple of (frequencies, power spectrum)
    """
    if isinstance(signal, Signal):
        data = signal.data
    else:
        data = list(signal)
    
    real, imag = fft(data)
    n = len(real)
    
    # Only positive frequencies
    n_pos = n // 2 + 1
    
    freqs = [i * sample_rate / n for i in range(n_pos)]
    power = [real[i]**2 + imag[i]**2 for i in range(n_pos)]
    
    # Scale
    power = [p / n**2 for p in power]
    power[1:-1] = [2 * p for p in power[1:-1]]  # Double non-DC and non-Nyquist
    
    return freqs, power


def spectral_centroid(signal: Union[List[float], Signal],
                      sample_rate: float) -> float:
    """Compute spectral centroid (center of mass of spectrum)."""
    freqs, power = power_spectrum(signal, sample_rate)
    
    total_power = sum(power)
    if total_power == 0:
        return 0.0
    
    weighted_sum = sum(f * p for f, p in zip(freqs, power))
    return weighted_sum / total_power


def spectral_bandwidth(signal: Union[List[float], Signal],
                       sample_rate: float) -> float:
    """Compute spectral bandwidth."""
    freqs, power = power_spectrum(signal, sample_rate)
    centroid = spectral_centroid(signal, sample_rate)
    
    total_power = sum(power)
    if total_power == 0:
        return 0.0
    
    variance = sum(p * (f - centroid)**2 for f, p in zip(freqs, power))
    return math.sqrt(variance / total_power)


# ============================================================================
# Signal Utilities
# ============================================================================

def normalize(signal: Union[List[float], Signal]) -> List[float]:
    """Normalize signal to [-1, 1]."""
    if isinstance(signal, Signal):
        data = signal.data
    else:
        data = list(signal)
    
    max_val = max(abs(x) for x in data) if data else 1.0
    if max_val > 0:
        return [x / max_val for x in data]
    return data


def zero_crossing_rate(signal: Union[List[float], Signal]) -> float:
    """Compute zero crossing rate."""
    if isinstance(signal, Signal):
        data = signal.data
    else:
        data = list(signal)
    
    if len(data) < 2:
        return 0.0
    
    crossings = sum(1 for i in range(1, len(data)) 
                    if data[i-1] * data[i] < 0)
    return crossings / (len(data) - 1)


def rms(signal: Union[List[float], Signal]) -> float:
    """Compute root mean square."""
    if isinstance(signal, Signal):
        data = signal.data
    else:
        data = list(signal)
    
    if not data:
        return 0.0
    
    return math.sqrt(sum(x**2 for x in data) / len(data))


def energy(signal: Union[List[float], Signal]) -> float:
    """Compute signal energy."""
    if isinstance(signal, Signal):
        data = signal.data
    else:
        data = list(signal)
    
    return sum(x**2 for x in data)


def autocorrelation(signal: Union[List[float], Signal],
                    max_lag: Optional[int] = None) -> List[float]:
    """Compute autocorrelation."""
    if isinstance(signal, Signal):
        data = signal.data
    else:
        data = list(signal)
    
    n = len(data)
    if max_lag is None:
        max_lag = n
    
    result = []
    for lag in range(min(max_lag, n)):
        val = 0.0
        for i in range(n - lag):
            val += data[i] * data[i + lag]
        result.append(val / n)
    
    return result


def cross_correlation(signal1: Union[List[float], Signal],
                      signal2: Union[List[float], Signal]) -> List[float]:
    """Compute cross-correlation between two signals."""
    if isinstance(signal1, Signal):
        data1 = signal1.data
    else:
        data1 = list(signal1)
    
    if isinstance(signal2, Signal):
        data2 = signal2.data
    else:
        data2 = list(signal2)
    
    n1, n2 = len(data1), len(data2)
    result = []
    
    for lag in range(-n2 + 1, n1):
        val = 0.0
        for i in range(n1):
            j = i - lag
            if 0 <= j < n2:
                val += data1[i] * data2[j]
        result.append(val)
    
    return result


def decimate(signal: Union[List[float], Signal],
             factor: int) -> List[float]:
    """Decimate signal by integer factor."""
    if isinstance(signal, Signal):
        data = signal.data
    else:
        data = list(signal)
    
    # Low-pass filter first to prevent aliasing
    # Use simple averaging
    filtered = moving_average(data, factor)
    
    return filtered[::factor]


def upsample(signal: Union[List[float], Signal],
             factor: int) -> List[float]:
    """Upsample signal by integer factor using linear interpolation."""
    if isinstance(signal, Signal):
        data = signal.data
    else:
        data = list(signal)
    
    if factor <= 1:
        return data
    
    result = []
    for i in range(len(data) - 1):
        result.append(data[i])
        for j in range(1, factor):
            t = j / factor
            result.append(data[i] * (1 - t) + data[i + 1] * t)
    result.append(data[-1])
    
    return result
