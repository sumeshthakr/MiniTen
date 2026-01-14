"""
Audio Processing Utilities

Audio processing for speech, music, and sound analysis.
Optimized for edge devices with minimal dependencies.

Features:
- Audio loading and saving (WAV format)
- Spectrograms (STFT, Mel-spectrogram) - custom FFT implementation
- MFCC features
- Audio augmentation
- Resampling
- All optimized for real-time processing on edge devices
- Minimal NumPy dependency - uses custom optimizations
"""

import math
import struct
import wave
from typing import List, Tuple, Optional, Union


# ============================================================================
# Core Audio Types and Utilities
# ============================================================================

class AudioTensor:
    """
    Lightweight audio tensor for edge devices.
    Stores audio samples as a flat list to minimize memory overhead.
    """
    
    def __init__(self, data: List[float], sample_rate: int = 16000):
        self.data = data
        self.sample_rate = sample_rate
        self.channels = 1  # Mono for now
    
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self.data) / self.sample_rate
    
    @property
    def num_samples(self) -> int:
        """Number of samples."""
        return len(self.data)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) -> float:
        return self.data[idx]


# ============================================================================
# Custom FFT Implementation (No NumPy dependency)
# ============================================================================

def _bit_reverse(n: int, bits: int) -> int:
    """Bit reversal for FFT."""
    result = 0
    for _ in range(bits):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result


def _fft_cooley_tukey(real: List[float], imag: List[float]) -> Tuple[List[float], List[float]]:
    """
    Cooley-Tukey FFT algorithm (in-place, radix-2).
    Optimized for edge devices.
    """
    n = len(real)
    if n <= 1:
        return real, imag
    
    # Check if n is power of 2
    bits = 0
    temp = n
    while temp > 1:
        temp >>= 1
        bits += 1
    
    if (1 << bits) != n:
        # Pad to next power of 2
        next_pow2 = 1 << (bits + 1)
        real = real + [0.0] * (next_pow2 - n)
        imag = imag + [0.0] * (next_pow2 - n)
        n = next_pow2
        bits += 1
    
    # Bit reversal permutation
    for i in range(n):
        j = _bit_reverse(i, bits)
        if i < j:
            real[i], real[j] = real[j], real[i]
            imag[i], imag[j] = imag[j], imag[i]
    
    # FFT computation
    size = 2
    while size <= n:
        half_size = size // 2
        angle_step = -2 * math.pi / size
        
        for start in range(0, n, size):
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


def _ifft_cooley_tukey(real: List[float], imag: List[float]) -> Tuple[List[float], List[float]]:
    """Inverse FFT using Cooley-Tukey algorithm."""
    n = len(real)
    
    # Conjugate
    imag = [-x for x in imag]
    
    # Forward FFT
    real, imag = _fft_cooley_tukey(real, imag)
    
    # Conjugate and scale
    real = [x / n for x in real]
    imag = [-x / n for x in imag]
    
    return real, imag


def fft(signal: Union[List[float], AudioTensor]) -> Tuple[List[float], List[float]]:
    """
    Compute Fast Fourier Transform.
    
    Args:
        signal: Input signal (list of samples or AudioTensor)
        
    Returns:
        Tuple of (real, imaginary) components
    """
    if isinstance(signal, AudioTensor):
        data = signal.data
    else:
        data = list(signal)
    
    real = list(data)
    imag = [0.0] * len(data)
    
    return _fft_cooley_tukey(real, imag)


def ifft(real: List[float], imag: List[float]) -> List[float]:
    """
    Compute Inverse Fast Fourier Transform.
    
    Args:
        real: Real part of spectrum
        imag: Imaginary part of spectrum
        
    Returns:
        Time domain signal
    """
    real_out, _ = _ifft_cooley_tukey(list(real), list(imag))
    return real_out


# ============================================================================
# Audio Loading and Saving
# ============================================================================

def load_audio(path: str, sample_rate: int = 16000) -> AudioTensor:
    """
    Load audio file (WAV format).
    
    Args:
        path: Path to audio file
        sample_rate: Target sample rate (default: 16000)
        
    Returns:
        AudioTensor with audio data
    """
    try:
        with wave.open(path, 'rb') as wav_file:
            n_channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()
            orig_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            
            # Read raw data
            raw_data = wav_file.readframes(n_frames)
            
            # Convert to samples
            if sampwidth == 1:
                # 8-bit unsigned
                samples = [((b - 128) / 128.0) for b in raw_data]
            elif sampwidth == 2:
                # 16-bit signed
                n_samples = len(raw_data) // 2
                samples = []
                for i in range(n_samples):
                    val = struct.unpack('<h', raw_data[i*2:(i+1)*2])[0]
                    samples.append(val / 32768.0)
            else:
                raise ValueError(f"Unsupported sample width: {sampwidth}")
            
            # Convert to mono if stereo
            if n_channels == 2:
                mono = []
                for i in range(0, len(samples), 2):
                    mono.append((samples[i] + samples[i+1]) / 2.0)
                samples = mono
            
            audio = AudioTensor(samples, orig_rate)
            
            # Resample if needed
            if orig_rate != sample_rate:
                audio = resample(audio, orig_rate, sample_rate)
            
            return audio
            
    except Exception as e:
        raise IOError(f"Failed to load audio: {e}")


def save_audio(audio: Union[AudioTensor, List[float]], path: str, 
               sample_rate: int = 16000):
    """
    Save audio tensor to WAV file.
    
    Args:
        audio: Audio tensor or list of samples
        path: Output path
        sample_rate: Sample rate
    """
    if isinstance(audio, AudioTensor):
        data = audio.data
        sample_rate = audio.sample_rate
    else:
        data = audio
    
    try:
        with wave.open(path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # Convert to 16-bit PCM
            pcm_data = b''
            for sample in data:
                # Clip to [-1, 1]
                sample = max(-1.0, min(1.0, sample))
                val = int(sample * 32767)
                pcm_data += struct.pack('<h', val)
            
            wav_file.writeframes(pcm_data)
            
    except Exception as e:
        raise IOError(f"Failed to save audio: {e}")


# ============================================================================
# Resampling
# ============================================================================

def resample(audio: Union[AudioTensor, List[float]], orig_sr: int, 
             target_sr: int) -> AudioTensor:
    """
    Resample audio to target sample rate using linear interpolation.
    Optimized for edge devices.
    
    Args:
        audio: Input audio
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled AudioTensor
    """
    if isinstance(audio, AudioTensor):
        data = audio.data
    else:
        data = list(audio)
    
    if orig_sr == target_sr:
        return AudioTensor(data, target_sr)
    
    # Calculate new length
    ratio = target_sr / orig_sr
    new_length = int(len(data) * ratio)
    
    # Linear interpolation
    resampled = []
    for i in range(new_length):
        # Source position
        src_pos = i / ratio
        src_idx = int(src_pos)
        frac = src_pos - src_idx
        
        if src_idx + 1 < len(data):
            val = data[src_idx] * (1 - frac) + data[src_idx + 1] * frac
        else:
            val = data[src_idx] if src_idx < len(data) else 0.0
        
        resampled.append(val)
    
    return AudioTensor(resampled, target_sr)


# ============================================================================
# Spectrograms
# ============================================================================

def _hann_window(size: int) -> List[float]:
    """Generate Hann window."""
    return [0.5 * (1 - math.cos(2 * math.pi * i / (size - 1))) for i in range(size)]


def _hamming_window(size: int) -> List[float]:
    """Generate Hamming window."""
    return [0.54 - 0.46 * math.cos(2 * math.pi * i / (size - 1)) for i in range(size)]


def spectrogram(audio: Union[AudioTensor, List[float]], 
                n_fft: int = 2048, 
                hop_length: int = 512,
                window: str = 'hann') -> List[List[float]]:
    """
    Compute Short-Time Fourier Transform (STFT) spectrogram.
    
    Args:
        audio: Input audio
        n_fft: FFT window size
        hop_length: Hop length between frames
        window: Window type ('hann' or 'hamming')
        
    Returns:
        Spectrogram as 2D list (frequency x time)
    """
    if isinstance(audio, AudioTensor):
        data = audio.data
    else:
        data = list(audio)
    
    # Generate window
    if window == 'hann':
        win = _hann_window(n_fft)
    elif window == 'hamming':
        win = _hamming_window(n_fft)
    else:
        win = [1.0] * n_fft
    
    # Number of frames
    n_frames = max(1, (len(data) - n_fft) // hop_length + 1)
    
    # Number of frequency bins (only positive frequencies)
    n_freqs = n_fft // 2 + 1
    
    spec = []
    for freq in range(n_freqs):
        spec.append([0.0] * n_frames)
    
    for frame_idx in range(n_frames):
        start = frame_idx * hop_length
        
        # Extract and window frame
        frame_real = []
        frame_imag = []
        for i in range(n_fft):
            if start + i < len(data):
                frame_real.append(data[start + i] * win[i])
            else:
                frame_real.append(0.0)
            frame_imag.append(0.0)
        
        # FFT
        fft_real, fft_imag = _fft_cooley_tukey(frame_real, frame_imag)
        
        # Compute magnitude for positive frequencies
        for freq in range(n_freqs):
            mag = math.sqrt(fft_real[freq]**2 + fft_imag[freq]**2)
            spec[freq][frame_idx] = mag
    
    return spec


def _hz_to_mel(hz: float) -> float:
    """Convert Hz to Mel scale."""
    return 2595 * math.log10(1 + hz / 700)


def _mel_to_hz(mel: float) -> float:
    """Convert Mel to Hz."""
    return 700 * (10 ** (mel / 2595) - 1)


def mel_filterbank(n_mels: int, n_fft: int, sample_rate: int,
                   fmin: float = 0.0, fmax: Optional[float] = None) -> List[List[float]]:
    """
    Create Mel filterbank.
    
    Args:
        n_mels: Number of Mel bands
        n_fft: FFT size
        sample_rate: Sample rate
        fmin: Minimum frequency
        fmax: Maximum frequency
        
    Returns:
        Filterbank as 2D list (n_mels x n_fft//2+1)
    """
    if fmax is None:
        fmax = sample_rate / 2
    
    # Mel points
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = [mel_min + (mel_max - mel_min) * i / (n_mels + 1) 
                  for i in range(n_mels + 2)]
    hz_points = [_mel_to_hz(m) for m in mel_points]
    
    # Frequency bins
    n_freqs = n_fft // 2 + 1
    freq_bins = [i * sample_rate / n_fft for i in range(n_freqs)]
    
    # Create filterbank
    filterbank = []
    for m in range(n_mels):
        filt = []
        for k in range(n_freqs):
            f = freq_bins[k]
            
            if f < hz_points[m]:
                val = 0.0
            elif f < hz_points[m + 1]:
                val = (f - hz_points[m]) / (hz_points[m + 1] - hz_points[m])
            elif f < hz_points[m + 2]:
                val = (hz_points[m + 2] - f) / (hz_points[m + 2] - hz_points[m + 1])
            else:
                val = 0.0
            
            filt.append(val)
        filterbank.append(filt)
    
    return filterbank


def mel_spectrogram(audio: Union[AudioTensor, List[float]], 
                    sample_rate: int = 16000,
                    n_fft: int = 2048, 
                    hop_length: int = 512,
                    n_mels: int = 128) -> List[List[float]]:
    """
    Compute Mel-scale spectrogram.
    
    Args:
        audio: Input audio
        sample_rate: Sample rate
        n_fft: FFT window size
        hop_length: Hop length
        n_mels: Number of Mel bands
        
    Returns:
        Mel-spectrogram as 2D list (n_mels x time)
    """
    # Get linear spectrogram
    spec = spectrogram(audio, n_fft, hop_length)
    
    # Get mel filterbank
    fb = mel_filterbank(n_mels, n_fft, sample_rate)
    
    n_freqs = len(spec)
    n_frames = len(spec[0])
    
    # Apply filterbank
    mel_spec = []
    for m in range(n_mels):
        mel_row = []
        for t in range(n_frames):
            val = 0.0
            for f in range(min(n_freqs, len(fb[m]))):
                val += fb[m][f] * spec[f][t]
            mel_row.append(val)
        mel_spec.append(mel_row)
    
    return mel_spec


# ============================================================================
# MFCC
# ============================================================================

def _dct_ii(x: List[float]) -> List[float]:
    """Type-II Discrete Cosine Transform."""
    n = len(x)
    result = []
    for k in range(n):
        val = 0.0
        for i in range(n):
            val += x[i] * math.cos(math.pi * k * (2 * i + 1) / (2 * n))
        result.append(val)
    return result


def mfcc(audio: Union[AudioTensor, List[float]], 
         sample_rate: int = 16000,
         n_mfcc: int = 13,
         n_mels: int = 40,
         n_fft: int = 2048,
         hop_length: int = 512) -> List[List[float]]:
    """
    Compute Mel-Frequency Cepstral Coefficients.
    
    Args:
        audio: Input audio
        sample_rate: Sample rate
        n_mfcc: Number of MFCC coefficients
        n_mels: Number of Mel bands
        n_fft: FFT size
        hop_length: Hop length
        
    Returns:
        MFCC features as 2D list (n_mfcc x time)
    """
    # Get mel spectrogram
    mel_spec = mel_spectrogram(audio, sample_rate, n_fft, hop_length, n_mels)
    
    n_frames = len(mel_spec[0])
    
    # Log mel spectrogram
    log_mel = []
    for row in mel_spec:
        log_mel.append([math.log(max(x, 1e-10)) for x in row])
    
    # DCT for each frame
    mfccs = []
    for k in range(n_mfcc):
        mfcc_row = []
        for t in range(n_frames):
            frame = [log_mel[m][t] for m in range(n_mels)]
            dct = _dct_ii(frame)
            mfcc_row.append(dct[k] if k < len(dct) else 0.0)
        mfccs.append(mfcc_row)
    
    return mfccs


# ============================================================================
# Audio Augmentation
# ============================================================================

def add_noise(audio: Union[AudioTensor, List[float]], 
              noise_level: float = 0.005) -> AudioTensor:
    """
    Add Gaussian noise to audio.
    
    Args:
        audio: Input audio
        noise_level: Noise amplitude
        
    Returns:
        Noisy audio
    """
    import random
    
    if isinstance(audio, AudioTensor):
        data = audio.data
        sr = audio.sample_rate
    else:
        data = list(audio)
        sr = 16000
    
    # Box-Muller transform for Gaussian noise
    noisy = []
    for i in range(len(data)):
        u1 = random.random()
        u2 = random.random()
        noise = math.sqrt(-2 * math.log(max(u1, 1e-12))) * math.cos(2 * math.pi * u2)
        noisy.append(data[i] + noise * noise_level)
    
    return AudioTensor(noisy, sr)


def time_stretch(audio: Union[AudioTensor, List[float]], 
                 rate: float = 1.0) -> AudioTensor:
    """
    Change audio speed without changing pitch (simple interpolation).
    
    Args:
        audio: Input audio
        rate: Speed factor (>1 = faster, <1 = slower)
        
    Returns:
        Time-stretched audio
    """
    if isinstance(audio, AudioTensor):
        data = audio.data
        sr = audio.sample_rate
    else:
        data = list(audio)
        sr = 16000
    
    if rate == 1.0:
        return AudioTensor(data.copy(), sr)
    
    # Simple resampling-based time stretch
    new_length = int(len(data) / rate)
    stretched = []
    
    for i in range(new_length):
        src_pos = i * rate
        src_idx = int(src_pos)
        frac = src_pos - src_idx
        
        if src_idx + 1 < len(data):
            val = data[src_idx] * (1 - frac) + data[src_idx + 1] * frac
        else:
            val = data[src_idx] if src_idx < len(data) else 0.0
        
        stretched.append(val)
    
    return AudioTensor(stretched, sr)


def pitch_shift(audio: Union[AudioTensor, List[float]], 
                n_steps: int = 0) -> AudioTensor:
    """
    Shift audio pitch by n semitones.
    Simple implementation using resampling.
    
    Args:
        audio: Input audio
        n_steps: Semitones to shift (positive = higher, negative = lower)
        
    Returns:
        Pitch-shifted audio
    """
    if isinstance(audio, AudioTensor):
        data = audio.data
        sr = audio.sample_rate
    else:
        data = list(audio)
        sr = 16000
    
    if n_steps == 0:
        return AudioTensor(data.copy(), sr)
    
    # Pitch shift ratio
    ratio = 2 ** (n_steps / 12)
    
    # Resample then time stretch
    resampled = resample(AudioTensor(data, sr), sr, int(sr * ratio))
    stretched = time_stretch(resampled, ratio)
    
    return AudioTensor(stretched.data, sr)


# ============================================================================
# Audio Utilities
# ============================================================================

def normalize(audio: Union[AudioTensor, List[float]], 
              peak: float = 1.0) -> AudioTensor:
    """Normalize audio to peak amplitude."""
    if isinstance(audio, AudioTensor):
        data = audio.data
        sr = audio.sample_rate
    else:
        data = list(audio)
        sr = 16000
    
    max_val = max(abs(x) for x in data) if data else 1.0
    if max_val > 0:
        scale = peak / max_val
        data = [x * scale for x in data]
    
    return AudioTensor(data, sr)


def trim_silence(audio: Union[AudioTensor, List[float]], 
                 threshold: float = 0.01) -> AudioTensor:
    """Remove silence from beginning and end."""
    if isinstance(audio, AudioTensor):
        data = audio.data
        sr = audio.sample_rate
    else:
        data = list(audio)
        sr = 16000
    
    # Find start
    start = 0
    for i, x in enumerate(data):
        if abs(x) > threshold:
            start = i
            break
    
    # Find end
    end = len(data)
    for i in range(len(data) - 1, -1, -1):
        if abs(data[i]) > threshold:
            end = i + 1
            break
    
    return AudioTensor(data[start:end], sr)


def power_to_db(power: Union[float, List[float]], 
                ref: float = 1.0, 
                amin: float = 1e-10) -> Union[float, List[float]]:
    """Convert power to decibels."""
    if isinstance(power, list):
        return [10 * math.log10(max(p / ref, amin)) for p in power]
    return 10 * math.log10(max(power / ref, amin))


def db_to_power(db: Union[float, List[float]], 
                ref: float = 1.0) -> Union[float, List[float]]:
    """Convert decibels to power."""
    if isinstance(db, list):
        return [ref * (10 ** (d / 10)) for d in db]
    return ref * (10 ** (db / 10))
