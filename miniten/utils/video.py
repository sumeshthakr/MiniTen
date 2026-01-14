"""
Video Processing Utilities

Video processing for computer vision and analysis tasks.
Optimized for edge devices with minimal dependencies.

Features:
- Frame extraction and manipulation
- Video loading (basic support)
- Temporal operations
- Motion detection
- Optical flow estimation
- Optimized for real-time edge processing
- Minimal external dependencies
"""

import math
import struct
from typing import List, Tuple, Optional, Union, Generator


# ============================================================================
# Core Video Types
# ============================================================================

class VideoFrame:
    """
    Single video frame (image).
    Stores pixel data as flat list for memory efficiency.
    """
    
    def __init__(self, data: List[float], width: int, height: int, 
                 channels: int = 3):
        self.data = data
        self.width = width
        self.height = height
        self.channels = channels
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return (height, width, channels)."""
        return (self.height, self.width, self.channels)
    
    def get_pixel(self, x: int, y: int) -> List[float]:
        """Get pixel value at (x, y)."""
        idx = (y * self.width + x) * self.channels
        return self.data[idx:idx + self.channels]
    
    def set_pixel(self, x: int, y: int, value: List[float]):
        """Set pixel value at (x, y)."""
        idx = (y * self.width + x) * self.channels
        for i, v in enumerate(value):
            self.data[idx + i] = v
    
    def to_grayscale(self) -> 'VideoFrame':
        """Convert to grayscale."""
        if self.channels == 1:
            return VideoFrame(self.data.copy(), self.width, self.height, 1)
        
        gray_data = []
        for i in range(0, len(self.data), self.channels):
            # Luminosity method
            gray = (0.299 * self.data[i] + 
                    0.587 * self.data[i + 1] + 
                    0.114 * self.data[i + 2])
            gray_data.append(gray)
        
        return VideoFrame(gray_data, self.width, self.height, 1)
    
    def resize(self, new_width: int, new_height: int) -> 'VideoFrame':
        """Resize frame using bilinear interpolation."""
        new_data = []
        
        scale_x = self.width / new_width
        scale_y = self.height / new_height
        
        for y in range(new_height):
            for x in range(new_width):
                # Source coordinates
                src_x = x * scale_x
                src_y = y * scale_y
                
                x0 = int(src_x)
                y0 = int(src_y)
                x1 = min(x0 + 1, self.width - 1)
                y1 = min(y0 + 1, self.height - 1)
                
                dx = src_x - x0
                dy = src_y - y0
                
                for c in range(self.channels):
                    # Bilinear interpolation
                    p00 = self.data[(y0 * self.width + x0) * self.channels + c]
                    p01 = self.data[(y0 * self.width + x1) * self.channels + c]
                    p10 = self.data[(y1 * self.width + x0) * self.channels + c]
                    p11 = self.data[(y1 * self.width + x1) * self.channels + c]
                    
                    val = (p00 * (1 - dx) * (1 - dy) +
                           p01 * dx * (1 - dy) +
                           p10 * (1 - dx) * dy +
                           p11 * dx * dy)
                    new_data.append(val)
        
        return VideoFrame(new_data, new_width, new_height, self.channels)
    
    def crop(self, x: int, y: int, width: int, height: int) -> 'VideoFrame':
        """Crop region from frame."""
        cropped = []
        for row in range(y, y + height):
            for col in range(x, x + width):
                if 0 <= row < self.height and 0 <= col < self.width:
                    idx = (row * self.width + col) * self.channels
                    cropped.extend(self.data[idx:idx + self.channels])
        
        return VideoFrame(cropped, width, height, self.channels)
    
    def copy(self) -> 'VideoFrame':
        """Create a copy of the frame."""
        return VideoFrame(self.data.copy(), self.width, self.height, self.channels)


class VideoClip:
    """
    Collection of video frames.
    """
    
    def __init__(self, frames: List[VideoFrame], fps: float = 30.0):
        self.frames = frames
        self.fps = fps
    
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self.frames) / self.fps
    
    @property
    def num_frames(self) -> int:
        """Number of frames."""
        return len(self.frames)
    
    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Return (num_frames, height, width, channels)."""
        if not self.frames:
            return (0, 0, 0, 0)
        f = self.frames[0]
        return (len(self.frames), f.height, f.width, f.channels)
    
    def __getitem__(self, idx: int) -> VideoFrame:
        return self.frames[idx]
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def __iter__(self) -> Generator[VideoFrame, None, None]:
        yield from self.frames
    
    def get_frame_at_time(self, time: float) -> VideoFrame:
        """Get frame at specific time (in seconds)."""
        frame_idx = int(time * self.fps)
        frame_idx = max(0, min(frame_idx, len(self.frames) - 1))
        return self.frames[frame_idx]
    
    def subclip(self, start_time: float, end_time: float) -> 'VideoClip':
        """Extract a subclip."""
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        
        start_frame = max(0, start_frame)
        end_frame = min(len(self.frames), end_frame)
        
        return VideoClip(self.frames[start_frame:end_frame], self.fps)


# ============================================================================
# Frame Extraction and Manipulation
# ============================================================================

def create_frames_from_data(data: List[List[float]], 
                            width: int, height: int,
                            channels: int = 3) -> List[VideoFrame]:
    """
    Create video frames from raw data.
    
    Args:
        data: List of frame data (each frame is flattened pixel data)
        width: Frame width
        height: Frame height
        channels: Number of color channels
        
    Returns:
        List of VideoFrame objects
    """
    return [VideoFrame(frame_data, width, height, channels) for frame_data in data]


def extract_keyframes(clip: VideoClip, 
                      threshold: float = 0.1,
                      min_interval: int = 5) -> List[Tuple[int, VideoFrame]]:
    """
    Extract keyframes based on frame difference.
    
    Args:
        clip: Input video clip
        threshold: Difference threshold for keyframe detection
        min_interval: Minimum frames between keyframes
        
    Returns:
        List of (frame_index, frame) tuples
    """
    if len(clip) == 0:
        return []
    
    keyframes = [(0, clip[0])]
    last_keyframe_idx = 0
    
    for i in range(1, len(clip)):
        if i - last_keyframe_idx < min_interval:
            continue
        
        # Compute frame difference
        diff = frame_difference(clip[i], clip[last_keyframe_idx])
        
        if diff > threshold:
            keyframes.append((i, clip[i]))
            last_keyframe_idx = i
    
    return keyframes


def frame_difference(frame1: VideoFrame, frame2: VideoFrame) -> float:
    """
    Compute normalized difference between two frames.
    
    Args:
        frame1, frame2: Input frames
        
    Returns:
        Difference score (0 to 1)
    """
    if len(frame1.data) != len(frame2.data):
        return 1.0
    
    total_diff = sum(abs(a - b) for a, b in zip(frame1.data, frame2.data))
    max_diff = len(frame1.data) * 255.0  # Assuming 0-255 range
    
    return total_diff / max_diff


def temporal_subsample(clip: VideoClip, factor: int) -> VideoClip:
    """
    Subsample video by taking every nth frame.
    
    Args:
        clip: Input video clip
        factor: Subsampling factor
        
    Returns:
        Subsampled video clip
    """
    subsampled = [clip[i] for i in range(0, len(clip), factor)]
    return VideoClip(subsampled, clip.fps / factor)


def temporal_smooth(clip: VideoClip, window_size: int = 3) -> VideoClip:
    """
    Apply temporal smoothing (averaging over time).
    
    Args:
        clip: Input video clip
        window_size: Number of frames to average
        
    Returns:
        Smoothed video clip
    """
    if len(clip) < window_size:
        return clip
    
    smoothed_frames = []
    half_window = window_size // 2
    
    for i in range(len(clip)):
        start = max(0, i - half_window)
        end = min(len(clip), i + half_window + 1)
        
        # Average frames in window
        avg_data = [0.0] * len(clip[i].data)
        count = end - start
        
        for j in range(start, end):
            for k in range(len(avg_data)):
                avg_data[k] += clip[j].data[k]
        
        avg_data = [v / count for v in avg_data]
        
        smoothed_frames.append(VideoFrame(
            avg_data, clip[i].width, clip[i].height, clip[i].channels
        ))
    
    return VideoClip(smoothed_frames, clip.fps)


# ============================================================================
# Motion Detection
# ============================================================================

def detect_motion(frame1: VideoFrame, frame2: VideoFrame,
                  threshold: float = 30.0) -> List[Tuple[int, int, int, int]]:
    """
    Detect motion between two frames.
    Returns bounding boxes of motion regions.
    
    Args:
        frame1, frame2: Consecutive frames
        threshold: Motion detection threshold
        
    Returns:
        List of bounding boxes (x, y, width, height)
    """
    # Convert to grayscale
    gray1 = frame1.to_grayscale()
    gray2 = frame2.to_grayscale()
    
    # Compute absolute difference
    diff_data = [abs(a - b) for a, b in zip(gray1.data, gray2.data)]
    
    # Threshold
    motion_mask = [1 if d > threshold else 0 for d in diff_data]
    
    # Find connected components (simple blob detection)
    return _find_motion_blobs(motion_mask, gray1.width, gray1.height)


def _find_motion_blobs(mask: List[int], width: int, height: int,
                       min_area: int = 100) -> List[Tuple[int, int, int, int]]:
    """Find bounding boxes of motion blobs."""
    visited = [False] * len(mask)
    blobs = []
    
    def flood_fill(start_x: int, start_y: int) -> Tuple[int, int, int, int]:
        """Flood fill to find blob bounds."""
        stack = [(start_x, start_y)]
        min_x, max_x = start_x, start_x
        min_y, max_y = start_y, start_y
        area = 0
        
        while stack:
            x, y = stack.pop()
            idx = y * width + x
            
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            if visited[idx] or mask[idx] == 0:
                continue
            
            visited[idx] = True
            area += 1
            
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            
            # 4-connectivity
            stack.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
        
        return (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1, area)
    
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            if mask[idx] == 1 and not visited[idx]:
                result = flood_fill(x, y)
                if result[4] >= min_area:  # Filter by area
                    blobs.append(result[:4])
    
    return blobs


def compute_motion_energy(clip: VideoClip) -> List[float]:
    """
    Compute motion energy for each frame.
    
    Args:
        clip: Input video clip
        
    Returns:
        List of motion energy values per frame
    """
    if len(clip) < 2:
        return [0.0] * len(clip)
    
    energies = [0.0]
    
    for i in range(1, len(clip)):
        diff = frame_difference(clip[i], clip[i-1])
        energies.append(diff)
    
    return energies


# ============================================================================
# Optical Flow (Lucas-Kanade)
# ============================================================================

def compute_optical_flow(frame1: VideoFrame, frame2: VideoFrame,
                         window_size: int = 5) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Compute dense optical flow using Lucas-Kanade method.
    
    Args:
        frame1, frame2: Consecutive frames
        window_size: Window size for gradient computation
        
    Returns:
        Tuple of (u, v) flow fields
    """
    gray1 = frame1.to_grayscale()
    gray2 = frame2.to_grayscale()
    
    w, h = gray1.width, gray1.height
    
    # Compute gradients
    ix = _compute_gradient_x(gray1)
    iy = _compute_gradient_y(gray1)
    it = [b - a for a, b in zip(gray1.data, gray2.data)]
    
    # Initialize flow fields
    u = [[0.0] * w for _ in range(h)]
    v = [[0.0] * w for _ in range(h)]
    
    half_win = window_size // 2
    
    for y in range(half_win, h - half_win):
        for x in range(half_win, w - half_win):
            # Collect window values
            a11 = a12 = a22 = b1 = b2 = 0.0
            
            for wy in range(-half_win, half_win + 1):
                for wx in range(-half_win, half_win + 1):
                    idx = (y + wy) * w + (x + wx)
                    
                    ix_val = ix[idx]
                    iy_val = iy[idx]
                    it_val = it[idx]
                    
                    a11 += ix_val * ix_val
                    a12 += ix_val * iy_val
                    a22 += iy_val * iy_val
                    b1 -= ix_val * it_val
                    b2 -= iy_val * it_val
            
            # Solve 2x2 system
            det = a11 * a22 - a12 * a12
            if abs(det) > 1e-6:
                u[y][x] = (a22 * b1 - a12 * b2) / det
                v[y][x] = (a11 * b2 - a12 * b1) / det
    
    return u, v


def _compute_gradient_x(frame: VideoFrame) -> List[float]:
    """Compute horizontal gradient."""
    w, h = frame.width, frame.height
    grad = [0.0] * len(frame.data)
    
    for y in range(h):
        for x in range(1, w - 1):
            idx = y * w + x
            grad[idx] = (frame.data[idx + 1] - frame.data[idx - 1]) / 2
    
    return grad


def _compute_gradient_y(frame: VideoFrame) -> List[float]:
    """Compute vertical gradient."""
    w, h = frame.width, frame.height
    grad = [0.0] * len(frame.data)
    
    for y in range(1, h - 1):
        for x in range(w):
            idx = y * w + x
            grad[idx] = (frame.data[idx + w] - frame.data[idx - w]) / 2
    
    return grad


def visualize_optical_flow(u: List[List[float]], v: List[List[float]],
                           scale: float = 10.0) -> VideoFrame:
    """
    Visualize optical flow as color image.
    
    Args:
        u, v: Flow fields
        scale: Visualization scale
        
    Returns:
        Color visualization of flow
    """
    h = len(u)
    w = len(u[0]) if h > 0 else 0
    
    data = []
    for y in range(h):
        for x in range(w):
            # Convert flow to angle and magnitude
            angle = math.atan2(v[y][x], u[y][x])
            mag = math.sqrt(u[y][x]**2 + v[y][x]**2) * scale
            
            # HSV to RGB (hue from angle, saturation=1, value from magnitude)
            hue = (angle + math.pi) / (2 * math.pi)
            sat = 1.0
            val = min(mag / 20, 1.0)
            
            r, g, b = _hsv_to_rgb(hue, sat, val)
            data.extend([r * 255, g * 255, b * 255])
    
    return VideoFrame(data, w, h, 3)


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    """Convert HSV to RGB."""
    if s == 0:
        return v, v, v
    
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    
    i = i % 6
    if i == 0:
        return v, t, p
    elif i == 1:
        return q, v, p
    elif i == 2:
        return p, v, t
    elif i == 3:
        return p, q, v
    elif i == 4:
        return t, p, v
    else:
        return v, p, q


# ============================================================================
# Video Processing Transforms
# ============================================================================

def resize_clip(clip: VideoClip, width: int, height: int) -> VideoClip:
    """Resize all frames in clip."""
    resized = [frame.resize(width, height) for frame in clip]
    return VideoClip(resized, clip.fps)


def grayscale_clip(clip: VideoClip) -> VideoClip:
    """Convert clip to grayscale."""
    gray = [frame.to_grayscale() for frame in clip]
    return VideoClip(gray, clip.fps)


def normalize_clip(clip: VideoClip, mean: List[float] = None, 
                   std: List[float] = None) -> VideoClip:
    """
    Normalize clip with mean and standard deviation.
    
    Args:
        clip: Input clip
        mean: Per-channel mean (default: compute from clip)
        std: Per-channel std (default: compute from clip)
        
    Returns:
        Normalized clip
    """
    if not clip.frames:
        return clip
    
    channels = clip.frames[0].channels
    
    # Compute mean and std if not provided
    if mean is None or std is None:
        channel_sums = [0.0] * channels
        channel_sq_sums = [0.0] * channels
        count = 0
        
        for frame in clip:
            for i, val in enumerate(frame.data):
                channel_sums[i % channels] += val
                channel_sq_sums[i % channels] += val * val
            count += len(frame.data) // channels
        
        if count > 0:
            mean = [s / count for s in channel_sums]
            variance = [(sq / count) - (m ** 2) 
                       for sq, m in zip(channel_sq_sums, mean)]
            std = [max(math.sqrt(v), 1e-6) for v in variance]
        else:
            mean = [0.0] * channels
            std = [1.0] * channels
    
    normalized_frames = []
    for frame in clip:
        norm_data = []
        for i, val in enumerate(frame.data):
            c = i % channels
            norm_data.append((val - mean[c]) / std[c])
        normalized_frames.append(VideoFrame(
            norm_data, frame.width, frame.height, channels
        ))
    
    return VideoClip(normalized_frames, clip.fps)


def crop_clip(clip: VideoClip, x: int, y: int, 
              width: int, height: int) -> VideoClip:
    """Crop all frames in clip."""
    cropped = [frame.crop(x, y, width, height) for frame in clip]
    return VideoClip(cropped, clip.fps)


def flip_horizontal_clip(clip: VideoClip) -> VideoClip:
    """Flip all frames horizontally."""
    flipped_frames = []
    for frame in clip:
        new_data = []
        for y in range(frame.height):
            row_start = y * frame.width * frame.channels
            for x in range(frame.width - 1, -1, -1):
                idx = row_start + x * frame.channels
                new_data.extend(frame.data[idx:idx + frame.channels])
        flipped_frames.append(VideoFrame(
            new_data, frame.width, frame.height, frame.channels
        ))
    return VideoClip(flipped_frames, clip.fps)


# ============================================================================
# Video Analysis
# ============================================================================

def compute_histogram(frame: VideoFrame, bins: int = 256) -> List[List[int]]:
    """
    Compute color histogram for frame.
    
    Args:
        frame: Input frame
        bins: Number of histogram bins
        
    Returns:
        List of histograms per channel
    """
    histograms = [[0] * bins for _ in range(frame.channels)]
    
    for i, val in enumerate(frame.data):
        c = i % frame.channels
        bin_idx = min(int(val / 256 * bins), bins - 1)
        histograms[c][bin_idx] += 1
    
    return histograms


def compute_temporal_histogram(clip: VideoClip, 
                               bins: int = 256) -> List[List[List[int]]]:
    """
    Compute histogram for each frame in clip.
    
    Returns:
        List of histograms per frame
    """
    return [compute_histogram(frame, bins) for frame in clip]


def scene_change_detection(clip: VideoClip, 
                           threshold: float = 0.3) -> List[int]:
    """
    Detect scene changes in video.
    
    Args:
        clip: Input video clip
        threshold: Scene change threshold
        
    Returns:
        List of frame indices where scenes change
    """
    if len(clip) < 2:
        return []
    
    changes = []
    prev_hist = compute_histogram(clip[0])
    
    for i in range(1, len(clip)):
        curr_hist = compute_histogram(clip[i])
        
        # Compare histograms
        diff = _histogram_difference(prev_hist, curr_hist)
        
        if diff > threshold:
            changes.append(i)
        
        prev_hist = curr_hist
    
    return changes


def _histogram_difference(hist1: List[List[int]], 
                          hist2: List[List[int]]) -> float:
    """Compute normalized histogram difference."""
    total_diff = 0.0
    total_sum = 0.0
    
    for c in range(len(hist1)):
        for b in range(len(hist1[c])):
            total_diff += abs(hist1[c][b] - hist2[c][b])
            total_sum += hist1[c][b] + hist2[c][b]
    
    return total_diff / max(total_sum, 1)
