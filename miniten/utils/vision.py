"""
Vision/Image Processing Utilities

Optimized image processing for computer vision tasks.

Features:
- Image loading and saving
- Preprocessing (resize, crop, normalize)
- Data augmentation
- Color space conversions
- Edge detection
"""

import numpy as np


def load_image(path):
    """
    Load image from file.
    
    Args:
        path: Path to image file
        
    Returns:
        Image as numpy array (H, W, C)
    """
    try:
        from PIL import Image
        img = Image.open(path)
        return np.array(img)
    except ImportError:
        try:
            import imageio
            return imageio.imread(path)
        except ImportError:
            raise ImportError("PIL or imageio required for image loading")


def save_image(array, path):
    """
    Save array as image.
    
    Args:
        array: Image array (H, W, C) or (H, W)
        path: Output path
    """
    array = np.asarray(array)
    
    # Ensure uint8
    if array.dtype != np.uint8:
        if array.max() <= 1:
            array = (array * 255).astype(np.uint8)
        else:
            array = array.astype(np.uint8)
    
    try:
        from PIL import Image
        img = Image.fromarray(array)
        img.save(path)
    except ImportError:
        try:
            import imageio
            imageio.imwrite(path, array)
        except ImportError:
            raise ImportError("PIL or imageio required for image saving")


def resize(image, size, interpolation='bilinear'):
    """
    Resize image to specified size.
    
    Args:
        image: Input image array (H, W, C) or (H, W)
        size: Target size (height, width)
        interpolation: Interpolation method ('nearest', 'bilinear')
        
    Returns:
        Resized image
    """
    image = np.asarray(image)
    target_h, target_w = size
    
    if len(image.shape) == 2:
        # Grayscale
        return _resize_2d(image, target_h, target_w, interpolation)
    else:
        # Color
        channels = []
        for c in range(image.shape[2]):
            channels.append(_resize_2d(image[:, :, c], target_h, target_w, interpolation))
        return np.stack(channels, axis=2)


def _resize_2d(image, target_h, target_w, interpolation='bilinear'):
    """Resize a 2D array."""
    src_h, src_w = image.shape
    
    output = np.zeros((target_h, target_w), dtype=image.dtype)
    
    scale_h = src_h / target_h
    scale_w = src_w / target_w
    
    for i in range(target_h):
        for j in range(target_w):
            src_i = i * scale_h
            src_j = j * scale_w
            
            if interpolation == 'nearest':
                output[i, j] = image[int(src_i), int(src_j)]
            else:  # bilinear
                i0 = int(src_i)
                j0 = int(src_j)
                i1 = min(i0 + 1, src_h - 1)
                j1 = min(j0 + 1, src_w - 1)
                
                di = src_i - i0
                dj = src_j - j0
                
                output[i, j] = (
                    image[i0, j0] * (1 - di) * (1 - dj) +
                    image[i0, j1] * (1 - di) * dj +
                    image[i1, j0] * di * (1 - dj) +
                    image[i1, j1] * di * dj
                )
    
    return output


def crop(image, top, left, height, width):
    """
    Crop image to specified region.
    
    Args:
        image: Input image
        top, left: Top-left corner coordinates
        height, width: Crop dimensions
        
    Returns:
        Cropped image
    """
    return image[top:top+height, left:left+width]


def center_crop(image, size):
    """
    Center crop image.
    
    Args:
        image: Input image
        size: (height, width) of crop
        
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    target_h, target_w = size
    
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    
    return crop(image, top, left, target_h, target_w)


def random_crop(image, size):
    """
    Random crop image.
    
    Args:
        image: Input image
        size: (height, width) of crop
        
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    target_h, target_w = size
    
    top = np.random.randint(0, h - target_h + 1)
    left = np.random.randint(0, w - target_w + 1)
    
    return crop(image, top, left, target_h, target_w)


def normalize(image, mean, std):
    """
    Normalize image with mean and standard deviation.
    
    Args:
        image: Input image (H, W, C)
        mean: Mean values for each channel
        std: Standard deviation for each channel
        
    Returns:
        Normalized image
    """
    image = np.asarray(image, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    
    return (image - mean) / std


def denormalize(image, mean, std):
    """
    Denormalize image.
    
    Args:
        image: Normalized image
        mean: Mean values used for normalization
        std: Std values used for normalization
        
    Returns:
        Denormalized image
    """
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    
    return image * std + mean


def to_grayscale(image):
    """
    Convert image to grayscale.
    
    Args:
        image: RGB image (H, W, 3)
        
    Returns:
        Grayscale image (H, W)
    """
    if len(image.shape) == 2:
        return image
    
    # Luminosity method
    return (0.299 * image[:, :, 0] + 
            0.587 * image[:, :, 1] + 
            0.114 * image[:, :, 2])


def flip_horizontal(image):
    """
    Flip image horizontally.
    
    Args:
        image: Input image
        
    Returns:
        Flipped image
    """
    return np.flip(image, axis=1)


def flip_vertical(image):
    """
    Flip image vertically.
    
    Args:
        image: Input image
        
    Returns:
        Flipped image
    """
    return np.flip(image, axis=0)


def rotate(image, angle):
    """
    Rotate image by specified angle.
    
    Args:
        image: Input image
        angle: Rotation angle in degrees
        
    Returns:
        Rotated image
    """
    # Simple rotation for 90, 180, 270 degrees
    k = int(angle / 90) % 4
    return np.rot90(image, k)


def adjust_brightness(image, factor):
    """
    Adjust image brightness.
    
    Args:
        image: Input image
        factor: Brightness factor (1.0 = no change)
        
    Returns:
        Adjusted image
    """
    return np.clip(image * factor, 0, 255).astype(image.dtype)


def adjust_contrast(image, factor):
    """
    Adjust image contrast.
    
    Args:
        image: Input image
        factor: Contrast factor (1.0 = no change)
        
    Returns:
        Adjusted image
    """
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    return np.clip((image - mean) * factor + mean, 0, 255).astype(image.dtype)


def random_brightness(image, max_delta=0.2):
    """Apply random brightness adjustment."""
    factor = 1.0 + np.random.uniform(-max_delta, max_delta)
    return adjust_brightness(image, factor)


def random_contrast(image, lower=0.8, upper=1.2):
    """Apply random contrast adjustment."""
    factor = np.random.uniform(lower, upper)
    return adjust_contrast(image, factor)


def random_flip(image, horizontal=True, vertical=False):
    """Randomly flip image."""
    if horizontal and np.random.random() > 0.5:
        image = flip_horizontal(image)
    if vertical and np.random.random() > 0.5:
        image = flip_vertical(image)
    return image


# Edge detection

def sobel_filter(image):
    """
    Apply Sobel edge detection filter.
    
    Args:
        image: Grayscale image
        
    Returns:
        Edge magnitude image
    """
    if len(image.shape) > 2:
        image = to_grayscale(image)
    
    image = image.astype(np.float64)
    
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    
    # Apply convolution
    gx = _convolve2d(image, sobel_x)
    gy = _convolve2d(image, sobel_y)
    
    # Magnitude
    magnitude = np.sqrt(gx**2 + gy**2)
    
    return magnitude


def _convolve2d(image, kernel):
    """Simple 2D convolution."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # Pad image
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    
    h, w = image.shape
    output = np.zeros((h, w), dtype=np.float64)
    
    for i in range(h):
        for j in range(w):
            output[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    
    return output


def canny_edge_detection(image, low_threshold, high_threshold):
    """
    Apply Canny edge detection.
    
    Args:
        image: Input image
        low_threshold: Low threshold for edge detection
        high_threshold: High threshold for edge detection
        
    Returns:
        Binary edge image
    """
    # Simplified Canny implementation
    
    # 1. Grayscale
    if len(image.shape) > 2:
        image = to_grayscale(image)
    
    # 2. Gaussian blur
    image = gaussian_blur(image, sigma=1.4)
    
    # 3. Gradient
    gx, gy = _sobel_gradients(image)
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)
    
    # 4. Non-maximum suppression
    nms = _non_max_suppression(magnitude, direction)
    
    # 5. Double threshold
    strong = nms > high_threshold
    weak = (nms >= low_threshold) & (nms <= high_threshold)
    
    # 6. Hysteresis
    edges = _hysteresis(strong, weak)
    
    return edges.astype(np.uint8) * 255


def gaussian_blur(image, sigma=1.0, kernel_size=None):
    """Apply Gaussian blur."""
    if kernel_size is None:
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
    
    # Create Gaussian kernel
    x = np.arange(kernel_size) - kernel_size // 2
    kernel_1d = np.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    
    return _convolve2d(image.astype(np.float64), kernel_2d)


def _sobel_gradients(image):
    """Get Sobel gradients."""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    
    return _convolve2d(image, sobel_x), _convolve2d(image, sobel_y)


def _non_max_suppression(magnitude, direction):
    """Non-maximum suppression for edge thinning."""
    h, w = magnitude.shape
    nms = np.zeros_like(magnitude)
    
    direction = direction * 180 / np.pi
    direction[direction < 0] += 180
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            angle = direction[i, j]
            
            # Determine neighbors to compare
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                n1, n2 = magnitude[i, j-1], magnitude[i, j+1]
            elif 22.5 <= angle < 67.5:
                n1, n2 = magnitude[i-1, j+1], magnitude[i+1, j-1]
            elif 67.5 <= angle < 112.5:
                n1, n2 = magnitude[i-1, j], magnitude[i+1, j]
            else:
                n1, n2 = magnitude[i-1, j-1], magnitude[i+1, j+1]
            
            if magnitude[i, j] >= n1 and magnitude[i, j] >= n2:
                nms[i, j] = magnitude[i, j]
    
    return nms


def _hysteresis(strong, weak):
    """Hysteresis thresholding."""
    edges = strong.copy()
    h, w = strong.shape
    
    # Iteratively connect weak edges to strong edges
    for _ in range(10):
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if weak[i, j] and np.any(edges[i-1:i+2, j-1:j+2]):
                    edges[i, j] = True
    
    return edges


# Preprocessing pipelines

class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class ToTensor:
    """Convert image to tensor format (C, H, W)."""
    
    def __call__(self, image):
        image = np.asarray(image, dtype=np.float32)
        
        if image.max() > 1:
            image = image / 255.0
        
        if len(image.shape) == 2:
            return image[np.newaxis, :, :]
        else:
            return np.transpose(image, (2, 0, 1))


class Normalize:
    """Normalize with mean and std."""
    
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
    
    def __call__(self, image):
        # Handle CHW format
        if len(image.shape) == 3 and image.shape[0] in [1, 3, 4]:
            mean = self.mean.reshape(-1, 1, 1)
            std = self.std.reshape(-1, 1, 1)
        else:
            mean = self.mean
            std = self.std
        
        return (image - mean) / std


class Resize:
    """Resize transform."""
    
    def __init__(self, size):
        self.size = size
    
    def __call__(self, image):
        return resize(image, self.size)


class CenterCrop:
    """Center crop transform."""
    
    def __init__(self, size):
        self.size = size
    
    def __call__(self, image):
        return center_crop(image, self.size)


class RandomCrop:
    """Random crop transform."""
    
    def __init__(self, size):
        self.size = size
    
    def __call__(self, image):
        return random_crop(image, self.size)


class RandomHorizontalFlip:
    """Random horizontal flip."""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image):
        if np.random.random() < self.p:
            return flip_horizontal(image)
        return image
