"""
Color utilities for MiniTen visualization.

Provides color manipulation, colormaps, and color schemes.
"""

import numpy as np


# Named colors (CSS colors)
NAMED_COLORS = {
    'red': (255, 0, 0),
    'green': (0, 128, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'gray': (128, 128, 128),
    'grey': (128, 128, 128),
    'orange': (255, 165, 0),
    'purple': (128, 0, 128),
    'pink': (255, 192, 203),
    'brown': (165, 42, 42),
    'navy': (0, 0, 128),
    'teal': (0, 128, 128),
    'olive': (128, 128, 0),
    'lime': (0, 255, 0),
    'aqua': (0, 255, 255),
    'maroon': (128, 0, 0),
    'silver': (192, 192, 192),
    'gold': (255, 215, 0),
}

# Default color cycle for plotting
DEFAULT_COLORS = [
    (31, 119, 180),   # Blue
    (255, 127, 14),   # Orange
    (44, 160, 44),    # Green
    (214, 39, 40),    # Red
    (148, 103, 189),  # Purple
    (140, 86, 75),    # Brown
    (227, 119, 194),  # Pink
    (127, 127, 127),  # Gray
    (188, 189, 34),   # Olive
    (23, 190, 207),   # Cyan
]


def get_color(index):
    """
    Get a color from the default color cycle.
    
    Args:
        index: Color index
        
    Returns:
        RGB tuple
    """
    return DEFAULT_COLORS[index % len(DEFAULT_COLORS)]


def get_named_color(name):
    """
    Get RGB values for a named color.
    
    Args:
        name: Color name (e.g., 'red', 'blue')
        
    Returns:
        RGB tuple as numpy array
    """
    name = name.lower().strip()
    
    # Handle hex colors
    if name.startswith('#'):
        return hex_to_rgb(name)
    
    if name in NAMED_COLORS:
        return np.array(NAMED_COLORS[name], dtype=np.uint8)
    
    # Default to blue if unknown
    return np.array((0, 0, 255), dtype=np.uint8)


def hex_to_rgb(hex_color):
    """
    Convert hex color to RGB.
    
    Args:
        hex_color: Hex color string (e.g., '#FF0000')
        
    Returns:
        RGB tuple as numpy array
    """
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    return np.array([r, g, b], dtype=np.uint8)


def rgb_to_hex(rgb):
    """
    Convert RGB to hex color.
    
    Args:
        rgb: RGB tuple (r, g, b)
        
    Returns:
        Hex color string
    """
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


# Colormaps
COLORMAPS = {
    'viridis': [
        (68, 1, 84), (72, 36, 117), (65, 68, 135), (53, 95, 141),
        (42, 120, 142), (33, 144, 141), (34, 168, 132), (68, 191, 112),
        (122, 209, 81), (189, 223, 38), (253, 231, 37)
    ],
    'plasma': [
        (13, 8, 135), (75, 3, 161), (125, 3, 168), (168, 34, 150),
        (203, 70, 121), (229, 107, 93), (248, 148, 65), (253, 195, 40),
        (240, 249, 33)
    ],
    'inferno': [
        (0, 0, 4), (40, 11, 84), (101, 21, 110), (159, 42, 99),
        (212, 72, 66), (245, 125, 21), (250, 193, 39), (252, 255, 164)
    ],
    'magma': [
        (0, 0, 4), (28, 16, 68), (79, 18, 123), (129, 37, 129),
        (181, 54, 122), (229, 80, 100), (251, 135, 97), (254, 194, 135),
        (252, 253, 191)
    ],
    'hot': [
        (0, 0, 0), (128, 0, 0), (255, 0, 0), (255, 128, 0),
        (255, 255, 0), (255, 255, 128), (255, 255, 255)
    ],
    'cool': [
        (0, 255, 255), (64, 192, 255), (128, 128, 255), (192, 64, 255),
        (255, 0, 255)
    ],
    'gray': [
        (0, 0, 0), (255, 255, 255)
    ],
    'jet': [
        (0, 0, 128), (0, 0, 255), (0, 128, 255), (0, 255, 255),
        (128, 255, 128), (255, 255, 0), (255, 128, 0), (255, 0, 0),
        (128, 0, 0)
    ],
}


def colormap(data, name='viridis', vmin=None, vmax=None):
    """
    Apply a colormap to data.
    
    Args:
        data: 2D array of values
        name: Colormap name
        vmin: Minimum value (default: data min)
        vmax: Maximum value (default: data max)
        
    Returns:
        RGB array of shape (height, width, 3)
    """
    data = np.asarray(data, dtype=np.float64)
    
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)
    
    # Normalize data to 0-1
    if vmax > vmin:
        normalized = (data - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(data)
    
    normalized = np.clip(normalized, 0, 1)
    
    # Get colormap
    cmap_colors = COLORMAPS.get(name, COLORMAPS['viridis'])
    n_colors = len(cmap_colors)
    
    # Create output array
    output = np.zeros((*data.shape, 3), dtype=np.uint8)
    
    # Apply colormap
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = normalized[i, j]
            idx = val * (n_colors - 1)
            idx_low = int(idx)
            idx_high = min(idx_low + 1, n_colors - 1)
            t = idx - idx_low
            
            # Interpolate between colors
            c1 = np.array(cmap_colors[idx_low])
            c2 = np.array(cmap_colors[idx_high])
            color = c1 + t * (c2 - c1)
            
            output[i, j] = color.astype(np.uint8)
    
    return output


def blend_colors(color1, color2, t):
    """
    Blend two colors.
    
    Args:
        color1: First RGB color
        color2: Second RGB color
        t: Blend factor (0 = color1, 1 = color2)
        
    Returns:
        Blended RGB color
    """
    c1 = np.asarray(color1, dtype=np.float64)
    c2 = np.asarray(color2, dtype=np.float64)
    return (c1 + t * (c2 - c1)).astype(np.uint8)
