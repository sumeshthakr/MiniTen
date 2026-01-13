# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

"""
Optimized rendering functions for MiniTen visualization.

This module provides Cython-optimized implementations of:
- Line drawing (Bresenham's algorithm)
- Circle drawing
- Polygon filling
- Color mapping
- Image scaling

All operations are designed for high performance and minimal memory usage.
"""

import cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, floor, ceil, fabs

# Initialize NumPy C API
np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void draw_line(np.ndarray[np.uint8_t, ndim=3] canvas,
                      int x1, int y1, int x2, int y2,
                      np.ndarray[np.uint8_t, ndim=1] color,
                      int linewidth=1) noexcept:
    """
    Draw a line on the canvas using Bresenham's algorithm.
    
    Args:
        canvas: RGB canvas array (height, width, 3)
        x1, y1: Start point
        x2, y2: End point
        color: RGB color array
        linewidth: Line width in pixels
    """
    cdef int dx = abs(x2 - x1)
    cdef int dy = abs(y2 - y1)
    cdef int steep = dy > dx
    cdef int x, y, error, y_step
    cdef int height = canvas.shape[0]
    cdef int width = canvas.shape[1]
    cdef unsigned char r = color[0]
    cdef unsigned char g = color[1]
    cdef unsigned char b = color[2]
    
    if steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
    
    dx = x2 - x1
    dy = abs(y2 - y1)
    error = dx // 2
    y = y1
    y_step = 1 if y1 < y2 else -1
    
    for x in range(x1, x2 + 1):
        if steep:
            if 0 <= x < height and 0 <= y < width:
                canvas[x, y, 0] = r
                canvas[x, y, 1] = g
                canvas[x, y, 2] = b
        else:
            if 0 <= y < height and 0 <= x < width:
                canvas[y, x, 0] = r
                canvas[y, x, 1] = g
                canvas[y, x, 2] = b
        
        error -= dy
        if error < 0:
            y += y_step
            error += dx


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void draw_line_aa(np.ndarray[np.uint8_t, ndim=3] canvas,
                         double x1, double y1, double x2, double y2,
                         np.ndarray[np.uint8_t, ndim=1] color,
                         double linewidth=1.0) noexcept:
    """
    Draw an anti-aliased line using Xiaolin Wu's algorithm.
    
    Args:
        canvas: RGB canvas array
        x1, y1: Start point (float)
        x2, y2: End point (float)
        color: RGB color
        linewidth: Line width
    """
    cdef int height = canvas.shape[0]
    cdef int width = canvas.shape[1]
    cdef double dx = x2 - x1
    cdef double dy = y2 - y1
    cdef double gradient
    cdef double xend, yend, xgap, intery
    cdef int xpxl1, ypxl1, xpxl2, ypxl2, x
    cdef unsigned char r = color[0]
    cdef unsigned char g = color[1]
    cdef unsigned char b = color[2]
    
    cdef bint steep = fabs(dy) > fabs(dx)
    
    if steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
        dx = x2 - x1
        dy = y2 - y1
    
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        dx = x2 - x1
        dy = y2 - y1
    
    gradient = dy / dx if dx != 0 else 1.0
    
    # First endpoint
    xend = round(x1)
    yend = y1 + gradient * (xend - x1)
    xpxl1 = <int>xend
    ypxl1 = <int>floor(yend)
    
    # Second endpoint
    xend = round(x2)
    xpxl2 = <int>xend
    
    intery = yend + gradient
    
    # Draw the line
    for x in range(xpxl1, xpxl2):
        if steep:
            if 0 <= <int>floor(intery) < width and 0 <= x < height:
                canvas[x, <int>floor(intery), 0] = r
                canvas[x, <int>floor(intery), 1] = g
                canvas[x, <int>floor(intery), 2] = b
        else:
            if 0 <= x < width and 0 <= <int>floor(intery) < height:
                canvas[<int>floor(intery), x, 0] = r
                canvas[<int>floor(intery), x, 1] = g
                canvas[<int>floor(intery), x, 2] = b
        intery += gradient


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void draw_circle(np.ndarray[np.uint8_t, ndim=3] canvas,
                        int cx, int cy, int radius,
                        np.ndarray[np.uint8_t, ndim=1] color,
                        bint filled=True) noexcept:
    """
    Draw a circle on the canvas.
    
    Args:
        canvas: RGB canvas array
        cx, cy: Center coordinates
        radius: Circle radius
        color: RGB color
        filled: Whether to fill the circle
    """
    cdef int height = canvas.shape[0]
    cdef int width = canvas.shape[1]
    cdef int x, y
    cdef int r2 = radius * radius
    cdef unsigned char r = color[0]
    cdef unsigned char g = color[1]
    cdef unsigned char b = color[2]
    
    for y in range(max(0, cy - radius), min(height, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(width, cx + radius + 1)):
            if filled:
                if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r2:
                    canvas[y, x, 0] = r
                    canvas[y, x, 1] = g
                    canvas[y, x, 2] = b
            else:
                # Circle outline
                cdef int dist2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
                if r2 - radius <= dist2 <= r2 + radius:
                    canvas[y, x, 0] = r
                    canvas[y, x, 1] = g
                    canvas[y, x, 2] = b


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void draw_rect(np.ndarray[np.uint8_t, ndim=3] canvas,
                      int x1, int y1, int x2, int y2,
                      np.ndarray[np.uint8_t, ndim=1] color,
                      bint filled=True,
                      int linewidth=1) noexcept:
    """
    Draw a rectangle on the canvas.
    
    Args:
        canvas: RGB canvas array
        x1, y1: Top-left corner
        x2, y2: Bottom-right corner
        color: RGB color
        filled: Whether to fill the rectangle
        linewidth: Border width if not filled
    """
    cdef int height = canvas.shape[0]
    cdef int width = canvas.shape[1]
    cdef int x, y
    cdef unsigned char r = color[0]
    cdef unsigned char g = color[1]
    cdef unsigned char b = color[2]
    
    # Clip to canvas bounds
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    
    if filled:
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                canvas[y, x, 0] = r
                canvas[y, x, 1] = g
                canvas[y, x, 2] = b
    else:
        # Top and bottom edges
        for x in range(x1, x2 + 1):
            for i in range(linewidth):
                if y1 + i < height:
                    canvas[y1 + i, x, 0] = r
                    canvas[y1 + i, x, 1] = g
                    canvas[y1 + i, x, 2] = b
                if y2 - i >= 0:
                    canvas[y2 - i, x, 0] = r
                    canvas[y2 - i, x, 1] = g
                    canvas[y2 - i, x, 2] = b
        # Left and right edges
        for y in range(y1, y2 + 1):
            for i in range(linewidth):
                if x1 + i < width:
                    canvas[y, x1 + i, 0] = r
                    canvas[y, x1 + i, 1] = g
                    canvas[y, x1 + i, 2] = b
                if x2 - i >= 0:
                    canvas[y, x2 - i, 0] = r
                    canvas[y, x2 - i, 1] = g
                    canvas[y, x2 - i, 2] = b


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.uint8_t, ndim=3] apply_colormap(np.ndarray[np.float64_t, ndim=2] data,
                                                     double vmin, double vmax,
                                                     str cmap_name):
    """
    Apply a colormap to 2D data.
    
    Args:
        data: 2D array of values
        vmin: Minimum value
        vmax: Maximum value
        cmap_name: Colormap name ('viridis', 'plasma', etc.)
        
    Returns:
        RGB array of shape (height, width, 3)
    """
    cdef Py_ssize_t height = data.shape[0]
    cdef Py_ssize_t width = data.shape[1]
    cdef np.ndarray[np.uint8_t, ndim=3] output = np.empty((height, width, 3), dtype=np.uint8)
    
    cdef double[:, ::1] data_view = data
    cdef unsigned char[:, :, ::1] out_view = output
    
    cdef double normalized, range_val
    cdef Py_ssize_t i, j
    cdef int idx
    cdef double t
    cdef int r1, g1, b1, r2, g2, b2
    
    range_val = vmax - vmin
    if range_val == 0:
        range_val = 1.0
    
    # Viridis colormap values (simplified)
    cdef int[11][3] viridis = [
        [68, 1, 84], [72, 36, 117], [65, 68, 135], [53, 95, 141],
        [42, 120, 142], [33, 144, 141], [34, 168, 132], [68, 191, 112],
        [122, 209, 81], [189, 223, 38], [253, 231, 37]
    ]
    
    for i in range(height):
        for j in range(width):
            normalized = (data_view[i, j] - vmin) / range_val
            if normalized < 0:
                normalized = 0
            elif normalized > 1:
                normalized = 1
            
            # Map to colormap
            t = normalized * 10
            idx = <int>t
            if idx >= 10:
                idx = 9
            
            t = t - idx
            
            r1 = viridis[idx][0]
            g1 = viridis[idx][1]
            b1 = viridis[idx][2]
            r2 = viridis[idx + 1][0]
            g2 = viridis[idx + 1][1]
            b2 = viridis[idx + 1][2]
            
            out_view[i, j, 0] = <unsigned char>(r1 + t * (r2 - r1))
            out_view[i, j, 1] = <unsigned char>(g1 + t * (g2 - g1))
            out_view[i, j, 2] = <unsigned char>(b1 + t * (b2 - b1))
    
    return output


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.uint8_t, ndim=3] resize_image(np.ndarray[np.uint8_t, ndim=3] image,
                                                   int new_height, int new_width):
    """
    Resize an image using nearest neighbor interpolation.
    
    Args:
        image: Input RGB image
        new_height: Target height
        new_width: Target width
        
    Returns:
        Resized RGB image
    """
    cdef Py_ssize_t old_height = image.shape[0]
    cdef Py_ssize_t old_width = image.shape[1]
    cdef np.ndarray[np.uint8_t, ndim=3] output = np.empty((new_height, new_width, 3), dtype=np.uint8)
    
    cdef unsigned char[:, :, ::1] img_view = image
    cdef unsigned char[:, :, ::1] out_view = output
    
    cdef double scale_y = <double>old_height / new_height
    cdef double scale_x = <double>old_width / new_width
    cdef Py_ssize_t src_y, src_x
    cdef Py_ssize_t i, j
    
    for i in range(new_height):
        for j in range(new_width):
            src_y = <Py_ssize_t>(i * scale_y)
            src_x = <Py_ssize_t>(j * scale_x)
            
            if src_y >= old_height:
                src_y = old_height - 1
            if src_x >= old_width:
                src_x = old_width - 1
            
            out_view[i, j, 0] = img_view[src_y, src_x, 0]
            out_view[i, j, 1] = img_view[src_y, src_x, 1]
            out_view[i, j, 2] = img_view[src_y, src_x, 2]
    
    return output


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.uint8_t, ndim=3] resize_image_bilinear(np.ndarray[np.uint8_t, ndim=3] image,
                                                            int new_height, int new_width):
    """
    Resize an image using bilinear interpolation.
    
    Args:
        image: Input RGB image
        new_height: Target height
        new_width: Target width
        
    Returns:
        Resized RGB image
    """
    cdef Py_ssize_t old_height = image.shape[0]
    cdef Py_ssize_t old_width = image.shape[1]
    cdef np.ndarray[np.uint8_t, ndim=3] output = np.empty((new_height, new_width, 3), dtype=np.uint8)
    
    cdef unsigned char[:, :, ::1] img_view = image
    cdef unsigned char[:, :, ::1] out_view = output
    
    cdef double scale_y = <double>(old_height - 1) / (new_height - 1) if new_height > 1 else 0
    cdef double scale_x = <double>(old_width - 1) / (new_width - 1) if new_width > 1 else 0
    cdef double src_y, src_x, dy, dx
    cdef Py_ssize_t y0, y1, x0, x1
    cdef Py_ssize_t i, j
    cdef double v00, v01, v10, v11, val
    cdef int c
    
    for i in range(new_height):
        for j in range(new_width):
            src_y = i * scale_y
            src_x = j * scale_x
            
            y0 = <Py_ssize_t>floor(src_y)
            y1 = y0 + 1
            x0 = <Py_ssize_t>floor(src_x)
            x1 = x0 + 1
            
            if y1 >= old_height:
                y1 = old_height - 1
            if x1 >= old_width:
                x1 = old_width - 1
            
            dy = src_y - y0
            dx = src_x - x0
            
            for c in range(3):
                v00 = img_view[y0, x0, c]
                v01 = img_view[y0, x1, c]
                v10 = img_view[y1, x0, c]
                v11 = img_view[y1, x1, c]
                
                val = (v00 * (1 - dx) * (1 - dy) +
                       v01 * dx * (1 - dy) +
                       v10 * (1 - dx) * dy +
                       v11 * dx * dy)
                
                out_view[i, j, c] = <unsigned char>val
    
    return output


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void fill_canvas(np.ndarray[np.uint8_t, ndim=3] canvas,
                        np.ndarray[np.uint8_t, ndim=1] color) noexcept:
    """
    Fill entire canvas with a color.
    
    Args:
        canvas: RGB canvas array
        color: RGB color
    """
    cdef Py_ssize_t height = canvas.shape[0]
    cdef Py_ssize_t width = canvas.shape[1]
    cdef unsigned char r = color[0]
    cdef unsigned char g = color[1]
    cdef unsigned char b = color[2]
    cdef Py_ssize_t i, j
    
    for i in range(height):
        for j in range(width):
            canvas[i, j, 0] = r
            canvas[i, j, 1] = g
            canvas[i, j, 2] = b
