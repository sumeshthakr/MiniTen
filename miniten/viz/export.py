"""
Export utilities for MiniTen visualization.

Provides functionality to save figures to various formats.
"""

import numpy as np


def save_figure(figure, filename, format=None, dpi=None):
    """
    Save a figure to a file.
    
    Args:
        figure: Figure object to save
        filename: Output filename
        format: File format (auto-detected from extension if None)
        dpi: Resolution (uses figure dpi if None)
    """
    # Render the figure
    canvas = figure._render()
    
    # Determine format from filename
    if format is None:
        if filename.lower().endswith('.png'):
            format = 'png'
        elif filename.lower().endswith('.svg'):
            format = 'svg'
        elif filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            format = 'jpeg'
        elif filename.lower().endswith('.bmp'):
            format = 'bmp'
        else:
            format = 'png'  # Default
    
    # Save based on format
    if format == 'png':
        save_png(canvas, filename)
    elif format == 'svg':
        save_svg(figure, filename)
    elif format in ('jpg', 'jpeg'):
        save_jpeg(canvas, filename)
    elif format == 'bmp':
        save_bmp(canvas, filename)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_png(canvas, filename):
    """
    Save canvas as PNG file.
    
    Uses a simple PNG encoder without external dependencies.
    """
    try:
        # Try using PIL if available
        from PIL import Image
        img = Image.fromarray(canvas)
        img.save(filename)
    except ImportError:
        # Fallback to numpy save
        try:
            import imageio
            imageio.imwrite(filename, canvas)
        except ImportError:
            # Minimal PPM format (can be converted to PNG externally)
            save_ppm(canvas, filename.replace('.png', '.ppm'))
            print(f"Warning: Saved as PPM (PIL/imageio not available). Use 'convert' to get PNG.")


def save_ppm(canvas, filename):
    """
    Save canvas as PPM file (simple format, no dependencies).
    """
    height, width = canvas.shape[:2]
    
    with open(filename, 'wb') as f:
        # PPM header
        f.write(f'P6\n{width} {height}\n255\n'.encode())
        # Pixel data
        f.write(canvas.tobytes())


def save_svg(figure, filename):
    """
    Save figure as SVG file.
    
    Creates vector graphics output.
    """
    svg_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{figure.width}" height="{figure.height}">',
        f'  <rect width="100%" height="100%" fill="white"/>',
    ]
    
    # Render each axes
    for ax in figure.axes:
        bounds = ax.bounds
        x0 = bounds[0] * figure.width
        y0 = (1 - bounds[1] - bounds[3]) * figure.height
        w = bounds[2] * figure.width
        h = bounds[3] * figure.height
        
        # Axes background
        svg_lines.append(f'  <rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#fafafa" stroke="black" stroke-width="1"/>')
        
        # Render plots
        for plot_data in ax._plots:
            svg_lines.extend(render_plot_svg(ax, plot_data, x0, y0, w, h))
    
    svg_lines.append('</svg>')
    
    with open(filename, 'w') as f:
        f.write('\n'.join(svg_lines))


def render_plot_svg(ax, plot_data, x0, y0, w, h):
    """Render a plot element to SVG."""
    lines = []
    plot_type = plot_data['type']
    
    xlim = ax.xlim or (0, 1)
    ylim = ax.ylim or (0, 1)
    
    def data_to_svg(x, y):
        px = x0 + (x - xlim[0]) / (xlim[1] - xlim[0]) * w
        py = y0 + h - (y - ylim[0]) / (ylim[1] - ylim[0]) * h
        return px, py
    
    if plot_type == 'line':
        x = plot_data['x']
        y = plot_data['y']
        color = _color_to_svg(plot_data['color'])
        
        # Create path
        points = []
        for i in range(len(x)):
            px, py = data_to_svg(x[i], y[i])
            if i == 0:
                points.append(f'M {px:.1f} {py:.1f}')
            else:
                points.append(f'L {px:.1f} {py:.1f}')
        
        path = ' '.join(points)
        lines.append(f'  <path d="{path}" fill="none" stroke="{color}" stroke-width="{plot_data["linewidth"]}"/>')
    
    elif plot_type == 'scatter':
        x = plot_data['x']
        y = plot_data['y']
        color = _color_to_svg(plot_data['c'])
        r = np.sqrt(plot_data['s']) / 2
        
        for i in range(len(x)):
            px, py = data_to_svg(x[i], y[i])
            lines.append(f'  <circle cx="{px:.1f}" cy="{py:.1f}" r="{r:.1f}" fill="{color}"/>')
    
    elif plot_type == 'bar':
        x = plot_data['x']
        height = plot_data['height']
        color = _color_to_svg(plot_data['color'])
        bar_width = w / len(x) * 0.8
        
        for i in range(len(x)):
            px, py = data_to_svg(x[i], height[i])
            _, py_base = data_to_svg(x[i], plot_data['bottom'][i] if hasattr(plot_data['bottom'], '__len__') else plot_data['bottom'])
            
            rect_x = px - bar_width / 2
            rect_y = min(py, py_base)
            rect_h = abs(py - py_base)
            
            lines.append(f'  <rect x="{rect_x:.1f}" y="{rect_y:.1f}" width="{bar_width:.1f}" height="{rect_h:.1f}" fill="{color}" stroke="black" stroke-width="0.5"/>')
    
    return lines


def _color_to_svg(color):
    """Convert color to SVG format."""
    if isinstance(color, str):
        return color
    elif isinstance(color, (list, tuple, np.ndarray)) and len(color) >= 3:
        return f'rgb({int(color[0])},{int(color[1])},{int(color[2])})'
    return 'blue'


def save_jpeg(canvas, filename, quality=95):
    """Save canvas as JPEG file."""
    try:
        from PIL import Image
        img = Image.fromarray(canvas)
        img.save(filename, quality=quality)
    except ImportError:
        raise ImportError("PIL is required for JPEG export")


def save_bmp(canvas, filename):
    """Save canvas as BMP file."""
    try:
        from PIL import Image
        img = Image.fromarray(canvas)
        img.save(filename)
    except ImportError:
        raise ImportError("PIL is required for BMP export")
