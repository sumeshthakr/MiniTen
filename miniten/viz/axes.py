"""
Axes class for MiniTen visualization.

Provides plotting methods and axes configuration.
"""

import numpy as np
from .colors import get_color


class Axes:
    """
    An axes object for creating plots.
    
    Similar to matplotlib.axes.Axes but optimized for performance.
    
    Attributes:
        figure: Parent Figure object
        bounds: Position and size [left, bottom, width, height]
        xlim: X-axis limits
        ylim: Y-axis limits
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Axes title
    """
    
    def __init__(self, figure, bounds):
        """
        Initialize Axes.
        
        Args:
            figure: Parent Figure
            bounds: [left, bottom, width, height] in figure coordinates (0-1)
        """
        self.figure = figure
        self.bounds = bounds
        self.xlim = None
        self.ylim = None
        self.xlabel_text = None
        self.ylabel_text = None
        self.title_text = None
        self._plots = []
        self._grid = False
        self._legend_data = []
    
    def plot(self, x, y=None, color=None, linestyle='-', linewidth=1.5, 
             marker=None, markersize=5, label=None, alpha=1.0):
        """
        Plot y vs x as lines and/or markers.
        
        Args:
            x: X coordinates (or y if y is None)
            y: Y coordinates
            color: Line color
            linestyle: Line style ('-', '--', '-.', ':')
            linewidth: Line width
            marker: Marker style ('o', 's', '^', etc.)
            markersize: Marker size
            label: Label for legend
            alpha: Transparency
            
        Returns:
            self for chaining
        """
        if y is None:
            y = np.asarray(x)
            x = np.arange(len(y))
        else:
            x = np.asarray(x)
            y = np.asarray(y)
        
        color = color or get_color(len(self._plots))
        
        plot_data = {
            'type': 'line',
            'x': x,
            'y': y,
            'color': color,
            'linestyle': linestyle,
            'linewidth': linewidth,
            'marker': marker,
            'markersize': markersize,
            'label': label,
            'alpha': alpha,
        }
        self._plots.append(plot_data)
        
        if label:
            self._legend_data.append((color, label))
        
        self._auto_limits(x, y)
        return self
    
    def scatter(self, x, y, s=20, c=None, marker='o', alpha=1.0, label=None):
        """
        Create a scatter plot.
        
        Args:
            x: X coordinates
            y: Y coordinates
            s: Marker size
            c: Marker color(s)
            marker: Marker style
            alpha: Transparency
            label: Label for legend
            
        Returns:
            self for chaining
        """
        x = np.asarray(x)
        y = np.asarray(y)
        c = c if c is not None else get_color(len(self._plots))
        
        plot_data = {
            'type': 'scatter',
            'x': x,
            'y': y,
            's': s,
            'c': c,
            'marker': marker,
            'alpha': alpha,
            'label': label,
        }
        self._plots.append(plot_data)
        
        if label:
            self._legend_data.append((c, label))
        
        self._auto_limits(x, y)
        return self
    
    def bar(self, x, height, width=0.8, bottom=None, color=None, 
            edgecolor='black', label=None, alpha=1.0):
        """
        Create a bar chart.
        
        Args:
            x: X positions of bars
            height: Height of bars
            width: Width of bars
            bottom: Y coordinate of bar bases
            color: Bar color
            edgecolor: Bar edge color
            label: Label for legend
            alpha: Transparency
            
        Returns:
            self for chaining
        """
        x = np.asarray(x)
        height = np.asarray(height)
        bottom = bottom if bottom is not None else np.zeros_like(height)
        color = color or get_color(len(self._plots))
        
        plot_data = {
            'type': 'bar',
            'x': x,
            'height': height,
            'width': width,
            'bottom': bottom,
            'color': color,
            'edgecolor': edgecolor,
            'label': label,
            'alpha': alpha,
        }
        self._plots.append(plot_data)
        
        if label:
            self._legend_data.append((color, label))
        
        self._auto_limits(x, bottom + height)
        return self
    
    def hist(self, x, bins=10, density=False, color=None, 
             edgecolor='black', label=None, alpha=1.0):
        """
        Create a histogram.
        
        Args:
            x: Input data
            bins: Number of bins or bin edges
            density: If True, normalize to form a probability density
            color: Bar color
            edgecolor: Edge color
            label: Label for legend
            alpha: Transparency
            
        Returns:
            (counts, bin_edges, patches)
        """
        x = np.asarray(x)
        counts, bin_edges = np.histogram(x, bins=bins, density=density)
        color = color or get_color(len(self._plots))
        
        plot_data = {
            'type': 'hist',
            'counts': counts,
            'bin_edges': bin_edges,
            'color': color,
            'edgecolor': edgecolor,
            'label': label,
            'alpha': alpha,
        }
        self._plots.append(plot_data)
        
        if label:
            self._legend_data.append((color, label))
        
        self._auto_limits(bin_edges, counts)
        return counts, bin_edges, None
    
    def imshow(self, data, cmap='viridis', vmin=None, vmax=None, 
               aspect='equal', origin='upper', alpha=1.0):
        """
        Display an image.
        
        Args:
            data: 2D array (grayscale) or 3D array (RGB)
            cmap: Colormap for grayscale images
            vmin: Minimum value for colormap
            vmax: Maximum value for colormap
            aspect: Aspect ratio ('equal', 'auto', or float)
            origin: Origin location ('upper' or 'lower')
            alpha: Transparency
            
        Returns:
            self for chaining
        """
        data = np.asarray(data)
        
        plot_data = {
            'type': 'image',
            'data': data,
            'cmap': cmap,
            'vmin': vmin if vmin is not None else np.min(data),
            'vmax': vmax if vmax is not None else np.max(data),
            'aspect': aspect,
            'origin': origin,
            'alpha': alpha,
        }
        self._plots.append(plot_data)
        
        # Set limits based on image size
        if self.xlim is None:
            self.xlim = (0, data.shape[1])
        if self.ylim is None:
            self.ylim = (0, data.shape[0])
        
        return self
    
    def heatmap(self, data, cmap='viridis', vmin=None, vmax=None,
                annot=False, fmt='.2f', cbar=True):
        """
        Create a heatmap.
        
        Args:
            data: 2D array of values
            cmap: Colormap
            vmin: Minimum value
            vmax: Maximum value
            annot: Whether to annotate cells with values
            fmt: String format for annotations
            cbar: Whether to show colorbar
            
        Returns:
            self for chaining
        """
        return self.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    
    def set_xlim(self, left=None, right=None):
        """Set x-axis limits."""
        if self.xlim is None:
            self.xlim = (0, 1)
        if left is not None:
            self.xlim = (left, self.xlim[1])
        if right is not None:
            self.xlim = (self.xlim[0], right)
        return self
    
    def set_ylim(self, bottom=None, top=None):
        """Set y-axis limits."""
        if self.ylim is None:
            self.ylim = (0, 1)
        if bottom is not None:
            self.ylim = (bottom, self.ylim[1])
        if top is not None:
            self.ylim = (self.ylim[0], top)
        return self
    
    def set_xlabel(self, label, fontsize=12):
        """Set x-axis label."""
        self.xlabel_text = label
        return self
    
    def set_ylabel(self, label, fontsize=12):
        """Set y-axis label."""
        self.ylabel_text = label
        return self
    
    def set_title(self, title, fontsize=14):
        """Set axes title."""
        self.title_text = title
        return self
    
    def legend(self, loc='best'):
        """Display legend."""
        self._show_legend = True
        self._legend_loc = loc
        return self
    
    def grid(self, visible=True, which='major', axis='both', 
             color='gray', alpha=0.5):
        """Configure grid lines."""
        self._grid = visible
        self._grid_color = color
        self._grid_alpha = alpha
        return self
    
    def axhline(self, y=0, color='black', linestyle='-', linewidth=1, alpha=1.0):
        """Add a horizontal line across the axes."""
        self._plots.append({
            'type': 'hline',
            'y': y,
            'color': color,
            'linestyle': linestyle,
            'linewidth': linewidth,
            'alpha': alpha,
        })
        return self
    
    def axvline(self, x=0, color='black', linestyle='-', linewidth=1, alpha=1.0):
        """Add a vertical line across the axes."""
        self._plots.append({
            'type': 'vline',
            'x': x,
            'color': color,
            'linestyle': linestyle,
            'linewidth': linewidth,
            'alpha': alpha,
        })
        return self
    
    def fill_between(self, x, y1, y2=0, color=None, alpha=0.5, label=None):
        """Fill area between two curves."""
        x = np.asarray(x)
        y1 = np.asarray(y1)
        y2 = np.asarray(y2) if hasattr(y2, '__len__') else np.full_like(y1, y2)
        color = color or get_color(len(self._plots))
        
        self._plots.append({
            'type': 'fill_between',
            'x': x,
            'y1': y1,
            'y2': y2,
            'color': color,
            'alpha': alpha,
            'label': label,
        })
        
        self._auto_limits(x, np.concatenate([y1, y2]))
        return self
    
    def text(self, x, y, s, fontsize=12, color='black', ha='left', va='bottom'):
        """Add text to the axes."""
        self._plots.append({
            'type': 'text',
            'x': x,
            'y': y,
            'text': s,
            'fontsize': fontsize,
            'color': color,
            'ha': ha,
            'va': va,
        })
        return self
    
    def _auto_limits(self, x, y):
        """Automatically adjust axis limits based on data."""
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        # Add padding
        x_pad = (x_max - x_min) * 0.05 or 0.5
        y_pad = (y_max - y_min) * 0.05 or 0.5
        
        if self.xlim is None:
            self.xlim = (x_min - x_pad, x_max + x_pad)
        else:
            self.xlim = (min(self.xlim[0], x_min - x_pad),
                        max(self.xlim[1], x_max + x_pad))
        
        if self.ylim is None:
            self.ylim = (y_min - y_pad, y_max + y_pad)
        else:
            self.ylim = (min(self.ylim[0], y_min - y_pad),
                        max(self.ylim[1], y_max + y_pad))
    
    def _update_bounds(self):
        """Update internal bounds for tight layout."""
        pass
    
    def _render(self, canvas):
        """Render this axes to the canvas."""
        # Calculate pixel coordinates
        fig = self.figure
        x0 = int(self.bounds[0] * fig.width)
        y0 = int((1 - self.bounds[1] - self.bounds[3]) * fig.height)
        w = int(self.bounds[2] * fig.width)
        h = int(self.bounds[3] * fig.height)
        
        # Draw axes background
        canvas[y0:y0+h, x0:x0+w] = 250  # Light gray background
        
        # Draw border
        canvas[y0, x0:x0+w] = 0
        canvas[y0+h-1, x0:x0+w] = 0
        canvas[y0:y0+h, x0] = 0
        canvas[y0:y0+h, x0+w-1] = 0
        
        # Render each plot
        for plot_data in self._plots:
            self._render_plot(canvas, plot_data, x0, y0, w, h)
    
    def _render_plot(self, canvas, plot_data, x0, y0, w, h):
        """Render a single plot element."""
        plot_type = plot_data['type']
        
        if plot_type == 'line':
            self._render_line(canvas, plot_data, x0, y0, w, h)
        elif plot_type == 'scatter':
            self._render_scatter(canvas, plot_data, x0, y0, w, h)
        elif plot_type == 'bar':
            self._render_bar(canvas, plot_data, x0, y0, w, h)
        elif plot_type == 'image':
            self._render_image(canvas, plot_data, x0, y0, w, h)
    
    def _data_to_pixel(self, x, y, x0, y0, w, h):
        """Convert data coordinates to pixel coordinates."""
        xlim = self.xlim or (0, 1)
        ylim = self.ylim or (0, 1)
        
        px = x0 + (x - xlim[0]) / (xlim[1] - xlim[0]) * w
        py = y0 + h - (y - ylim[0]) / (ylim[1] - ylim[0]) * h
        
        return px.astype(int) if hasattr(px, 'astype') else int(px), \
               py.astype(int) if hasattr(py, 'astype') else int(py)
    
    def _render_line(self, canvas, data, x0, y0, w, h):
        """Render a line plot."""
        x = data['x']
        y = data['y']
        color = self._parse_color(data['color'])
        
        px, py = self._data_to_pixel(x, y, x0, y0, w, h)
        
        # Draw line segments
        for i in range(len(px) - 1):
            self._draw_line(canvas, px[i], py[i], px[i+1], py[i+1], color)
    
    def _render_scatter(self, canvas, data, x0, y0, w, h):
        """Render a scatter plot."""
        x = data['x']
        y = data['y']
        color = self._parse_color(data['c'])
        s = data['s']
        
        px, py = self._data_to_pixel(x, y, x0, y0, w, h)
        
        # Draw circles
        r = int(np.sqrt(s) / 2)
        for i in range(len(px)):
            self._draw_circle(canvas, px[i], py[i], r, color)
    
    def _render_bar(self, canvas, data, x0, y0, w, h):
        """Render a bar chart."""
        x = data['x']
        height = data['height']
        color = self._parse_color(data['color'])
        
        px, py = self._data_to_pixel(x, height, x0, y0, w, h)
        _, py_base = self._data_to_pixel(x, data['bottom'], x0, y0, w, h)
        
        bar_width = max(1, w // len(x) - 2)
        
        for i in range(len(px)):
            x1 = max(x0, px[i] - bar_width // 2)
            x2 = min(x0 + w - 1, px[i] + bar_width // 2)
            y1 = min(py[i], py_base[i])
            y2 = max(py[i], py_base[i])
            
            canvas[y1:y2, x1:x2] = color
    
    def _render_image(self, canvas, data, x0, y0, w, h):
        """Render an image."""
        img = data['data']
        
        if len(img.shape) == 2:
            # Grayscale - apply colormap
            from .colors import colormap
            img = colormap(img, data['cmap'], data['vmin'], data['vmax'])
        
        # Resize to fit using nearest neighbor (no scipy dependency)
        scale_y = h / img.shape[0]
        scale_x = w / img.shape[1]
        
        resized = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                src_i = min(int(i / scale_y), img.shape[0] - 1)
                src_j = min(int(j / scale_x), img.shape[1] - 1)
                resized[i, j] = img[src_i, src_j]
        
        canvas[y0:y0+h, x0:x0+w] = resized
    
    def _draw_line(self, canvas, x1, y1, x2, y2, color):
        """Draw a line using Bresenham's algorithm."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        steep = dy > dx
        
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
                if 0 <= x < canvas.shape[0] and 0 <= y < canvas.shape[1]:
                    canvas[x, y] = color
            else:
                if 0 <= y < canvas.shape[0] and 0 <= x < canvas.shape[1]:
                    canvas[y, x] = color
            
            error -= dy
            if error < 0:
                y += y_step
                error += dx
    
    def _draw_circle(self, canvas, cx, cy, r, color):
        """Draw a filled circle."""
        for y in range(max(0, cy - r), min(canvas.shape[0], cy + r + 1)):
            for x in range(max(0, cx - r), min(canvas.shape[1], cx + r + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                    canvas[y, x] = color
    
    def _parse_color(self, color):
        """Parse color to RGB tuple."""
        if isinstance(color, str):
            from .colors import get_named_color
            return get_named_color(color)
        elif isinstance(color, (list, tuple, np.ndarray)):
            if len(color) == 3:
                return np.array(color, dtype=np.uint8)
        return np.array([0, 0, 255], dtype=np.uint8)  # Default blue
