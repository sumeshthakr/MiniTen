"""
Metrics logging for MiniTen training monitor.

Provides efficient logging of training metrics with support for:
- Scalar metrics (loss, accuracy, etc.)
- Histograms
- Images
- Text logs
- Experiment tracking
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np


class MetricsLogger:
    """
    Efficient metrics logger for training monitoring.
    
    Logs metrics to files for later visualization and analysis.
    Optimized for low memory usage on edge devices.
    
    Args:
        log_dir: Directory to save logs
        experiment_name: Name of the experiment
        comment: Additional comment for the run
        flush_secs: How often to flush to disk (in seconds)
        
    Example:
        >>> logger = MetricsLogger("./runs", experiment_name="mnist_cnn")
        >>> logger.log("loss", 0.5, step=1)
        >>> logger.log("accuracy", 0.85, step=1)
        >>> logger.close()
    """
    
    def __init__(self, log_dir="./runs", experiment_name=None, comment="", flush_secs=30):
        """Initialize the metrics logger."""
        self.log_dir = Path(log_dir)
        self.flush_secs = flush_secs
        
        # Create unique run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            run_name = f"{experiment_name}_{timestamp}"
        else:
            run_name = timestamp
        
        if comment:
            run_name = f"{run_name}_{comment}"
        
        self.run_dir = self.log_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self._metrics = {}  # metric_name -> list of (step, value, timestamp)
        self._scalars_file = self.run_dir / "scalars.jsonl"
        self._last_flush = time.time()
        self._closed = False
        
        # Metadata
        self._metadata = {
            "experiment_name": experiment_name,
            "comment": comment,
            "start_time": timestamp,
            "metrics": [],
        }
        self._save_metadata()
    
    def log(self, name, value, step=None, timestamp=None):
        """
        Log a scalar metric.
        
        Args:
            name: Metric name
            value: Metric value
            step: Training step (auto-incremented if None)
            timestamp: Timestamp (current time if None)
        """
        if self._closed:
            raise RuntimeError("Logger is closed")
        
        if name not in self._metrics:
            self._metrics[name] = []
            self._metadata["metrics"].append(name)
        
        # Auto-increment step if not provided
        if step is None:
            if self._metrics[name]:
                step = self._metrics[name][-1][0] + 1
            else:
                step = 0
        
        timestamp = timestamp or time.time()
        
        # Handle numpy types
        if hasattr(value, 'item'):
            value = value.item()
        
        self._metrics[name].append((step, value, timestamp))
        
        # Append to file
        with open(self._scalars_file, "a") as f:
            entry = {"name": name, "step": step, "value": value, "timestamp": timestamp}
            f.write(json.dumps(entry) + "\n")
        
        # Flush if needed
        if time.time() - self._last_flush > self.flush_secs:
            self.flush()
    
    def log_metrics(self, metrics_dict, step=None):
        """
        Log multiple metrics at once.
        
        Args:
            metrics_dict: Dictionary of {metric_name: value}
            step: Training step
        """
        for name, value in metrics_dict.items():
            self.log(name, value, step=step)
    
    def log_histogram(self, name, values, step=None, bins=64):
        """
        Log a histogram of values.
        
        Args:
            name: Histogram name
            values: Array of values
            step: Training step
            bins: Number of bins
        """
        values = np.asarray(values)
        hist, bin_edges = np.histogram(values, bins=bins)
        
        hist_file = self.run_dir / "histograms.jsonl"
        with open(hist_file, "a") as f:
            entry = {
                "name": name,
                "step": step,
                "counts": hist.tolist(),
                "bin_edges": bin_edges.tolist(),
                "timestamp": time.time(),
            }
            f.write(json.dumps(entry) + "\n")
    
    def log_image(self, name, image, step=None):
        """
        Log an image.
        
        Args:
            name: Image name
            image: Image array (H, W, C) or (H, W)
            step: Training step
        """
        image = np.asarray(image)
        
        # Save image
        images_dir = self.run_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        step = step or 0
        filename = f"{name}_step{step}.npy"
        np.save(images_dir / filename, image)
        
        # Log metadata
        image_meta_file = self.run_dir / "images.jsonl"
        with open(image_meta_file, "a") as f:
            entry = {
                "name": name,
                "step": step,
                "filename": filename,
                "shape": list(image.shape),
                "timestamp": time.time(),
            }
            f.write(json.dumps(entry) + "\n")
    
    def log_text(self, name, text, step=None):
        """
        Log text.
        
        Args:
            name: Text name
            text: Text content
            step: Training step
        """
        text_file = self.run_dir / "text.jsonl"
        with open(text_file, "a") as f:
            entry = {
                "name": name,
                "step": step,
                "text": text,
                "timestamp": time.time(),
            }
            f.write(json.dumps(entry) + "\n")
    
    def log_hyperparams(self, hparams):
        """
        Log hyperparameters.
        
        Args:
            hparams: Dictionary of hyperparameters
        """
        self._metadata["hyperparams"] = hparams
        self._save_metadata()
        
        hparams_file = self.run_dir / "hparams.json"
        with open(hparams_file, "w") as f:
            json.dump(hparams, f, indent=2)
    
    def get_metric(self, name):
        """
        Get all values for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            List of (step, value, timestamp) tuples
        """
        return self._metrics.get(name, [])
    
    def get_latest(self, name):
        """
        Get the latest value for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Latest value or None
        """
        values = self._metrics.get(name, [])
        return values[-1][1] if values else None
    
    def flush(self):
        """Flush all buffered data to disk."""
        self._save_metadata()
        self._last_flush = time.time()
    
    def _save_metadata(self):
        """Save metadata to disk."""
        metadata_file = self.run_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(self._metadata, f, indent=2)
    
    def close(self):
        """Close the logger and finalize logs."""
        if not self._closed:
            self._metadata["end_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.flush()
            self._closed = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class Experiment:
    """
    Experiment tracker for comparing multiple runs.
    
    Args:
        experiments_dir: Directory containing experiment runs
        
    Example:
        >>> exp = Experiment("./runs")
        >>> runs = exp.list_runs()
        >>> comparison = exp.compare(["run1", "run2"], metric="loss")
    """
    
    def __init__(self, experiments_dir="./runs"):
        """Initialize experiment tracker."""
        self.experiments_dir = Path(experiments_dir)
    
    def list_runs(self):
        """
        List all experiment runs.
        
        Returns:
            List of run names
        """
        if not self.experiments_dir.exists():
            return []
        
        runs = []
        for d in self.experiments_dir.iterdir():
            if d.is_dir() and (d / "metadata.json").exists():
                runs.append(d.name)
        
        return sorted(runs)
    
    def get_run_metadata(self, run_name):
        """
        Get metadata for a run.
        
        Args:
            run_name: Name of the run
            
        Returns:
            Metadata dictionary
        """
        metadata_file = self.experiments_dir / run_name / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                return json.load(f)
        return {}
    
    def get_run_metrics(self, run_name):
        """
        Get all metrics for a run.
        
        Args:
            run_name: Name of the run
            
        Returns:
            Dictionary of metric_name -> list of (step, value)
        """
        scalars_file = self.experiments_dir / run_name / "scalars.jsonl"
        metrics = {}
        
        if scalars_file.exists():
            with open(scalars_file) as f:
                for line in f:
                    entry = json.loads(line)
                    name = entry["name"]
                    if name not in metrics:
                        metrics[name] = []
                    metrics[name].append((entry["step"], entry["value"]))
        
        return metrics
    
    def compare(self, run_names, metric):
        """
        Compare runs on a specific metric.
        
        Args:
            run_names: List of run names to compare
            metric: Metric to compare
            
        Returns:
            Dictionary of run_name -> metric values
        """
        comparison = {}
        for run_name in run_names:
            metrics = self.get_run_metrics(run_name)
            if metric in metrics:
                comparison[run_name] = metrics[metric]
        return comparison
    
    def get_best_run(self, metric, mode="min"):
        """
        Get the best run based on a metric.
        
        Args:
            metric: Metric to evaluate
            mode: 'min' or 'max'
            
        Returns:
            (run_name, best_value)
        """
        best_run = None
        best_value = float('inf') if mode == 'min' else float('-inf')
        
        for run_name in self.list_runs():
            metrics = self.get_run_metrics(run_name)
            if metric in metrics and metrics[metric]:
                values = [v for _, v in metrics[metric]]
                final_value = values[-1]
                
                if mode == 'min' and final_value < best_value:
                    best_value = final_value
                    best_run = run_name
                elif mode == 'max' and final_value > best_value:
                    best_value = final_value
                    best_run = run_name
        
        return best_run, best_value
    
    def delete_run(self, run_name):
        """
        Delete a run.
        
        Args:
            run_name: Name of the run to delete
        """
        import shutil
        run_dir = self.experiments_dir / run_name
        if run_dir.exists():
            shutil.rmtree(run_dir)
