"""
Training callbacks for MiniTen.

Provides callbacks for training monitoring and control:
- EarlyStopping
- ModelCheckpoint
- ProgressBar
- Custom callbacks
"""

import time
import sys
import numpy as np
from pathlib import Path


class Callback:
    """
    Base class for training callbacks.
    
    Subclass this to create custom callbacks.
    
    Methods to override:
        on_train_begin(logs)
        on_train_end(logs)
        on_epoch_begin(epoch, logs)
        on_epoch_end(epoch, logs)
        on_batch_begin(batch, logs)
        on_batch_end(batch, logs)
    """
    
    def __init__(self, logger=None):
        """
        Initialize callback.
        
        Args:
            logger: Optional MetricsLogger for logging
        """
        self.logger = logger
        self.model = None
        self.params = {}
    
    def set_model(self, model):
        """Set the model reference."""
        self.model = model
    
    def set_params(self, params):
        """Set training parameters."""
        self.params = params
    
    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, logs=None):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        # Log metrics if logger is available
        if self.logger and logs:
            self.logger.log_metrics(logs, step=epoch)
    
    def on_batch_begin(self, batch, logs=None):
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, batch, logs=None):
        """Called at the end of each batch."""
        pass


class CallbackList:
    """Container for managing multiple callbacks."""
    
    def __init__(self, callbacks=None):
        """
        Initialize callback list.
        
        Args:
            callbacks: List of callbacks
        """
        self.callbacks = callbacks or []
    
    def append(self, callback):
        """Add a callback."""
        self.callbacks.append(callback)
    
    def set_model(self, model):
        """Set model for all callbacks."""
        for callback in self.callbacks:
            callback.set_model(model)
    
    def set_params(self, params):
        """Set params for all callbacks."""
        for callback in self.callbacks:
            callback.set_params(params)
    
    def on_train_begin(self, logs=None):
        """Called at training start."""
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs=None):
        """Called at training end."""
        for callback in self.callbacks:
            callback.on_train_end(logs)
    
    def on_epoch_begin(self, epoch, logs=None):
        """Called at epoch start."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at epoch end."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch, logs=None):
        """Called at batch start."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch, logs=None):
        """Called at batch end."""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric stops improving.
    
    Args:
        monitor: Metric to monitor
        patience: Number of epochs with no improvement to wait
        min_delta: Minimum change to qualify as improvement
        mode: 'min' or 'max'
        restore_best_weights: Whether to restore model weights from best epoch
        
    Example:
        >>> early_stop = EarlyStopping(monitor='val_loss', patience=5)
        >>> # Use with training loop
    """
    
    def __init__(self, monitor='val_loss', patience=5, min_delta=0, 
                 mode='auto', restore_best_weights=False, logger=None):
        super().__init__(logger)
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        # Auto-detect mode
        if mode == 'auto':
            if 'acc' in monitor or 'accuracy' in monitor:
                mode = 'max'
            else:
                mode = 'min'
        self.mode = mode
        
        self.best = None
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.stop_training = False
    
    def on_train_begin(self, logs=None):
        """Reset state at training start."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best = None
        self.stop_training = False
    
    def on_epoch_end(self, epoch, logs=None):
        """Check if training should stop."""
        super().on_epoch_end(epoch, logs)
        
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        if self.best is None:
            self.best = current
            if self.restore_best_weights and self.model:
                self.best_weights = self._get_weights()
            return
        
        # Check for improvement
        if self.mode == 'min':
            improved = current < self.best - self.min_delta
        else:
            improved = current > self.best + self.min_delta
        
        if improved:
            self.best = current
            self.wait = 0
            if self.restore_best_weights and self.model:
                self.best_weights = self._get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
                if self.restore_best_weights and self.best_weights:
                    self._set_weights(self.best_weights)
    
    def _get_weights(self):
        """Get model weights (placeholder)."""
        if hasattr(self.model, 'get_weights'):
            return self.model.get_weights()
        return None
    
    def _set_weights(self, weights):
        """Set model weights (placeholder)."""
        if hasattr(self.model, 'set_weights') and weights is not None:
            self.model.set_weights(weights)


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.
    
    Args:
        filepath: Path to save checkpoints (can include formatting)
        monitor: Metric to monitor
        save_best_only: Only save when model improves
        mode: 'min' or 'max'
        save_freq: Frequency to save ('epoch' or int)
        
    Example:
        >>> checkpoint = ModelCheckpoint(
        ...     'checkpoints/model_{epoch:02d}_{val_loss:.4f}.npz',
        ...     monitor='val_loss',
        ...     save_best_only=True
        ... )
    """
    
    def __init__(self, filepath, monitor='val_loss', save_best_only=False,
                 mode='auto', save_freq='epoch', logger=None):
        super().__init__(logger)
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        
        # Auto-detect mode
        if mode == 'auto':
            if 'acc' in monitor or 'accuracy' in monitor:
                mode = 'max'
            else:
                mode = 'min'
        self.mode = mode
        
        self.best = None
        self.epoch_count = 0
        
        # Create directory
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, epoch, logs=None):
        """Save checkpoint if needed."""
        super().on_epoch_end(epoch, logs)
        
        logs = logs or {}
        self.epoch_count += 1
        
        if self.save_freq == 'epoch':
            current = logs.get(self.monitor)
            
            if self.save_best_only:
                if current is None:
                    return
                
                save = False
                if self.best is None:
                    save = True
                elif self.mode == 'min' and current < self.best:
                    save = True
                elif self.mode == 'max' and current > self.best:
                    save = True
                
                if save:
                    self.best = current
                    self._save_model(epoch, logs)
            else:
                self._save_model(epoch, logs)
    
    def _save_model(self, epoch, logs):
        """Save the model."""
        # Format filepath
        filepath_str = str(self.filepath)
        
        # Replace placeholders
        filepath_str = filepath_str.format(
            epoch=epoch,
            **{k: v for k, v in logs.items() if isinstance(v, (int, float))}
        )
        
        filepath = Path(filepath_str)
        
        # Save model (placeholder - actual implementation depends on model)
        if self.model and hasattr(self.model, 'save'):
            self.model.save(filepath)
        else:
            # Save a marker file
            filepath.touch()
            if self.logger:
                self.logger.log_text("checkpoint", f"Saved checkpoint: {filepath}", step=epoch)


class ProgressBar(Callback):
    """
    Display a progress bar during training.
    
    Args:
        total_epochs: Total number of epochs
        width: Progress bar width in characters
        metrics: List of metrics to display
        
    Example:
        >>> progress = ProgressBar(total_epochs=100)
        >>> # Progress bar will be shown during training
    """
    
    def __init__(self, total_epochs=None, width=30, metrics=None, logger=None):
        super().__init__(logger)
        self.total_epochs = total_epochs
        self.width = width
        self.metrics = metrics or ['loss', 'accuracy', 'val_loss', 'val_accuracy']
        
        self.start_time = None
        self.epoch_start_time = None
    
    def on_train_begin(self, logs=None):
        """Record training start time."""
        self.start_time = time.time()
        print("Training started...")
        print("-" * 60)
    
    def on_epoch_begin(self, epoch, logs=None):
        """Record epoch start time."""
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        """Display progress."""
        super().on_epoch_end(epoch, logs)
        
        logs = logs or {}
        epoch_time = time.time() - self.epoch_start_time
        
        # Build progress bar
        if self.total_epochs:
            progress = (epoch + 1) / self.total_epochs
            filled = int(self.width * progress)
            bar = '█' * filled + '░' * (self.width - filled)
            epoch_str = f"Epoch {epoch + 1}/{self.total_epochs}"
        else:
            bar = '█' * (epoch % self.width + 1)
            epoch_str = f"Epoch {epoch + 1}"
        
        # Build metrics string
        metrics_str = " - ".join(
            f"{m}: {logs[m]:.4f}" 
            for m in self.metrics 
            if m in logs
        )
        
        # Display
        line = f"\r{epoch_str} [{bar}] {epoch_time:.1f}s"
        if metrics_str:
            line += f" - {metrics_str}"
        
        print(line)
        sys.stdout.flush()
    
    def on_train_end(self, logs=None):
        """Display final summary."""
        total_time = time.time() - self.start_time
        print("-" * 60)
        print(f"Training completed in {total_time:.1f}s")


class LearningRateScheduler(Callback):
    """
    Adjust learning rate during training.
    
    Args:
        schedule: Function that takes epoch and returns learning rate
        
    Example:
        >>> def schedule(epoch):
        ...     if epoch < 10:
        ...         return 0.001
        ...     return 0.0001
        >>> scheduler = LearningRateScheduler(schedule)
    """
    
    def __init__(self, schedule, logger=None):
        super().__init__(logger)
        self.schedule = schedule
    
    def on_epoch_begin(self, epoch, logs=None):
        """Adjust learning rate."""
        if self.model and hasattr(self.model, 'optimizer'):
            new_lr = self.schedule(epoch)
            if hasattr(self.model.optimizer, 'lr'):
                self.model.optimizer.lr = new_lr
            if self.logger:
                self.logger.log("learning_rate", new_lr, step=epoch)


class ReduceLROnPlateau(Callback):
    """
    Reduce learning rate when a metric stops improving.
    
    Args:
        monitor: Metric to monitor
        factor: Factor to reduce learning rate by
        patience: Epochs to wait before reducing
        min_lr: Minimum learning rate
        
    Example:
        >>> reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    """
    
    def __init__(self, monitor='val_loss', factor=0.1, patience=10,
                 min_lr=1e-7, mode='auto', logger=None):
        super().__init__(logger)
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        
        if mode == 'auto':
            mode = 'min' if 'loss' in monitor else 'max'
        self.mode = mode
        
        self.best = None
        self.wait = 0
        self.current_lr = None
    
    def on_epoch_end(self, epoch, logs=None):
        """Check if learning rate should be reduced."""
        super().on_epoch_end(epoch, logs)
        
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        if self.best is None:
            self.best = current
            return
        
        # Check for improvement
        if self.mode == 'min':
            improved = current < self.best
        else:
            improved = current > self.best
        
        if improved:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self._reduce_lr(epoch)
                self.wait = 0
    
    def _reduce_lr(self, epoch):
        """Reduce learning rate."""
        if self.model and hasattr(self.model, 'optimizer'):
            optimizer = self.model.optimizer
            if hasattr(optimizer, 'lr'):
                old_lr = optimizer.lr
                new_lr = max(old_lr * self.factor, self.min_lr)
                optimizer.lr = new_lr
                if self.logger:
                    self.logger.log("learning_rate", new_lr, step=epoch)
                print(f"\nReducing learning rate: {old_lr:.6f} -> {new_lr:.6f}")
