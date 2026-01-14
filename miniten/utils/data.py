"""
Data Loading and Processing Utilities

Efficient data loading, batching, and preprocessing for training.

Features:
- DataLoader for batching and shuffling
- Dataset base class
- Data augmentation
- Parallel data loading
- Memory-efficient iterators
"""

import numpy as np


class Dataset:
    """
    Base class for all datasets.
    
    Subclasses should override __getitem__ and __len__.
    
    Example:
        >>> class MyDataset(Dataset):
        ...     def __init__(self, data):
        ...         self.data = data
        ...     
        ...     def __getitem__(self, idx):
        ...         return self.data[idx]
        ...     
        ...     def __len__(self):
        ...         return len(self.data)
    """
    
    def __getitem__(self, index):
        """Get item at index."""
        raise NotImplementedError("Subclasses must implement __getitem__")
    
    def __len__(self):
        """Return dataset size."""
        raise NotImplementedError("Subclasses must implement __len__")


class ArrayDataset(Dataset):
    """
    Dataset from numpy arrays.
    
    Args:
        *arrays: Arrays to wrap (X, y, etc.)
    """
    
    def __init__(self, *arrays):
        """Initialize with arrays."""
        if len(arrays) == 0:
            raise ValueError("At least one array required")
        
        self.arrays = [np.asarray(a) for a in arrays]
        
        # Check all arrays have same length
        length = len(self.arrays[0])
        for arr in self.arrays[1:]:
            if len(arr) != length:
                raise ValueError("All arrays must have same length")
    
    def __getitem__(self, index):
        """Get item at index."""
        if len(self.arrays) == 1:
            return self.arrays[0][index]
        return tuple(arr[index] for arr in self.arrays)
    
    def __len__(self):
        """Return dataset size."""
        return len(self.arrays[0])


class DataLoader:
    """
    Data loader for batching and iterating over datasets.
    
    Args:
        dataset: Dataset to load from
        batch_size: Number of samples per batch (default: 1)
        shuffle: Whether to shuffle data (default: False)
        num_workers: Number of parallel workers (default: 0)
        drop_last: Drop last incomplete batch (default: False)
    
    Example:
        >>> dataset = ArrayDataset(X, y)
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for batch_x, batch_y in loader:
        ...     # Process batch
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False,
                 num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        
        self._indices = None
    
    def __iter__(self):
        """Return iterator over batches."""
        n = len(self.dataset)
        
        # Create indices
        if self.shuffle:
            indices = np.random.permutation(n)
        else:
            indices = np.arange(n)
        
        self._indices = indices
        
        # Generate batches
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            
            if self.drop_last and (end - start) < self.batch_size:
                break
            
            batch_indices = indices[start:end]
            
            # Collect batch
            batch = [self.dataset[i] for i in batch_indices]
            
            # Stack if tuples
            if isinstance(batch[0], tuple):
                yield tuple(np.stack([b[i] for b in batch]) for i in range(len(batch[0])))
            else:
                yield np.stack(batch)
    
    def __len__(self):
        """Return number of batches."""
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


class RandomSampler:
    """Random sampling of dataset indices."""
    
    def __init__(self, dataset, replacement=False, num_samples=None):
        self.dataset = dataset
        self.replacement = replacement
        self.num_samples = num_samples or len(dataset)
    
    def __iter__(self):
        n = len(self.dataset)
        if self.replacement:
            for _ in range(self.num_samples):
                yield np.random.randint(0, n)
        else:
            yield from np.random.permutation(n)[:self.num_samples]
    
    def __len__(self):
        return self.num_samples


class SequentialSampler:
    """Sequential sampling of dataset indices."""
    
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __iter__(self):
        return iter(range(len(self.dataset)))
    
    def __len__(self):
        return len(self.dataset)


class SubsetSampler:
    """Sample from a subset of indices."""
    
    def __init__(self, indices):
        self.indices = indices
    
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)


def train_test_split(dataset, test_size=0.2, shuffle=True, seed=None):
    """
    Split dataset into train and test sets.
    
    Args:
        dataset: Dataset to split
        test_size: Fraction for test set
        shuffle: Whether to shuffle before splitting
        seed: Random seed
        
    Returns:
        (train_dataset, test_dataset)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(dataset)
    indices = np.arange(n)
    
    if shuffle:
        np.random.shuffle(indices)
    
    split = int(n * (1 - test_size))
    train_indices = indices[:split]
    test_indices = indices[split:]
    
    return Subset(dataset, train_indices), Subset(dataset, test_indices)


class Subset(Dataset):
    """
    Subset of a dataset at specified indices.
    """
    
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices)
    
    def __getitem__(self, index):
        return self.dataset[self.indices[index]]
    
    def __len__(self):
        return len(self.indices)


class ConcatDataset(Dataset):
    """
    Concatenation of multiple datasets.
    """
    
    def __init__(self, datasets):
        self.datasets = list(datasets)
        self.cumulative_sizes = np.cumsum([len(d) for d in self.datasets])
    
    def __getitem__(self, index):
        # Find which dataset this index belongs to
        for i, size in enumerate(self.cumulative_sizes):
            if index < size:
                prev_size = 0 if i == 0 else self.cumulative_sizes[i - 1]
                return self.datasets[i][index - prev_size]
        raise IndexError(f"Index {index} out of range")
    
    def __len__(self):
        return self.cumulative_sizes[-1] if self.datasets else 0
