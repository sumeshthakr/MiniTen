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
        >>> dataset = MyDataset(data)
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for batch in loader:
        ...     # Process batch
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False,
                 num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        raise NotImplementedError("DataLoader to be implemented")
    
    def __iter__(self):
        """Return iterator over batches."""
        raise NotImplementedError("To be implemented")
    
    def __len__(self):
        """Return number of batches."""
        raise NotImplementedError("To be implemented")


class RandomSampler:
    """Random sampling of dataset indices."""
    
    def __init__(self, dataset):
        self.dataset = dataset
        raise NotImplementedError("RandomSampler to be implemented")


class SequentialSampler:
    """Sequential sampling of dataset indices."""
    
    def __init__(self, dataset):
        self.dataset = dataset
        raise NotImplementedError("SequentialSampler to be implemented")
