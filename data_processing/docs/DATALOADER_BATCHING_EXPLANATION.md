# DataLoader Batching Explanation

## Understanding the Data Flow

### 1. Dataset Level: IterableDataset (Data Stream)

`EEGDataset` inherits from `IterableDataset`, which means it's a **data stream**:

```python
class EEGDataset(IterableDataset):
    def __iter__(self):
        # ... setup code ...
        for task_type, sample_name in all_samples:
            # Load sample data
            sample_data = group[sample_name][:]  # Shape: (60, timepoints)
            
            # Create sample dictionary
            sample = self._create_sample_dict(...)
            yield sample  # ← Yields ONE sample at a time
```

**Key Point**: The dataset yields **individual samples** one at a time (stream).

### 2. DataLoader Level: Automatic Batching

PyTorch's `DataLoader` automatically collects samples from the stream and groups them into batches:

```python
loader_kwargs = {
    'dataset': dataset,           # IterableDataset (stream)
    'batch_size': batch_size,     # e.g., 32 samples per batch
    'collate_fn': EEGDataLoader.collate_fn  # Function to combine samples into batch
}
return DataLoader(**loader_kwargs)
```

**How it works**:
1. DataLoader internally calls `iter(dataset)` to get the stream
2. It collects `batch_size` samples from the stream (e.g., 32 samples)
3. It calls `collate_fn([sample1, sample2, ..., sample32])` to combine them
4. Returns a batched dictionary

### 3. Collate Function: Combining Samples into Batch

The `collate_fn` takes a **list of individual samples** and creates a **batched dictionary**:

```python
@staticmethod
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Args:
        batch: List of sample dictionaries (e.g., 32 samples)
    
    Returns:
        Dictionary containing batched tensors
    """
    # Stack EEG data from all samples
    eeg_data = torch.stack([sample['eeg_data'] for sample in batch])
    # Shape: (batch_size, 60, timepoints) - e.g., (32, 60, 200)
    
    # Stack labels
    gender = torch.stack([...])  # Shape: (batch_size,) - e.g., (32,)
    age = torch.stack([...])    # Shape: (batch_size,) - e.g., (32,)
    
    # Collect lists (not tensors)
    task_types = [sample['task_type'] for sample in batch]  # List of 32 strings
    
    return {
        'eeg_data': eeg_data,      # Tensor: (batch_size, 60, timepoints)
        'gender': gender,           # Tensor: (batch_size,)
        'age': age,                 # Tensor: (batch_size,)
        'task_types': task_types,   # List: [batch_size strings]
        # ... other fields
    }
```

### 4. Usage in Saliency Code

In `compute_batch_saliency`, the code iterates over batches:

```python
for batch_idx, batch in enumerate(data_loader):
    # batch is a dictionary with batched tensors
    inputs = batch['eeg_data']      # Shape: (batch_size, 60, timepoints)
    labels = batch[target_key]      # Shape: (batch_size,)
    task_types = batch['task_types'] # List of batch_size strings
    
    # Process batch...
    for i in range(batch_size):
        sample_input = inputs[i:i+1]  # Extract one sample: (1, 60, timepoints)
        sample_label = labels[i:i+1]  # Extract one label: (1,)
        # ... compute saliency for this sample
```

## Summary

**Yes, the DataLoader DOES create batches!**

- **Dataset**: Streams individual samples (`yield sample`)
- **DataLoader**: Collects `batch_size` samples from the stream
- **collate_fn**: Combines samples into batched tensors
- **Result**: Each iteration of `data_loader` returns a batch dictionary

The key insight is that PyTorch's `DataLoader` handles the batching automatically, even for `IterableDataset` (stream-based datasets). The `collate_fn` is responsible for combining the streamed samples into batched tensors.
