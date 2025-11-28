# EEG Data Processing Pipeline

This comprehensive guide covers the entire EEG data processing pipeline, from raw data preprocessing to model training with PyTorch. The system uses HDF5 format for efficient storage and streaming of large EEG datasets.

## Quick Start

### Basic Usage
```python
from eeg_data_preprocessing import EEGDataPreprocessor
from eeg_dataset import EEGDataset, EEGDataLoader

# 1. Preprocess data (creates HDF5 files with train/val/test splits)
preprocessor = EEGDataPreprocessor(
    data_root="/path/to/preprocessed_new",
    output_dir="/path/to/processed_eeg_data_hdf5"
)
participants = preprocessor.process_all_participants(n_jobs=64)  # Use all CPU cores

# 2. Create dataset from HDF5 file
dataset = EEGDataset(
    hdf5_file="/path/to/processed_eeg_data_hdf5/eeg_data_train_1s.h5",
    task_type="both",  # "active", "passive", or "both"
    shuffle=True  # Shuffle for training
)

# 3. Create data loader
dataloader = EEGDataLoader.create_dataloader(
    dataset=dataset,
    batch_size=32,
    num_workers=4
)

# 4. Use in training
for batch in dataloader:
    eeg_data = batch['eeg_data']      # Shape: (batch_size, 60, 200) or (batch_size, 60, 800)
    gender = batch['gender']          # Shape: (batch_size,)
    age = batch['age']                # Shape: (batch_size,)
    participant_ids = batch['participant_ids']
    task_types = batch['task_types']
```

### Requirements
- Python 3.7+
- PyTorch
- NumPy
- Pandas
- h5py
- scikit-learn

## Table of Contents

1. [Overview](#overview)
2. [Data Structure](#data-structure)
3. [Data Preprocessing](#data-preprocessing)
4. [Dataset and DataLoader Classes](#dataset-and-dataloader-classes)
5. [Target Transforms](#target-transforms)
6. [Cross-Validation and Cross-Task Evaluation](#cross-validation-and-cross-task-evaluation)
7. [Usage Examples](#usage-examples)
8. [File Structure](#file-structure)
9. [Best Practices](#best-practices)

## Overview

This pipeline processes EEG data from the preprocessed_new directory structure and creates PyTorch datasets and data loaders for machine learning tasks. The system supports:

- **Gender Classification**: Binary classification (female/male)
- **Age Classification**: 3-class classification (<8.5, 8.5-12.5, >12.5 years)
- **Age Regression**: Continuous age prediction with multiple normalization strategies
- **Combined Classification**: Gender+age combinations (6 classes)
- **HDF5 Storage**: Efficient storage without compression for faster data loading (3-5x speedup)
- **Streaming Data Loading**: Memory-efficient IterableDataset for large datasets
- **Cross-Task Evaluation**: Train on one task type, test on another
- **Participant-Level Filtering**: Flexible participant isolation or mixing for training

## Key Features

- **HDF5 Storage**: Efficient hierarchical storage without compression for faster training
- **Data Processing**: Converts (2, 60, 240) EEG data to (60, 200) or (60, 800) by concatenating runs
- **Task Separation**: Separates active (ccd) and passive (sus) tasks into separate groups
- **Streaming Data Loading**: Memory-efficient IterableDataset for large datasets
- **PyTorch Integration**: Full PyTorch Dataset and DataLoader support
- **Pre-split Data**: Train/validation/test splits created during preprocessing
- **Participant Metadata**: Each segment stores participant_id for flexible filtering
- **Cross-Task Evaluation**: Train on one task type, test on another
- **Participant-Level Filtering**: Filter by participant IDs for flexible training strategies
- **Multiprocessing**: Parallel preprocessing using all available CPU cores
- **Type Safety**: Comprehensive type hints and validation

### Important Notes
- The pipeline creates pre-split HDF5 files (train/val/test) during preprocessing
- Each EEG segment stores participant_id in metadata for participant-level filtering
- The dataset supports filtering by task type (active/passive/both) and participant IDs
- All EEG data is converted to PyTorch tensors automatically
- Training data is shuffled by default to avoid participant-level batch correlations
- Worker sharding is implemented for multiprocessing DataLoader workers

## Data Structure

### Input Data Structure
```
preprocessed_new/
├── cmi_bids_R1/
│   ├── sub-NDARAC904DMU/
│   │   ├── demographics.csv
│   │   ├── ccd_data_trial_0.npy  # Active task data (2, 60, 240)
│   │   ├── ccd_data_trial_1.npy
│   │   ├── sus_data_trial_0.npy  # Passive task data (2, 60, 240)
│   │   ├── sus_data_trial_1.npy
│   │   └── ...
│   └── ...
└── cmi_bids_R2/
    └── ...
```

### Output Data Structure (HDF5 Format)
The preprocessing creates HDF5 files with the following structure:

```
eeg_data_{split}_{segment_length}.h5  # e.g., eeg_data_train_1s.h5
├── active/          (Group)
│   ├── sample_XXXXXX (Dataset: EEG array, shape (60, 200) or (60, 800))
│   └── metadata_XXXXXX (Group with attributes)
│       ├── participant_id: str
│       ├── gender: int (1 = male, 0 = female)
│       ├── age: float
│       ├── task_type: str ("active")
│       └── task_number: int
└── passive/         (Group)
    ├── sample_XXXXXX (Dataset: EEG array, shape (60, 200) or (60, 800))
    └── metadata_XXXXXX (Group with attributes)
        ├── participant_id: str
        ├── gender: int (1 = male, 0 = female)
        ├── age: float
        ├── task_type: str ("passive")
        └── task_number: int

File-level attributes:
- split_name: str ("train", "val", or "test")
- segment_length: str ("1s", "2s", or "4s")
- num_participants: int
- total_samples: int
```

Each sample is stored as a separate dataset without compression for faster data loading during training (trade-off: ~8-10% larger files but 3-5x faster loading).

**Performance Consideration**: For 1s segments with ~955K samples, storing each sample as a separate dataset creates a large HDF5 B-tree index. This can cause slower data loading due to B-tree traversal overhead. The dataset code automatically optimizes for this by:
- Using larger HDF5 chunk cache (500MB) for datasets with >500K samples
- Skipping sorting for large datasets during cache building
- Recommending `num_workers=0` for 1s segments to avoid file contention

## Data Preprocessing

### Data Processing Details
1. **EEG Data Processing**:
   - Input: (2, 60, 240) - 2 channels, 60 electrodes, 240 time points
   - Output: Two separate (60, 200) arrays - one per channel
   - Processing: Drop first 40 time points from each channel

2. **Dual Segment Creation**:
   - **1-second segments**: (60, 200) - single runs
   - **4-second segments**: (60, 800) - concatenated from 4 consecutive runs
   - **Concatenation logic**: Groups 4 consecutive runs of the same task type and participant

3. **Task Naming**:
   - Active tasks: `ccd_trial_{trial_num}_run{run_num}`
   - Passive tasks: `sus_trial_{trial_num}_run{run_num}`

4. **HDF5 Storage**:
   - Creates separate HDF5 files for train/val/test splits
   - Stores segments without compression for faster data loading (3-5x speedup)
   - Trade-off: ~8-10% larger files but significantly faster training
   - Each segment includes participant_id in metadata
   - Supports multiple segment lengths (1s, 2s, 4s)
   - **Performance Note**: Each sample is stored as a separate HDF5 dataset. For 1s segments with ~955K samples, this creates a large B-tree index that can cause slower data loading. Consider using `num_workers=0` for 1s segments to avoid file contention, or re-preprocess with batched storage for better performance.

5. **Multiprocessing**:
   - Parallel processing of participants using all available CPU cores
   - Two-pass approach: first pass collects metadata, second pass saves to HDF5

### Preprocessing Usage
```python
from eeg_data_preprocessing import EEGDataPreprocessor

# Initialize preprocessor
preprocessor = EEGDataPreprocessor(
    data_root="/path/to/preprocessed_new",
    output_dir="/path/to/processed_eeg_data_hdf5"
)

# Process all participants with multiprocessing
# n_jobs=None uses all available CPU cores
participants = preprocessor.process_all_participants(n_jobs=64)

# Creates HDF5 files:
# - eeg_data_train_1s.h5, eeg_data_val_1s.h5, eeg_data_test_1s.h5
# - eeg_data_train_2s.h5, eeg_data_val_2s.h5, eeg_data_test_2s.h5
# - eeg_data_train_4s.h5, eeg_data_val_4s.h5, eeg_data_test_4s.h5
```

## Dataset and DataLoader Classes

### EEGDataset Class
The main IterableDataset class for loading and processing EEG data from HDF5 files.

```python
from eeg_dataset import EEGDataset

# Basic usage
dataset = EEGDataset(
    hdf5_file="/path/to/processed_eeg_data_hdf5/eeg_data_train_1s.h5",
    task_type="both",  # "active", "passive", or "both"
    target_type="both",  # "gender", "age", "both", "combined"
    transform=None,  # Optional input transform
    gender_transform=None,  # Optional gender target transform
    age_transform=None,  # Optional age target transform
    combined_transform=None,  # Optional combined target transform
    participant_filter=None,  # Optional list of participant IDs to include
    shuffle=True,  # Shuffle samples (recommended for training)
    random_seed=42  # Random seed for reproducibility
)
```

### Specialized Dataset Creation
```python
from target_transforms import (
    gender_classification_transform,
    age_classification_transform,
    age_regression_transform,
    combined_gender_age_classification_transform
)

# Gender classification dataset
gender_dataset = EEGDataset(
    hdf5_file="/path/to/processed_eeg_data_hdf5/eeg_data_train_1s.h5",
    task_type="both",
    target_type="gender",
    gender_transform=gender_classification_transform
)

# Age classification dataset
age_dataset = EEGDataset(
    hdf5_file="/path/to/processed_eeg_data_hdf5/eeg_data_train_1s.h5",
    task_type="both",
    target_type="age",
    age_transform=age_classification_transform
)

# Age regression dataset
regression_dataset = EEGDataset(
    hdf5_file="/path/to/processed_eeg_data_hdf5/eeg_data_train_1s.h5",
    task_type="both",
    target_type="age",
    age_transform=age_regression_transform
)

# Combined classification dataset
combined_dataset = EEGDataset(
    hdf5_file="/path/to/processed_eeg_data_hdf5/eeg_data_train_1s.h5",
    task_type="both",
    target_type="combined",
    combined_transform=combined_gender_age_classification_transform
)

# Multi-task dataset (separate heads for gender and age)
multi_task_dataset = EEGDataset(
    hdf5_file="/path/to/processed_eeg_data_hdf5/eeg_data_train_1s.h5",
    task_type="both",
    target_type="both",
    gender_transform=gender_classification_transform,
    age_transform=age_classification_transform
)
```

### Key Features
- **PyTorch Integration**: Full PyTorch IterableDataset and DataLoader support
- **Streaming Data Loading**: Memory-efficient loading of large datasets
- **Custom Transforms**: Support for both input and target transforms
- **Participant Filtering**: Filter by participant IDs for flexible training strategies
- **Task Type Filtering**: Filter by task type (active/passive/both)
- **Worker Sharding**: Automatic data distribution across DataLoader workers
- **Shuffling**: Optional shuffling to avoid participant-level batch correlations

## Target Transforms

### Available Transform Types

#### 1. Gender Classification
- **Function**: `gender_classification_transform`
- **Classes**: 0 (female), 1 (male)
- **Output**: Single integer tensor
- **Use case**: Binary gender classification

#### 2. Age Classification
- **Function**: `age_classification_transform`
- **Classes**: 
  - 0: < 8.5 years
  - 1: 8.5-12.5 years
  - 2: > 12.5 years
- **Output**: Single integer tensor
- **Use case**: Age group classification

#### 3. Age Regression
- **Function**: `age_regression_transform`
- **Output**: Normalized age [0, 1]
- **Use case**: Age regression

#### 4. Combined Gender+Age Classification
- **Function**: `combined_gender_age_classification_transform`
- **Classes**:
  - 0: (female, <8.5)
  - 1: (female, 8.5-12.5)
  - 2: (female, >12.5)
  - 3: (male, <8.5)
  - 4: (male, 8.5-12.5)
  - 5: (male, >12.5)
- **Output**: Single integer tensor
- **Use case**: Combined gender and age classification

### Usage Examples

```python
from target_transforms import (
    gender_classification_transform,
    age_classification_transform,
    age_regression_transform,
    combined_gender_age_classification_transform
)

# Gender classification
gender_dataset = EEGDataset(
    hdf5_file="/path/to/processed_eeg_data_hdf5/eeg_data_train_1s.h5",
    gender_transform=gender_classification_transform,
    target_type="gender"
)

# Age classification
age_dataset = EEGDataset(
    hdf5_file="/path/to/processed_eeg_data_hdf5/eeg_data_train_1s.h5",
    age_transform=age_classification_transform,
    target_type="age"
)

# Age regression
age_reg_dataset = EEGDataset(
    hdf5_file="/path/to/processed_eeg_data_hdf5/eeg_data_train_1s.h5",
    age_transform=age_regression_transform,
    target_type="age"
)

# Combined classification
combined_dataset = EEGDataset(
    hdf5_file="/path/to/processed_eeg_data_hdf5/eeg_data_train_1s.h5",
    combined_transform=combined_gender_age_classification_transform,
    target_type="combined"
)
```

## Cross-Task Evaluation

### Cross-Task Evaluation
Train on one task type and test on another by creating datasets with different task types:

```python
from eeg_dataset import EEGDataset, EEGDataLoader
from pathlib import Path

# Get HDF5 file paths
hdf5_dir = Path("/path/to/processed_eeg_data_hdf5")
train_file = hdf5_dir / "eeg_data_train_1s.h5"
val_file = hdf5_dir / "eeg_data_val_1s.h5"
test_file = hdf5_dir / "eeg_data_test_1s.h5"

# Create datasets: train on active, test on passive
train_dataset = EEGDataset(
    hdf5_file=str(train_file),
    task_type="active",  # Train on active tasks
    shuffle=True
)
val_dataset = EEGDataset(
    hdf5_file=str(val_file),
    task_type="passive",  # Validate on passive tasks
    shuffle=False
)
test_dataset = EEGDataset(
    hdf5_file=str(test_file),
    task_type="passive",  # Test on passive tasks
    shuffle=False
)

# Create loaders
train_loader = EEGDataLoader.create_dataloader(train_dataset, batch_size=32)
val_loader = EEGDataLoader.create_dataloader(val_dataset, batch_size=32)
test_loader = EEGDataLoader.create_dataloader(test_dataset, batch_size=32)
```

### Participant-Level Filtering
Filter by participant IDs for flexible training strategies (e.g., participant isolation):

```python
# Get list of participant IDs from train split
# (You would extract this from the HDF5 file or maintain a separate list)

# Train on specific participants
train_dataset = EEGDataset(
    hdf5_file="/path/to/processed_eeg_data_hdf5/eeg_data_train_1s.h5",
    participant_filter=["sub-NDARAC904DMU", "sub-NDARAB514MAJ"],  # Only these participants
    shuffle=True
)
```

## Usage Examples

### Complete Training Pipeline
```python
from eeg_data_preprocessing import EEGDataPreprocessor
from eeg_dataset import EEGDataset, EEGDataLoader
from target_transforms import gender_classification_transform

# 1. Preprocess data (creates HDF5 files with train/val/test splits)
preprocessor = EEGDataPreprocessor(
    data_root="/path/to/preprocessed_new",
    output_dir="/path/to/processed_eeg_data_hdf5"
)
participants = preprocessor.process_all_participants(n_jobs=64)

# 2. Create data loaders from pre-split HDF5 files
train_loader, val_loader, test_loader = EEGDataLoader.create_train_val_test_loaders(
    hdf5_dir="/path/to/processed_eeg_data_hdf5",
    segment_length="1s",
    task_type="both",
    target_type="gender",
    gender_transform=gender_classification_transform,
    batch_size=32,
    num_workers=4,
    shuffle_train=True
)

# 3. Train model
for epoch in range(num_epochs):
    for batch in train_loader:
        eeg_data = batch['eeg_data']  # (batch_size, 60, 200)
        gender = batch['gender']      # (batch_size,)
        participant_ids = batch['participant_ids']
        # ... training code ...
```

### Task-Type-Specific Evaluation
Evaluate model performance separately for active and passive tasks:

```python
from evaluator import EEGEvaluator

# Create datasets for each task type
active_test_dataset = EEGDataset(
    hdf5_file="/path/to/processed_eeg_data_hdf5/eeg_data_test_1s.h5",
    task_type="active",
    shuffle=False
)
passive_test_dataset = EEGDataset(
    hdf5_file="/path/to/processed_eeg_data_hdf5/eeg_data_test_1s.h5",
    task_type="passive",
    shuffle=False
)

# Create loaders
active_loader = EEGDataLoader.create_dataloader(active_test_dataset, batch_size=32)
passive_loader = EEGDataLoader.create_dataloader(passive_test_dataset, batch_size=32)

# Evaluate separately
evaluator = EEGEvaluator()
active_results = evaluator.evaluate_by_task_type(
    model=model,
    data_loaders={"active": active_loader}
)
passive_results = evaluator.evaluate_by_task_type(
    model=model,
    data_loaders={"passive": passive_loader}
)
```

## File Structure

```
EEG/data_processing/
├── eeg_data_preprocessing.py    # Main preprocessing pipeline
├── eeg_dataset.py              # PyTorch IterableDataset and DataLoader classes
├── target_transforms.py        # Target transformation functions
├── constants.py                # Shared constants and utilities
├── processed_eeg_data_hdf5/    # HDF5 files with train/val/test splits
│   ├── eeg_data_train_1s.h5
│   ├── eeg_data_val_1s.h5
│   ├── eeg_data_test_1s.h5
│   ├── eeg_data_train_2s.h5
│   ├── eeg_data_val_2s.h5
│   ├── eeg_data_test_2s.h5
│   ├── eeg_data_train_4s.h5
│   ├── eeg_data_val_4s.h5
│   └── eeg_data_test_4s.h5
└── README.md                   # This file
```

## Best Practices

### 1. Data Preprocessing
- Use multiprocessing (`n_jobs`) to speed up preprocessing on multi-core systems
- Verify data integrity after preprocessing by checking HDF5 file attributes
- The preprocessing creates pre-split train/val/test files automatically

### 2. Dataset Usage
- Choose the appropriate segment length for your model (1s, 2s, or 4s)
- Use `shuffle=True` for training data to avoid participant-level batch correlations
- Use `shuffle=False` for validation and test data
- Filter by participant IDs if you need participant-level isolation

### 3. Data Loading
- Use `num_workers > 0` for parallel data loading (worker sharding is automatic)
- Set `pin_memory=True` for GPU training (default in `create_dataloader`)
- Adjust batch size based on your GPU memory

### 4. Cross-Task Evaluation
- Create separate datasets with different `task_type` parameters
- Use the same HDF5 file but filter by task type at dataset level
- Evaluate model performance separately for each task type

### 5. Performance
- Use appropriate batch sizes and number of workers
- HDF5 streaming is memory-efficient for large datasets
- Worker sharding ensures no data duplication with `num_workers > 0`
- **1s Segment Performance**: 1s segments have ~955K samples (3.1x more than 4s), each stored as a separate HDF5 dataset. This creates a large B-tree index that can cause slower data loading. Recommended optimizations:
  - Use `num_workers=0` to avoid HDF5 file contention
  - Increase HDF5 chunk cache size (automatically set to 500MB for large datasets)
  - Consider re-preprocessing with batched storage (multiple samples per dataset) for better performance

### 6. Reproducibility
- Set `random_seed` for reproducible shuffling
- Pre-split HDF5 files ensure consistent train/val/test splits
- Document your preprocessing and evaluation procedures

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or number of workers
2. **HDF5 File Not Found**: Verify the HDF5 file path and that preprocessing completed successfully
3. **Data Loading Errors**: Verify HDF5 file structure and that groups (active/passive) exist
4. **Transform Errors**: Ensure transforms match your target type
5. **Worker Sharding Issues**: If using `num_workers > 0`, ensure worker sharding is working correctly
6. **Slow Data Loading for 1s Segments**: 1s segments have ~955K samples, each stored as a separate HDF5 dataset. This creates a large B-tree index that can cause exponential slowdown. Solutions:
   - Use `num_workers=0` to avoid file contention (already optimized in dataset code)
   - HDF5 cache is automatically increased to 500MB for large datasets
   - For long-term solution, consider re-preprocessing with batched storage (store multiple samples per dataset)

### Debugging Tips

1. **Check HDF5 Structure**: Inspect HDF5 files using `h5py` to verify structure
2. **Check Data Distribution**: Use dataset statistics to verify data loading
3. **Monitor Memory Usage**: Use appropriate batch sizes for your hardware
4. **Verify Transforms**: Test transforms on sample data before training
5. **Test Worker Sharding**: Verify that samples are not duplicated when using multiple workers

## API Reference

### EEGDataPreprocessor
- `__init__(data_root, output_dir)`: Initialize preprocessor
- `process_all_participants(n_jobs=None)`: Process all participants and create HDF5 files
  - `n_jobs`: Number of parallel workers (None = use all CPU cores)

### EEGDataset
- `__init__(hdf5_file, task_type, target_type, ...)`: Initialize IterableDataset
  - `hdf5_file`: Path to HDF5 file (e.g., "eeg_data_train_1s.h5")
  - `task_type`: "active", "passive", or "both"
  - `target_type`: "gender", "age", "both", "combined"
  - `participant_filter`: Optional list of participant IDs to include
  - `shuffle`: Whether to shuffle samples (default: False)
  - `random_seed`: Random seed for shuffling

### EEGDataLoader
- `create_dataloader(dataset, batch_size, num_workers, ...)`: Create PyTorch DataLoader
  - Automatically handles worker sharding for `num_workers > 0`
- `create_train_val_test_loaders(hdf5_dir, segment_length, ...)`: Create train/val/test loaders
  - Loads from pre-split HDF5 files
  - `shuffle_train`: Whether to shuffle training data (default: True)

### Target Transforms
- `gender_classification_transform(gender)`: Convert gender to classification target (0/1)
- `age_classification_transform(age)`: Convert age to classification target (0/1/2)
- `age_regression_transform(age)`: Convert age to regression target (normalized 0-1)
- `combined_gender_age_classification_transform(gender, age)`: Convert to combined target (0-5)