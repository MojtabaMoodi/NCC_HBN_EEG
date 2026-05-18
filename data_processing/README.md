# EEG Data Processing Pipeline

This comprehensive guide covers the entire EEG data processing pipeline, from raw data preprocessing to model training with PyTorch. The system uses HDF5 format for efficient storage and streaming of large EEG datasets.

## Recent Updates

- **128-Channel Support**: Now supports both 60-channel `.npy` files and 128-channel `.h5` files
- **Configurable Window Sizes**: Choose which segment lengths to process (1s, 2s, 4s) to save time and storage
- **Multi-Part File Support**: Automatic file splitting for all window sizes prevents large file performance issues
- **Improved Performance**: Large datasets are automatically split into multiple files (100K samples per file)
- **Command-Line Script**: `preprocess_eeg_to_hdf5.py` for preprocessing 60ch .npy or 128ch .h5 to HDF5 (age/gender)
- **Imbalanced data (gender/age)**: Optional stratified train batches or oversampling via `train_balance_method` in `create_train_val_test_loaders`; `compute_class_weights_from_train_hdf5()` for inverse-frequency class weights (e.g. for use in loss when not oversampling)

## Quick Start

### Basic Usage

**For 60-channel .npy files (legacy format)**:
```python
from eeg_data_preprocessing import EEGDataPreprocessor
from eeg_dataset import EEGDataset, EEGDataLoader

# 1. Preprocess data (creates HDF5 files with train/val/test splits)
preprocessor = EEGDataPreprocessor(
    data_root="/path/to/preprocessed_new",
    output_dir="/path/to/processed_eeg_data_hdf5"
    # Default: num_channels=60, window_sizes=['1s', '2s', '4s']
)
preprocessor.process_all_participants(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
    stratify_by="gender"
)

# 2. Create dataset from HDF5 file (automatically handles multi-part files)
dataset = EEGDataset(
    hdf5_file="/path/to/processed_eeg_data_hdf5/eeg_data_train_1s_part00.h5",
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
    eeg_data = batch['eeg_data']      # Shape: (batch_size, 60, timepoints)
    gender = batch['gender']          # Shape: (batch_size,)
    age = batch['age']                # Shape: (batch_size,)
    participant_ids = batch['participant_ids']
    task_types = batch['task_types']
```

**For 128-channel .h5 files (new format)**:
```python
from eeg_data_preprocessing import EEGDataPreprocessor
from eeg_dataset import EEGDataset, EEGDataLoader

# 1. Preprocess 128-channel data
preprocessor = EEGDataPreprocessor(
    data_root="/path/to/preprocessed_128_channels",
    output_dir="/path/to/processed_eeg_data_128ch",
    num_channels=128,
    window_sizes=['1s', '4s']  # Choose which windows to process
)
preprocessor.process_all_participants(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
    stratify_by="both"
)

# 2. Create dataset (automatically handles multi-part files)
# The loader automatically detects and loads from all part files
train_loader, val_loader, test_loader = EEGDataLoader.create_train_val_test_loaders(
    hdf5_dir="/path/to/processed_eeg_data_128ch",
    segment_length="4s",
    task_type="both",
    target_type="both",
    batch_size=32,
    num_workers=4,
    shuffle_train=True
)

# 3. Use in training
for batch in train_loader:
    eeg_data = batch['eeg_data']      # Shape: (batch_size, 128, 800) for 4s
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
10. [Additional documentation](#additional-documentation)

## Overview

This pipeline processes EEG data from multiple input formats and creates PyTorch datasets and data loaders for machine learning tasks. The system supports:

- **Multiple Input Formats**: 
  - 60-channel data from `.npy` files (legacy format)
  - 128-channel data from `.h5` files (new format)
- **Configurable Window Sizes**: Choose which segment lengths to process (1s, 2s, 4s)
- **Gender Classification**: Binary classification (female/male)
- **Age Classification**: 3-class classification (<8.5, 8.5-12.5, >12.5 years)
- **Age Regression**: Continuous age prediction with multiple normalization strategies
- **Combined Classification**: Gender+age combinations (6 classes)
- **Multi-Part HDF5 Storage**: Automatic file splitting for optimal performance (prevents large file issues)
- **Streaming Data Loading**: Memory-efficient IterableDataset for large datasets
- **Cross-Task Evaluation**: Train on one task type, test on another
- **Participant-Level Filtering**: Flexible participant isolation or mixing for training

## Key Features

- **Multiple Input Formats**: Supports both 60-channel `.npy` files and 128-channel `.h5` files
- **Configurable Window Sizes**: Process only the segment lengths you need (1s, 2s, 4s)
- **Multi-Part HDF5 Storage**: Automatic file splitting (e.g., `part00.h5`, `part01.h5`) prevents large file performance issues
- **HDF5 Storage**: Efficient hierarchical storage without compression for faster training
- **Data Processing**: Converts EEG data to segments by concatenating runs and handling timepoint alignment
- **Task Separation**: Separates active (ccd) and passive (sus) tasks into separate groups
- **Streaming Data Loading**: Memory-efficient IterableDataset for large datasets
- **PyTorch Integration**: Full PyTorch Dataset and DataLoader support
- **Pre-split Data**: Train/validation/test splits created during preprocessing
- **Participant Metadata**: Each segment stores participant_id for flexible filtering
- **Cross-Task Evaluation**: Train on one task type, test on another
- **Participant-Level Filtering**: Filter by participant IDs for flexible training strategies
- **Imbalanced data**: For gender/age classification, optional stratified batching or oversampling (`train_balance_method`), and `compute_class_weights_from_train_hdf5()` for class weights in the loss
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

**Option 1: 60-channel data (legacy format - .npy files)**
```
preprocessed_new/
├── cmi_bids_R1/
│   ├── sub-NDARAC904DMU/
│   │   ├── demographics.csv
│   │   ├── ccd_data_trial_0.npy  # Active task data (N, 60, 240)
│   │   ├── ccd_data_trial_1.npy
│   │   ├── sus_data_trial_0.npy  # Passive task data (N, 60, 240)
│   │   ├── sus_data_trial_1.npy
│   │   └── ...
│   └── ...
└── cmi_bids_R2/
    └── ...
```

**Option 2: 128-channel data (new format - .h5 files)**
```
preprocessed_128_channels/
├── cmi_bids_R1/
│   ├── sub-NDARAC904DMU/
│   │   ├── demographics.csv
│   │   ├── ccd_1_data_trial_0.h5  # Active task data (N, 128, 240)
│   │   ├── ccd_1_data_trial_1.h5
│   │   ├── sus_1_data_trial_0.h5  # Passive task data (N, 128, 240)
│   │   ├── sus_1_data_trial_1.h5
│   │   └── ...
│   └── ...
└── cmi_bids_R2/
    └── ...
```

**H5 File Structure**: Each `.h5` file contains a `'data'` key with shape `(N, num_channels, 240)` where:
- `N` = number of runs (typically 2)
- `num_channels` = 60 (for .npy) or 128 (for .h5)
- `240` = timepoints (first 40 are dropped to get 200 per run)

### Output Data Structure (HDF5 Format)
The preprocessing creates HDF5 files with multi-part support for optimal performance:

**Single File Format** (for smaller datasets):
```
eeg_data_{split}_{segment_length}.h5  # e.g., eeg_data_train_4s.h5
```

**Multi-Part File Format** (for large datasets, automatic splitting):
```
eeg_data_{split}_{segment_length}_part00.h5  # First 100K samples
eeg_data_{split}_{segment_length}_part01.h5  # Next 100K samples
eeg_data_{split}_{segment_length}_part02.h5  # And so on...
```

**File Structure** (same for both formats):
```
eeg_data_train_4s_part00.h5
├── active/          (Group)
│   ├── sample_XXXXXX (Dataset: EEG array, shape (num_channels, timepoints))
│   │                 # For 1s: (60 or 128, 200)
│   │                 # For 2s: (60 or 128, 400)
│   │                 # For 4s: (60 or 128, 800)
│   └── metadata_XXXXXX (Group with attributes)
│       ├── participant_id: str
│       ├── gender: int (1 = male, 0 = female)
│       ├── age: float
│       ├── task_type: str ("active")
│       └── task_number: int
└── passive/         (Group)
    ├── sample_XXXXXX (Dataset: EEG array, same shape as above)
    └── metadata_XXXXXX (Group with attributes)
        ├── participant_id: str
        ├── gender: int (1 = male, 0 = female)
        ├── age: float
        ├── task_type: str ("passive")
        └── task_number: int

File-level attributes:
- split_name: str ("train", "val", or "test")
- segment_length: str ("1s", "2s", or "4s")
- part_idx: int (for multi-part files, 0, 1, 2, ...)
- num_participants: int
- total_samples: int (per file, not total across all parts)
```

**Key Points**:
- Each sample is stored as a separate dataset without compression for faster data loading (trade-off: ~8-10% larger files but 3-5x faster loading)
- Files automatically split when reaching 100K samples per file to prevent large file performance issues
- Participant data CAN span multiple files (this is acceptable and handled transparently)
- The dataset loader automatically detects and handles both single-file and multi-part formats

## Data Preprocessing

### Data Processing Details
1. **EEG Data Processing**:
   - **Input Formats**:
     - `.npy` files: Shape `(N, 60, 240)` - N runs, 60 channels, 240 timepoints
     - `.h5` files: Shape `(N, 128, 240)` - N runs, 128 channels, 240 timepoints
   - **Processing**: 
     - Drop first 40 timepoints to get 200 per run
     - Concatenate all runs along time axis
     - Split into segments of specified length (1s, 2s, or 4s)
   - **Output**: 
     - 1s segments: `(num_channels, 200)` - 200 timepoints
     - 2s segments: `(num_channels, 400)` - 400 timepoints
     - 4s segments: `(num_channels, 800)` - 800 timepoints

2. **Segment Creation**:
   - **1-second segments**: Single runs (200 timepoints)
   - **2-second segments**: Two consecutive runs (400 timepoints)
   - **4-second segments**: Four consecutive runs (800 timepoints)
   - **Remainder handling**: If total timepoints don't divide evenly, remainder is either dropped (if < half segment) or zero-padded (if >= half segment)

3. **Task Type Detection**:
   - Active tasks: Files starting with `ccd` (e.g., `ccd_1_data_trial_0.npy` or `ccd_1_data_trial_0.h5`)
   - Passive tasks: Files starting with `sus` (e.g., `sus_1_data_trial_0.npy` or `sus_1_data_trial_0.h5`)

4. **HDF5 Storage**:
   - Creates separate HDF5 files for train/val/test splits
   - **Multi-part file support**: Automatically splits into multiple files when reaching 100K samples per file
   - Stores segments without compression for faster data loading (3-5x speedup)
   - Trade-off: ~8-10% larger files but significantly faster training
   - Each segment includes participant_id in metadata
   - Supports configurable segment lengths (choose 1s, 2s, 4s, or all)
   - **Performance**: Multi-part files prevent large file performance issues. The dataset loader automatically handles both single-file and multi-part formats.

5. **Two-Pass Processing**:
   - First pass: Collect participant metadata for stratified train/val/test splits
   - Second pass: Process participants and save to appropriate split files

### Preprocessing Usage

**For 60-channel .npy files (legacy format)**:
```python
from eeg_data_preprocessing import EEGDataPreprocessor

# Initialize preprocessor (default: 60 channels, all window sizes)
preprocessor = EEGDataPreprocessor(
    data_root="/path/to/preprocessed_new",
    output_dir="/path/to/processed_eeg_data_hdf5"
)

# Process all participants
preprocessor.process_all_participants(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
    stratify_by="gender"
)
```

**For 128-channel .h5 files (new format)**:
```python
from eeg_data_preprocessing import EEGDataPreprocessor

# Initialize preprocessor with 128 channels
preprocessor = EEGDataPreprocessor(
    data_root="/path/to/preprocessed_128_channels",
    output_dir="/path/to/processed_eeg_data_128ch",
    num_channels=128,
    window_sizes=['1s', '4s']  # Choose which windows to process
)

# Process all participants
preprocessor.process_all_participants(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
    stratify_by="both"  # Stratify by both gender and age
)
```

**Using the command-line script** (`preprocess_eeg_to_hdf5.py`):
```bash
# 128ch .h5: 1s and 4s windows
python preprocess_eeg_to_hdf5.py --window_sizes 1s 4s

# 60ch .npy (preprocessed_new) for age and gender, 1s/2s/4s
python preprocess_eeg_to_hdf5.py --data_root /path/to/preprocessed_new --output_dir /path/to/out --num_channels 60 --window_sizes 1s 2s 4s

# Custom paths (128ch)
python preprocess_eeg_to_hdf5.py --data_root /path/to/preprocessed_128_channels --output_dir /path/to/output --window_sizes 1s 4s --num_channels 128 --stratify_by both
```
Train/val/test splits are stratified by age and/or gender (`--stratify_by`). Training data is shuffled at load time (see Dataset and DataLoader) so batches mix participants for age/gender prediction.

**Output Files**:
- Creates HDF5 files with multi-part support (automatic splitting at 100K samples):
  - `eeg_data_train_1s_part00.h5`, `eeg_data_train_1s_part01.h5`, ...
  - `eeg_data_val_1s_part00.h5`, `eeg_data_val_1s_part01.h5`, ...
  - `eeg_data_test_1s_part00.h5`, `eeg_data_test_1s_part01.h5`, ...
  - Same pattern for 2s and 4s segments

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
    shuffle_train=True,
    random_seed=42,
    use_stratified_train_batches=True,   # optional: balanced batches for gender/age
    train_balance_method=None,            # or "stratified" or "oversample"
    prediction_type="classification",
)

# 3. Train model
for epoch in range(num_epochs):
    for batch in train_loader:
        eeg_data = batch['eeg_data']  # (batch_size, 60, 200)
        gender = batch['gender']      # (batch_size,)
        participant_ids = batch['participant_ids']
        # ... training code ...
```

### Imbalanced data (gender/age classification)

For imbalanced classes you can (1) balance via data or (2) use class weights in the loss:

**Option 1 — Balanced batches or oversampling** (no class weights in loss):

```python
train_loader, val_loader, test_loader = EEGDataLoader.create_train_val_test_loaders(
    hdf5_dir="/path/to/processed_eeg_data_hdf5",
    segment_length="4s",
    task_type="both",
    target_type="gender",
    gender_transform=gender_classification_transform,
    train_balance_method="stratified",  # or "oversample"
    prediction_type="classification",
    random_seed=42,
    batch_size=128,
    num_workers=4,
    shuffle_train=True,
)
```

- **stratified**: Each train batch has (roughly) equal counts per class; participant round-robin within class.
- **oversample**: Sampling with replacement so each class is seen in proportion to inverse frequency.

**Option 2 — Class weights** (when not using `train_balance_method="oversample"`):

```python
from eeg_dataset import compute_class_weights_from_train_hdf5

weights = compute_class_weights_from_train_hdf5(
    hdf5_dir="/path/to/processed_eeg_data_hdf5",
    segment_length_str="4s",
    target_type="gender",
    task_type="both",
    weight_power=1.0,  # use >1 (e.g. 1.5) to upweight minority more
)
# Pass weights to your loss (e.g. CrossEntropyLoss(weight=torch.tensor(weights)))
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
├── eeg_data_preprocessing.py   # Main preprocessing pipeline
├── eeg_dataset.py              # PyTorch Dataset/DataLoader: EEGDataset, EEGMapDataset, EEGDataLoader, StratifiedBatchSampler, get_train_sample_list_with_classes, compute_class_weights_from_train_hdf5
├── target_transforms.py        # Target transformation functions
├── constants.py                 # Shared constants (e.g. age/gender class bounds)
├── aggregation.py              # Segment→participant aggregation (majority vote, median)
├── preprocess_eeg_to_hdf5.py    # Preprocess 60ch .npy or 128ch .h5 to HDF5 (age/gender)
├── preprocess_user_identification.py  # User identification preprocessing
├── multi_file_dataset.py       # Multi-file dataset support
├── validate_128channels.py     # Validation for 128-channel data
├── verify_train_val_test_user_split.py  # Check participant-level train/val/test splits (gender/age HDF5)
├── verify_user_id_fold_partition.py     # Check 5-fold user-ID unknown partitions from JSON metadata
├── processed_eeg_data_hdf5/    # HDF5 files with train/val/test splits (60-channel)
│   ├── eeg_data_train_1s_part00.h5
│   ├── eeg_data_val_1s_part00.h5
│   ├── eeg_data_test_1s_part00.h5
│   ├── eeg_data_train_4s_part00.h5
│   └── ...
├── processed_eeg_data_128ch/   # HDF5 files for 128-channel data (same structure)
└── README.md                   # This file
```

## Best Practices

### 1. Data Preprocessing
- Choose the appropriate input format (60-channel .npy or 128-channel .h5)
- Select which window sizes to process (1s, 2s, 4s) to save time and storage
- The preprocessing creates pre-split train/val/test files automatically
- Multi-part files are created automatically when datasets are large (>100K samples)
- Verify data integrity after preprocessing by checking HDF5 file attributes

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
- **Multi-Part Files**: Large datasets are automatically split into multiple files (100K samples per file) to prevent large file performance issues
- **Automatic Detection**: The dataset loader automatically detects and handles both single-file and multi-part formats
- **File Rotation**: Participant data can span multiple files (this is acceptable and handled transparently)
- **1s Segment Performance**: 1s segments have ~955K samples, automatically split into ~10 part files for optimal performance

### 6. Reproducibility
- Set `random_seed` for reproducible shuffling
- Pre-split HDF5 files ensure consistent train/val/test splits
- Document your preprocessing and evaluation procedures

## Additional documentation

- [docs/DATALOADER_BATCHING_EXPLANATION.md](docs/DATALOADER_BATCHING_EXPLANATION.md) — How `EEGDataset` (stream), `DataLoader`, and `collate_fn` form batches.
- [docs/MULTI_FILE_SOLUTION.md](docs/MULTI_FILE_SOLUTION.md) — Multi-file HDF5 layout for 1s segments (e.g. part0, part1), loading, and performance.

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or number of workers
2. **HDF5 File Not Found**: Verify the HDF5 file path and that preprocessing completed successfully
3. **Data Loading Errors**: Verify HDF5 file structure and that groups (active/passive) exist
4. **Transform Errors**: Ensure transforms match your target type
5. **Worker Sharding Issues**: If using `num_workers > 0`, ensure worker sharding is working correctly
6. **Slow Data Loading for 1s Segments**: This is now handled automatically with multi-part files. The preprocessing automatically splits large datasets into multiple files (100K samples per file), preventing large file performance issues. The dataset loader automatically detects and handles multi-part files.

### Debugging Tips

1. **Check HDF5 Structure**: Inspect HDF5 files using `h5py` to verify structure
2. **Check Data Distribution**: Use dataset statistics to verify data loading
3. **Monitor Memory Usage**: Use appropriate batch sizes for your hardware
4. **Verify Transforms**: Test transforms on sample data before training
5. **Test Worker Sharding**: Verify that samples are not duplicated when using multiple workers

## API Reference

### EEGDataPreprocessor
- `__init__(data_root, output_dir, num_channels=60, window_sizes=None)`: Initialize preprocessor
  - `data_root`: Path to input data directory
  - `output_dir`: Path to output directory for HDF5 files
  - `num_channels`: Number of EEG channels (60 for .npy files, 128 for .h5 files)
  - `window_sizes`: List of window sizes to process (e.g., `['1s', '4s']`). Default: `['1s', '2s', '4s']`
- `process_all_participants(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42, stratify_by='gender')`: Process all participants and create HDF5 files
  - Creates multi-part files automatically when datasets are large
  - Supports both 60-channel and 128-channel data

### EEGDataset
- `__init__(hdf5_file, task_type, target_type, ...)`: Initialize IterableDataset
  - `hdf5_file`: Path to HDF5 file (e.g., "eeg_data_train_1s.h5") or list of paths for multi-file
  - `task_type`: "active", "passive", or "both"
  - `target_type`: "gender", "age", "both", "combined", "user_identification"
  - `participant_filter`: Optional list of participant IDs to include
  - `shuffle`: Whether to shuffle samples (default: False)
  - `random_seed`: Random seed for shuffling

### EEGDataLoader
- `create_dataloader(dataset, batch_size, num_workers, ...)`: Create PyTorch DataLoader
  - Automatically handles worker sharding for `num_workers > 0`
- `create_train_val_test_loaders(hdf5_dir, segment_length, ...)`: Create train/val/test loaders
  - Loads from pre-split HDF5 files (single or multi-part); auto-detects file layout
  - `shuffle_train`: Whether to shuffle training data (default: True)
  - `random_seed`: For reproducible shuffling and balanced sampling
  - `use_stratified_train_batches`: If True and target is gender/age, use stratified batches (unless overridden by `train_balance_method`)
  - `train_balance_method`: `None` | `'stratified'` | `'oversample'`. For **gender/age classification only**: stratified = balanced batches per class (participant round-robin); oversample = sampling with replacement (inverse class frequency). Not used for combined, regression, or user_identification
  - `prediction_type`: `'classification'` | `'regression'` | None. Stratified/oversample apply only when classification. None is treated as classification
  - `user_identification_transform`: Optional for `target_type="user_identification"`

### Class weights (imbalanced gender/age)
- `compute_class_weights_from_train_hdf5(hdf5_dir, segment_length_str, target_type, task_type="both", weight_power=1.0)`: Compute inverse-frequency class weights from train HDF5 metadata (no EEG data loaded). Returns a list of per-class weights. Use when **not** using `train_balance_method="oversample"` (e.g. in the loss). `weight_power` > 1 upweights minority classes more.

### Target Transforms
- `gender_classification_transform(gender)`: Convert gender to classification target (0/1)
- `age_classification_transform(age)`: Convert age to classification target (0/1/2)
- `age_regression_transform(age)`: Convert age to regression target (normalized 0-1)
- `combined_gender_age_classification_transform(gender, age)`: Convert to combined target (0-5)