# EEG Data Processing Pipeline

This comprehensive guide covers the entire EEG data processing pipeline, from raw data preprocessing to model training with PyTorch. The system has been completely refactored to eliminate code duplication and provide a clean, maintainable architecture.

## Quick Start

### Basic Usage
```python
from eeg_data_preprocessing import EEGDataPreprocessor
from eeg_dataset import EEGDataset, EEGDataLoader

# 1. Preprocess data (creates both 1s and 4s segments)
preprocessor = EEGDataPreprocessor(
    data_root="/path/to/preprocessed_new",
    output_dir_1s="/path/to/1s_segments",
    output_dir_4s="/path/to/4s_segments"
)
participants = preprocessor.process_all_participants()

# 2. Create dataset
dataset = EEGDataset(
    pickle_dir="/path/to/1s_segments",  # or "/path/to/4s_segments"
    task_type="both"  # "active", "passive", or "both"
)

# 3. Create data loader
dataloader = EEGDataLoader.create_dataloader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
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
- **Dual Segment Processing**: Creates both 1-second and 4-second segments
- **Cross-Task Evaluation**: Train on one task type, test on another
- **N-Fold Cross-Validation**: With stratification by gender, age, or both

## Key Features

- **Dual Segment Processing**: Automatically creates both 1s and 4s segments during preprocessing
- **Data Processing**: Converts (2, 60, 240) EEG data to (60, 200) or (60, 800) by concatenating runs
- **Task Separation**: Separates active (ccd) and passive (sus) tasks
- **Participant Batching**: Groups 10 participants per pickle file for efficient loading
- **PyTorch Integration**: Full PyTorch Dataset and DataLoader support
- **Flexible Splits**: Easy train/validation/test splitting by participants
- **Cross-Validation**: N-fold cross-validation with stratification
- **Cross-Task Evaluation**: Train on one task type, test on another
- **Participant Isolation**: Ensures no data leakage between splits
- **Type Safety**: Comprehensive type hints and validation

### Important Notes
- The pipeline preserves participant-level splits to avoid data leakage
- Each pickle file contains exactly 10 participants for efficient loading
- The dataset supports filtering by task type (active/passive/both)
- All EEG data is converted to PyTorch tensors automatically
- Both 1-second and 4-second segments are created during preprocessing

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

### Output Data Structure
Each pickle file contains 10 participants with the following structure:
```python
{
    'participant_id': str,
    'gender': float,  # 1.0 = male, 0.0 = female
    'age': float,
    'passive_eeg': [eeg_data_1, eeg_data_2, ...],  # List of (60, 200) or (60, 800) arrays
    'active_eeg': [eeg_data_1, eeg_data_2, ...],   # List of (60, 200) or (60, 800) arrays
    'demographics': {'age': float, 'gender': float}
}
```

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

4. **Participant Batching**:
   - Groups exactly 10 participants per pickle file
   - Maintains participant-level isolation

### Preprocessing Usage
```python
from eeg_data_preprocessing import EEGDataPreprocessor

# Initialize preprocessor
preprocessor = EEGDataPreprocessor(
    data_root="/path/to/preprocessed_new",
    output_dir_1s="/path/to/1s_segments",
    output_dir_4s="/path/to/4s_segments"
)

# Process all participants
participants = preprocessor.process_all_participants()

# The preprocessor automatically creates both 1s and 4s segments
```

## Dataset and DataLoader Classes

### EEGDataset Class
The main dataset class for loading and processing EEG data.

```python
from eeg_dataset import EEGDataset

# Basic usage
dataset = EEGDataset(
    pickle_dir="/path/to/pickles",
    task_type="both",  # "active", "passive", or "both"
    target_type="both",  # "gender", "age", "both", "combined"
    transform=None,  # Optional input transform
    gender_transform=None,  # Optional gender target transform
    age_transform=None,  # Optional age target transform
    combined_transform=None  # Optional combined target transform
)
```

### Specialized Dataset Creation
```python
# Gender classification dataset
gender_dataset = EEGDataset.create_gender_classification_dataset(
    pickle_dir="/path/to/pickles"
)

# Age classification dataset
age_dataset = EEGDataset.create_age_classification_dataset(
    pickle_dir="/path/to/pickles"
)

# Age regression dataset
regression_dataset = EEGDataset.create_age_regression_dataset(
    pickle_dir="/path/to/pickles"
)

# Combined classification dataset
combined_dataset = EEGDataset.create_combined_classification_dataset(
    pickle_dir="/path/to/pickles"
)

# Multi-task dataset
multi_task_dataset = EEGDataset.create_multi_task_dataset(
    pickle_dir="/path/to/pickles"
)
```

### Key Features
- **PyTorch Integration**: Full PyTorch Dataset and DataLoader support
- **Flexible Splits**: Easy train/validation/test splitting by participants
- **Custom Transforms**: Support for both input and target transforms
- **Cross-Validation**: N-fold cross-validation with stratification
- **Cross-Task Evaluation**: Train on one task type, test on another
- **Participant Isolation**: Ensures no data leakage between splits

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
    pickle_dir="/path/to/pickles",
    gender_transform=gender_classification_transform,
    target_type="gender"
)

# Age classification
age_dataset = EEGDataset(
    pickle_dir="/path/to/pickles",
    age_transform=age_classification_transform,
    target_type="age"
)

# Age regression
age_reg_dataset = EEGDataset(
    pickle_dir="/path/to/pickles",
    age_transform=age_regression_transform,
    target_type="age"
)

# Combined classification
combined_dataset = EEGDataset(
    pickle_dir="/path/to/pickles",
    combined_transform=combined_gender_age_classification_transform,
    target_type="combined"
)
```

## Cross-Validation and Cross-Task Evaluation

### Cross-Validation
The system supports stratified cross-validation with multiple stratification options:

```python
from eeg_dataset import EEGDataLoader

# 5-fold cross-validation with gender stratification
fold_loaders = EEGDataLoader.create_cross_validation_loaders(
    pickle_dir="/path/to/pickles",
    n_folds=5,
    stratify_by="gender",  # "gender", "age", or "both"
    task_type="both",
    target_type="both"
)

# Use the fold loaders
for fold, (train_loader, val_loader, test_loader) in enumerate(fold_loaders):
    print(f"Fold {fold + 1}:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
```

### Cross-Task Evaluation
Train on one task type and test on another:

```python
# Train on active tasks, test on passive tasks
train_loader, val_loader, test_loader = EEGDataLoader.create_cross_task_loaders(
    pickle_dir="/path/to/pickles",
    train_task_type="active",
    val_test_task_type="passive",
    task_type="both",
    target_type="both"
)
```

### Cross-Task Cross-Validation
Combine cross-validation with cross-task evaluation:

```python
# 5-fold cross-validation with cross-task evaluation
fold_loaders = EEGDataLoader.create_cross_task_cross_validation_loaders(
    pickle_dir="/path/to/pickles",
    train_task_type="active",
    val_test_task_type="passive",
    n_folds=5,
    stratify_by="gender",
    task_type="both",
    target_type="both"
)
```

## Usage Examples

### Complete Training Pipeline
```python
from eeg_data_preprocessing import EEGDataPreprocessor
from eeg_dataset import EEGDataset, EEGDataLoader
from target_transforms import gender_classification_transform

# 1. Preprocess data
preprocessor = EEGDataPreprocessor(
    data_root="/path/to/preprocessed_new",
    output_dir_1s="/path/to/1s_segments",
    output_dir_4s="/path/to/4s_segments"
)
participants = preprocessor.process_all_participants()

# 2. Create dataset
dataset = EEGDataset(
    pickle_dir="/path/to/1s_segments",
    task_type="both",
    gender_transform=gender_classification_transform,
    target_type="gender"
)

# 3. Create data loaders
train_loader, val_loader, test_loader = EEGDataLoader.create_train_val_test_loaders(
    pickle_dir="/path/to/1s_segments",
    task_type="both",
    target_type="gender"
)

# 4. Train model
for epoch in range(num_epochs):
    for batch in train_loader:
        eeg_data = batch['eeg_data']  # (batch_size, 60, 200)
        gender = batch['gender']      # (batch_size,)
        # ... training code ...
```

### Cross-Validation Example
```python
# 5-fold cross-validation
fold_loaders = EEGDataLoader.create_cross_validation_loaders(
    pickle_dir="/path/to/1s_segments",
    n_folds=5,
    stratify_by="gender",
    task_type="both",
    target_type="gender"
)

# Train on each fold
for fold, (train_loader, val_loader, test_loader) in enumerate(fold_loaders):
    print(f"Training fold {fold + 1}")
    # ... training code for this fold ...
```

## File Structure

```
EEG/data_processing/
├── eeg_data_preprocessing.py    # Main preprocessing pipeline
├── eeg_dataset.py              # PyTorch Dataset and DataLoader classes
├── target_transforms.py        # Target transformation functions
├── constants.py                # Shared constants and utilities
├── processed_eeg_data_1s_segments/  # 1-second segment pickle files
├── processed_eeg_data_4s_segments/  # 4-second segment pickle files
└── README.md                   # This file
```

## Best Practices

### 1. Data Preprocessing
- Always use the dual preprocessing to create both 1s and 4s segments
- Verify data integrity after preprocessing
- Use appropriate batch sizes for your memory constraints

### 2. Dataset Usage
- Choose the appropriate segment length for your model (1s vs 4s)
- Use stratified cross-validation for balanced evaluation
- Ensure participant-level splits to avoid data leakage

### 3. Cross-Validation
- Use appropriate stratification based on your target variable
- Consider the class distribution when choosing the number of folds
- Validate that stratification is working correctly

### 4. Performance
- Use appropriate batch sizes and number of workers
- Consider memory constraints when loading large datasets
- Use pin_memory=True for GPU training

### 5. Reproducibility
- Set random seeds for reproducible results
- Use consistent train/val/test splits across experiments
- Document your preprocessing and evaluation procedures

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or number of workers
2. **Stratification Errors**: Check that you have enough samples per class
3. **Data Loading Errors**: Verify pickle file paths and structure
4. **Transform Errors**: Ensure transforms match your target type

### Debugging Tips

1. **Check Data Distribution**: Use dataset statistics to verify data loading
2. **Validate Stratification**: Test cross-validation with small datasets first
3. **Monitor Memory Usage**: Use appropriate batch sizes for your hardware
4. **Verify Transforms**: Test transforms on sample data before training

## API Reference

### EEGDataPreprocessor
- `__init__(data_root, output_dir_1s, output_dir_4s)`: Initialize preprocessor
- `process_all_participants()`: Process all participants and create segments
- `process_participant(participant_id)`: Process a single participant

### EEGDataset
- `__init__(pickle_dir, task_type, target_type, ...)`: Initialize dataset
- `create_gender_classification_dataset(pickle_dir)`: Create gender classification dataset
- `create_age_classification_dataset(pickle_dir)`: Create age classification dataset
- `create_age_regression_dataset(pickle_dir)`: Create age regression dataset
- `create_combined_classification_dataset(pickle_dir)`: Create combined classification dataset
- `create_multi_task_dataset(pickle_dir)`: Create multi-task dataset

### EEGDataLoader
- `create_dataloader(dataset, batch_size, ...)`: Create PyTorch DataLoader
- `create_train_val_test_loaders(pickle_dir, ...)`: Create train/val/test loaders
- `create_cross_validation_loaders(pickle_dir, ...)`: Create cross-validation loaders
- `create_cross_task_loaders(pickle_dir, ...)`: Create cross-task loaders
- `create_cross_task_cross_validation_loaders(pickle_dir, ...)`: Create cross-task CV loaders

### Target Transforms
- `gender_classification_transform(gender)`: Convert gender to classification target
- `age_classification_transform(age)`: Convert age to classification target
- `age_regression_transform(age)`: Convert age to regression target
- `combined_gender_age_classification_transform(gender, age)`: Convert to combined target