# EEG Data Processing Pipeline

This comprehensive guide covers the entire EEG data processing pipeline, from raw data preprocessing to model training with PyTorch. The system has been completely refactored to eliminate code duplication and provide a clean, maintainable architecture.

## Quick Start

### Basic Usage
```python
from eeg_data_preprocessing import EEGDataPreprocessor
from eeg_dataset import EEGDataset, EEGDataLoader

# 1. Preprocess data
preprocessor = EEGDataPreprocessor(
    data_root="/path/to/preprocessed_new",
    output_dir="/path/to/output"
)
participants = preprocessor.process_all_participants()
preprocessor.create_pickle_batches(participants, batch_size=10)

# 2. Create dataset
dataset = EEGDataset(
    pickle_dir="/path/to/output",
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
    eeg_data = batch['eeg_data']      # Shape: (batch_size, 60, 200)
    gender = batch['gender']          # Shape: (batch_size,)
    age = batch['age']                # Shape: (batch_size,)
    participant_ids = batch['participant_ids']
    task_names = batch['task_names']
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
6. [Dynamic Transforms](#dynamic-transforms)
7. [Shared Constants and Utilities](#shared-constants-and-utilities)
8. [Configuration System](#configuration-system)
9. [Cross-Validation and Cross-Task Evaluation](#cross-validation-and-cross-task-evaluation)
10. [Usage Examples](#usage-examples)
11. [File Structure](#file-structure)
12. [Best Practices](#best-practices)

## Overview

This pipeline processes EEG data from the preprocessed_new directory structure and creates PyTorch datasets and data loaders for machine learning tasks. The system supports:

- **Gender Classification**: Binary classification (female/male)
- **Age Classification**: 3-class classification (<8.5, 8.5-12.5, >12.5 years)
- **Age Regression**: Continuous age prediction with multiple normalization strategies
- **Combined Classification**: Gender+age combinations (6 classes)
- **Dynamic Transforms**: Dataset-specific transforms that adapt to your data
- **Cross-Task Evaluation**: Train on one task type, test on another
- **N-Fold Cross-Validation**: With stratification by gender, age, or both

## Key Features

- **Data Processing**: Converts (2, 60, 240) EEG data to (60, 200) by dropping first 40 time points
- **Task Separation**: Separates active (ccd) and passive (sus) tasks
- **Run Separation**: Creates separate runs for each channel (run1, run2)
- **Participant Batching**: Groups 10 participants per pickle file for efficient loading
- **PyTorch Integration**: Full PyTorch Dataset and DataLoader support
- **Flexible Splits**: Easy train/validation/test splitting by participants
- **Statistics**: Built-in dataset statistics and analysis tools
- **Dynamic Transforms**: Dataset-specific transforms that adapt to your data
- **Cross-Validation**: N-fold cross-validation with stratification
- **Cross-Task Evaluation**: Train on one task type, test on another
- **Participant Isolation**: Ensures no data leakage between splits
- **Type Safety**: Comprehensive type hints and validation
- **Configuration System**: Easy-to-use configuration for all transform types

### Important Notes
- The pipeline preserves participant-level splits to avoid data leakage
- Each pickle file contains exactly 10 participants for efficient loading
- The dataset supports filtering by task type (active/passive/both)
- All EEG data is converted to PyTorch tensors automatically

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
    'passive_eeg': {
        'sus_trial_0_run1': [eeg_data_1, eeg_data_2, ...],  # Each eeg_data: (60, 200)
        'sus_trial_0_run2': [eeg_data_1, eeg_data_2, ...],
        'sus_trial_1_run1': [eeg_data_1, eeg_data_2, ...],
        ...
    },
    'active_eeg': {
        'ccd_trial_0_run1': [eeg_data_1, eeg_data_2, ...],
        'ccd_trial_0_run2': [eeg_data_1, eeg_data_2, ...],
        'ccd_trial_1_run1': [eeg_data_1, eeg_data_2, ...],
        ...
    }
}
```

## Data Preprocessing

### Data Processing Details
1. **EEG Data Processing**:
   - Input: (2, 60, 240) - 2 channels, 60 electrodes, 240 time points
   - Output: Two separate (60, 200) arrays - one per channel
   - Processing: Drop first 40 time points from each channel

2. **Task Naming**:
   - Active tasks: `ccd_trial_{trial_num}_run{run_num}`
   - Passive tasks: `sus_trial_{trial_num}_run{run_num}`
   - Each trial creates two runs (one per channel)

3. **Demographics**:
   - Age: Continuous value
   - Gender: 1.0 (male), 0.0 (female)
   - Additional fields: handedness, p_factor, attention, internalizing, externalizing

### Usage
```python
from eeg_data_preprocessing import EEGDataPreprocessor

# Initialize preprocessor
preprocessor = EEGDataPreprocessor(
    data_root="/path/to/preprocessed_new",
    output_dir="/path/to/output"
)

# Process all participants
participants = preprocessor.process_all_participants()

# Create pickle batches (10 participants per file)
preprocessor.create_pickle_batches(participants, batch_size=10)
```

## Dataset and DataLoader Classes

### EEGDataset Class
```python
from eeg_dataset import EEGDataset, EEGDataLoader

# Create dataset with separate transforms
dataset = EEGDataset(
    pickle_dir="/path/to/pickle/files",
    task_type="both",  # "active", "passive", or "both"
    gender_transform=your_gender_transform_function,  # Optional
    age_transform=your_age_transform_function,        # Optional
    target_type="both"  # "gender", "age", "both", "combined"
)

# Create data loader
dataloader = EEGDataLoader.create_dataloader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

### Factory Methods for Common Use Cases
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
- **Statistics**: Built-in dataset statistics and analysis tools
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

# Multi-task learning (both gender and age)
multi_task_dataset = EEGDataset(
    pickle_dir="/path/to/pickles",
    gender_transform=gender_classification_transform,
    age_transform=age_regression_transform,
    target_type="both"
)
```

## Dynamic Transforms

The system now supports dynamic transforms that adapt to your specific dataset characteristics.

### Factory Functions

```python
from target_transforms import (
    create_age_regression_transform,
    create_age_standardized_regression_transform,
    create_age_binned_regression_transform,
    create_age_regression_transform_from_dataset,
    create_age_standardized_regression_transform_from_dataset,
    create_age_binned_regression_transform_from_dataset
)

# Method 1: Create from age list
ages = [6.2, 7.8, 9.1, 10.5, 11.2, 13.8, 15.1, 16.7, 18.3, 20.1]
age_transform = create_age_regression_transform(ages)

# Method 2: Create from dataset (recommended)
dataset = EEGDataset(pickle_dir="/path/to/pickles", task_type="both")
age_transform = create_age_regression_transform_from_dataset(dataset)

# Use the dynamic transform
dynamic_dataset = EEGDataset(
    pickle_dir="/path/to/pickles",
    age_transform=age_transform,
    target_type="age"
)
```

### Benefits of Dynamic Transforms
- **Dataset-Specific**: Automatically adapts to your data's age range
- **No Hardcoding**: Eliminates need for manual age range specification
- **Consistent Normalization**: Ensures proper scaling across different datasets
- **Easy Integration**: Works seamlessly with existing pipeline

## Shared Constants and Utilities

The system now uses a centralized `constants.py` module to eliminate code duplication and ensure consistency.

### Key Constants
```python
from constants import (
    AGE_CLASS_1_MAX,      # 8.5
    AGE_CLASS_2_MAX,      # 12.5
    DEFAULT_MIN_AGE,      # 5.0
    DEFAULT_MAX_AGE,      # 22.0
    get_age_class,        # Get age class index
    get_gender_class,     # Get gender class index
    get_combined_class,   # Get combined class index
    get_stratify_label,   # Get stratification label
    validate_gender,      # Validate gender value
    validate_stratify_by  # Validate stratify_by parameter
)
```

### Utility Functions
```python
# Age classification
age_class = get_age_class(12.0)  # Returns 1 (8.5-12.5)

# Gender classification
gender_class = get_gender_class(1.0)  # Returns 1 (male)

# Combined classification
combined_class = get_combined_class(1.0, 12.0)  # Returns 4 (male, 8.5-12.5)

# Stratification
participant_data = {'gender': 1.0, 'age': 12.0}
stratify_label = get_stratify_label(participant_data, "both")  # Returns 4
```

## Model Examples

### Gender Classification Model
```python
import torch
import torch.nn as nn

class GenderClassifier(nn.Module):
    def __init__(self, input_size=60*200):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # 2 classes: female, male
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Usage
model = GenderClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

### Age Classification Model
```python
class AgeClassifier(nn.Module):
    def __init__(self, input_size=60*200):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3)  # 3 classes: <8.5, 8.5-12.5, >12.5
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```

### Age Regression Model
```python
class AgeRegressor(nn.Module):
    def __init__(self, input_size=60*200):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)  # 1 output for age
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.regressor(x)
```

### Combined Gender+Age Classification Model
```python
class CombinedGenderAgeClassifier(nn.Module):
    def __init__(self, input_size=60*200):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 6)  # 6 classes: gender+age combinations
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```

## Configuration System

The updated configuration system provides easy access to all transform types and model configurations.

### Quick Setup
```python
from transform_config import create_dataset_with_transform, create_dynamic_dataset

# Create gender classification dataset
gender_dataset = create_dataset_with_transform(
    pickle_dir="/path/to/pickles",
    transform_type="gender_classification",
    task_type="both",
    target_type="gender"
)

# Create age classification dataset
age_dataset = create_dataset_with_transform(
    pickle_dir="/path/to/pickles",
    transform_type="age_classification",
    task_type="both",
    target_type="age"
)

# Create dynamic age regression dataset
dynamic_dataset = create_dynamic_dataset(
    pickle_dir="/path/to/pickles",
    transform_type="dynamic_age_regression",
    task_type="both",
    target_type="age"
)
```

### Available Transform Types
```python
from transform_config import list_available_transforms, list_available_models

# List all available transforms
list_available_transforms()

# List all available model types
list_available_models()
```

### Transform Types (10 total)
- `gender_classification` - Gender classification (0/1)
- `age_regression` - Age regression (normalized)
- `age_standardized_regression` - Age regression (standardized)
- `age_binned_regression` - Age regression (binned)
- `age_classification` - Age classification (3 classes)
- `combined_gender_age_classification` - Combined classification (6 classes)
- `dynamic_age_regression` - Dynamic age regression
- `dynamic_age_standardized_regression` - Dynamic standardized regression
- `dynamic_age_binned_regression` - Dynamic binned regression
- `no_transform` - Raw data

### Model Types (9 total)
- `gender_classifier` - Gender classification
- `age_classifier` - Age classification
- `age_regressor` - Age regression
- `standardized_age_regressor` - Standardized age regression
- `binned_age_regressor` - Binned age regression
- `combined_classifier` - Combined classification
- `dynamic_age_regressor` - Dynamic age regression
- `dynamic_standardized_age_regressor` - Dynamic standardized regression
- `dynamic_binned_age_regressor` - Dynamic binned regression

## Cross-Validation and Cross-Task Evaluation

The system now supports advanced evaluation strategies for robust model assessment.

### N-Fold Cross-Validation
```python
from eeg_dataset import EEGDataLoader

# Create 5-fold cross-validation with stratification
fold_loaders = EEGDataLoader.create_cross_validation_loaders(
    pickle_dir="/path/to/pickles",
    n_folds=5,
    stratify_by="both",  # "gender", "age", or "both"
    batch_size=32,
    task_type="both",
    target_type="age",
    age_transform=your_age_transform
)

# Use the fold loaders
for fold, (train_loader, val_loader, test_loader) in enumerate(fold_loaders):
    print(f"Fold {fold + 1}:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")
```

### Cross-Task Evaluation
```python
# Train on active tasks, test on passive tasks
train_loader, val_loader, test_loader = EEGDataLoader.create_cross_task_loaders(
    pickle_dir="/path/to/pickles",
    train_task_type="active",
    val_test_task_type="passive",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    batch_size=32,
    target_type="age",
    age_transform=your_age_transform
)

# Cross-task evaluation with cross-validation
fold_loaders = EEGDataLoader.create_cross_task_cross_validation_loaders(
    pickle_dir="/path/to/pickles",
    train_task_type="active",
    val_test_task_type="passive",
    n_folds=5,
    stratify_by="both",
    batch_size=32,
    target_type="age",
    age_transform=your_age_transform
)
```

### Stratification Options
- **`"gender"`**: Stratify by gender (2 groups)
- **`"age"`**: Stratify by age groups (3 groups: <8.5, 8.5-12.5, >12.5)
- **`"both"`**: Stratify by combined gender+age (6 groups)

### Key Benefits
- **No Data Leakage**: Participant-level splits ensure no overlap
- **Balanced Splits**: Stratification maintains demographic ratios
- **Robust Evaluation**: Multiple folds provide reliable performance estimates
- **Cross-Task Generalization**: Test model's ability to generalize across task types

## Usage Examples

### Complete Pipeline Example
```python
from eeg_data_preprocessing import EEGDataPreprocessor
from eeg_dataset import EEGDataset, EEGDataLoader
from target_transforms import gender_classification_transform

# Step 1: Preprocess data
preprocessor = EEGDataPreprocessor(
    data_root="/path/to/preprocessed_new",
    output_dir="/path/to/output"
)
participants = preprocessor.process_all_participants()
preprocessor.create_pickle_batches(participants, batch_size=10)

# Step 2: Create dataset
dataset = EEGDataset(
    pickle_dir="/path/to/output",
    task_type="both",
    gender_transform=gender_classification_transform
)

# Step 3: Create data loader
dataloader = EEGDataLoader.create_dataloader(
    dataset=dataset,
    batch_size=32,
    shuffle=True
)

# Step 4: Train model
model = GenderClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for batch in dataloader:
        eeg_data = batch['eeg_data']
        gender_labels = batch['gender']
        
        optimizer.zero_grad()
        outputs = model(eeg_data)
        loss = criterion(outputs, gender_labels)
        loss.backward()
        optimizer.step()
```

### Train/Validation/Test Splits
```python
# Create train/val/test splits
train_loader, val_loader, test_loader = EEGDataLoader.create_train_val_test_loaders(
    pickle_dir="/path/to/pickles",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    batch_size=32
)
```

## File Structure

```
EEG/data_processing/
├── eeg_data_preprocessing.py         # Data preprocessing script
├── eeg_dataset.py                   # PyTorch Dataset and DataLoader classes
├── target_transforms.py             # Transform functions and factory methods
├── constants.py                     # Shared constants and utilities
├── transform_config.py              # Configuration system
└── README.md                        # This comprehensive guide
```

### Key Files Description

#### Core Processing Files
- **`eeg_data_preprocessing.py`**: Converts raw EEG data to pickle format
- **`eeg_dataset.py`**: PyTorch Dataset and DataLoader with advanced splitting
- **`target_transforms.py`**: Transform functions and dynamic factory methods
- **`constants.py`**: Shared constants and utility functions

#### Configuration and Utilities
- **`transform_config.py`**: Easy configuration for all transform types
- **`README.md`**: This comprehensive guide

## Best Practices

### 1. Data Preprocessing
- Always validate data shapes before processing
- Use appropriate batch sizes for pickle files
- Handle missing data gracefully

### 2. Model Design
- Use appropriate output dimensions for your task
- Apply dropout for regularization
- Use batch normalization if needed

### 3. Training
- Use appropriate loss functions (CrossEntropyLoss for classification, MSELoss for regression)
- Monitor training and validation loss
- Use early stopping to prevent overfitting

### 4. Evaluation
- Use participant-level splits to avoid data leakage
- Evaluate on held-out test set
- Report metrics appropriate for your task

### 5. Transform Selection
- Choose transforms based on your model requirements
- Use dynamic transforms for dataset-specific normalization
- Test transforms before training
- Document your transform choices

### 6. Cross-Validation
- Always use participant-level splits to avoid data leakage
- Use stratification to maintain demographic balance
- Consider cross-task evaluation for generalization testing
- Use multiple folds for robust performance estimates

### 7. Dynamic Transforms
- Prefer dynamic transforms over hardcoded parameters
- Use `create_*_transform_from_dataset()` for automatic adaptation
- Test transforms on your specific dataset before training

## Model Configurations Summary

| Task | Transform Function | Model Outputs | Classes | Loss Function | Dynamic Support |
|------|-------------------|---------------|---------|---------------|-----------------|
| Gender Classification | `gender_classification_transform` | 2 | 0=female, 1=male | CrossEntropyLoss | No |
| Age Classification | `age_classification_transform` | 3 | 0=<8.5, 1=8.5-12.5, 2=>12.5 | CrossEntropyLoss | No |
| Age Regression | `age_regression_transform` | 1 | Normalized [0,1] | MSELoss | Yes |
| Age Standardized Regression | `age_standardized_regression_transform` | 1 | Standardized | MSELoss | Yes |
| Age Binned Regression | `age_binned_regression_transform` | 1 | Binned [0,1] | MSELoss | Yes |
| Combined Classification | `combined_gender_age_classification_transform` | 6 | 0-5: gender+age combinations | CrossEntropyLoss | No |

## Testing

```python
from target_transforms import test_transforms

# Test all transform functions
test_transforms()
```

## Key Benefits

1. **Complete Pipeline**: From raw data to trained models with advanced evaluation
2. **Flexible**: Easy to switch between different tasks, transforms, and evaluation strategies
3. **PyTorch Integration**: Full compatibility with PyTorch ecosystem
4. **Efficient**: Optimized data loading and preprocessing
5. **Well-Documented**: Comprehensive examples and documentation
6. **Modular**: Easy to extend and customize
7. **DRY Architecture**: Eliminated code duplication with shared constants and utilities
8. **Dynamic Transforms**: Dataset-specific transforms that adapt to your data
9. **Advanced Evaluation**: N-fold cross-validation and cross-task evaluation
10. **Robust Splitting**: Participant-level splits with stratification to prevent data leakage
11. **Type Safety**: Comprehensive type hints and validation
12. **Configuration System**: Easy-to-use configuration for all transform types

## Recent Improvements

### Code Quality
- **Eliminated Duplication**: Created `constants.py` to centralize shared logic
- **Better Organization**: Clear separation of concerns across modules
- **Type Safety**: Added comprehensive type hints throughout
- **Error Handling**: Improved validation and error messages

### New Features
- **Dynamic Transforms**: Factory functions for dataset-specific transforms
- **Cross-Validation**: N-fold cross-validation with stratification
- **Cross-Task Evaluation**: Train on one task type, test on another
- **Enhanced Configuration**: Updated `transform_config.py` with all new features
- **Factory Methods**: Convenient dataset creation methods

### Performance
- **Faster Transforms**: Optimized transform functions
- **Better Memory Usage**: Efficient data loading and batching
- **Parallel Processing**: Support for multi-worker data loading

This pipeline provides everything you need to process EEG data and train machine learning models for gender classification, age prediction, and combined tasks with state-of-the-art evaluation strategies.
