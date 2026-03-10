"""
Dataset configuration module for LaBraM experiments.

This module provides dynamic configuration generation for all experiment types,
following DRY principles to eliminate redundant configuration dictionaries.
"""

import os

# Comprehensive list of all valid dataset names (matching CNN experiment naming)
VALID_DATASET_NAMES = [
    # === AGE CLASSIFICATION EXPERIMENTS ===
    'age_baseline',
    'age_classification',  # alias for age prediction (same as age_baseline; used in eval scripts)
    'age_cv_age_stratified',
    'age_cross_task_active_to_passive',
    'age_cross_task_passive_to_active',
    'age_cross_task_cv_active_to_passive_age_stratified',
    'age_cross_task_cv_passive_to_active_age_stratified',
    
    # === GENDER CLASSIFICATION EXPERIMENTS ===
    'gender_baseline',
    'gender_cv_gender_stratified',
    'gender_cross_task_active_to_passive',
    'gender_cross_task_passive_to_active',
    'gender_cross_task_cv_active_to_passive_gender_stratified',
    'gender_cross_task_cv_passive_to_active_gender_stratified',
    
    # === COMBINED AGE+GENDER CLASSIFICATION EXPERIMENTS ===
    'combined_baseline',
    'combined_cv_combined_stratified',
    'combined_cross_task_active_to_passive',
    'combined_cross_task_passive_to_active',
    'combined_cross_task_cv_active_to_passive_combined_stratified',
    'combined_cross_task_cv_passive_to_active_combined_stratified',
    
    # === MULTI-OUTPUT AGE+GENDER CLASSIFICATION EXPERIMENTS ===
    'multi_output_baseline',
    'multi_output_cv_gender_stratified',
    'multi_output_cross_task_active_to_passive',
    'multi_output_cross_task_passive_to_active',
    'multi_output_cross_task_cv_active_to_passive_gender_stratified',
    'multi_output_cross_task_cv_passive_to_active_gender_stratified',
]

# Base dataset type configurations
DATASET_TYPE_CONFIGS = {
    'age': {'nb_classes': 3, 'metrics': ["accuracy", "balanced_accuracy", "f1_weighted"]},
    'gender': {'nb_classes': 1, 'metrics': ["accuracy", "balanced_accuracy", "pr_auc", "roc_auc"]},  # Binary classification metrics
    'combined': {'nb_classes': 6, 'metrics': ["accuracy", "balanced_accuracy", "f1_weighted"]},
    'multi_output': {'nb_classes': 2, 'metrics': ["accuracy", "balanced_accuracy", "f1_weighted"]},
}

# Data paths for different segment lengths (HDF5 format). Same dir for all; multipart: eeg_data_{train,val,test}_{1s|2s|4s}_part*.h5
# Override with environment variable EEG_HDF5_DIR or --data_path when running.
DATA_PATHS = {
    '1s': "/home/mojtabam/scratch/processed_eeg_data_hdf5",
    '2s': "/home/mojtabam/scratch/processed_eeg_data_hdf5",
    '4s': "/home/mojtabam/scratch/processed_eeg_data_hdf5"
}

def get_dataset_type_and_params(dataset_name):
    """
    Extract dataset type and parameters from dataset name.
    
    Args:
        dataset_name: Name like 'age_baseline', 'gender_cv_gender_stratified', etc.
        
    Returns:
        Tuple of (dataset_type, kwargs) where dataset_type is the base type
        and kwargs contains experiment-specific parameters
    """
    # Convert to lowercase for easier parsing
    name_lower = dataset_name.lower()
    
    # Extract base dataset type (first part before first underscore)
    if name_lower.startswith('age_'):
        dataset_type = 'age'
    elif name_lower.startswith('gender_'):
        dataset_type = 'gender'
    elif name_lower.startswith('combined_'):
        dataset_type = 'combined'
    elif name_lower.startswith('multi_output_'):
        dataset_type = 'multi_output'
    else:
        raise ValueError(f"Unknown dataset type in: {dataset_name}")
    
    # Initialize kwargs
    kwargs = {}
    
    # Parse cross-task cross-validation (most specific)
    if '_cross_task_cv_' in name_lower:
        # Extract task direction
        if 'active_to_passive' in name_lower:
            kwargs['train_task_type'] = 'active'
            kwargs['val_test_task_type'] = 'passive'
        elif 'passive_to_active' in name_lower:
            kwargs['train_task_type'] = 'passive'
            kwargs['val_test_task_type'] = 'active'
        else:
            raise ValueError(f"Unknown task direction in: {dataset_name}")
        
        # Extract stratification
        if 'age_stratified' in name_lower:
            kwargs['stratify_by'] = 'age'
        elif 'gender_stratified' in name_lower:
            kwargs['stratify_by'] = 'gender'
        elif 'combined_stratified' in name_lower:
            kwargs['stratify_by'] = 'both'
        else:
            raise ValueError(f"Unknown stratification in: {dataset_name}")
        
        kwargs['n_folds'] = 5
    
    # Parse cross-task only
    elif '_cross_task_' in name_lower:
        # Extract task direction
        if 'active_to_passive' in name_lower:
            kwargs['train_task_type'] = 'active'
            kwargs['val_test_task_type'] = 'passive'
        elif 'passive_to_active' in name_lower:
            kwargs['train_task_type'] = 'passive'
            kwargs['val_test_task_type'] = 'active'
        else:
            raise ValueError(f"Unknown task direction in: {dataset_name}")
    
    # Parse cross-validation only
    elif '_cv_' in name_lower:
        # Extract stratification
        if 'age_stratified' in name_lower:
            kwargs['stratify_by'] = 'age'
        elif 'gender_stratified' in name_lower:
            kwargs['stratify_by'] = 'gender'
        elif 'combined_stratified' in name_lower:
            kwargs['stratify_by'] = 'both'
        else:
            raise ValueError(f"Unknown stratification in: {dataset_name}")
        
        kwargs['n_folds'] = 5
    
    # Handle baseline (no additional parameters needed)
    elif name_lower.endswith('_baseline'):
        pass  # No additional kwargs needed
    
    return dataset_type, kwargs

def get_dataset_config(dataset_name):
    """
    Get dataset configuration based on dataset name.
    
    Args:
        dataset_name: Name like 'age_baseline', 'gender_cv_gender_stratified', etc.
        
    Returns:
        Dictionary with 'nb_classes' and 'metrics' keys
    """
    # Extract base dataset type from the name
    dataset_type, _ = get_dataset_type_and_params(dataset_name)
    
    # Return the configuration for that dataset type
    return DATASET_TYPE_CONFIGS[dataset_type]

def get_data_path(segment_length='1s'):
    """
    Get HDF5 data directory for the given segment length.
    Directory must contain multipart or single HDF5 files:
    eeg_data_train_{1s|2s|4s}[_part*.h5], eeg_data_val_*, eeg_data_test_*.

    Uses EEG_HDF5_DIR if set, otherwise DATA_PATHS[segment_length] (same as CNN).

    Args:
        segment_length: '1s', '2s', or '4s'

    Returns:
        Path to the HDF5 data directory (string)

    Raises:
        ValueError: If segment_length is invalid.
    """
    if segment_length not in DATA_PATHS:
        raise ValueError(f"Invalid segment length: {segment_length!r}. Must be '1s', '2s', or '4s'")
    path = os.environ.get("EEG_HDF5_DIR")
    if path and path.strip():
        return path.strip()
    return DATA_PATHS[segment_length]

# Generate the comprehensive dataset configs dynamically
CUSTOM_DATASET_CONFIGS = {name: get_dataset_config(name) for name in VALID_DATASET_NAMES}
