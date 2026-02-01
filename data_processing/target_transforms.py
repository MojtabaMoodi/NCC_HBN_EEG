#!/usr/bin/env python3
"""
Target Transform Functions for EEG Data

This module provides various target transform functions for different
machine learning tasks on EEG data. It includes both basic transforms
and factory functions for creating dataset-specific transforms.
"""

import json
import logging
from pathlib import Path
import h5py
import torch
import numpy as np
from typing import Union, Callable, Any, List, Dict, Tuple, Optional
from constants import (
    AGE_CLASS_1_MAX, AGE_CLASS_2_MAX, DEFAULT_MIN_AGE, DEFAULT_MAX_AGE,
    get_age_class, get_gender_class, get_combined_class, validate_gender
)

logger = logging.getLogger(__name__)

# =============================================================================
# BASIC TRANSFORM FUNCTIONS
# =============================================================================

def gender_classification_transform(gender: Union[float, int]) -> torch.Tensor:
    """
    Transform gender for classification task.
    
    Args:
        gender: Gender value (0.0 = female, 1.0 = male)
        
    Returns:
        Class index tensor: 0 (female), 1 (male)
    """
    validate_gender(gender)
    gender_class = get_gender_class(gender)
    return torch.tensor(gender_class, dtype=torch.long)

def age_classification_transform(age: Union[float, int]) -> torch.Tensor:
    """
    Transform age for classification task.
    
    Class 0: < 8.5 years
    Class 1: 8.5-12.5 years  
    Class 2: > 12.5 years
    
    Args:
        age: Age value in years
        
    Returns:
        Class index tensor: 0 (<8.5), 1 (8.5-12.5), 2 (>12.5)
    """
    age_class = get_age_class(age)
    return torch.tensor(age_class, dtype=torch.long)

def age_regression_transform(age: Union[float, int], 
                           min_age: float = DEFAULT_MIN_AGE, 
                           max_age: float = DEFAULT_MAX_AGE) -> torch.Tensor:
    """
    Transform age for regression task using min-max normalization.
    
    Args:
        age: Age value in years
        min_age: Minimum age in the dataset (default: 5.0)
        max_age: Maximum age in the dataset (default: 22.0)
        
    Returns:
        Normalized age tensor (0-1 range)
    """
    # Normalize age to [0, 1] range
    normalized_age = (age - min_age) / (max_age - min_age)
    return torch.tensor(max(0.0, min(1.0, normalized_age)))  # Clamp to [0, 1]

def age_standardized_regression_transform(age: Union[float, int], 
                                        mean_age: float = 10.5, 
                                        std_age: float = 3.4) -> torch.Tensor:
    """
    Transform age for regression task using standardization.
    
    Args:
        age: Age value in years
        mean_age: Mean age in the dataset (default: 10.5)
        std_age: Standard deviation of age in the dataset (default: 3.4)
        
    Returns:
        Standardized age tensor (mean=0, std=1)
    """
    standardized_age = (age - mean_age) / std_age
    return torch.tensor(standardized_age)

def age_binned_regression_transform(age: Union[float, int], 
                                  min_age: float = DEFAULT_MIN_AGE, 
                                  max_age: float = DEFAULT_MAX_AGE) -> torch.Tensor:
    """
    Transform age for regression task using binning.
    
    Args:
        age: Age value in years
        min_age: Minimum age in the dataset (default: 5.0)
        max_age: Maximum age in the dataset (default: 22.0)
        
    Returns:
        Binned age tensor (0-1 range within each bin)
    """
    if age < AGE_CLASS_1_MAX:
        # Bin 1: min_age-8.5 years -> 0-1
        normalized = (age - min_age) / (AGE_CLASS_1_MAX - min_age)
        return torch.tensor(max(0.0, min(1.0, normalized)))
    elif age <= AGE_CLASS_2_MAX:
        # Bin 2: 8.5-12.5 years -> 0-1
        normalized = (age - AGE_CLASS_1_MAX) / (AGE_CLASS_2_MAX - AGE_CLASS_1_MAX)
        return torch.tensor(max(0.0, min(1.0, normalized)))
    else:
        # Bin 3: 12.5-max_age years -> 0-1
        normalized = (age - AGE_CLASS_2_MAX) / (max_age - AGE_CLASS_2_MAX)
        return torch.tensor(max(0.0, min(1.0, normalized)))

def combined_gender_age_classification_transform(gender: Union[float, int], 
                                               age: Union[float, int]) -> torch.Tensor:
    """
    Transform gender and age for combined classification task.
    
    Combined classes:
    0: (female, <8.5)
    1: (female, 8.5-12.5)
    2: (female, >12.5)
    3: (male, <8.5)
    4: (male, 8.5-12.5)
    5: (male, >12.5)
    
    Args:
        gender: Gender value (0.0 = female, 1.0 = male)
        age: Age value in years
        
    Returns:
        Combined class index tensor
    """
    combined_class = get_combined_class(gender, age)
    return torch.tensor(combined_class, dtype=torch.long)

class UserIdentificationTransform:
    """
    Picklable user identification transform class for multiprocessing compatibility.
    Maps participant_id to class index for user identification task.
    """
    # Sentinel label for unknown (held-out) users when consider_unknown_users=True
    UNKNOWN_LABEL = -1

    def __init__(self, participant_id_to_class_idx: Dict[str, int], unknown_label: Optional[int] = None):
        """
        Initialize user identification transform.
        
        Args:
            participant_id_to_class_idx: Dictionary mapping participant_id to class index
            unknown_label: If set, return this label for participant_id not in mapping (e.g. -1 for unknown split).
                          If None, raise KeyError for unknown participant_id (default).
        """
        self.participant_id_to_class_idx = participant_id_to_class_idx
        self.unknown_label = unknown_label
        # Create reverse mapping for validation
        self.class_idx_to_participant_id = {v: k for k, v in participant_id_to_class_idx.items()}
    
    def __call__(self, participant_id: str) -> torch.Tensor:
        """
        Transform participant_id to class index.
        
        Args:
            participant_id: Participant ID string
            
        Returns:
            Class index tensor (or unknown_label if participant not in mapping and unknown_label is set)
            
        Raises:
            KeyError: If participant_id is not in the mapping and unknown_label is None
        """
        if participant_id not in self.participant_id_to_class_idx:
            if self.unknown_label is not None:
                return torch.tensor(self.unknown_label, dtype=torch.long)
            raise KeyError(
                f"Participant ID '{participant_id}' not found in participant_id_to_class_idx mapping. "
                f"This indicates a data inconsistency issue."
            )
        class_idx = self.participant_id_to_class_idx[participant_id]
        return torch.tensor(class_idx, dtype=torch.long)
    
    def get_num_classes(self) -> int:
        """Get the number of user classes."""
        return len(self.participant_id_to_class_idx)
    
    def get_participant_id(self, class_idx: int) -> str:
        """
        Get participant_id from class index (reverse lookup).
        
        Args:
            class_idx: Class index
            
        Returns:
            Participant ID string
        """
        if class_idx not in self.class_idx_to_participant_id:
            raise KeyError(f"Class index {class_idx} not found in mapping")
        return self.class_idx_to_participant_id[class_idx]

def create_user_identification_transform_from_mapping_file(mapping_file_path: str) -> UserIdentificationTransform:
    """
    Create user identification transform from a JSON mapping file.
    
    Args:
        mapping_file_path: Path to JSON file containing participant_id to class_idx mapping
        
    Returns:
        UserIdentificationTransform instance
    """
    
    mapping_path = Path(mapping_file_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file_path}")
    
    with open(mapping_path, 'r') as f:
        participant_id_to_class_idx = json.load(f)
    
    return UserIdentificationTransform(participant_id_to_class_idx)

def create_user_identification_transform_from_hdf5(
    hdf5_dir: str,
    unknown_label: Optional[int] = None
) -> Tuple[UserIdentificationTransform, int]:
    """
    Create user identification transform from HDF5 directory.
    Looks for participant_id_to_class_idx.json in the HDF5 directory.
    
    Args:
        hdf5_dir: Directory containing HDF5 files and mapping JSON
        unknown_label: If set (e.g. UserIdentificationTransform.UNKNOWN_LABEL), return this label
                      for participant_id not in mapping. Use when loading "unknown" split.
        
    Returns:
        Tuple of (transform, num_classes)
    """
    hdf5_dir_path = Path(hdf5_dir)
    mapping_file = hdf5_dir_path / 'participant_id_to_class_idx.json'
    
    if not mapping_file.exists():
        raise FileNotFoundError(
            f"Mapping file not found: {mapping_file}. "
            f"Please run preprocessing with process_all_participants_user_identification() first."
        )
    
    with open(mapping_file, 'r') as f:
        participant_id_to_class_idx = json.load(f)
    
    transform = UserIdentificationTransform(
        participant_id_to_class_idx,
        unknown_label=unknown_label
    )
    num_classes = transform.get_num_classes()
    
    return transform, num_classes

# =============================================================================
# FACTORY FUNCTIONS FOR DATASET-SPECIFIC TRANSFORMS
# =============================================================================

def _validate_age_list(ages: List[Union[float, int]]) -> None:
    """Validate that age list is not empty."""
    if not ages:
        raise ValueError("Cannot create transform from empty age list")

def _extract_ages_from_dataset(dataset) -> List[Union[float, int]]:
    """Extract age values from EEGDataset."""
    return [sample['age'] for sample in dataset.samples]

def create_age_regression_transform(ages: List[Union[float, int]]) -> Callable:
    """
    Create an age regression transform with dataset-specific age range.
    
    Args:
        ages: List of age values from the dataset
        
    Returns:
        Age regression transform (picklable AgeRegressionTransform instance)
    """
    _validate_age_list(ages)
    min_age = min(ages)
    max_age = max(ages)
    
    # Use picklable class instead of closure for multiprocessing compatibility
    return AgeRegressionTransform(min_age, max_age)

def create_age_standardized_regression_transform(ages: List[Union[float, int]]) -> Callable:
    """
    Create an age standardized regression transform with dataset-specific statistics.
    
    Args:
        ages: List of age values from the dataset
        
    Returns:
        Age standardized regression transform function with dataset-specific statistics
    """
    _validate_age_list(ages)
    mean_age = np.mean(ages)
    std_age = np.std(ages)
    
    def age_standardized_regression_transform_with_stats(age: Union[float, int]) -> torch.Tensor:
        """Age standardized regression transform with dataset-specific statistics."""
        standardized_age = (age - mean_age) / std_age
        return torch.tensor(standardized_age)
    
    return age_standardized_regression_transform_with_stats

def create_age_binned_regression_transform(ages: List[Union[float, int]]) -> Callable:
    """
    Create an age binned regression transform with dataset-specific age range.
    
    Args:
        ages: List of age values from the dataset
        
    Returns:
        Age binned regression transform function with dataset-specific range
    """
    _validate_age_list(ages)
    min_age = min(ages)
    max_age = max(ages)
    
    def age_binned_regression_transform_with_range(age: Union[float, int]) -> torch.Tensor:
        """Age binned regression transform with dataset-specific range."""
        if age < AGE_CLASS_1_MAX:
            # Bin 1: min_age-8.5 years -> 0-1
            normalized = (age - min_age) / (AGE_CLASS_1_MAX - min_age)
            return torch.tensor(max(0.0, min(1.0, normalized)))
        elif age <= AGE_CLASS_2_MAX:
            # Bin 2: 8.5-12.5 years -> 0-1
            normalized = (age - AGE_CLASS_1_MAX) / (AGE_CLASS_2_MAX - AGE_CLASS_1_MAX)
            return torch.tensor(max(0.0, min(1.0, normalized)))
        else:
            # Bin 3: 12.5-max_age years -> 0-1
            normalized = (age - AGE_CLASS_2_MAX) / (max_age - AGE_CLASS_2_MAX)
            return torch.tensor(max(0.0, min(1.0, normalized)))
    
    return age_binned_regression_transform_with_range

class AgeRegressionTransform:
    """
    Picklable age regression transform class for multiprocessing compatibility.
    This class can be used instead of closures to ensure compatibility with
    DataLoader multiprocessing (spawn context).
    """
    def __init__(self, min_age: float, max_age: float):
        self.min_age = min_age
        self.max_age = max_age
    
    def __call__(self, age: Union[float, int]) -> torch.Tensor:
        """Age regression transform with dataset-specific range."""
        normalized_age = (age - self.min_age) / (self.max_age - self.min_age)
        return torch.tensor(max(0.0, min(1.0, normalized_age)))

# Dataset integration functions
def create_age_regression_transform_from_dataset(dataset) -> Callable:
    """Create age regression transform from EEGDataset."""
    ages = _extract_ages_from_dataset(dataset)
    return create_age_regression_transform(ages)

def compute_age_range_from_hdf5(hdf5_file_path_or_files: Union[str, List[str], Path, List[Path]]) -> Tuple[float, float]:
    """
    Compute age min/max from HDF5 file(s) by reading metadata.
    
    Important: For regression normalization, this should be called on the TRAINING file(s) only.
    The computed min/max should then be used for normalizing all splits (train/val/test)
    to avoid data leakage. Never compute min/max from validation or test data.
    
    For 1s segments with multiple files, this function will read from ALL files to compute
    the correct age range across the entire training dataset.
    
    Args:
        hdf5_file_path_or_files: Path to HDF5 file or list of paths (should be training file(s) for proper normalization)
        
    Returns:
        Tuple of (min_age, max_age)
    """
    
    # Normalize input to list of paths
    if isinstance(hdf5_file_path_or_files, (str, Path)):
        file_paths = [Path(hdf5_file_path_or_files)]
    else:
        file_paths = [Path(p) for p in hdf5_file_path_or_files]
    
    ages = []
    try:
        # Check all files for age_min/age_max attributes first (fast path)
        all_have_attributes = True
        attribute_ages = []
        
        for file_path in file_paths:
            if not file_path.exists():
                all_have_attributes = False
                break
            with h5py.File(file_path, 'r') as f:
                if 'age_min' in f.attrs and 'age_max' in f.attrs:
                    file_min = float(f.attrs['age_min'])
                    file_max = float(f.attrs['age_max'])
                    attribute_ages.extend([file_min, file_max])
                else:
                    all_have_attributes = False
        
        # If all files have age_min/age_max attributes, use them
        if all_have_attributes and attribute_ages:
            min_age = float(min(attribute_ages))
            max_age = float(max(attribute_ages))
            return min_age, max_age
        
        # Otherwise, iterate through metadata in all files to collect all ages
        # This is more accurate as it reads actual age values from all samples
        for file_path in file_paths:
            if not file_path.exists():
                continue
            with h5py.File(file_path, 'r') as f:
                for task_group_name in ['active', 'passive']:
                    if task_group_name in f:
                        task_group = f[task_group_name]
                        for key in task_group.keys():
                            if key.startswith('metadata_'):
                                metadata = task_group[key]
                                if 'age' in metadata.attrs:
                                    age = float(metadata.attrs['age'])
                                    ages.append(age)
        
        if ages:
            min_age = float(min(ages))
            max_age = float(max(ages))
            return min_age, max_age
        else:
            return DEFAULT_MIN_AGE, DEFAULT_MAX_AGE
    except Exception as e:
        file_str = str(file_paths[0]) if len(file_paths) == 1 else f"{len(file_paths)} files"
        logger.warning(f"Error computing age range from {file_str}: {e}, using defaults")
        return DEFAULT_MIN_AGE, DEFAULT_MAX_AGE

def create_age_regression_transform_from_hdf5(hdf5_file_path_or_files: Union[str, List[str], Path, List[Path]]) -> Tuple[Callable, float, float]:
    """
    Create age regression transform from HDF5 file(s).
    
    Important: This should be called on the TRAINING file(s) only. The computed min/max
    will be used to normalize all data splits (train/val/test) to prevent data leakage.
    
    For 1s segments with multiple files, this function will read from ALL files to compute
    the correct age range across the entire training dataset.
    
    Args:
        hdf5_file_path_or_files: Path to HDF5 file or list of paths (MUST be training file(s) for proper normalization)
        
    Returns:
        Tuple of (transform_function, min_age, max_age)
        The transform_function is a picklable AgeRegressionTransform instance
    """
    min_age, max_age = compute_age_range_from_hdf5(hdf5_file_path_or_files)
    
    # Return a picklable class instance instead of a closure
    transform = AgeRegressionTransform(min_age, max_age)
    
    return transform, min_age, max_age

def create_age_standardized_regression_transform_from_dataset(dataset) -> Callable:
    """Create age standardized regression transform from EEGDataset."""
    ages = _extract_ages_from_dataset(dataset)
    return create_age_standardized_regression_transform(ages)

def create_age_binned_regression_transform_from_dataset(dataset) -> Callable:
    """Create age binned regression transform from EEGDataset."""
    ages = _extract_ages_from_dataset(dataset)
    return create_age_binned_regression_transform(ages)

# =============================================================================
# CONVENIENCE FUNCTIONS AND CONSTANTS
# =============================================================================

def create_combined_transform(gender_transform: Callable = None, 
                            age_transform: Callable = None) -> Callable:
    """
    Create a combined transform function for both gender and age.
    
    Args:
        gender_transform: Function to transform gender
        age_transform: Function to transform age
        
    Returns:
        Combined transform function
    """
    def combined_transform(target):
        # This is a simplified approach - in practice, you'd need to know
        # which target you're transforming. This function assumes you're
        # calling it separately for each target.
        if gender_transform and age_transform:
            # Try to determine which transform to use based on target value
            if isinstance(target, (int, float)):
                if target in [0.0, 1.0]:
                    return gender_transform(target)  # Gender
                else:
                    return age_transform(target)  # Age
        elif gender_transform:
            return gender_transform(target)
        elif age_transform:
            return age_transform(target)
        
        return torch.tensor(target) if target is not None else torch.tensor(0.0)
    
    return combined_transform

# Pre-defined transform combinations for common use cases
GENDER_CLASSIFICATION = gender_classification_transform
AGE_REGRESSION = age_regression_transform
AGE_CLASSIFICATION = age_classification_transform
AGE_STANDARDIZED_REGRESSION = age_standardized_regression_transform
AGE_BINNED_REGRESSION = age_binned_regression_transform
COMBINED_GENDER_AGE_CLASSIFICATION = combined_gender_age_classification_transform

# Combined transforms for multi-task learning
GENDER_CLASS_AGE_REGR = create_combined_transform(
    gender_classification_transform, 
    age_regression_transform
)

GENDER_CLASS_AGE_CLASS = create_combined_transform(
    gender_classification_transform, 
    age_classification_transform
)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_transform_info(transform_name: str) -> Dict[str, str]:
    """
    Get information about a transform function.
    
    Args:
        transform_name: Name of the transform
        
    Returns:
        Dictionary with transform information
    """
    transforms_info = {
        'gender_classification': {
            'description': 'Gender classification (0/1 -> class indices)',
            'output_shape': '[1]',
            'output_range': '[0,1]',
            'use_case': 'Gender classification models'
        },
        'age_regression': {
            'description': 'Age regression (normalized to 0-1)',
            'output_shape': '[1]',
            'output_range': '[0,1]',
            'use_case': 'Age regression models'
        },
        'age_classification': {
            'description': 'Age classification (3 classes: <8.5, 8.5-12.5, >12.5)',
            'output_shape': '[1]',
            'output_range': '[0,1,2]',
            'use_case': 'Age classification models'
        },
        'age_standardized_regression': {
            'description': 'Age regression (standardized, mean=0, std=1)',
            'output_shape': '[1]',
            'output_range': '[-∞,∞]',
            'use_case': 'Age regression with standardization'
        },
        'age_binned_regression': {
            'description': 'Age regression (binned and normalized)',
            'output_shape': '[1]',
            'output_range': '[0,1]',
            'use_case': 'Age regression with binning'
        },
        'combined_gender_age_classification': {
            'description': 'Combined gender+age classification (6 classes)',
            'output_shape': '[1]',
            'output_range': '[0,1,2,3,4,5]',
            'use_case': 'Combined gender+age classification models'
        }
    }
    
    return transforms_info.get(transform_name, {})

def test_transforms():
    """Test all transform functions."""
    print("Testing Target Transform Functions")
    print("=" * 50)
    
    # Test gender transforms
    print("Gender Classification Transform:")
    print(f"  Female (0.0): {gender_classification_transform(0.0)} (class 0)")
    print(f"  Male (1.0): {gender_classification_transform(1.0)} (class 1)")
    
    # Test age transforms
    test_ages = [6.0, 10.0, 15.0]
    
    print("\nAge Regression Transform (normalized):")
    for age in test_ages:
        result = age_regression_transform(age)
        print(f"  Age {age}: {result}")
    
    print("\nAge Classification Transform:")
    for age in test_ages:
        result = age_classification_transform(age)
        class_name = "Class 0 (<8.5)" if result.item() == 0 else \
                    "Class 1 (8.5-12.5)" if result.item() == 1 else \
                    "Class 2 (>12.5)"
        print(f"  Age {age}: {result} ({class_name})")
    
    print("\nAge Standardized Regression Transform:")
    for age in test_ages:
        result = age_standardized_regression_transform(age)
        print(f"  Age {age}: {result}")
    
    print("\nAge Binned Regression Transform:")
    for age in test_ages:
        result = age_binned_regression_transform(age)
        print(f"  Age {age}: {result}")
    
    # Test combined gender+age classification
    print("\nCombined Gender+Age Classification Transform:")
    test_combinations = [
        (0.0, 6.0),   # Female, <8.5
        (0.0, 10.0),  # Female, 8.5-12.5
        (0.0, 15.0),  # Female, >12.5
        (1.0, 6.0),   # Male, <8.5
        (1.0, 10.0),  # Male, 8.5-12.5
        (1.0, 15.0),  # Male, >12.5
    ]
    
    for gender, age in test_combinations:
        result = combined_gender_age_classification_transform(gender, age)
        gender_name = "Female" if result.item() < 3 else "Male"
        age_class = result.item() % 3
        age_name = "<8.5" if age_class == 0 else "8.5-12.5" if age_class == 1 else ">12.5"
        class_name = f"({gender_name}, {age_name})"
        print(f"  Gender {gender}, Age {age}: {result} ({class_name})")

if __name__ == "__main__":
    test_transforms()
