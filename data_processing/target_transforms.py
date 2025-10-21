#!/usr/bin/env python3
"""
Target Transform Functions for EEG Data

This module provides various target transform functions for different
machine learning tasks on EEG data. It includes both basic transforms
and factory functions for creating dataset-specific transforms.
"""

import torch
import numpy as np
from typing import Union, Callable, Any, List, Dict
from constants import (
    AGE_CLASS_1_MAX, AGE_CLASS_2_MAX, DEFAULT_MIN_AGE, DEFAULT_MAX_AGE,
    get_age_class, get_gender_class, get_combined_class, validate_gender
)

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
        Age regression transform function with dataset-specific normalization
    """
    _validate_age_list(ages)
    min_age = min(ages)
    max_age = max(ages)
    
    def age_regression_transform_with_range(age: Union[float, int]) -> torch.Tensor:
        """Age regression transform with dataset-specific range."""
        normalized_age = (age - min_age) / (max_age - min_age)
        return torch.tensor(max(0.0, min(1.0, normalized_age)))
    
    return age_regression_transform_with_range

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

# Dataset integration functions
def create_age_regression_transform_from_dataset(dataset) -> Callable:
    """Create age regression transform from EEGDataset."""
    ages = _extract_ages_from_dataset(dataset)
    return create_age_regression_transform(ages)

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
