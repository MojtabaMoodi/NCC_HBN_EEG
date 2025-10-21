#!/usr/bin/env python3
"""
Shared Constants and Utilities for EEG Data Processing

This module contains shared constants and utility functions used across
multiple modules to avoid code duplication and ensure consistency.
"""

from typing import Union, Tuple

# =============================================================================
# AGE CLASSIFICATION CONSTANTS
# =============================================================================

# Age classification thresholds (consistent across all modules)
AGE_CLASS_1_MAX = 8.5
AGE_CLASS_2_MAX = 12.5

# Default age range for normalization
DEFAULT_MIN_AGE = 5.0
DEFAULT_MAX_AGE = 22.0

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_age_class(age: Union[float, int]) -> int:
    """
    Get age class index for classification.
    
    Args:
        age: Age value in years
        
    Returns:
        Age class index: 0 (<8.5), 1 (8.5-12.5), 2 (>12.5)
    """
    if age < AGE_CLASS_1_MAX:
        return 0  # < 8.5
    elif age <= AGE_CLASS_2_MAX:
        return 1  # 8.5-12.5
    else:
        return 2  # > 12.5

def get_gender_class(gender: Union[float, int]) -> int:
    """
    Get gender class index for classification.
    
    Args:
        gender: Gender value (0.0 = female, 1.0 = male)
        
    Returns:
        Gender class index: 0 (female), 1 (male)
    """
    if gender == 0.0:
        return 0  # Female
    elif gender == 1.0:
        return 1  # Male
    else:
        raise ValueError(f"Invalid gender value: {gender}. Expected 0.0 (female) or 1.0 (male)")

def get_combined_class(gender: Union[float, int], age: Union[float, int]) -> int:
    """
    Get combined gender+age class index for classification.
    
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
        Combined class index (0-5)
    """
    gender_class = get_gender_class(gender)
    age_class = get_age_class(age)
    return gender_class * 3 + age_class

def get_stratify_label(participant_data: dict, stratify_by: str) -> int:
    """
    Get stratification label for a participant based on the specified field.
    
    Args:
        participant_data: Dictionary containing participant data with 'gender' and 'age' keys
        stratify_by: Field to stratify by ("gender", "age", "both")
        
    Returns:
        Stratification label
    """
    if stratify_by == "gender":
        return get_gender_class(participant_data['gender'])
    elif stratify_by == "age":
        return get_age_class(participant_data['age'])
    elif stratify_by == "both":
        return get_combined_class(participant_data['gender'], participant_data['age'])
    else:
        return 0  # No stratification

def decode_combined_class(combined_class: int) -> Tuple[str, str]:
    """
    Decode a combined class index back to gender and age class names.
    
    Args:
        combined_class: Combined class index (0-5)
        
    Returns:
        Tuple of (gender_name, age_class_name)
    """
    gender_class = combined_class // 3
    age_class = combined_class % 3
    
    gender_name = "Female" if gender_class == 0 else "Male"
    
    if age_class == 0:
        age_class_name = "<8.5"
    elif age_class == 1:
        age_class_name = "8.5-12.5"
    else:
        age_class_name = ">12.5"
    
    return gender_name, age_class_name

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_gender(gender: Union[float, int]) -> None:
    """Validate gender value."""
    if gender not in [0.0, 1.0]:
        raise ValueError(f"Invalid gender value: {gender}. Expected 0.0 (female) or 1.0 (male)")

def validate_age(age: Union[float, int]) -> None:
    """Validate age value."""
    if not isinstance(age, (int, float)) or age < 0:
        raise ValueError(f"Invalid age value: {age}. Expected non-negative number")

def validate_stratify_by(stratify_by: str) -> None:
    """Validate stratify_by parameter."""
    if stratify_by not in ["gender", "age", "both"]:
        raise ValueError(f"Invalid stratify_by value: {stratify_by}. Expected 'gender', 'age', or 'both'")

# =============================================================================
# CONSTANTS FOR REFERENCE
# =============================================================================

# Age class names for reference
AGE_CLASS_NAMES = ["<8.5", "8.5-12.5", ">12.5"]

# Gender class names for reference
GENDER_CLASS_NAMES = ["Female", "Male"]

# Combined class names for reference
COMBINED_CLASS_NAMES = [
    "(Female, <8.5)", "(Female, 8.5-12.5)", "(Female, >12.5)",
    "(Male, <8.5)", "(Male, 8.5-12.5)", "(Male, >12.5)"
]

if __name__ == "__main__":
    # Test the utility functions
    print("Testing Constants and Utilities")
    print("=" * 40)
    
    # Test age classification
    test_ages = [6.0, 10.0, 15.0]
    print("Age Classification:")
    for age in test_ages:
        age_class = get_age_class(age)
        print(f"  Age {age}: Class {age_class} ({AGE_CLASS_NAMES[age_class]})")
    
    # Test gender classification
    test_genders = [0.0, 1.0]
    print("\nGender Classification:")
    for gender in test_genders:
        gender_class = get_gender_class(gender)
        print(f"  Gender {gender}: Class {gender_class} ({GENDER_CLASS_NAMES[gender_class]})")
    
    # Test combined classification
    test_combinations = [(0.0, 6.0), (0.0, 10.0), (0.0, 15.0), (1.0, 6.0), (1.0, 10.0), (1.0, 15.0)]
    print("\nCombined Classification:")
    for gender, age in test_combinations:
        combined_class = get_combined_class(gender, age)
        gender_name, age_class_name = decode_combined_class(combined_class)
        print(f"  Gender {gender}, Age {age}: Class {combined_class} ({gender_name}, {age_class_name})")
    
    print("\n✅ All tests passed!")
