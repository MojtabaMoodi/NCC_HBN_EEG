#!/usr/bin/env python3
"""
Configuration file for target transforms

This file provides easy configuration for different transform types
and model combinations.
"""

from target_transforms import (
    gender_classification_transform,
    age_regression_transform,
    age_classification_transform,
    age_standardized_regression_transform,
    age_binned_regression_transform,
    combined_gender_age_classification_transform,
    # Factory functions for dynamic transforms
    create_age_regression_transform,
    create_age_standardized_regression_transform,
    create_age_binned_regression_transform,
    create_age_regression_transform_from_dataset,
    create_age_standardized_regression_transform_from_dataset,
    create_age_binned_regression_transform_from_dataset
)

# Transform configurations for different model types
TRANSFORM_CONFIGS = {
    # Gender classification models
    'gender_classification': {
        'transform': gender_classification_transform,
        'output_shape': (1,),
        'output_range': '[0, 1]',
        'loss_function': 'CrossEntropyLoss',
        'description': 'Gender classification (0=female, 1=male)'
    },
    
    # Age regression models
    'age_regression': {
        'transform': age_regression_transform,
        'output_shape': (1,),
        'output_range': '[0, 1]',
        'loss_function': 'MSELoss',
        'description': 'Age regression (normalized to 0-1)'
    },
    
    'age_standardized_regression': {
        'transform': age_standardized_regression_transform,
        'output_shape': (1,),
        'output_range': '[-∞, ∞]',
        'loss_function': 'MSELoss',
        'description': 'Age regression (standardized)'
    },
    
    'age_binned_regression': {
        'transform': age_binned_regression_transform,
        'output_shape': (1,),
        'output_range': '[0, 1]',
        'loss_function': 'MSELoss',
        'description': 'Age regression (binned)'
    },
    
    # Age classification models
    'age_classification': {
        'transform': age_classification_transform,
        'output_shape': (1,),
        'output_range': '[0, 1, 2]',
        'loss_function': 'CrossEntropyLoss',
        'description': 'Age classification (0=<8.5, 1=8.5-12.5, 2=>12.5)'
    },
    
    # Combined gender+age classification models
    'combined_gender_age_classification': {
        'transform': combined_gender_age_classification_transform,
        'output_shape': (1,),
        'output_range': '[0, 1, 2, 3, 4, 5]',
        'loss_function': 'CrossEntropyLoss',
        'description': 'Combined gender+age classification (0-5: gender+age combinations)'
    },
    
    # Dynamic age regression transforms (factory functions)
    'dynamic_age_regression': {
        'transform': create_age_regression_transform_from_dataset,
        'output_shape': (1,),
        'output_range': '[0, 1]',
        'loss_function': 'MSELoss',
        'description': 'Dynamic age regression (dataset-specific normalization)',
        'is_factory': True
    },
    
    'dynamic_age_standardized_regression': {
        'transform': create_age_standardized_regression_transform_from_dataset,
        'output_shape': (1,),
        'output_range': '[-∞, ∞]',
        'loss_function': 'MSELoss',
        'description': 'Dynamic age standardized regression (dataset-specific statistics)',
        'is_factory': True
    },
    
    'dynamic_age_binned_regression': {
        'transform': create_age_binned_regression_transform_from_dataset,
        'output_shape': (1,),
        'output_range': '[0, 1]',
        'loss_function': 'MSELoss',
        'description': 'Dynamic age binned regression (dataset-specific range)',
        'is_factory': True
    },
    
    # No transform (raw data)
    'no_transform': {
        'transform': None,
        'output_shape': (1,),
        'output_range': 'raw',
        'loss_function': 'MSELoss',
        'description': 'No transform (raw data)'
    }
}

# Model-specific configurations
MODEL_CONFIGS = {
    'gender_classifier': {
        'transform_type': 'gender_classification',
        'model_type': 'classification',
        'num_classes': 2,
        'target_name': 'gender'
    },
    
    'age_regressor': {
        'transform_type': 'age_regression',
        'model_type': 'regression',
        'num_outputs': 1,
        'target_name': 'age'
    },
    
    'age_classifier': {
        'transform_type': 'age_classification',
        'model_type': 'classification',
        'num_classes': 3,
        'target_name': 'age'
    },
    
    'standardized_age_regressor': {
        'transform_type': 'age_standardized_regression',
        'model_type': 'regression',
        'num_outputs': 1,
        'target_name': 'age'
    },
    
    'binned_age_regressor': {
        'transform_type': 'age_binned_regression',
        'model_type': 'regression',
        'num_outputs': 1,
        'target_name': 'age'
    },
    
    'combined_classifier': {
        'transform_type': 'combined_gender_age_classification',
        'model_type': 'classification',
        'num_classes': 6,
        'target_name': 'combined'
    },
    
    'dynamic_age_regressor': {
        'transform_type': 'dynamic_age_regression',
        'model_type': 'regression',
        'num_outputs': 1,
        'target_name': 'age',
        'is_dynamic': True
    },
    
    'dynamic_standardized_age_regressor': {
        'transform_type': 'dynamic_age_standardized_regression',
        'model_type': 'regression',
        'num_outputs': 1,
        'target_name': 'age',
        'is_dynamic': True
    },
    
    'dynamic_binned_age_regressor': {
        'transform_type': 'dynamic_age_binned_regression',
        'model_type': 'regression',
        'num_outputs': 1,
        'target_name': 'age',
        'is_dynamic': True
    }
}

def get_transform_config(transform_type: str) -> dict:
    """
    Get configuration for a specific transform type.
    
    Args:
        transform_type: Type of transform ('gender_classification', 'age_regression', etc.)
        
    Returns:
        Dictionary with transform configuration
    """
    return TRANSFORM_CONFIGS.get(transform_type, {})

def get_model_config(model_name: str) -> dict:
    """
    Get configuration for a specific model type.
    
    Args:
        model_name: Name of the model ('gender_classifier', 'age_regressor', etc.)
        
    Returns:
        Dictionary with model configuration
    """
    return MODEL_CONFIGS.get(model_name, {})

def get_transform_function(transform_type: str):
    """
    Get the transform function for a specific type.
    
    Args:
        transform_type: Type of transform
        
    Returns:
        Transform function or None
    """
    config = get_transform_config(transform_type)
    return config.get('transform', None)

def list_available_transforms():
    """List all available transform types."""
    print("Available Transform Types:")
    print("=" * 40)
    for transform_type, config in TRANSFORM_CONFIGS.items():
        print(f"{transform_type}: {config['description']}")
        print(f"  Output shape: {config['output_shape']}")
        print(f"  Output range: {config['output_range']}")
        print(f"  Loss function: {config['loss_function']}")
        print()

def list_available_models():
    """List all available model types."""
    print("Available Model Types:")
    print("=" * 40)
    for model_name, config in MODEL_CONFIGS.items():
        transform_config = get_transform_config(config['transform_type'])
        print(f"{model_name}: {transform_config['description']}")
        print(f"  Model type: {config['model_type']}")
        print(f"  Target: {config['target_name']}")
        print(f"  Output shape: {transform_config['output_shape']}")
        print()

def create_dataset_with_transform(pickle_dir: str, transform_type: str, task_type: str = "both", 
                                target_type: str = "both"):
    """
    Create a dataset with a specific transform type.
    
    Args:
        pickle_dir: Directory containing pickle files
        transform_type: Type of transform to use
        task_type: Type of tasks to include
        target_type: Type of target to return ("gender", "age", "both", "combined")
        
    Returns:
        EEGDataset instance
    """
    from eeg_dataset import EEGDataset
    
    config = get_transform_config(transform_type)
    transform_function = config.get('transform')
    
    if not transform_function:
        # No transform
        return EEGDataset(
            pickle_dir=pickle_dir,
            task_type=task_type,
            target_type=target_type
        )
    
    # Check if it's a factory function
    if config.get('is_factory', False):
        # For factory functions, we need to create the transform from dataset
        temp_dataset = EEGDataset(pickle_dir=pickle_dir, task_type=task_type)
        transform_function = transform_function(temp_dataset)
    
    # Determine which transform to use based on target_type
    if target_type == "gender":
        return EEGDataset(
            pickle_dir=pickle_dir,
            task_type=task_type,
            gender_transform=transform_function,
            target_type=target_type
        )
    elif target_type == "age":
        return EEGDataset(
            pickle_dir=pickle_dir,
            task_type=task_type,
            age_transform=transform_function,
            target_type=target_type
        )
    elif target_type == "combined":
        # For combined classification, we need a special approach
        if transform_type == "combined_gender_age_classification":
            return EEGDataset(
                pickle_dir=pickle_dir,
                task_type=task_type,
                gender_transform=lambda g, a: transform_function(g, a),
                target_type=target_type
            )
        else:
            raise ValueError(f"Transform type {transform_type} not compatible with target_type 'combined'")
    else:  # "both"
        # For both, we need to determine which transform to use
        if transform_type.startswith('gender'):
            return EEGDataset(
                pickle_dir=pickle_dir,
                task_type=task_type,
                gender_transform=transform_function,
                target_type=target_type
            )
        elif transform_type.startswith('age') or transform_type.startswith('dynamic_age'):
            return EEGDataset(
                pickle_dir=pickle_dir,
                task_type=task_type,
                age_transform=transform_function,
                target_type=target_type
            )
        else:
            raise ValueError(f"Transform type {transform_type} not compatible with target_type 'both'")

def create_dynamic_dataset(pickle_dir: str, transform_type: str, task_type: str = "both", 
                          target_type: str = "age"):
    """
    Create a dataset with a dynamic transform (requires dataset to be loaded first).
    
    Args:
        pickle_dir: Directory containing pickle files
        transform_type: Type of dynamic transform to use
        task_type: Type of tasks to include
        target_type: Type of target to return
        
    Returns:
        EEGDataset instance with dynamic transform
    """
    from eeg_dataset import EEGDataset
    
    config = get_transform_config(transform_type)
    if not config.get('is_factory', False):
        raise ValueError(f"Transform type {transform_type} is not a factory function")
    
    # Create transform from dataset
    temp_dataset = EEGDataset(pickle_dir=pickle_dir, task_type=task_type)
    transform_function = config['transform'](temp_dataset)
    
    # Create final dataset with dynamic transform
    if target_type == "age":
        return EEGDataset(
            pickle_dir=pickle_dir,
            task_type=task_type,
            age_transform=transform_function,
            target_type=target_type
        )
    else:
        raise ValueError(f"Dynamic transforms only support target_type 'age', got {target_type}")

# Example usage
if __name__ == "__main__":
    print("Transform Configuration Examples")
    print("=" * 50)
    
    # List available options
    list_available_transforms()
    list_available_models()
    
    # Example: Create a gender classification dataset
    print("Example: Gender Classification Dataset")
    print("-" * 40)
    gender_config = get_transform_config('gender_classification')
    print(f"Transform: {gender_config['description']}")
    print(f"Output shape: {gender_config['output_shape']}")
    print(f"Loss function: {gender_config['loss_function']}")
    
    # Example: Create an age classification dataset
    print("\nExample: Age Classification Dataset")
    print("-" * 40)
    age_config = get_transform_config('age_classification')
    print(f"Transform: {age_config['description']}")
    print(f"Output shape: {age_config['output_shape']}")
    print(f"Loss function: {age_config['loss_function']}")
    
    # Example: Create a dataset
    print("\nExample: Creating Dataset")
    print("-" * 40)
    print("dataset = create_dataset_with_transform(")
    print("    pickle_dir='/path/to/pickles',")
    print("    transform_type='age_classification',")
    print("    task_type='both',")
    print("    target_type='age'")
    print(")")
    
    # Example: Create a dynamic dataset
    print("\nExample: Creating Dynamic Dataset")
    print("-" * 40)
    print("dynamic_dataset = create_dynamic_dataset(")
    print("    pickle_dir='/path/to/pickles',")
    print("    transform_type='dynamic_age_regression',")
    print("    task_type='both',")
    print("    target_type='age'")
    print(")")
    
    # Example: List available options
    print("\nExample: Available Options")
    print("-" * 40)
    print("Available transform types:")
    for transform_type in TRANSFORM_CONFIGS.keys():
        print(f"  - {transform_type}")
    
    print("\nAvailable model types:")
    for model_name in MODEL_CONFIGS.keys():
        print(f"  - {model_name}")
