"""
Shared utility functions for EEG classification experiments.
Keeps code DRY by centralizing common functionality.
"""

import os
import json
import numpy as np
from typing import Any, Dict, List, Union
from config import DataConfig
from data_processing.eeg_dataset import get_recommended_num_workers

# Data paths and segment lengths (HDF5 format). All segment lengths use the same dir (multipart: part00, part01, ...).
# Set to scratch output from: python preprocess_eeg_to_hdf5.py --data_root .../preprocessed_new --output_dir .../processed_eeg_data_hdf5 --num_channels 60 --window_sizes 1s 2s 4s
DATA_PATHS = {
    '1s': "/home/mojtabam/scratch/processed_eeg_data_hdf5",
    '2s': "/home/mojtabam/scratch/processed_eeg_data_hdf5",
    '4s': "/home/mojtabam/scratch/processed_eeg_data_hdf5"
}

SEGMENT_LENGTHS = {
    '1s': 200,  # 1 second = 200 samples at 200Hz
    '2s': 400,  # 2 second = 400 samples at 200Hz
    '4s': 800   # 4 seconds = 800 samples at 200Hz
}


def create_data_config_for_segment_length(segment_length: str, batch_size: int = None, 
                                        random_seed: int = 42, num_gpus: int = 1) -> DataConfig:
    """
    Create DataConfig for a specific segment length.
    Centralized function to avoid code duplication.
    
    Args:
        segment_length: '1s', '2s', or '4s'
        batch_size: Batch size (if None, uses default: 6144 for 1s, 256 for 2s, 192 for 4s)
        random_seed: Random seed
        num_gpus: Number of GPUs to use
        
    Returns:
        DataConfig configured for the specified segment length
    """
    if segment_length not in DATA_PATHS:
        raise ValueError(f"Invalid segment_length: {segment_length}. Must be '1s', '2s', or '4s'")
    
    # Use larger batch sizes for better GPU utilization and fewer iterations
    # With 4 GPUs and L40S (48GB each), we have plenty of GPU memory
    # But need to balance with CPU memory (spawn workers use more RAM)
    if batch_size is None:
        if segment_length == '1s':
            batch_size = 6144  # Maximum batch size to minimize number of batches and HDF5 I/O operations
            # With 955K samples, this gives ~155 batches per epoch (vs 233 with 4096)
        elif segment_length == '2s':
            batch_size = 256  # Balanced for memory and performance
        else:  # 4s
            batch_size = 192  # Reduced to prevent OOM
    
    num_workers = get_recommended_num_workers(segment_length, num_gpus)

    config = DataConfig(
        hdf5_dir=DATA_PATHS[segment_length],
        segment_length=SEGMENT_LENGTHS[segment_length],
        batch_size=batch_size,
        random_seed=random_seed,
        num_workers=num_workers
    )
    
    return config


def get_resnet_model_type(resnet_type: int, task: str = "age") -> tuple:
    """
    Get the model type string and configuration for a given ResNet architecture and task.
    
    Args:
        resnet_type: ResNet architecture type (18, 34, or 50)
        task: Task type ('age', 'gender', 'combined', 'multi_output', 'age_regression')
        
    Returns:
        Tuple of (model_type, model_config_dict) where:
        - model_type: Model type string for ModelFactory
        - model_config_dict: Additional config parameters (num_classes, etc.)
        
    Raises:
        ValueError: If resnet_type or task is invalid
    """
    valid_resnet_types = [18, 34, 50]
    if resnet_type not in valid_resnet_types:
        raise ValueError(f"Invalid resnet_type: {resnet_type}. Must be one of {valid_resnet_types}")
    
    valid_tasks = ['age', 'gender', 'combined', 'multi_output', 'age_regression']
    if task not in valid_tasks:
        raise ValueError(f"Invalid task: {task}. Must be one of {valid_tasks}")
    
    # Map ResNet type and task to model type
    if task == 'age':
        if resnet_type == 18:
            return ('age_resnet', {'num_classes': 3})  # Uses ResNet18
        elif resnet_type == 34:
            return ('age_resnet34', {'num_classes': 3})  # ResNet34 for age
        elif resnet_type == 50:
            return ('age_resnet50', {'num_classes': 3})  # ResNet50 for age
    elif task == 'gender':
        if resnet_type == 18:
            return ('gender_resnet', {'num_classes': 2})  # Uses ResNet18
        elif resnet_type == 34:
            return ('gender_resnet34', {'num_classes': 2})  # ResNet34 for gender
        elif resnet_type == 50:
            return ('gender_resnet50', {'num_classes': 2})  # ResNet50 for gender
    elif task == 'combined':
        return ('combined_resnet', {'num_classes': 6})  # Uses ResNet18
    elif task == 'multi_output':
        return ('multi_output_resnet', {'num_classes': 2})  # Uses ResNet18
    elif task == 'age_regression':
        return ('age_regression_resnet', {'num_classes': 1})  # Uses ResNet18
    
    # Fallback (should not reach here)
    return ('age_resnet', {'num_classes': 3})


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def safe_json_dump(data: Dict[str, Any], filepath: str, indent: int = 2):
    """
    Safely dump data to JSON file, creating directories if needed.
    
    Args:
        data: Dictionary to save as JSON
        filepath: Path to output JSON file
        indent: JSON indentation level (default: 2)
    """
    # Create directory if it doesn't exist
    dir_path = os.path.dirname(filepath)
    if dir_path:  # Only create directory if filepath has a directory component
        os.makedirs(dir_path, exist_ok=True)
    
    # Convert numpy types and dump to JSON
    json_data = convert_numpy_types(data)
    
    with open(filepath, 'w') as f:
        json.dump(json_data, f, indent=indent)

