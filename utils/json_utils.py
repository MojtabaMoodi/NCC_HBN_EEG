"""
JSON Serialization Utilities for EEG Classification Framework
Handles NumPy types and other non-serializable objects.
"""

import json
import numpy as np
import torch
from typing import Any, Dict, List, Union

def _convert_single_numpy_type(obj: Any) -> Any:
    """
    Convert a single NumPy/PyTorch object to Python native type.
    
    Args:
        obj: Object to convert
        
    Returns:
        Converted object or original if not convertible
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj

def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert NumPy types to Python native types.
    
    Args:
        obj: Object to convert
        
    Returns:
        Object with NumPy types converted to Python types
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return _convert_single_numpy_type(obj)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types."""
    
    def default(self, obj):
        return _convert_single_numpy_type(obj)

def safe_json_dump(data: Any, file_path: str, indent: int = 2) -> None:
    """
    Safely dump data to JSON file, handling NumPy types.
    
    Args:
        data: Data to serialize
        file_path: Path to output file
        indent: JSON indentation level
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder)

def safe_json_load(file_path: str) -> Any:
    """
    Safely load data from JSON file.
    
    Args:
        file_path: Path to input file
        
    Returns:
        Loaded data
    """
    with open(file_path, 'r') as f:
        return json.load(f)
