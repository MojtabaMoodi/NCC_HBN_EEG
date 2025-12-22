#!/usr/bin/env python3
"""
LaBraM-compatible EEG Dataset Classes

This module provides LaBraM-compatible wrapper around EEGDataset that returns
the format expected by LaBraM: (X, Y) tuples instead of dictionaries.

The wrapper reuses all the data loading and preprocessing logic from the
original EEGDataset while providing LaBraM's expected interface.
"""

import numpy as np
import sys
import os
from pathlib import Path
import torch
from scipy.signal import resample
from typing import Dict, List, Tuple, Any, Optional, Union, Iterator
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import IterableDataset

# Add the data_processing directory to the path
sys.path.append(str(Path(__file__).parent.parent / "data_processing"))

from eeg_dataset import EEGDataset, EEGDataLoader
from target_transforms import (
    age_classification_transform,
    combined_gender_age_classification_transform,
    gender_classification_transform
)
from constants import (
    get_stratify_label,
    validate_stratify_by
)

class LaBraMEEGDataset(IterableDataset):
    """
    LaBraM-compatible EEG Dataset wrapper.
    
    This class wraps EEGDataset (IterableDataset) to provide LaBraM's expected interface:
    - __iter__ returns (X, Y) tuples instead of dictionaries
    - X is the EEG data tensor
    - Y is the label (int for single classification, tuple for multi-output)
    """
    
    def __init__(self, 
                 eeg_dataset: EEGDataset,
                 sampling_rate: int = 200):
        """
        Initialize the LaBraM EEG Dataset wrapper.
        
        Args:
            eeg_dataset: EEGDataset instance to wrap
            sampling_rate: Sampling rate for resampling (default: 200 Hz)
        """
        super().__init__()
        self.eeg_dataset = eeg_dataset
        self.sampling_rate = sampling_rate
        self.default_rate = 200
        
        # Copy attributes for compatibility
        self.hdf5_file = eeg_dataset.hdf5_file_str if hasattr(eeg_dataset, 'hdf5_file_str') else None
        self.task_type = eeg_dataset.task_type
        self.transform = eeg_dataset.transform
        self.gender_transform = eeg_dataset.gender_transform
        self.age_transform = eeg_dataset.age_transform
        self.combined_transform = eeg_dataset.combined_transform
        self.target_type = eeg_dataset.target_type
        self.participant_filter = eeg_dataset.participant_filter
    
    def _resample_if_needed(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """
        Resample EEG data if sampling rate differs from default.
        
        Note: Our data is already preprocessed to 200Hz, so no resampling is needed
        unless we want to use a different sampling rate for experimentation.
        
        Args:
            eeg_data: EEG data tensor of shape (60, time_points)
            
        Returns:
            EEG data tensor (resampled if needed)
        """
        if self.sampling_rate != self.default_rate:
            # Convert to numpy for resampling
            eeg_np = eeg_data.numpy()
            # Calculate the target length: (current_length * target_rate) / current_rate
            current_length = eeg_np.shape[-1]
            target_length = int((current_length * self.sampling_rate) / self.default_rate)
            # Resample along the time dimension (axis=-1)
            resampled = resample(eeg_np, target_length, axis=-1)
            return torch.FloatTensor(resampled)
        return eeg_data
    
    def _extract_label(self, sample: Dict[str, Any]) -> Any:
        """
        Extract label from sample dictionary based on target type.
        
        Note: EEGDataset already applies transforms, so we use the transformed values directly.
        
        Args:
            sample: Sample dictionary from EEGDataset (already contains transformed values)
            
        Returns:
            Label (int for single classification, tuple for multi-output)
        """
        if self.target_type == "gender":
            # EEGDataset already applied gender_transform, so use it directly
            if 'gender' in sample:
                return sample['gender']
            else:
                raise ValueError("gender not found in sample")
        elif self.target_type == "age":
            # EEGDataset already applied age_transform, so use it directly
            if 'age' in sample:
                return sample['age']
            else:
                raise ValueError("age not found in sample")
        elif self.target_type == "combined":
            # Use combined label if available (EEGDataset sets this when target_type is "combined" and combined_transform is provided)
            if 'combined' in sample:
                return sample['combined']
            else:
                raise ValueError(
                    "combined label not found in sample. "
                    "Ensure target_type is 'combined' and combined_transform is provided when creating EEGDataset."
                )
        elif self.target_type == "both":
            # Multi-output case: EEGDataset already applied transforms, so use them directly
            if 'gender' not in sample or 'age' not in sample:
                raise ValueError("gender and age must be in sample for multi-output target type")
            gender_label = sample['gender']
            age_label = sample['age']
            return (gender_label, age_label)
        else:
            raise ValueError(f"Unknown target_type: {self.target_type}")
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, Any]]:
        """
        Iterate through the dataset, yielding (X, Y) tuples in LaBraM format.
        
        Yields:
            Tuple of (X, Y) where X is EEG data tensor and Y is the label
        """
        for sample in self.eeg_dataset:
            # Extract EEG data
            eeg_data = sample['eeg_data']
            
            # Apply resampling if needed
            eeg_data = self._resample_if_needed(eeg_data)
            
            # Extract label based on target type
            label = self._extract_label(sample)
            
            yield eeg_data, label


def _get_dataset_config(dataset_type: str) -> Tuple[str, Optional[callable], Optional[callable], Optional[callable]]:
    """
    Get dataset configuration based on dataset type.
    
    Args:
        dataset_type: Type of dataset ("gender", "age", "combined", "multi_output")
        
    Returns:
        Tuple of (target_type, gender_transform, age_transform, combined_transform)
    """
    if dataset_type == "gender":
        return "gender", gender_classification_transform, None, None
    elif dataset_type == "age":
        return "age", None, age_classification_transform, None
    elif dataset_type == "combined":
        return "combined", None, None, combined_gender_age_classification_transform
    elif dataset_type == "multi_output":
        return "both", gender_classification_transform, age_classification_transform, None
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

def _split_participants(participants: List[str],
                       train_ratio: float,
                       val_ratio: float,
                       test_ratio: float,
                       random_seed: int) -> Tuple[List[str], List[str], List[str]]:
    """
    Split participants into train, validation, and test sets.
    
    Args:
        participants: List of participant IDs
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_participants, val_participants, test_participants)
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    np.random.seed(random_seed)
    np.random.shuffle(participants)
    
    n_participants = len(participants)
    n_train = int(n_participants * train_ratio)
    n_val = int(n_participants * val_ratio)
    
    train_participants = participants[:n_train]
    val_participants = participants[n_train:n_train + n_val]
    test_participants = participants[n_train + n_val:]
    
    return train_participants, val_participants, test_participants


def prepare_labram_dataset(dataset_type: str, hdf5_dir: str, segment_length: str = "1s",
                          task_type: str = "both",
                          random_seed: int = 42,
                          **kwargs) -> Tuple[LaBraMEEGDataset, LaBraMEEGDataset, LaBraMEEGDataset]:
    """
    Prepare LaBraM-compatible datasets for training, validation, and testing using HDF5 files.
    
    This function creates datasets directly from pre-split HDF5 files (train/val/test),
    since LaBraM expects raw Dataset objects. The splits are already created during preprocessing.
    
    Args:
        dataset_type: Type of dataset ("gender", "age", "combined", "multi_output")
        hdf5_dir: Directory containing HDF5 files (e.g., "processed_eeg_data_hdf5")
        segment_length: Length of segments ('1s', '2s', or '4s')
        task_type: Type of tasks to include ("active", "passive", "both")
        random_seed: Random seed for shuffling (default: 42)
        **kwargs: Additional arguments passed to dataset constructor (e.g., sampling_rate)
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    
    # Get dataset configuration
    target_type, gender_transform, age_transform, combined_transform = _get_dataset_config(dataset_type)
    
    # Get HDF5 file paths
    hdf5_dir_path = Path(hdf5_dir)
    train_file, val_file, test_file = EEGDataLoader._get_hdf5_file_paths(
        hdf5_dir_path, segment_length
    )
    EEGDataLoader._verify_hdf5_files(train_file, val_file, test_file)
    
    # Create EEGDataset instances from pre-split HDF5 files
    # Note: HDF5 files are already split into train/val/test, so no participant filtering needed
    train_eeg_dataset = EEGDataset(
        hdf5_file=str(train_file),
        task_type=task_type,
        target_type=target_type,
        gender_transform=gender_transform,
        age_transform=age_transform,
        combined_transform=combined_transform,
        shuffle=True,  # Shuffle training data
        random_seed=random_seed
    )
    
    val_eeg_dataset = EEGDataset(
        hdf5_file=str(val_file),
        task_type=task_type,
        target_type=target_type,
        gender_transform=gender_transform,
        age_transform=age_transform,
        combined_transform=combined_transform,
        shuffle=False  # Don't shuffle validation data
    )
    
    test_eeg_dataset = EEGDataset(
        hdf5_file=str(test_file),
        task_type=task_type,
        target_type=target_type,
        gender_transform=gender_transform,
        age_transform=age_transform,
        combined_transform=combined_transform,
        shuffle=False  # Don't shuffle test data
    )
    
    # Wrap with LaBraMEEGDataset
    sampling_rate = kwargs.get('sampling_rate', 200)
    train_dataset = LaBraMEEGDataset(train_eeg_dataset, sampling_rate=sampling_rate)
    val_dataset = LaBraMEEGDataset(val_eeg_dataset, sampling_rate=sampling_rate)
    test_dataset = LaBraMEEGDataset(test_eeg_dataset, sampling_rate=sampling_rate)
    
    return train_dataset, val_dataset, test_dataset

# COMMENTED OUT: Cross-validation and cross-task experiments not supported yet
# These functions are disabled until HDF5 support is added for cross-validation/cross-task scenarios
"""
def prepare_labram_cross_validation_datasets(dataset_type: str, hdf5_dir: str, segment_length: str,
                                           n_folds: int = 5, task_type: str = "both",
                                           stratify_by: str = "gender",
                                           random_seed: int = 42,
                                           transform: Optional[callable] = None,
                                           **kwargs) -> List[Tuple[LaBraMEEGDataset, LaBraMEEGDataset, LaBraMEEGDataset]]:
    # Function body commented out - not supported with HDF5 yet
    raise NotImplementedError("Cross-validation not yet supported with HDF5")
"""

# COMMENTED OUT: Cross-task experiments not supported yet
"""
def prepare_labram_cross_task_datasets(dataset_type: str, pickle_dir: str,
                                     train_task_type: str, val_test_task_type: str,
                                     task_type: str = "both",
                                     train_ratio: float = 0.7,
                                     val_ratio: float = 0.15,
                                     test_ratio: float = 0.15,
                                     random_seed: int = 42,
                                     **kwargs) -> Tuple[LaBraMEEGDataset, LaBraMEEGDataset, LaBraMEEGDataset]:
    # Function body commented out - not supported with HDF5 yet
    pass
"""

# COMMENTED OUT: Cross-task cross-validation experiments not supported yet
"""
def prepare_labram_cross_task_cross_validation_datasets(dataset_type: str, hdf5_dir: str, segment_length: str,
                                                       train_task_type: str, val_test_task_type: str,
                                                       n_folds: int = 5, stratify_by: str = "gender",
                                                       random_seed: int = 42,
                                                       **kwargs) -> List[Tuple[LaBraMEEGDataset, LaBraMEEGDataset, LaBraMEEGDataset]]:
    # Function body commented out - not supported with HDF5 yet
    raise NotImplementedError("Cross-task cross-validation not yet supported with HDF5")
"""
