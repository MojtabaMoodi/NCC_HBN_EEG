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
        self.pickle_dir = eeg_dataset.pickle_dir
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


def prepare_labram_dataset(dataset_type: str, pickle_dir: str,
                          task_type: str = "both",
                          train_ratio: float = 0.7,
                          val_ratio: float = 0.15,
                          test_ratio: float = 0.15,
                          random_seed: int = 42,
                          **kwargs) -> Tuple[LaBraMEEGDataset, LaBraMEEGDataset, LaBraMEEGDataset]:
    """
    Prepare LaBraM-compatible datasets for training, validation, and testing.
    
    This function creates datasets directly without going through DataLoaders,
    since LaBraM expects raw Dataset objects.
    
    Args:
        dataset_type: Type of dataset ("gender", "age", "combined", "multi_output")
        pickle_dir: Directory containing pickle files
        task_type: Type of tasks to include ("active", "passive", "both")
        train_ratio: Ratio of data for training (default: 0.7)
        val_ratio: Ratio of data for validation (default: 0.15)
        test_ratio: Ratio of data for testing (default: 0.15)
        random_seed: Random seed for reproducible splits (default: 42)
        **kwargs: Additional arguments passed to dataset constructor (e.g., sampling_rate)
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    
    # Get dataset configuration
    target_type, gender_transform, age_transform, combined_transform = _get_dataset_config(dataset_type)
    
    # Set random seed BEFORE collecting participants (matching CNN behavior)
    # This ensures consistent participant collection order
    import torch
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Get unique participants by iterating through a temporary dataset
    participants = set()
    temp_dataset = EEGDataset(
        pickle_dir=pickle_dir,
        task_type="both",  # Get all participants regardless of task_type
        target_type="both"
    )
    for sample in temp_dataset:
        participants.add(sample['participant_id'])
    participants = list(participants)
    np.random.shuffle(participants)  # Shuffle after collection (matching CNN behavior)
    
    # Diagnostic: Log participant count
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Found {len(participants)} unique participants for dataset preparation")
    
    # Split participants (now just does the split, random seed already set)
    train_participants, val_participants, test_participants = _split_participants(
        participants, train_ratio, val_ratio, test_ratio, random_seed
    )
    
    # Diagnostic: Log split counts
    logger.info(f"Participant split: Train={len(train_participants)}, Val={len(val_participants)}, Test={len(test_participants)}")
    logger.info(f"Split ratios: Train={len(train_participants)/len(participants):.2f}, Val={len(val_participants)/len(participants):.2f}, Test={len(test_participants)/len(participants):.2f}")
    
    # Create EEGDataset instances with participant filtering
    train_eeg_dataset = EEGDataset(
        pickle_dir=pickle_dir,
        task_type=task_type,
        target_type=target_type,
        gender_transform=gender_transform,
        age_transform=age_transform,
        combined_transform=combined_transform,
        participant_filter=train_participants
    )
    
    val_eeg_dataset = EEGDataset(
        pickle_dir=pickle_dir,
        task_type=task_type,
        target_type=target_type,
        gender_transform=gender_transform,
        age_transform=age_transform,
        combined_transform=combined_transform,
        participant_filter=val_participants
    )
    
    test_eeg_dataset = EEGDataset(
        pickle_dir=pickle_dir,
        task_type=task_type,
        target_type=target_type,
        gender_transform=gender_transform,
        age_transform=age_transform,
        combined_transform=combined_transform,
        participant_filter=test_participants
    )
    
    # Wrap with LaBraMEEGDataset
    sampling_rate = kwargs.get('sampling_rate', 200)
    train_dataset = LaBraMEEGDataset(train_eeg_dataset, sampling_rate=sampling_rate)
    val_dataset = LaBraMEEGDataset(val_eeg_dataset, sampling_rate=sampling_rate)
    test_dataset = LaBraMEEGDataset(test_eeg_dataset, sampling_rate=sampling_rate)
    
    return train_dataset, val_dataset, test_dataset

def prepare_labram_cross_validation_datasets(dataset_type: str, pickle_dir: str,
                                           n_folds: int = 5, task_type: str = "both",
                                           stratify_by: str = "gender",
                                           random_seed: int = 42,
                                           transform: Optional[callable] = None,
                                           **kwargs) -> List[Tuple[LaBraMEEGDataset, LaBraMEEGDataset, LaBraMEEGDataset]]:
    """
    Prepare LaBraM-compatible datasets for cross-validation.
    
    This function replicates the same logic as EEGDataLoader.create_cross_validation_loaders
    but returns LaBraM-compatible datasets instead of DataLoaders.
    
    Args:
        dataset_type: Type of dataset ("gender", "age", "combined", "multi_output")
        pickle_dir: Directory containing pickle files
        n_folds: Number of folds for cross-validation
        task_type: Type of tasks to include ("active", "passive", "both")
        stratify_by: Field to stratify by ("gender", "age", "both")
        random_seed: Random seed for reproducible splits (default: 42)
        transform: Optional transform to be applied on EEG data
        **kwargs: Additional arguments passed to dataset constructor (e.g., sampling_rate)
        
    Returns:
        List of (train_dataset, val_dataset, test_dataset) tuples for each fold
    """
    
    # Validate stratify_by parameter
    validate_stratify_by(stratify_by)
    
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Get dataset configuration
    target_type, gender_transform, age_transform, combined_transform = _get_dataset_config(dataset_type)
    
    # Get unique participants and their stratification labels by iterating through dataset
    participants = set()
    participant_data = {}
    temp_dataset = EEGDataset(
        pickle_dir=pickle_dir,
        task_type="both",  # Get all participants regardless of task_type
        target_type="both"
    )
    for sample in temp_dataset:
        participant_id = sample['participant_id']
        if participant_id not in participants:
            participants.add(participant_id)
            stratify_label = get_stratify_label(sample, stratify_by)
            participant_data[participant_id] = stratify_label
    
    # Prepare data for stratification
    participant_ids = list(participant_data.keys())
    stratify_labels = [participant_data[pid] for pid in participant_ids]
    np.random.shuffle(participant_ids)
    
    # Create stratified k-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    fold_datasets = []
    sampling_rate = kwargs.get('sampling_rate', 200)
    
    for fold, (train_val_indices, test_indices) in enumerate(skf.split(participant_ids, stratify_labels)):
        # Split train_val into train and val
        train_val_participants = [participant_ids[i] for i in train_val_indices]
        test_participants = [participant_ids[i] for i in test_indices]
        
        # Get stratification labels for train_val participants
        train_val_labels = [stratify_labels[i] for i in train_val_indices]
        
        # Use StratifiedShuffleSplit for train/val split to maintain stratification
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=random_seed)
        
        train_indices, val_indices = next(sss.split(train_val_participants, train_val_labels))
        
        train_participants = [train_val_participants[i] for i in train_indices]
        val_participants = [train_val_participants[i] for i in val_indices]
        
        # Create EEGDataset instances with participant filtering
        train_eeg_dataset = EEGDataset(
            pickle_dir=pickle_dir,
            task_type=task_type,
            target_type=target_type,
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            participant_filter=train_participants
        )
        
        val_eeg_dataset = EEGDataset(
            pickle_dir=pickle_dir,
            task_type=task_type,
            target_type=target_type,
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            participant_filter=val_participants
        )
        
        test_eeg_dataset = EEGDataset(
            pickle_dir=pickle_dir,
            task_type=task_type,
            target_type=target_type,
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            participant_filter=test_participants
        )
        
        # Wrap with LaBraMEEGDataset
        train_dataset = LaBraMEEGDataset(train_eeg_dataset, sampling_rate=sampling_rate)
        val_dataset = LaBraMEEGDataset(val_eeg_dataset, sampling_rate=sampling_rate)
        test_dataset = LaBraMEEGDataset(test_eeg_dataset, sampling_rate=sampling_rate)
        
        fold_datasets.append((train_dataset, val_dataset, test_dataset))
    
    return fold_datasets

def prepare_labram_cross_task_datasets(dataset_type: str, pickle_dir: str,
                                     train_task_type: str, val_test_task_type: str,
                                     task_type: str = "both",
                                     train_ratio: float = 0.7,
                                     val_ratio: float = 0.15,
                                     test_ratio: float = 0.15,
                                     random_seed: int = 42,
                                     **kwargs) -> Tuple[LaBraMEEGDataset, LaBraMEEGDataset, LaBraMEEGDataset]:
    """
    Prepare LaBraM-compatible datasets for cross-task evaluation.
    
    This function replicates the same logic as EEGDataLoader.create_cross_task_loaders
    but returns LaBraM-compatible datasets instead of DataLoaders.
    
    Args:
        dataset_type: Type of dataset ("gender", "age", "combined", "multi_output")
        pickle_dir: Directory containing pickle files
        train_task_type: Task type for training ("active" or "passive")
        val_test_task_type: Task type for validation and testing ("active" or "passive")
        task_type: Type of tasks to include ("active", "passive", "both")
        train_ratio: Ratio of data for training (default: 0.7)
        val_ratio: Ratio of data for validation (default: 0.15)
        test_ratio: Ratio of data for testing (default: 0.15)
        random_seed: Random seed for reproducible splits (default: 42)
        **kwargs: Additional arguments passed to dataset constructor (e.g., sampling_rate)
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    
    # Validate task types
    if train_task_type not in ["active", "passive"]:
        raise ValueError(f"train_task_type must be 'active' or 'passive', got {train_task_type}")
    if val_test_task_type not in ["active", "passive"]:
        raise ValueError(f"val_test_task_type must be 'active' or 'passive', got {val_test_task_type}")
    if train_task_type == val_test_task_type:
        raise ValueError(f"train_task_type and val_test_task_type must be different, both are {train_task_type}")
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Get dataset configuration
    target_type, gender_transform, age_transform, combined_transform = _get_dataset_config(dataset_type)
    
    # Get unique participants by iterating through a temporary dataset
    participants = set()
    temp_dataset = EEGDataset(
        pickle_dir=pickle_dir,
        task_type="both",  # Get all participants regardless of task_type
        target_type="both"
    )
    for sample in temp_dataset:
        participants.add(sample['participant_id'])
    participants = list(participants)
    np.random.shuffle(participants)
    
    # Split participants
    n_participants = len(participants)
    n_train = int(n_participants * train_ratio)
    n_val = int(n_participants * val_ratio)
    
    train_participants = participants[:n_train]
    val_participants = participants[n_train:n_train + n_val]
    test_participants = participants[n_train + n_val:]
    
    # Create EEGDataset instances with correct task_type and participant filtering
    train_eeg_dataset = EEGDataset(
        pickle_dir=pickle_dir,
        task_type=train_task_type,  # Only train_task_type
        target_type=target_type,
        gender_transform=gender_transform,
        age_transform=age_transform,
        combined_transform=combined_transform,
        participant_filter=train_participants
    )
    
    val_eeg_dataset = EEGDataset(
        pickle_dir=pickle_dir,
        task_type=val_test_task_type,  # Only val_test_task_type
        target_type=target_type,
        gender_transform=gender_transform,
        age_transform=age_transform,
        combined_transform=combined_transform,
        participant_filter=val_participants
    )
    
    test_eeg_dataset = EEGDataset(
        pickle_dir=pickle_dir,
        task_type=val_test_task_type,  # Only val_test_task_type
        target_type=target_type,
        gender_transform=gender_transform,
        age_transform=age_transform,
        combined_transform=combined_transform,
        participant_filter=test_participants
    )
    
    # Wrap with LaBraMEEGDataset
    sampling_rate = kwargs.get('sampling_rate', 200)
    train_dataset = LaBraMEEGDataset(train_eeg_dataset, sampling_rate=sampling_rate)
    val_dataset = LaBraMEEGDataset(val_eeg_dataset, sampling_rate=sampling_rate)
    test_dataset = LaBraMEEGDataset(test_eeg_dataset, sampling_rate=sampling_rate)
    
    return train_dataset, val_dataset, test_dataset

def prepare_labram_cross_task_cross_validation_datasets(dataset_type: str, pickle_dir: str,
                                                       train_task_type: str, val_test_task_type: str,
                                                       n_folds: int = 5, stratify_by: str = "gender",
                                                       random_seed: int = 42,
                                                       **kwargs) -> List[Tuple[LaBraMEEGDataset, LaBraMEEGDataset, LaBraMEEGDataset]]:
    """
    Prepare LaBraM-compatible datasets for cross-task cross-validation.
    
    This function combines cross-validation with cross-task evaluation:
    - Training uses one task type (e.g., "active")
    - Validation and testing use another task type (e.g., "passive")
    - Stratified k-fold cross-validation on participants
    - No participant overlap between train and val/test sets
    
    This function replicates the same logic as EEGDataLoader.create_cross_task_cross_validation_loaders
    but returns LaBraM-compatible datasets instead of DataLoaders.
    
    Args:
        dataset_type: Type of dataset ("gender", "age", "combined", "multi_output")
        pickle_dir: Directory containing pickle files
        train_task_type: Task type for training ("active" or "passive")
        val_test_task_type: Task type for validation and testing ("active" or "passive")
        n_folds: Number of folds for cross-validation
        stratify_by: Field to stratify by ("gender", "age", "both")
        random_seed: Random seed for reproducible splits (default: 42)
        **kwargs: Additional arguments passed to dataset constructor (e.g., sampling_rate)
        
    Returns:
        List of (train_dataset, val_dataset, test_dataset) tuples for each fold
    """
    
    # Validate task types
    if train_task_type not in ["active", "passive"]:
        raise ValueError(f"train_task_type must be 'active' or 'passive', got {train_task_type}")
    if val_test_task_type not in ["active", "passive"]:
        raise ValueError(f"val_test_task_type must be 'active' or 'passive', got {val_test_task_type}")
    if train_task_type == val_test_task_type:
        raise ValueError(f"train_task_type and val_test_task_type must be different, both are {train_task_type}")
    
    # Validate stratify_by parameter
    validate_stratify_by(stratify_by)
    
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Get dataset configuration
    target_type, gender_transform, age_transform, combined_transform = _get_dataset_config(dataset_type)
    
    # Get unique participants and their stratification labels by iterating through dataset
    participants = set()
    participant_data = {}
    temp_dataset = EEGDataset(
        pickle_dir=pickle_dir,
        task_type="both",  # Get all participants regardless of task_type
        target_type="both"
    )
    for sample in temp_dataset:
        participant_id = sample['participant_id']
        if participant_id not in participants:
            participants.add(participant_id)
            stratify_label = get_stratify_label(sample, stratify_by)
            participant_data[participant_id] = stratify_label
    
    # Prepare data for stratification
    participant_ids = list(participant_data.keys())
    stratify_labels = [participant_data[pid] for pid in participant_ids]
    np.random.shuffle(participant_ids)
    
    # Create stratified k-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    fold_datasets = []
    sampling_rate = kwargs.get('sampling_rate', 200)
    
    for fold, (train_val_indices, test_indices) in enumerate(skf.split(participant_ids, stratify_labels)):
        # Split train_val into train and val
        train_val_participants = [participant_ids[i] for i in train_val_indices]
        test_participants = [participant_ids[i] for i in test_indices]
        
        # Get stratification labels for train_val participants
        train_val_labels = [stratify_labels[i] for i in train_val_indices]
        
        # Use StratifiedShuffleSplit for train/val split to maintain stratification
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=random_seed)
        
        train_indices, val_indices = next(sss.split(train_val_participants, train_val_labels))
        
        train_participants = [train_val_participants[i] for i in train_indices]
        val_participants = [train_val_participants[i] for i in val_indices]
        
        # Create EEGDataset instances with correct task_type and participant filtering
        train_eeg_dataset = EEGDataset(
            pickle_dir=pickle_dir,
            task_type=train_task_type,  # Only train_task_type
            target_type=target_type,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            participant_filter=train_participants
        )
        
        val_eeg_dataset = EEGDataset(
            pickle_dir=pickle_dir,
            task_type=val_test_task_type,  # Only val_test_task_type
            target_type=target_type,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            participant_filter=val_participants
        )
        
        test_eeg_dataset = EEGDataset(
            pickle_dir=pickle_dir,
            task_type=val_test_task_type,  # Only val_test_task_type
            target_type=target_type,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            participant_filter=test_participants
        )
        
        # Wrap with LaBraMEEGDataset
        train_dataset = LaBraMEEGDataset(train_eeg_dataset, sampling_rate=sampling_rate)
        val_dataset = LaBraMEEGDataset(val_eeg_dataset, sampling_rate=sampling_rate)
        test_dataset = LaBraMEEGDataset(test_eeg_dataset, sampling_rate=sampling_rate)
        
        fold_datasets.append((train_dataset, val_dataset, test_dataset))
    
    return fold_datasets
