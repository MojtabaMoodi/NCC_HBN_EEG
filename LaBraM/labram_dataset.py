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
import warnings
from pathlib import Path
import torch
from scipy.signal import resample
from typing import Dict, List, Tuple, Any, Optional, Union, Iterator
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import IterableDataset

# Add project root and data_processing so that data_processing package and its
# internal imports (constants, target_transforms) resolve when LaBraM runs from its own dir.
_project_root = Path(__file__).resolve().parent.parent
_data_processing_dir = _project_root / "data_processing"
for _path in (_project_root, _data_processing_dir):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from data_processing.eeg_dataset import EEGDataset, EEGDataLoader
# Import MultiFileEEGDataset for type hints (optional, may not be available)
try:
    from data_processing.multi_file_dataset import MultiFileEEGDataset
except ImportError:
    MultiFileEEGDataset = None  # type: ignore

from data_processing.target_transforms import (
    age_classification_transform,
    combined_gender_age_classification_transform,
    create_age_regression_transform_from_hdf5,
    gender_classification_transform,
)
from data_processing.constants import (
    get_stratify_label,
    validate_stratify_by
)

class LaBraMEEGDataset(IterableDataset):
    """
    LaBraM-compatible EEG Dataset wrapper.
    
    This class wraps EEGDataset or MultiFileEEGDataset (IterableDataset) to provide 
    LaBraM's expected interface:
    - __iter__ returns (X, Y) tuples instead of dictionaries
    - X is the EEG data tensor
    - Y is the label (int for single classification, tuple for multi-output)
    
    Works with both single-file EEGDataset and multi-file MultiFileEEGDataset instances.
    """
    
    def __init__(self, 
                 eeg_dataset: Union[EEGDataset, Any],  # Can be EEGDataset or MultiFileEEGDataset
                 sampling_rate: int = 200):
        """
        Initialize the LaBraM EEG Dataset wrapper.
        
        Args:
            eeg_dataset: EEGDataset or MultiFileEEGDataset instance to wrap
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
        elif self.target_type == "user_identification":
            # User identification: EEGDataset already applied user_identification_transform
            if 'user_identification' in sample:
                return sample['user_identification']
            else:
                raise ValueError(
                    "user_identification label not found in sample. "
                    "Ensure target_type is 'user_identification' and user_identification_transform is provided when creating EEGDataset."
                )
        else:
            raise ValueError(f"Unknown target_type: {self.target_type}")
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, Any, Optional[str]]]:
        """
        Iterate through the dataset, yielding (X, Y, participant_id) in LaBraM format.
        participant_id is used for participant-level aggregation (e.g. majority vote).
        
        Yields:
            Tuple of (X, Y, participant_id) where X is EEG data, Y is the label,
            and participant_id is from metadata (None if absent).
        """
        for sample in self.eeg_dataset:
            # Extract EEG data
            eeg_data = sample['eeg_data']
            if not isinstance(eeg_data, torch.Tensor) or eeg_data.dim() < 2:
                raise ValueError(
                    f"Expected eeg_data to be a 2D+ tensor (channels x time). "
                    f"Got type={type(eeg_data).__name__}, dim={getattr(eeg_data, 'dim', lambda: None)()}. "
                    "Check HDF5 data: each sample dataset should have shape (n_channels, n_timepoints), e.g. (60, 800) for 4s at 200Hz."
                )
            eeg_data = self._resample_if_needed(eeg_data)
            label = self._extract_label(sample)
            participant_id = sample.get('participant_id')
            yield eeg_data, label, participant_id


def collate_labram_with_participant_ids(batch, label_mode='classification'):
    """
    Collate batch of (eeg, label, participant_id) into (eeg_stacked, label_stacked, participant_ids).
    Use with DataLoader when participant-level aggregation (e.g. majority vote) is needed.
    Expects eeg to be (n_channels, n_timepoints) per sample; labels scalar. Does not change dimensions.

    Args:
        label_mode: 'classification' (long class indices) or 'regression' (float tensor targets, stacked as (B, 1)).
    """
    if not batch:
        raise ValueError("collate_labram_with_participant_ids received an empty batch. Check DataLoader and dataset.")
    for i, b in enumerate(batch):
        x = b[0]
        if not torch.is_tensor(x) or x.dim() < 2:
            raise ValueError(
                f"Batch item {i}: expected eeg tensor with dim >= 2 (channels x time). "
                f"Got type={type(x).__name__}, shape={getattr(x, 'shape', None)}. "
                "Check HDF5 data: sample datasets should have shape (n_channels, n_timepoints), e.g. (60, 800) for 4s."
            )
    eeg = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    if isinstance(labels[0], tuple):
        # Multi-output: (gender, age)
        label = (
            torch.stack([l[0] for l in labels]),
            torch.stack([l[1] for l in labels]),
        )
    else:
        if label_mode == 'regression':
            float_labels = []
            for i, L in enumerate(labels):
                if torch.is_tensor(L):
                    v = L.flatten()[0].float() if L.numel() > 0 else torch.tensor(0.0, dtype=torch.float32)
                    float_labels.append(v)
                elif isinstance(L, (float, int)):
                    float_labels.append(torch.tensor(float(L), dtype=torch.float32))
                else:
                    raise TypeError(
                        f"Regression label at index {i} must be a tensor or numeric; got {type(L).__name__}."
                    )
            label = torch.stack(float_labels, dim=0).unsqueeze(1)
        elif label_mode != 'classification':
            raise ValueError(f"label_mode must be 'classification' or 'regression'; got {label_mode!r}.")
        else:
            # Single task: ensure each label is scalar so stacking gives (B,)
            scalar_labels = []
            for i, L in enumerate(labels):
                if torch.is_tensor(L):
                    if L.numel() > 1:
                        warnings.warn(
                            f"Label at index {i} had {L.numel()} elements (shape {L.shape}); expected scalar. "
                            "Using first element as class index. Check dataset _extract_label and sample.",
                            UserWarning,
                            stacklevel=2,
                        )
                        val = int(round(L.flatten()[0].item()))
                        scalar_labels.append(torch.tensor(val, dtype=torch.long))
                    else:
                        scalar_labels.append(L.flatten()[0].long() if L.numel() > 0 else torch.tensor(0, dtype=torch.long))
                elif isinstance(L, (list, tuple)):
                    if len(L) > 1:
                        warnings.warn(
                            f"Label at index {i} had len {len(L)}; expected scalar. Using first element. Check dataset.",
                            UserWarning,
                            stacklevel=2,
                        )
                    scalar_labels.append(torch.tensor(L[0] if len(L) > 0 else 0, dtype=torch.long))
                else:
                    scalar_labels.append(torch.tensor(L, dtype=torch.long))
            label = torch.stack(scalar_labels)
    participant_ids = [b[2] for b in batch]
    return (eeg, label, participant_ids)


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
    elif dataset_type == "age_regression":
        raise ValueError(
            "dataset_type 'age_regression' is configured in prepare_labram_dataset via "
            "create_age_regression_transform_from_hdf5(train_files); do not call _get_dataset_config for it."
        )
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
                          user_identification_transform=None,
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
    
    # Get HDF5 file paths first (age regression needs train files for leakage-safe min/max).
    hdf5_dir_path = Path(hdf5_dir)
    train_files, val_files, test_files = EEGDataLoader._get_hdf5_file_paths(
        hdf5_dir_path, segment_length
    )
    EEGDataLoader._verify_hdf5_files(train_files, val_files, test_files)

    # Get dataset configuration
    age_min: Optional[float] = None
    age_max: Optional[float] = None
    if dataset_type == "user_identification":
        target_type = "user_identification"
        gender_transform = None
        age_transform = None
        combined_transform = None
        if user_identification_transform is None:
            raise ValueError("user_identification_transform is required for user_identification dataset_type")
    elif dataset_type == "age_regression":
        age_transform, age_min, age_max = create_age_regression_transform_from_hdf5(train_files)
        target_type = "age"
        gender_transform = None
        combined_transform = None
    else:
        target_type, gender_transform, age_transform, combined_transform = _get_dataset_config(dataset_type)
    
    # DIAGNOSTIC: Log file discovery
    import logging
    logger = logging.getLogger(__name__)
    if isinstance(train_files, (list, tuple)):
        logger.info(f"📁 Found {len(train_files)} training file(s) for {segment_length} segments:")
        for i, f in enumerate(train_files):
            logger.info(f"   [{i}] {Path(f).name}")
    else:
        logger.info(f"📁 Found 1 training file for {segment_length} segments: {Path(train_files).name}")
    
    # Handle both single files and lists of files (for 1s segments with multi-file HDF5)
    # Use _create_dataset_from_hdf5 which handles both cases automatically
    train_eeg_dataset = EEGDataLoader._create_dataset_from_hdf5(
        hdf5_file_or_files=train_files,
        task_type=task_type,
        target_type=target_type,
        transform=None,
        gender_transform=gender_transform,
        age_transform=age_transform,
        combined_transform=combined_transform,
        user_identification_transform=user_identification_transform,
        shuffle=True,  # Shuffle training data
        random_seed=random_seed
    )
    
    val_eeg_dataset = EEGDataLoader._create_dataset_from_hdf5(
        hdf5_file_or_files=val_files,
        task_type=task_type,
        target_type=target_type,
        transform=None,
        gender_transform=gender_transform,
        age_transform=age_transform,
        combined_transform=combined_transform,
        user_identification_transform=user_identification_transform,
        shuffle=False  # Don't shuffle validation data
    )
    
    test_eeg_dataset = EEGDataLoader._create_dataset_from_hdf5(
        hdf5_file_or_files=test_files,
        task_type=task_type,
        target_type=target_type,
        transform=None,
        gender_transform=gender_transform,
        age_transform=age_transform,
        combined_transform=combined_transform,
        user_identification_transform=user_identification_transform,
        shuffle=False  # Don't shuffle test data
    )
    
    # Wrap with LaBraMEEGDataset
    sampling_rate = kwargs.get('sampling_rate', 200)
    train_dataset = LaBraMEEGDataset(train_eeg_dataset, sampling_rate=sampling_rate)
    val_dataset = LaBraMEEGDataset(val_eeg_dataset, sampling_rate=sampling_rate)
    test_dataset = LaBraMEEGDataset(test_eeg_dataset, sampling_rate=sampling_rate)

    if dataset_type == "age_regression":
        for ds in (train_dataset, val_dataset, test_dataset):
            ds.age_min = age_min
            ds.age_max = age_max
    
    return train_dataset, val_dataset, test_dataset


def prepare_labram_unknown_dataset(
    hdf5_dir: str,
    segment_length: str,
    task_type: str = "both",
    sampling_rate: int = 200,
    random_seed: int = 42
) -> Optional[LaBraMEEGDataset]:
    """
    Create LaBraM dataset for the "unknown" split when consider_unknown_users was used.
    Unknown participants are not in the class mapping; samples get label -1 (UNKNOWN_LABEL).
    
    Args:
        hdf5_dir: Directory containing HDF5 files (and participant_id_to_class_idx.json)
        segment_length: '1s', '2s', or '4s'
        task_type: 'active', 'passive', or 'both'
        sampling_rate: Sampling rate for LaBraM
        random_seed: Random seed for shuffling (optional)
        
    Returns:
        LaBraMEEGDataset for unknown split, or None if no unknown split exists.
    """
    from data_processing.target_transforms import (
        UserIdentificationTransform,
        create_user_identification_transform_from_hdf5
    )
    
    hdf5_dir_path = Path(hdf5_dir)
    unknown_files = EEGDataLoader._get_unknown_hdf5_file_paths(hdf5_dir_path, segment_length)
    if not unknown_files:
        return None
    
    # Transform with unknown_label so participant_id not in mapping returns -1
    transform, _ = create_user_identification_transform_from_hdf5(
        hdf5_dir,
        unknown_label=UserIdentificationTransform.UNKNOWN_LABEL
    )
    
    unknown_eeg_dataset = EEGDataLoader._create_dataset_from_hdf5(
        hdf5_file_or_files=unknown_files,
        task_type=task_type,
        target_type="user_identification",
        transform=None,
        gender_transform=None,
        age_transform=None,
        combined_transform=None,
        user_identification_transform=transform,
        shuffle=False
    )
    
    return LaBraMEEGDataset(unknown_eeg_dataset, sampling_rate=sampling_rate)


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
