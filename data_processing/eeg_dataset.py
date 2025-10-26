#!/usr/bin/env python3
"""
PyTorch Dataset and DataLoader Classes for EEG Data

This module provides PyTorch Dataset and DataLoader classes for loading
and processing the preprocessed EEG data from pickle files.
"""

import os
import pickle
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from pathlib import Path
from constants import (
    get_stratify_label, validate_stratify_by
)
from target_transforms import (
    gender_classification_transform,
    age_classification_transform,
    age_regression_transform,
    combined_gender_age_classification_transform
)

logger = logging.getLogger(__name__)

class EEGDataset(Dataset):
    """
    PyTorch Dataset class for EEG data.
    
    This class loads preprocessed EEG data from pickle files and provides
    access to individual samples for training, validation, and testing.
    """
    
    def __init__(self, 
                 pickle_dir: str,
                 task_type: str = "both",  # "active", "passive", or "both"
                 transform: Optional[callable] = None,
                 gender_transform: Optional[callable] = None,
                 age_transform: Optional[callable] = None,
                 combined_transform: Optional[callable] = None,
                 target_type: str = "both"):  # "gender", "age", "both", "combined"
        """
        Initialize the EEG Dataset.
        
        Args:
            pickle_dir: Directory containing pickle files
            task_type: Type of tasks to include ("active", "passive", or "both")
            transform: Optional transform to be applied on EEG data
            gender_transform: Optional transform to be applied on gender targets
            age_transform: Optional transform to be applied on age targets
            combined_transform: Optional transform to be applied on combined targets (gender + age)
            target_type: Type of target to return ("gender", "age", "both", "combined")
        """
        self.pickle_dir = Path(pickle_dir)
        self.task_type = task_type
        self.transform = transform
        self.gender_transform = gender_transform
        self.age_transform = age_transform
        self.combined_transform = combined_transform
        self.target_type = target_type
        
        # Load all pickle files and create samples
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples from {task_type} tasks")
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """
        Load all samples from pickle files.
        
        Returns:
            List of sample dictionaries
        """
        samples = []
        
        # Find all pickle files
        pickle_files = list(self.pickle_dir.glob("eeg_batch_*.pkl"))
        pickle_files.sort()
        
        logger.info(f"Found {len(pickle_files)} pickle files")
        
        for pickle_file in pickle_files:
            logger.info(f"Loading {pickle_file.name}")
            
            try:
                with open(pickle_file, 'rb') as f:
                    batch_data = pickle.load(f)
                
                # Process each participant in the batch
                for participant in batch_data:
                    participant_samples = self._create_participant_samples(participant)
                    samples.extend(participant_samples)
                    
            except Exception as e:
                logger.error(f"Error loading {pickle_file}: {e}")
                continue
        
        return samples
    
    def _create_participant_samples(self, participant: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create samples for a single participant.
        
        Args:
            participant: Participant data dictionary
            
        Returns:
            List of sample dictionaries for this participant
        """
        samples = []
        
        # Get participant metadata
        participant_id = participant['participant_id']
        gender = participant['gender']
        age = participant['age']
        
        # Process active tasks
        if self.task_type in ["active", "both"]:
            active_eeg = participant.get('active_eeg', [])
            
            # Process list of arrays
            for i, eeg_data in enumerate(active_eeg):
                sample = {
                    'eeg_data': eeg_data,  # Shape: (60, 200) for 1s or (60, 800) for 4s
                    'task_type': 'active',
                    'participant_id': participant_id,
                    'gender': gender,
                    'age': age,
                    'sample_id': f"{participant_id}_active_{i}"
                }
                samples.append(sample)
        
        # Process passive tasks
        if self.task_type in ["passive", "both"]:
            passive_eeg = participant.get('passive_eeg', [])
            
            # Process list of arrays
            for i, eeg_data in enumerate(passive_eeg):
                sample = {
                    'eeg_data': eeg_data,  # Shape: (60, 200) for 1s or (60, 800) for 4s
                    'task_type': 'passive',
                    'participant_id': participant_id,
                    'gender': gender,
                    'age': age,
                    'sample_id': f"{participant_id}_passive_{i}"
                }
                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing the sample data
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
        
        sample = self.samples[idx].copy()
        
        # Convert EEG data to tensor
        eeg_data = torch.FloatTensor(sample['eeg_data'])
        
        # Apply transforms if provided
        if self.transform:
            eeg_data = self.transform(eeg_data)
        
        sample['eeg_data'] = eeg_data
        
        # Apply individual transforms
        if self.gender_transform and 'gender' in sample:
            sample['gender'] = self.gender_transform(sample['gender'])
        
        if self.age_transform and 'age' in sample:
            sample['age'] = self.age_transform(sample['age'])
        
        return sample

    @classmethod
    def _create_from_samples(cls, samples: List[Dict[str, Any]], pickle_dir: str, 
                           task_type: str, transform: Optional[callable], 
                           gender_transform: Optional[callable], age_transform: Optional[callable],
                           combined_transform: Optional[callable], target_type: str) -> 'EEGDataset':
        """
        Create a dataset from pre-loaded samples (for cross-validation).
        
        Args:
            samples: Pre-loaded sample data
            pickle_dir: Directory containing pickle files
            task_type: Type of tasks to include
            transform: Optional transform to be applied on EEG data
            gender_transform: Optional transform to be applied on gender targets
            age_transform: Optional transform to be applied on age targets
            combined_transform: Optional transform to be applied on combined targets (gender + age)
            target_type: Type of target to return

        Returns:
            EEGDataset instance
        """
        # Create instance without calling __init__
        instance = cls.__new__(cls)
        
        # Set attributes manually
        instance.pickle_dir = Path(pickle_dir)
        instance.task_type = task_type
        instance.transform = transform
        instance.gender_transform = gender_transform
        instance.age_transform = age_transform
        instance.combined_transform = combined_transform
        instance.target_type = target_type
        instance.samples = samples
        
        return instance
    
    @classmethod
    def create_gender_classification_dataset(cls, pickle_dir: str, **kwargs):
        """Create dataset for gender classification."""
        return cls(
            pickle_dir=pickle_dir,
            gender_transform=gender_classification_transform,
            target_type="gender",
            **kwargs
        )
    
    @classmethod
    def create_age_classification_dataset(cls, pickle_dir: str, **kwargs):
        """Create dataset for age classification."""
        return cls(
            pickle_dir=pickle_dir,
            age_transform=age_classification_transform,
            target_type="age",
            **kwargs
        )
    
    @classmethod
    def create_age_regression_dataset(cls, pickle_dir: str, **kwargs):
        """Create dataset for age regression."""
        return cls(
            pickle_dir=pickle_dir,
            age_transform=age_regression_transform,
            target_type="age",
            **kwargs
        )
    
    @classmethod
    def create_combined_classification_dataset(cls, pickle_dir: str, **kwargs):
        """Create dataset for combined gender+age classification."""
        return cls(
            pickle_dir=pickle_dir,
            combined_transform=combined_gender_age_classification_transform,
            target_type="combined",
            **kwargs
        )
    
    @classmethod
    def create_multi_task_dataset(cls, pickle_dir: str, **kwargs):
        """Create dataset for multi-task learning (both gender and age)."""
        return cls(
            pickle_dir=pickle_dir,
            gender_transform=gender_classification_transform,
            age_transform=age_regression_transform,
            target_type="both",
            **kwargs
        )


class EEGDataLoader:
    """
    Custom DataLoader class for EEG data with additional utilities.
    """
    
    @staticmethod
    def create_dataloader(dataset: EEGDataset,
                         batch_size: int = 32,
                         shuffle: bool = True,
                         num_workers: int = 4,
                         pin_memory: bool = True,
                         drop_last: bool = False) -> DataLoader:
        """
        Create a PyTorch DataLoader for the EEG dataset.
        
        Args:
            dataset: EEGDataset instance
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            drop_last: Whether to drop the last incomplete batch
            
        Returns:
            PyTorch DataLoader instance
        """
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=EEGDataLoader.collate_fn
        )
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for batching EEG data.
        
        Args:
            batch: List of sample dictionaries
            
        Returns:
            Dictionary containing batched tensors
        """
        # Stack EEG data
        eeg_data = torch.stack([sample['eeg_data'] for sample in batch])
        
        # Stack other tensors - handle both scalar and tensor values
        gender_list = []
        age_list = []
        
        for sample in batch:
            if isinstance(sample['gender'], torch.Tensor):
                gender_list.append(sample['gender'])
            else:
                # Gender is typically used for classification, so use long dtype
                gender_list.append(torch.tensor(sample['gender'], dtype=torch.long))
            
            if isinstance(sample['age'], torch.Tensor):
                age_list.append(sample['age'])
            else:
                # Age is typically used for regression, so use float32 dtype
                age_list.append(torch.tensor(sample['age'], dtype=torch.float32))
        
        # Stack tensors
        gender = torch.stack(gender_list)
        age = torch.stack(age_list)
        
        return {
            'eeg_data': eeg_data,  # Shape: (batch_size, 60, 200), dtype: float32
            'gender': gender,       # Shape: (batch_size,), dtype: long (for classification)
            'age': age,            # Shape: (batch_size,), dtype: float32 (for regression)
            'participant_ids': [sample['participant_id'] for sample in batch],
            'task_types': [sample['task_type'] for sample in batch],
            'sample_ids': [sample['sample_id'] for sample in batch]
        }
    
    @staticmethod
    def create_train_val_test_loaders(pickle_dir: str,
                                    train_ratio: float = 0.7,
                                    val_ratio: float = 0.15,
                                    test_ratio: float = 0.15,
                                    batch_size: int = 32,
                                    num_workers: int = 4,
                                    random_seed: int = 42,
                                    task_type: str = "both",
                                    target_type: str = "both",
                                    transform: Optional[callable] = None,
                                    gender_transform: Optional[callable] = None,
                                    age_transform: Optional[callable] = None,
                                    combined_transform: Optional[callable] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test data loaders.
        
        Args:
            pickle_dir: Directory containing pickle files
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            batch_size: Batch size for all loaders
            num_workers: Number of workers for all loaders
            random_seed: Random seed for reproducible splits
            task_type: Type of tasks to include ("active", "passive", "both")
            target_type: Type of target to return ("gender", "age", "both", "combined")
            transform: Optional transform to be applied on EEG data
            gender_transform: Optional transform to be applied on gender targets
            age_transform: Optional transform to be applied on age targets
            combined_transform: Optional transform to be applied on combined targets (gender + age)
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Set random seed for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Create full dataset with specified parameters
        full_dataset = EEGDataset(
            pickle_dir=pickle_dir, 
            task_type=task_type,
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            target_type=target_type
        )
        
        # Get unique participants for splitting
        participants = list(set(sample['participant_id'] for sample in full_dataset.samples))
        np.random.shuffle(participants)
        
        # Split participants
        n_participants = len(participants)
        n_train = int(n_participants * train_ratio)
        n_val = int(n_participants * val_ratio)
        
        train_participants = participants[:n_train]
        val_participants = participants[n_train:n_train + n_val]
        test_participants = participants[n_train + n_val:]
        
        # Create datasets for each split
        train_samples = [s for s in full_dataset.samples if s['participant_id'] in train_participants]
        val_samples = [s for s in full_dataset.samples if s['participant_id'] in val_participants]
        test_samples = [s for s in full_dataset.samples if s['participant_id'] in test_participants]
        
        # Create datasets using proper initialization
        train_dataset = EEGDataset._create_from_samples(
            samples=train_samples,
            pickle_dir=pickle_dir,
            task_type=task_type,
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            target_type=target_type
        )
        
        val_dataset = EEGDataset._create_from_samples(
            samples=val_samples,
            pickle_dir=pickle_dir,
            task_type=task_type,
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            target_type=target_type
        )
        
        test_dataset = EEGDataset._create_from_samples(
            samples=test_samples,
            pickle_dir=pickle_dir,
            task_type=task_type,
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            target_type=target_type
        )
        
        # Create data loaders
        train_loader = EEGDataLoader.create_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = EEGDataLoader.create_dataloader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = EEGDataLoader.create_dataloader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        logger.info(f"Created data loaders:")
        logger.info(f"  Train: {len(train_dataset)} samples from {len(train_participants)} participants")
        logger.info(f"  Val: {len(val_dataset)} samples from {len(val_participants)} participants")
        logger.info(f"  Test: {len(test_dataset)} samples from {len(test_participants)} participants")
        
        return train_loader, val_loader, test_loader
    
    @staticmethod
    def create_cross_validation_loaders(pickle_dir: str,
                                      n_folds: int = 5,
                                      batch_size: int = 32,
                                      num_workers: int = 4,
                                      random_seed: int = 42,
                                      stratify_by: str = "gender",
                                      task_type: str = "both",
                                      target_type: str = "both",
                                      transform: Optional[callable] = None,
                                      gender_transform: Optional[callable] = None,
                                      age_transform: Optional[callable] = None,
                                      combined_transform: Optional[callable] = None) -> List[Tuple[DataLoader, DataLoader, DataLoader]]:
        """
        Create n-fold cross-validation data loaders with stratification.
        
        Args:
            pickle_dir: Directory containing pickle files
            n_folds: Number of folds for cross-validation
            batch_size: Batch size for all loaders
            num_workers: Number of workers for all loaders
            random_seed: Random seed for reproducible splits
            stratify_by: Field to stratify by ("gender", "age", "both")
            task_type: Type of tasks to include ("active", "passive", "both")
            target_type: Type of target to return ("gender", "age", "both", "combined")
            transform: Optional transform to be applied on EEG data
            gender_transform: Optional transform to be applied on gender targets
            age_transform: Optional transform to be applied on age targets
            combined_transform: Optional transform to be applied on combined targets (gender + age)
            
        Returns:
            List of (train_loader, val_loader, test_loader) tuples for each fold
        """        
        # Validate stratify_by parameter
        validate_stratify_by(stratify_by)
        
        # Set random seed for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Create full dataset with specified parameters
        full_dataset = EEGDataset(
            pickle_dir=pickle_dir, 
            task_type=task_type,
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            target_type=target_type
        )
        
        # Get unique participants and their stratification labels
        participants = list(set(sample['participant_id'] for sample in full_dataset.samples))
        participant_data = {}
        
        for participant_id in participants:
            participant_samples = [s for s in full_dataset.samples if s['participant_id'] == participant_id]
            if participant_samples:
                sample = participant_samples[0]  # Get first sample for metadata
                stratify_label = get_stratify_label(sample, stratify_by)
                participant_data[participant_id] = stratify_label
        
        # Prepare data for stratification
        participant_ids = list(participant_data.keys())
        stratify_labels = [participant_data[pid] for pid in participant_ids]
        
        # Create stratified k-fold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        
        fold_loaders = []
        
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
            
            # Create datasets for this fold using proper initialization
            train_samples = [s for s in full_dataset.samples if s['participant_id'] in train_participants]
            val_samples = [s for s in full_dataset.samples if s['participant_id'] in val_participants]
            test_samples = [s for s in full_dataset.samples if s['participant_id'] in test_participants]
            
            # Create dataset objects with proper initialization
            train_dataset = EEGDataset._create_from_samples(
                samples=train_samples,
                pickle_dir=pickle_dir,
                task_type=full_dataset.task_type,
                transform=full_dataset.transform,
                gender_transform=full_dataset.gender_transform,
                age_transform=full_dataset.age_transform,
                combined_transform=full_dataset.combined_transform,
                target_type=full_dataset.target_type
            )
            
            val_dataset = EEGDataset._create_from_samples(
                samples=val_samples,
                pickle_dir=pickle_dir,
                task_type=full_dataset.task_type,
                transform=full_dataset.transform,
                gender_transform=full_dataset.gender_transform,
                age_transform=full_dataset.age_transform,
                combined_transform=full_dataset.combined_transform,
                target_type=full_dataset.target_type
            )
            
            test_dataset = EEGDataset._create_from_samples(
                samples=test_samples,
                pickle_dir=pickle_dir,
                task_type=full_dataset.task_type,
                transform=full_dataset.transform,
                gender_transform=full_dataset.gender_transform,
                age_transform=full_dataset.age_transform,
                combined_transform=full_dataset.combined_transform,
                target_type=full_dataset.target_type
            )
            
            # Create data loaders
            train_loader = EEGDataLoader.create_dataloader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
            )
            val_loader = EEGDataLoader.create_dataloader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
            test_loader = EEGDataLoader.create_dataloader(
                test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
            
            fold_loaders.append((train_loader, val_loader, test_loader))
            
            logger.info(f"Fold {fold + 1}/{n_folds}:")
            logger.info(f"  Train: {len(train_dataset)} samples from {len(train_participants)} participants")
            logger.info(f"  Val: {len(val_dataset)} samples from {len(val_participants)} participants")
            logger.info(f"  Test: {len(test_dataset)} samples from {len(test_participants)} participants")
        
        return fold_loaders
    
    @staticmethod
    def create_cross_task_loaders(pickle_dir: str,
                                train_task_type: str,
                                val_test_task_type: str,
                                train_ratio: float = 0.7,
                                val_ratio: float = 0.15,
                                test_ratio: float = 0.15,
                                batch_size: int = 32,
                                num_workers: int = 4,
                                random_seed: int = 42,
                                target_type: str = "both",
                                transform: Optional[callable] = None,
                                gender_transform: Optional[callable] = None,
                                age_transform: Optional[callable] = None,
                                combined_transform: Optional[callable] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create cross-task data loaders for training on one task type and testing on another.
        
        This ensures:
        1. Training uses one task type (e.g., active tasks)
        2. Validation and testing use the other task type (e.g., passive tasks)
        3. No participant overlap between train and val/test sets
        4. Proper generalization evaluation across task types
        
        Args:
            pickle_dir: Directory containing pickle files
            train_task_type: Task type for training ("active" or "passive")
            val_test_task_type: Task type for validation and testing ("active" or "passive")
            train_ratio: Ratio of participants for training
            val_ratio: Ratio of participants for validation
            test_ratio: Ratio of participants for testing
            batch_size: Batch size for all loaders
            num_workers: Number of workers for all loaders
            random_seed: Random seed for reproducible splits
            target_type: Type of target to return ("gender", "age", "both", "combined")
            transform: Optional transform to be applied on EEG data
            gender_transform: Optional transform to be applied on gender targets
            age_transform: Optional transform to be applied on age targets
            combined_transform: Optional transform to be applied on combined targets (gender + age)
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
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
        
        # Create full dataset for both task types to get all participants
        full_dataset = EEGDataset(
            pickle_dir=pickle_dir, 
            task_type="both",  # Load both to get all participants
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            target_type=target_type
        )
        
        # Get unique participants
        participants = list(set(sample['participant_id'] for sample in full_dataset.samples))
        np.random.shuffle(participants)
        
        # Split participants
        n_participants = len(participants)
        n_train = int(n_participants * train_ratio)
        n_val = int(n_participants * val_ratio)
        
        train_participants = participants[:n_train]
        val_participants = participants[n_train:n_train + n_val]
        test_participants = participants[n_train + n_val:]
        
        # Create training dataset (only train_task_type)
        train_samples = [s for s in full_dataset.samples 
                        if s['participant_id'] in train_participants 
                        and s['task_type'] == train_task_type]
        
        # Create validation dataset (only val_test_task_type)
        val_samples = [s for s in full_dataset.samples 
                      if s['participant_id'] in val_participants 
                      and s['task_type'] == val_test_task_type]
        
        # Create test dataset (only val_test_task_type)
        test_samples = [s for s in full_dataset.samples 
                       if s['participant_id'] in test_participants 
                       and s['task_type'] == val_test_task_type]
        
        # Create datasets
        train_dataset = EEGDataset._create_from_samples(
            samples=train_samples,
            pickle_dir=pickle_dir,
            task_type=train_task_type,
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            target_type=target_type
        )
        
        val_dataset = EEGDataset._create_from_samples(
            samples=val_samples,
            pickle_dir=pickle_dir,
            task_type=val_test_task_type,
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            target_type=target_type
        )
        
        test_dataset = EEGDataset._create_from_samples(
            samples=test_samples,
            pickle_dir=pickle_dir,
            task_type=val_test_task_type,
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            target_type=target_type
        )
        
        # Create data loaders
        train_loader = EEGDataLoader.create_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = EEGDataLoader.create_dataloader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = EEGDataLoader.create_dataloader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        # Log statistics
        logger.info(f"Cross-task evaluation setup:")
        logger.info(f"  Training: {len(train_dataset)} samples from {len(train_participants)} participants ({train_task_type} tasks)")
        logger.info(f"  Validation: {len(val_dataset)} samples from {len(val_participants)} participants ({val_test_task_type} tasks)")
        logger.info(f"  Test: {len(test_dataset)} samples from {len(test_participants)} participants ({val_test_task_type} tasks)")
        logger.info(f"  No participant overlap between train and val/test sets")
        
        return train_loader, val_loader, test_loader
    
    @staticmethod
    def create_cross_task_cross_validation_loaders(pickle_dir: str,
                                                 train_task_type: str,
                                                 val_test_task_type: str,
                                                 n_folds: int = 5,
                                                 batch_size: int = 32,
                                                 num_workers: int = 4,
                                                 random_seed: int = 42,
                                                 stratify_by: str = "gender",
                                                 target_type: str = "both",
                                                 transform: Optional[callable] = None,
                                                 gender_transform: Optional[callable] = None,
                                                 age_transform: Optional[callable] = None,
                                                 combined_transform: Optional[callable] = None) -> List[Tuple[DataLoader, DataLoader, DataLoader]]:
        """
        Create n-fold cross-validation with cross-task evaluation.
        
        This ensures:
        1. Training uses one task type (e.g., active tasks)
        2. Validation and testing use the other task type (e.g., passive tasks)
        3. No participant overlap between train and val/test sets
        4. Stratified participant-level splits
        5. Proper generalization evaluation across task types
        
        Args:
            pickle_dir: Directory containing pickle files
            train_task_type: Task type for training ("active" or "passive")
            val_test_task_type: Task type for validation and testing ("active" or "passive")
            n_folds: Number of folds for cross-validation
            batch_size: Batch size for all loaders
            num_workers: Number of workers for all loaders
            random_seed: Random seed for reproducible splits
            stratify_by: Field to stratify by ("gender", "age", "both")
            target_type: Type of target to return ("gender", "age", "both", "combined")
            transform: Optional transform to be applied on EEG data
            gender_transform: Optional transform to be applied on gender targets
            age_transform: Optional transform to be applied on age targets
            combined_transform: Optional transform to be applied on combined targets (gender + age)
            
        Returns:
            List of (train_loader, val_loader, test_loader) tuples for each fold
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
        
        # Create full dataset for both task types to get all participants
        full_dataset = EEGDataset(
            pickle_dir=pickle_dir, 
            task_type="both",  # Load both to get all participants
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            target_type=target_type
        )
        
        # Get unique participants and their stratification labels
        participants = list(set(sample['participant_id'] for sample in full_dataset.samples))
        participant_data = {}
        
        for participant_id in participants:
            participant_samples = [s for s in full_dataset.samples if s['participant_id'] == participant_id]
            if participant_samples:
                sample = participant_samples[0]  # Get first sample for metadata
                stratify_label = get_stratify_label(sample, stratify_by)
                participant_data[participant_id] = stratify_label
        
        # Prepare data for stratification
        participant_ids = list(participant_data.keys())
        stratify_labels = [participant_data[pid] for pid in participant_ids]
        
        # Create stratified k-fold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        
        fold_loaders = []
        
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
            
            # Create training dataset (only train_task_type)
            train_samples = [s for s in full_dataset.samples 
                            if s['participant_id'] in train_participants 
                            and s['task_type'] == train_task_type]
            
            # Create validation dataset (only val_test_task_type)
            val_samples = [s for s in full_dataset.samples 
                          if s['participant_id'] in val_participants 
                          and s['task_type'] == val_test_task_type]
            
            # Create test dataset (only val_test_task_type)
            test_samples = [s for s in full_dataset.samples 
                           if s['participant_id'] in test_participants 
                           and s['task_type'] == val_test_task_type]
            
            # Create datasets for this fold
            train_dataset = EEGDataset._create_from_samples(
                samples=train_samples,
                pickle_dir=pickle_dir,
                task_type=train_task_type,
                transform=transform,
                gender_transform=gender_transform,
                age_transform=age_transform,
                combined_transform=combined_transform,
                target_type=target_type
            )
            
            val_dataset = EEGDataset._create_from_samples(
                samples=val_samples,
                pickle_dir=pickle_dir,
                task_type=val_test_task_type,
                transform=transform,
                gender_transform=gender_transform,
                age_transform=age_transform,
                combined_transform=combined_transform,
                target_type=target_type
            )
            
            test_dataset = EEGDataset._create_from_samples(
                samples=test_samples,
                pickle_dir=pickle_dir,
                task_type=val_test_task_type,
                transform=transform,
                gender_transform=gender_transform,
                age_transform=age_transform,
                combined_transform=combined_transform,
                target_type=target_type
            )
            
            # Create data loaders
            train_loader = EEGDataLoader.create_dataloader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
            )
            val_loader = EEGDataLoader.create_dataloader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
            test_loader = EEGDataLoader.create_dataloader(
                test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
            
            fold_loaders.append((train_loader, val_loader, test_loader))
            
            logger.info(f"Fold {fold + 1}/{n_folds} (Cross-task evaluation):")
            logger.info(f"  Train: {len(train_dataset)} samples from {len(train_participants)} participants ({train_task_type} tasks)")
            logger.info(f"  Val: {len(val_dataset)} samples from {len(val_participants)} participants ({val_test_task_type} tasks)")
            logger.info(f"  Test: {len(test_dataset)} samples from {len(test_participants)} participants ({val_test_task_type} tasks)")
            logger.info(f"  No participant overlap between train and val/test sets")
        
        return fold_loaders
