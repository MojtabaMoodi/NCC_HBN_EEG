#!/usr/bin/env python3
"""
PyTorch Dataset and DataLoader Classes for EEG Data

This module provides PyTorch Dataset and DataLoader classes for loading
and processing the preprocessed EEG data from pickle files.
"""

import os
import pickle
import time
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import IterableDataset, DataLoader
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

class EEGDataset(IterableDataset):
    """
    PyTorch IterableDataset class for EEG data.
    
    This class loads preprocessed EEG data from pickle files and provides
    an iterator that yields samples for training, validation, and testing.
    
    Supports datasets of any size by iterating through pickle files on-demand.
    """
    
    def __init__(self, 
                 pickle_dir: str,
                 task_type: str = "both",  # "active", "passive", or "both"
                 transform: Optional[callable] = None,
                 gender_transform: Optional[callable] = None,
                 age_transform: Optional[callable] = None,
                 combined_transform: Optional[callable] = None,
                 target_type: str = "both",  # "gender", "age", "both", "combined"
                 participant_filter: Optional[List[str]] = None):  
        """
        Initialize the EEG IterableDataset.
        
        Args:
            pickle_dir: Directory containing pickle files
            task_type: Type of tasks to include ("active", "passive", or "both")
            transform: Optional transform to be applied on EEG data
            gender_transform: Optional transform to be applied on gender targets
            age_transform: Optional transform to be applied on age targets
            combined_transform: Optional transform to be applied on combined targets (gender + age)
            target_type: Type of target to return ("gender", "age", "both", "combined")
            participant_filter: Optional list of participant IDs to include (None = all participants)
        """
        self.pickle_dir = Path(pickle_dir)
        self.task_type = task_type
        self.transform = transform
        self.gender_transform = gender_transform
        self.age_transform = age_transform
        self.combined_transform = combined_transform
        self.target_type = target_type
        self.participant_filter = participant_filter if participant_filter is None else set(participant_filter)
        
        logger.info(f"Initialized EEGDataset with task_type={task_type}, target_type={target_type}")
        if self.participant_filter is not None:
            logger.info(f"Filtering to {len(self.participant_filter)} participants")
    
    def __iter__(self):
        """
        Iterate through the dataset, yielding samples from pickle files.
        
        Yields:
            Dictionary containing sample data with transforms applied
        """
        # Find all pickle files
        pickle_files = list(self.pickle_dir.glob("eeg_batch_*.pkl"))
        pickle_files.sort()
        
        for pickle_file in pickle_files:
            try:
                # Load pickle file
                with open(pickle_file, 'rb') as f:
                    batch_data = pickle.load(f)
                
                # Process each participant in the batch
                for participant in batch_data:
                    participant_id = str(participant['participant_id'])
                    
                    # Filter by participant if specified
                    if self.participant_filter is not None and participant_id not in self.participant_filter:
                        continue
                    
                    # Get participant metadata
                    gender = participant['gender']
                    age = participant['age']
                    
                    # Process active tasks
                    if self.task_type in ["active", "both"]:
                        active_eeg = participant.get('active_eeg', [])
                        for i, eeg_data in enumerate(active_eeg):
                            sample = self._create_sample_dict(
                                eeg_data=eeg_data,
                                task_type='active',
                                participant_id=participant_id,
                                gender=gender,
                                age=age,
                                sample_idx=i
                            )
                            yield sample
                    
                    # Process passive tasks
                    if self.task_type in ["passive", "both"]:
                        passive_eeg = participant.get('passive_eeg', [])
                        for i, eeg_data in enumerate(passive_eeg):
                            sample = self._create_sample_dict(
                                eeg_data=eeg_data,
                                task_type='passive',
                                participant_id=participant_id,
                                gender=gender,
                                age=age,
                                sample_idx=i
                            )
                            yield sample
                    
            except Exception as e:
                logger.error(f"Error loading {pickle_file}: {e}")
                continue
    
    def _create_sample_dict(self, eeg_data: np.ndarray, task_type: str, 
                           participant_id: str, gender: int, age: float, 
                           sample_idx: int) -> Dict[str, Any]:
        """
        Create a sample dictionary and apply transforms.
        
        Args:
            eeg_data: EEG data array
            task_type: Task type ('active' or 'passive')
            participant_id: Participant ID
            gender: Gender value
            age: Age value
            sample_idx: Sample index within participant
            
        Returns:
            Dictionary containing sample data with transforms applied
        """
        # Convert EEG data to tensor
        eeg_tensor = torch.FloatTensor(eeg_data)
        
        # Apply transforms if provided
        if self.transform:
            eeg_tensor = self.transform(eeg_tensor)
        
        # Create sample dict
        sample = {
            'eeg_data': eeg_tensor,
            'task_type': task_type,
            'participant_id': participant_id,
            'gender': gender,
            'age': age,
            'sample_id': f"{participant_id}_{task_type}_{sample_idx}"
        }
        
        # Apply individual transforms
        if self.gender_transform and 'gender' in sample:
            sample['gender'] = self.gender_transform(sample['gender'])
        
        if self.age_transform and 'age' in sample:
            sample['age'] = self.age_transform(sample['age'])
        
        # Apply combined transform if target_type is "combined"
        # Note: Combined transform uses raw gender and age values before individual transforms
        if self.target_type == "combined" and self.combined_transform:
            # Apply combined transform
            sample['combined'] = self.combined_transform(gender, age)
        
        return sample


class EEGDataLoader:
    """
    Custom DataLoader class for EEG data with additional utilities.
    """
    
    @staticmethod
    def create_dataloader(dataset: EEGDataset,
                         batch_size: int = 32,
                         num_workers: int = 4,
                         pin_memory: bool = True,
                         drop_last: bool = False,
                         prefetch_factor: int = 2) -> DataLoader:
        """
        Create a PyTorch DataLoader for the EEG IterableDataset.
        
        Args:
            dataset: EEGDataset instance (IterableDataset)
            batch_size: Number of samples per batch
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            drop_last: Whether to drop the last incomplete batch
            prefetch_factor: Number of batches to prefetch (only used when num_workers > 0)
            
        Returns:
            PyTorch DataLoader instance
        """
        loader_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'drop_last': drop_last,
            'collate_fn': EEGDataLoader.collate_fn
        }
        
        # prefetch_factor only works with num_workers > 0
        if num_workers > 0:
            # Increase prefetch_factor for better GPU utilization
            loader_kwargs['prefetch_factor'] = max(prefetch_factor, 8)
            loader_kwargs['persistent_workers'] = True  # Keep workers alive between epochs
        
        return DataLoader(**loader_kwargs)
    
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
        
        # Get unique participants by iterating through the dataset
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
        
        # Create datasets for each split with participant filtering
        train_dataset = EEGDataset(
            pickle_dir=pickle_dir,
            task_type=task_type,
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            target_type=target_type,
            participant_filter=train_participants
        )
        
        val_dataset = EEGDataset(
            pickle_dir=pickle_dir,
            task_type=task_type,
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            target_type=target_type,
            participant_filter=val_participants
        )
        
        test_dataset = EEGDataset(
            pickle_dir=pickle_dir,
            task_type=task_type,
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            target_type=target_type,
            participant_filter=test_participants
        )
        
        # Create data loaders
        train_loader = EEGDataLoader.create_dataloader(
            train_dataset, batch_size=batch_size, num_workers=num_workers
        )
        val_loader = EEGDataLoader.create_dataloader(
            val_dataset, batch_size=batch_size, num_workers=num_workers
        )
        test_loader = EEGDataLoader.create_dataloader(
            test_dataset, batch_size=batch_size, num_workers=num_workers
        )
        
        logger.info(f"Created data loaders:")
        logger.info(f"  Train: {len(train_participants)} participants")
        logger.info(f"  Val: {len(val_participants)} participants")
        logger.info(f"  Test: {len(test_participants)} participants")
        
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
        
        # Get unique participants and their stratification labels by iterating through the dataset
        participant_data = {}
        temp_dataset = EEGDataset(
            pickle_dir=pickle_dir,
            task_type="both",  # Get all participants regardless of task_type
            target_type="both"
        )
        for sample in temp_dataset:
            participant_id = sample['participant_id']
            if participant_id not in participant_data:
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
            
            # Create datasets for this fold with participant filtering
            train_dataset = EEGDataset(
                pickle_dir=pickle_dir,
                task_type=task_type,
                transform=transform,
                gender_transform=gender_transform,
                age_transform=age_transform,
                combined_transform=combined_transform,
                target_type=target_type,
                participant_filter=train_participants
            )
            
            val_dataset = EEGDataset(
                pickle_dir=pickle_dir,
                task_type=task_type,
                transform=transform,
                gender_transform=gender_transform,
                age_transform=age_transform,
                combined_transform=combined_transform,
                target_type=target_type,
                participant_filter=val_participants
            )
            
            test_dataset = EEGDataset(
                pickle_dir=pickle_dir,
                task_type=task_type,
                transform=transform,
                gender_transform=gender_transform,
                age_transform=age_transform,
                combined_transform=combined_transform,
                target_type=target_type,
                participant_filter=test_participants
            )
            
            # Create data loaders
            train_loader = EEGDataLoader.create_dataloader(
                train_dataset, batch_size=batch_size, num_workers=num_workers
            )
            val_loader = EEGDataLoader.create_dataloader(
                val_dataset, batch_size=batch_size, num_workers=num_workers
            )
            test_loader = EEGDataLoader.create_dataloader(
                test_dataset, batch_size=batch_size, num_workers=num_workers
            )
            
            fold_loaders.append((train_loader, val_loader, test_loader))
            
            logger.info(f"Fold {fold + 1}/{n_folds}:")
            logger.info(f"  Train: {len(train_participants)} participants")
            logger.info(f"  Val: {len(val_participants)} participants")
            logger.info(f"  Test: {len(test_participants)} participants")
        
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
        
        # Get unique participants by iterating through the dataset
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
        
        # Create datasets with correct task_type and participant filtering
        train_dataset = EEGDataset(
            pickle_dir=pickle_dir,
            task_type=train_task_type,  # Only train_task_type
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            target_type=target_type,
            participant_filter=train_participants
        )
        
        val_dataset = EEGDataset(
            pickle_dir=pickle_dir,
            task_type=val_test_task_type,  # Only val_test_task_type
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            target_type=target_type,
            participant_filter=val_participants
        )
        
        test_dataset = EEGDataset(
            pickle_dir=pickle_dir,
            task_type=val_test_task_type,  # Only val_test_task_type
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            target_type=target_type,
            participant_filter=test_participants
        )
        
        # Create data loaders
        train_loader = EEGDataLoader.create_dataloader(
            train_dataset, batch_size=batch_size, num_workers=num_workers
        )
        val_loader = EEGDataLoader.create_dataloader(
            val_dataset, batch_size=batch_size, num_workers=num_workers
        )
        test_loader = EEGDataLoader.create_dataloader(
            test_dataset, batch_size=batch_size, num_workers=num_workers
        )
        
        # Log statistics
        logger.info(f"Cross-task evaluation setup:")
        logger.info(f"  Training: {len(train_participants)} participants ({train_task_type} tasks)")
        logger.info(f"  Validation: {len(val_participants)} participants ({val_test_task_type} tasks)")
        logger.info(f"  Test: {len(test_participants)} participants ({val_test_task_type} tasks)")
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
        
        # Get unique participants and their stratification labels by iterating through the dataset
        participant_data = {}
        temp_dataset = EEGDataset(
            pickle_dir=pickle_dir,
            task_type="both",  # Get all participants regardless of task_type
            target_type="both"
        )
        for sample in temp_dataset:
            participant_id = sample['participant_id']
            if participant_id not in participant_data:
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
            
            # Create datasets for this fold with correct task_type and participant filtering
            train_dataset = EEGDataset(
                pickle_dir=pickle_dir,
                task_type=train_task_type,  # Only train_task_type
                transform=transform,
                gender_transform=gender_transform,
                age_transform=age_transform,
                combined_transform=combined_transform,
                target_type=target_type,
                participant_filter=train_participants
            )
            
            val_dataset = EEGDataset(
                pickle_dir=pickle_dir,
                task_type=val_test_task_type,  # Only val_test_task_type
                transform=transform,
                gender_transform=gender_transform,
                age_transform=age_transform,
                combined_transform=combined_transform,
                target_type=target_type,
                participant_filter=val_participants
            )
            
            test_dataset = EEGDataset(
                pickle_dir=pickle_dir,
                task_type=val_test_task_type,  # Only val_test_task_type
                transform=transform,
                gender_transform=gender_transform,
                age_transform=age_transform,
                combined_transform=combined_transform,
                target_type=target_type,
                participant_filter=test_participants
            )
            
            # Create data loaders
            train_loader = EEGDataLoader.create_dataloader(
                train_dataset, batch_size=batch_size, num_workers=num_workers
            )
            val_loader = EEGDataLoader.create_dataloader(
                val_dataset, batch_size=batch_size, num_workers=num_workers
            )
            test_loader = EEGDataLoader.create_dataloader(
                test_dataset, batch_size=batch_size, num_workers=num_workers
            )
            
            fold_loaders.append((train_loader, val_loader, test_loader))
            
            logger.info(f"Fold {fold + 1}/{n_folds} (Cross-task evaluation):")
            logger.info(f"  Train: {len(train_participants)} participants ({train_task_type} tasks)")
            logger.info(f"  Val: {len(val_participants)} participants ({val_test_task_type} tasks)")
            logger.info(f"  Test: {len(test_participants)} participants ({val_test_task_type} tasks)")
            logger.info(f"  No participant overlap between train and val/test sets")
        
        return fold_loaders
