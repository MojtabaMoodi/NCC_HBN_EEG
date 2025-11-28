#!/usr/bin/env python3
"""
PyTorch Dataset and DataLoader Classes for EEG Data

This module provides PyTorch Dataset and DataLoader classes for loading
and processing the preprocessed EEG data from HDF5 files.
"""

import os
import h5py
import time
import traceback
import multiprocessing
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
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
    
    This class loads preprocessed EEG data from HDF5 files and provides
    an iterator that yields samples for training, validation, and testing.
    
    The HDF5 files are created by the preprocessing pipeline and contain
    pre-split data (train/val/test) with segments of specified lengths.
    """
    
    def __init__(self, 
                 hdf5_file: str,
                 task_type: str = "both",  # "active", "passive", or "both"
                 transform: Optional[callable] = None,
                 gender_transform: Optional[callable] = None,
                 age_transform: Optional[callable] = None,
                 combined_transform: Optional[callable] = None,
                 target_type: str = "both",  # "gender", "age", "both", "combined"
                 participant_filter: Optional[List[str]] = None,
                 shuffle: bool = False,
                 random_seed: Optional[int] = None):  
        """
        Initialize the EEG IterableDataset.
        
        Args:
            hdf5_file: Path to HDF5 file (e.g., "eeg_data_train_1s.h5")
            task_type: Type of tasks to include ("active", "passive", or "both")
            transform: Optional transform to be applied on EEG data
            gender_transform: Optional transform to be applied on gender targets
            age_transform: Optional transform to be applied on age targets
            combined_transform: Optional transform to be applied on combined targets (gender + age)
            target_type: Type of target to return ("gender", "age", "both", "combined")
            participant_filter: Optional list of participant IDs to include (None = all participants)
            shuffle: Whether to shuffle samples during iteration (recommended for training)
            random_seed: Random seed for shuffling (None = use current random state)
        """
        # Convert to Path and then to string for multiprocessing compatibility
        # 'spawn' context requires picklable objects, and Path objects need to be strings
        self.hdf5_file = Path(hdf5_file)
        if not self.hdf5_file.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_file}")
        # Store as string for multiprocessing compatibility (spawn context)
        self.hdf5_file_str = str(self.hdf5_file.resolve())
        
        self.task_type = task_type
        self.transform = transform
        self.gender_transform = gender_transform
        self.age_transform = age_transform
        self.combined_transform = combined_transform
        self.target_type = target_type
        self.participant_filter = participant_filter if participant_filter is None else set(participant_filter)
        self.shuffle = shuffle
        self.random_seed = random_seed
        
        # Cache sample names to avoid collecting them every epoch
        # This is a significant speedup for large datasets
        self._cached_sample_names = None
        self._cache_file_mtime = None
        
        logger.info(f"Initialized EEGDataset from {self.hdf5_file.name} with task_type={task_type}, target_type={target_type}, shuffle={shuffle}")
        if self.participant_filter is not None:
            logger.info(f"Filtering to {len(self.participant_filter)} participants")
    
    def __iter__(self):
        """
        Iterate through the dataset, yielding samples from HDF5 file.
        
        Samples are shuffled if self.shuffle=True to avoid feeding all samples
        from one participant consecutively, which can lead to batch-level
        correlations and training instability.
        
        Yields:
            Dictionary containing sample data with transforms applied
        """
        try:
            # Handle worker sharding for multiprocessing (num_workers > 0)
            # We need to determine sharding BEFORE opening the file, so each worker
            # processes a different subset of samples
            worker_info = get_worker_info()
            
            # Check if we need to refresh the cache (file modified or cache doesn't exist)
            # Use string path for HDF5 (required for spawn multiprocessing)
            hdf5_path = Path(self.hdf5_file_str)
            current_mtime = hdf5_path.stat().st_mtime
            if (self._cached_sample_names is None or 
                self._cache_file_mtime is None or 
                self._cache_file_mtime != current_mtime):
                # Cache sample names (only done once per file modification)
                # Use string path for HDF5 compatibility with multiprocessing
                with h5py.File(self.hdf5_file_str, 'r') as f:
                    task_groups = []
                    if self.task_type in ["active", "both"]:
                        if 'active' in f:
                            task_groups.append(('active', f['active']))
                    if self.task_type in ["passive", "both"]:
                        if 'passive' in f:
                            task_groups.append(('passive', f['passive']))
                    
                    # Collect all (task_type, sample_name) pairs
                    # Optimize: Don't sort if not needed (sorting 955K items is slow)
                    all_samples = []
                    for task_type, group in task_groups:
                        # Use list comprehension for faster collection
                        # Only sort if we have a reasonable number of samples (< 100K)
                        # For large datasets (1s with 955K), sorting adds significant overhead
                        sample_names = [name for name in group.keys() if name.startswith('sample_')]
                        if len(sample_names) < 100000:
                            sample_names = sorted(sample_names)  # Only sort for smaller datasets
                        for sample_name in sample_names:
                            all_samples.append((task_type, sample_name))
                    
                    # Cache the results
                    self._cached_sample_names = all_samples
                    self._cache_file_mtime = current_mtime
                    if worker_info is None or worker_info.id == 0:
                        logger.info(f"Cached {len(all_samples)} sample names from HDF5 file")
            else:
                # Use cached sample names (much faster!)
                all_samples = self._cached_sample_names.copy()
                if worker_info is None or worker_info.id == 0:
                    logger.debug(f"Using cached sample names ({len(all_samples)} samples)")
            
            # Shuffle if requested (do this BEFORE sharding for proper randomization)
            # Important for training to avoid participant-level batch correlations
            if self.shuffle:
                # Create a local random state to avoid affecting global random state
                # Use worker_id in seed to ensure different shuffling per worker
                seed = self.random_seed
                if worker_info is not None:
                    seed = (seed + worker_info.id) if seed is not None else None
                rng = np.random.RandomState(seed)
                rng.shuffle(all_samples)
            
            # Handle worker sharding AFTER shuffling (if applicable)
            # This ensures each worker gets a random subset, not a contiguous chunk
            if worker_info is not None:
                # We're in a worker process - shard the data
                num_workers = worker_info.num_workers
                worker_id = worker_info.id
                # Split samples across workers
                total_samples = len(all_samples)
                samples_per_worker = total_samples // num_workers
                start_idx = worker_id * samples_per_worker
                # Last worker gets any remaining samples
                end_idx = start_idx + samples_per_worker if worker_id < num_workers - 1 else total_samples
                all_samples = all_samples[start_idx:end_idx]
                logger.debug(f"Worker {worker_id}/{num_workers}: processing samples {start_idx}-{end_idx} ({len(all_samples)} samples)")
            
            # Open HDF5 file and cache group references (avoid repeated lookups)
            # Use larger chunk cache for 1s datasets (955K samples need more cache)
            # Increase cache size for better performance with many small reads
            cache_size = 1024*1024*500 if len(all_samples) > 500000 else 1024*1024*100  # 500MB for large datasets, 100MB otherwise
            with h5py.File(self.hdf5_file_str, 'r', rdcc_nbytes=cache_size, rdcc_nslots=50000) as f:
                # Cache group references to avoid repeated dictionary lookups
                active_group = f.get('active', None)
                passive_group = f.get('passive', None)
                
                # Process samples in the determined order
                for task_type, sample_name in all_samples:
                    # Get the group for this task type (use cached reference)
                    if task_type == 'active':
                        group = active_group
                    else:  # passive
                        group = passive_group
                    
                    # Load sample data
                    sample_data = group[sample_name][:]  # Shape: (60, timepoints)
                    
                    # Get metadata (optimize string replacement)
                    metadata_name = sample_name.replace('sample_', 'metadata_', 1)  # Only replace first occurrence
                    if metadata_name in group:
                        metadata = group[metadata_name]
                        # Optimize participant_id extraction
                        participant_id_attr = metadata.attrs['participant_id']
                        participant_id = participant_id_attr.decode() if isinstance(participant_id_attr, bytes) else str(participant_id_attr)
                        
                        # Filter by participant if specified (do this early to avoid unnecessary processing)
                        if self.participant_filter is not None and participant_id not in self.participant_filter:
                            continue
                        
                        # Extract metadata attributes (cache to avoid repeated lookups)
                        gender = int(metadata.attrs['gender'])
                        age = float(metadata.attrs['age'])
                        task_number = int(metadata.attrs.get('task_number', -1))
                        
                        # Create sample dictionary
                        sample = self._create_sample_dict(
                            eeg_data=sample_data,
                            task_type=task_type,
                            participant_id=participant_id,
                            gender=gender,
                            age=age,
                            task_number=task_number,
                            sample_idx=sample_name
                        )
                        yield sample
                            
        except Exception as e:
            # Provide more detailed error information for debugging
            worker_info = get_worker_info()
            worker_id = worker_info.id if worker_info else "main"
            error_msg = f"Error loading HDF5 file {self.hdf5_file_str} in worker {worker_id}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg) from e
    
    def _create_sample_dict(self, eeg_data: np.ndarray, task_type: str, 
                           participant_id: str, gender: int, age: float, 
                           task_number: int, sample_idx: str) -> Dict[str, Any]:
        """
        Create a sample dictionary and apply transforms.
        
        Args:
            eeg_data: EEG data array
            task_type: Task type ('active' or 'passive')
            participant_id: Participant ID
            gender: Gender value
            age: Age value
            task_number: Task number from metadata
            sample_idx: Sample identifier (from HDF5 dataset name)
            
        Returns:
            Dictionary containing sample data with transforms applied
        """
        # Check for NaN/Inf values in input data
        if np.any(np.isnan(eeg_data)) or np.any(np.isinf(eeg_data)):
            raise ValueError(f"NaN/Inf detected in EEG data for participant {participant_id}, task {task_type}, sample {sample_idx}. "
                             f"NaN count: {np.sum(np.isnan(eeg_data))}, Inf count: {np.sum(np.isinf(eeg_data))}. "
                             f"Replacing with zeros.")
        
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
            'task_number': task_number,
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
    def _get_hdf5_file_paths(hdf5_dir: Path, segment_length: str) -> Tuple[Path, Path, Path]:
        """
        Helper method to construct HDF5 file paths for train/val/test splits.
        
        Args:
            hdf5_dir: Directory containing HDF5 files
            segment_length: Length of segments ('1s', '2s', or '4s')
            
        Returns:
            Tuple of (train_file, val_file, test_file) paths
        """
        train_file = hdf5_dir / f"eeg_data_train_{segment_length}.h5"
        val_file = hdf5_dir / f"eeg_data_val_{segment_length}.h5"
        test_file = hdf5_dir / f"eeg_data_test_{segment_length}.h5"
        
        return train_file, val_file, test_file
    
    @staticmethod
    def _verify_hdf5_files(train_file: Path, val_file: Path, test_file: Path) -> None:
        """
        Helper method to verify that HDF5 files exist.
        
        Args:
            train_file: Path to train HDF5 file
            val_file: Path to val HDF5 file
            test_file: Path to test HDF5 file
            
        Raises:
            FileNotFoundError: If any file is missing
        """
        for file_path, split_name in [(train_file, "train"), (val_file, "val"), (test_file, "test")]:
            if not file_path.exists():
                raise FileNotFoundError(f"{split_name} HDF5 file not found: {file_path}")
    
    @staticmethod
    def _create_dataset_from_hdf5(hdf5_file: str,
                                  task_type: str,
                                  target_type: str,
                                  transform: Optional[callable],
                                  gender_transform: Optional[callable],
                                  age_transform: Optional[callable],
                                  combined_transform: Optional[callable],
                                  shuffle: bool = False,
                                  random_seed: Optional[int] = None) -> EEGDataset:
        """
        Helper method to create an EEGDataset from an HDF5 file.
        
        Args:
            hdf5_file: Path to HDF5 file
            task_type: Type of tasks to include ("active", "passive", "both")
            target_type: Type of target to return ("gender", "age", "both", "combined")
            transform: Optional transform to be applied on EEG data
            gender_transform: Optional transform to be applied on gender targets
            age_transform: Optional transform to be applied on age targets
            combined_transform: Optional transform to be applied on combined targets
            shuffle: Whether to shuffle samples during iteration
            random_seed: Random seed for shuffling
            
        Returns:
            EEGDataset instance
        """
        return EEGDataset(
            hdf5_file=hdf5_file,
            task_type=task_type,
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            target_type=target_type,
            shuffle=shuffle,
            random_seed=random_seed
        )
    
    @staticmethod
    def _create_loaders_from_files(train_file: Path, val_file: Path, test_file: Path,
                                   train_task_type: str, val_task_type: str, test_task_type: str,
                                   target_type: str, transform: Optional[callable],
                                   gender_transform: Optional[callable], age_transform: Optional[callable],
                                   combined_transform: Optional[callable], batch_size: int,
                                   num_workers: int, shuffle_train: bool = True,
                                   random_seed: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Helper method to create train/val/test loaders from HDF5 files.
        
        Args:
            train_file: Path to train HDF5 file
            val_file: Path to val HDF5 file
            test_file: Path to test HDF5 file
            train_task_type: Task type for training ("active", "passive", "both")
            val_task_type: Task type for validation ("active", "passive", "both")
            test_task_type: Task type for testing ("active", "passive", "both")
            target_type: Type of target to return ("gender", "age", "both", "combined")
            transform: Optional transform to be applied on EEG data
            gender_transform: Optional transform to be applied on gender targets
            age_transform: Optional transform to be applied on age targets
            combined_transform: Optional transform to be applied on combined targets
            batch_size: Batch size for all loaders
            num_workers: Number of workers for all loaders
            shuffle_train: Whether to shuffle training data (default: True)
            random_seed: Random seed for shuffling (None = use current random state)
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_dataset = EEGDataLoader._create_dataset_from_hdf5(
            hdf5_file=str(train_file),
            task_type=train_task_type,
            target_type=target_type,
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            shuffle=shuffle_train,  # Shuffle training data to avoid participant-level batch correlations
            random_seed=random_seed
        )
        
        val_dataset = EEGDataLoader._create_dataset_from_hdf5(
            hdf5_file=str(val_file),
            task_type=val_task_type,
            target_type=target_type,
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            shuffle=False  # Don't shuffle validation data
        )
        
        test_dataset = EEGDataLoader._create_dataset_from_hdf5(
            hdf5_file=str(test_file),
            task_type=test_task_type,
            target_type=target_type,
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            shuffle=False  # Don't shuffle test data
        )
        
        train_loader = EEGDataLoader.create_dataloader(
            train_dataset, batch_size=batch_size, num_workers=num_workers
        )
        val_loader = EEGDataLoader.create_dataloader(
            val_dataset, batch_size=batch_size, num_workers=num_workers
        )
        test_loader = EEGDataLoader.create_dataloader(
            test_dataset, batch_size=batch_size, num_workers=num_workers
        )
        
        return train_loader, val_loader, test_loader
    
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
            # Increase prefetch_factor for better GPU utilization (higher = more prefetching)
            # Higher prefetch means more batches ready when GPU needs them, reducing idle time
            # Reduced for spawn context to prevent OOM (each worker prefetches independently)
            loader_kwargs['prefetch_factor'] = max(prefetch_factor, 4)  # Reduced from 16 to prevent OOM with spawn
            loader_kwargs['persistent_workers'] = True  # Keep workers alive between epochs (avoids worker restart overhead)
            # Use 'spawn' for better HDF5 compatibility with multiprocessing
            # 'fork' can cause issues with HDF5 file handles
            loader_kwargs['multiprocessing_context'] = multiprocessing.get_context('spawn')
        
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
        
        result = {
            'eeg_data': eeg_data,  # Shape: (batch_size, 60, timepoints), dtype: float32
            'gender': gender,       # Shape: (batch_size,), dtype: long (for classification)
            'age': age,            # Shape: (batch_size,), dtype: float32 (for regression)
            'participant_ids': [sample['participant_id'] for sample in batch],
            'task_types': [sample['task_type'] for sample in batch],
            'task_numbers': [sample.get('task_number', -1) for sample in batch],
            'sample_ids': [sample['sample_id'] for sample in batch]
        }
        
        # Add combined target if present
        if 'combined' in batch[0]:
            combined_list = []
            for sample in batch:
                if isinstance(sample['combined'], torch.Tensor):
                    combined_list.append(sample['combined'])
                else:
                    combined_list.append(torch.tensor(sample['combined'], dtype=torch.long))
            result['combined'] = torch.stack(combined_list)
        
        return result
    
    @staticmethod
    def create_train_val_test_loaders(hdf5_dir: str,
                                    segment_length: str = "1s",
                                    batch_size: int = 32,
                                    num_workers: int = 4,
                                    task_type: str = "both",
                                    target_type: str = "both",
                                    transform: Optional[callable] = None,
                                    gender_transform: Optional[callable] = None,
                                    age_transform: Optional[callable] = None,
                                    combined_transform: Optional[callable] = None,
                                    shuffle_train: bool = True,
                                    random_seed: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test data loaders from pre-split HDF5 files.
        
        Note: The train/val/test splits are already created during preprocessing.
        This method simply loads the pre-split HDF5 files.
        
        Training data is shuffled by default to avoid feeding all samples from one
        participant consecutively, which can lead to batch-level correlations and
        training instability. Validation and test data are not shuffled.
        
        Args:
            hdf5_dir: Directory containing HDF5 files (e.g., "processed_eeg_data_hdf5")
            segment_length: Length of segments ('1s', '2s', or '4s')
            batch_size: Batch size for all loaders
            num_workers: Number of workers for all loaders
            task_type: Type of tasks to include ("active", "passive", "both")
            target_type: Type of target to return ("gender", "age", "both", "combined")
            transform: Optional transform to be applied on EEG data
            gender_transform: Optional transform to be applied on gender targets
            age_transform: Optional transform to be applied on age targets
            combined_transform: Optional transform to be applied on combined targets (gender + age)
            shuffle_train: Whether to shuffle training data (default: True, recommended)
            random_seed: Random seed for shuffling (None = use current random state)
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        hdf5_dir_path = Path(hdf5_dir)
        
        # Construct and verify HDF5 file paths
        train_file, val_file, test_file = EEGDataLoader._get_hdf5_file_paths(
            hdf5_dir_path, segment_length
        )
        EEGDataLoader._verify_hdf5_files(train_file, val_file, test_file)
        
        # Create loaders using helper method
        train_loader, val_loader, test_loader = EEGDataLoader._create_loaders_from_files(
            train_file=train_file,
            val_file=val_file,
            test_file=test_file,
            train_task_type=task_type,
            val_task_type=task_type,
            test_task_type=task_type,
            target_type=target_type,
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle_train=shuffle_train,
            random_seed=random_seed
        )
        
        logger.info(f"Created data loaders from HDF5 files:")
        logger.info(f"  Train: {train_file.name}")
        logger.info(f"  Val: {val_file.name}")
        logger.info(f"  Test: {test_file.name}")
        
        return train_loader, val_loader, test_loader
