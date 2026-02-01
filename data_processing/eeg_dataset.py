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
    combined_gender_age_classification_transform,
    UserIdentificationTransform
)

# Import MultiFileEEGDataset for 1s segments
try:
    from multi_file_dataset import MultiFileEEGDataset
except ImportError:
    MultiFileEEGDataset = None

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
                 user_identification_transform: Optional[UserIdentificationTransform] = None,
                 target_type: str = "both",  # "gender", "age", "both", "combined", "user_identification"
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
            user_identification_transform: Optional transform to be applied on user identification targets
            target_type: Type of target to return ("gender", "age", "both", "combined", "user_identification")
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
        self.user_identification_transform = user_identification_transform
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
                    # Get total_samples from file attributes (fast, no B-tree traversal)
                    total_samples = f.attrs.get('total_samples', 0)
                    
                    # For very large datasets (>500K), collecting keys via group.keys() is extremely slow
                    # (requires full B-tree traversal which can take hours for 955K items)
                    # We must still collect keys, but we'll add progress logging
                    if total_samples > 500000:
                        if worker_info is None or worker_info.id == 0:
                            logger.info(f"Large dataset detected ({total_samples:,} samples).")
                            logger.info("Collecting sample names (this may take 10-30 minutes for 955K samples)...")
                            logger.info("This is a one-time operation - cache will be saved for future use.")
                        
                        task_groups = []
                        if self.task_type in ["active", "both"]:
                            if 'active' in f:
                                task_groups.append(('active', f['active']))
                        if self.task_type in ["passive", "both"]:
                            if 'passive' in f:
                                task_groups.append(('passive', f['passive']))
                        
                        all_samples = []
                        # Unfortunately, we still need to collect keys to know which samples exist in which group
                        # But we'll add progress indication
                        for task_type, group in task_groups:
                            if worker_info is None or worker_info.id == 0:
                                logger.info(f"Collecting {task_type} sample names...")
                            # Collect keys - this is the slow part but necessary
                            sample_names = []
                            count = 0
                            for name in group.keys():
                                if name.startswith('sample_'):
                                    sample_names.append(name)
                                    count += 1
                                    # Log progress every 100K samples
                                    if count % 100000 == 0 and (worker_info is None or worker_info.id == 0):
                                        logger.info(f"  Collected {count:,} {task_type} sample names...")
                            
                            if worker_info is None or worker_info.id == 0:
                                logger.info(f"  Collected {len(sample_names):,} {task_type} sample names total")
                            
                            # Don't sort for large datasets (saves time)
                            for sample_name in sample_names:
                                all_samples.append((task_type, sample_name))
                    else:
                        # Small/medium datasets: use original approach (collect keys)
                        task_groups = []
                        if self.task_type in ["active", "both"]:
                            if 'active' in f:
                                task_groups.append(('active', f['active']))
                        if self.task_type in ["passive", "both"]:
                            if 'passive' in f:
                                task_groups.append(('passive', f['passive']))
                        
                        # Collect all (task_type, sample_name) pairs
                        all_samples = []
                        for task_type, group in task_groups:
                            # Use list comprehension for faster collection
                            # Only sort if we have a reasonable number of samples (< 100K)
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
            # For very large datasets, use 1GB cache to reduce disk I/O
            if len(all_samples) > 500000:
                cache_size = 1024*1024*1024  # 1GB for very large datasets (1s segments)
                cache_slots = 100000  # More slots for better cache hit rate
            else:
                cache_size = 1024*1024*100  # 100MB for smaller datasets
                cache_slots = 50000
            with h5py.File(self.hdf5_file_str, 'r', rdcc_nbytes=cache_size, rdcc_nslots=cache_slots) as f:
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
                    # Note: sample_name is guaranteed to exist in group because we collected it
                    # from group.keys() during cache building, so no existence check needed
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
                        # For user_identification target_type, gender and age are optional
                        if self.target_type == "user_identification":
                            # User identification only needs participant_id, task_type, and task_number
                            gender = metadata.attrs.get('gender', None)
                            if gender is not None:
                                gender = int(gender)
                            age = metadata.attrs.get('age', None)
                            if age is not None:
                                age = float(age)
                        else:
                            # For other target types, gender and age are required
                            if 'gender' not in metadata.attrs:
                                raise KeyError(
                                    f"Missing required 'gender' attribute in metadata for sample {sample_name} "
                                    f"in file {self.hdf5_file_str}. This is required for target_type '{self.target_type}'."
                                )
                            if 'age' not in metadata.attrs:
                                raise KeyError(
                                    f"Missing required 'age' attribute in metadata for sample {sample_name} "
                                    f"in file {self.hdf5_file_str}. This is required for target_type '{self.target_type}'."
                                )
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
                           participant_id: str, gender: Optional[int], age: Optional[float], 
                           task_number: int, sample_idx: str) -> Dict[str, Any]:
        """
        Create a sample dictionary and apply transforms.
        
        Args:
            eeg_data: EEG data array
            task_type: Task type ('active' or 'passive')
            participant_id: Participant ID
            gender: Gender value (optional, None for user_identification)
            age: Age value (optional, None for user_identification)
            task_number: Task number from metadata
            sample_idx: Sample identifier (from HDF5 dataset name)
            
        Returns:
            Dictionary containing sample data with transforms applied
        """
        # Check for NaN/Inf values in input data
        # Best practice: Fail fast at dataset level to alert user to data quality issues
        # This ensures data quality problems are caught early, not hidden during training
        nan_count = np.sum(np.isnan(eeg_data))
        inf_count = np.sum(np.isinf(eeg_data))
        if nan_count > 0 or inf_count > 0:
            error_msg = (
                f"NaN/Inf detected in EEG data for participant {participant_id}, "
                f"task {task_type}, sample {sample_idx}. "
                f"NaN count: {nan_count}, Inf count: {inf_count}. "
                f"This indicates a data quality issue that should be fixed at the preprocessing stage. "
                f"Please check the preprocessing pipeline and source data."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
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
            'task_number': task_number,
            'sample_id': f"{participant_id}_{task_type}_{sample_idx}"
        }
        
        # Only include gender and age if they are not None
        if gender is not None:
            sample['gender'] = gender
        if age is not None:
            sample['age'] = age
        
        # Apply individual transforms
        if self.gender_transform and 'gender' in sample and sample['gender'] is not None:
            sample['gender'] = self.gender_transform(sample['gender'])
        
        if self.age_transform and 'age' in sample and sample['age'] is not None:
            sample['age'] = self.age_transform(sample['age'])
        
        # Apply combined transform if target_type is "combined"
        # Note: Combined transform uses raw gender and age values before individual transforms
        if self.target_type == "combined" and self.combined_transform:
            if gender is None or age is None:
                raise ValueError(
                    f"Combined transform requires both gender and age, but got gender={gender}, age={age} "
                    f"for participant {participant_id}, sample {sample_idx}"
                )
            # Apply combined transform
            sample['combined'] = self.combined_transform(gender, age)
        
        # Apply user identification transform if target_type is "user_identification"
        if self.target_type == "user_identification" and self.user_identification_transform:
            sample['user_identification'] = self.user_identification_transform(participant_id)
        
        return sample


class EEGDataLoader:
    """
    Custom DataLoader class for EEG data with additional utilities.
    """
    
    @staticmethod
    def _get_hdf5_file_paths(hdf5_dir: Path, segment_length: str) -> Tuple[Union[Path, List[Path]], Union[Path, List[Path]], Union[Path, List[Path]]]:
        """
        Helper method to construct HDF5 file paths for train/val/test splits.
        
        Checks for multiple part files for all segment lengths and returns list of paths if found.
        Falls back to single file paths if no part files exist.
        
        Args:
            hdf5_dir: Directory containing HDF5 files
            segment_length: Length of segments ('1s', '2s', or '4s')
            
        Returns:
            Tuple of (train_file(s), val_file(s), test_file(s)) paths
            Returns lists of paths if multi-file structure exists, single paths otherwise.
        """
        # Check for multiple part files for all segment lengths
        train_files = sorted(hdf5_dir.glob(f"eeg_data_train_{segment_length}_part*.h5"))
        val_files = sorted(hdf5_dir.glob(f"eeg_data_val_{segment_length}_part*.h5"))
        test_files = sorted(hdf5_dir.glob(f"eeg_data_test_{segment_length}_part*.h5"))
        
        # Fallback to single file if no part files found
        if not train_files:
            train_file = hdf5_dir / f"eeg_data_train_{segment_length}.h5"
            if train_file.exists():
                train_files = [train_file]
        if not val_files:
            val_file = hdf5_dir / f"eeg_data_val_{segment_length}.h5"
            if val_file.exists():
                val_files = [val_file]
        if not test_files:
            test_file = hdf5_dir / f"eeg_data_test_{segment_length}.h5"
            if test_file.exists():
                test_files = [test_file]
        
        # Return lists if multi-file, single paths if single file
        if len(train_files) == 1 and len(val_files) == 1 and len(test_files) == 1:
            # All are single files, return as single paths for backward compatibility
            return train_files[0], val_files[0], test_files[0]
        else:
            # Multi-file structure, return as lists
            return train_files, val_files, test_files

    @staticmethod
    def _get_unknown_hdf5_file_paths(hdf5_dir: Path, segment_length: str):
        """
        Get HDF5 file path(s) for the "unknown" split when consider_unknown_users was used.
        
        Returns:
            List of paths if multi-file, single path if single file, or empty list if no unknown split.
        """
        part_files = sorted(hdf5_dir.glob(f"eeg_data_unknown_{segment_length}_part*.h5"))
        if part_files:
            return part_files[0] if len(part_files) == 1 else part_files
        single = hdf5_dir / f"eeg_data_unknown_{segment_length}.h5"
        if single.exists():
            return single
        return []
    
    @staticmethod
    def _verify_hdf5_files(train_files, val_files, test_files) -> None:
        """
        Helper method to verify that HDF5 files exist.
        
        Args:
            train_files: Path(s) to train HDF5 file(s) (single Path or list of Paths)
            val_files: Path(s) to val HDF5 file(s) (single Path or list of Paths)
            test_files: Path(s) to test HDF5 file(s) (single Path or list of Paths)
            
        Raises:
            FileNotFoundError: If any file is missing
        """
        # Handle both single files and lists of files
        for files, split_name in [(train_files, "train"), (val_files, "val"), (test_files, "test")]:
            if isinstance(files, (list, tuple)):
                if len(files) == 0:
                    raise FileNotFoundError(f"{split_name} HDF5 files not found")
                for file_path in files:
                    if not file_path.exists():
                        raise FileNotFoundError(f"{split_name} HDF5 file not found: {file_path}")
            else:
                if not files.exists():
                    raise FileNotFoundError(f"{split_name} HDF5 file not found: {files}")
    
    @staticmethod
    def _create_dataset_from_hdf5(hdf5_file_or_files,
                                  task_type: str,
                                  target_type: str,
                                  transform: Optional[callable],
                                  gender_transform: Optional[callable],
                                  age_transform: Optional[callable],
                                  combined_transform: Optional[callable],
                                  user_identification_transform: Optional[UserIdentificationTransform] = None,
                                  shuffle: bool = False,
                                  random_seed: Optional[int] = None):
        """
        Create dataset from HDF5 file(s).
        
        Args:
            hdf5_file_or_files: Single file path (str) or list of file paths (for multi-file)
            task_type: Type of tasks to include
            target_type: Type of target to return
            transform: Optional transform to be applied on EEG data
            gender_transform: Optional transform to be applied on gender targets
            age_transform: Optional transform to be applied on age targets
            combined_transform: Optional transform to be applied on combined targets
            user_identification_transform: Optional transform for user identification
            shuffle: Whether to shuffle samples during iteration
            random_seed: Random seed for shuffling
            
        Returns:
            EEGDataset or MultiFileEEGDataset instance
        """
        # Handle multiple files (for 1s segments)
        if isinstance(hdf5_file_or_files, (list, tuple)):
            if MultiFileEEGDataset is None:
                raise ImportError("MultiFileEEGDataset not available. Cannot load multiple files.")
            
            # Convert Path objects to strings
            hdf5_files = [str(f) if isinstance(f, Path) else f for f in hdf5_file_or_files]
            
            return MultiFileEEGDataset(
                hdf5_files=hdf5_files,
                task_type=task_type,
                transform=transform,
                gender_transform=gender_transform,
                age_transform=age_transform,
                combined_transform=combined_transform,
                user_identification_transform=user_identification_transform,
                target_type=target_type,
                shuffle=shuffle,
                random_seed=random_seed
            )
        else:
            # Single file
            hdf5_file = str(hdf5_file_or_files) if isinstance(hdf5_file_or_files, Path) else hdf5_file_or_files
            
            return EEGDataset(
                hdf5_file=hdf5_file,
                task_type=task_type,
                transform=transform,
                gender_transform=gender_transform,
                age_transform=age_transform,
                combined_transform=combined_transform,
                user_identification_transform=user_identification_transform,
                target_type=target_type,
                shuffle=shuffle,
                random_seed=random_seed
            )
    
    @staticmethod
    def _create_loaders_from_files(train_file, val_file, test_file,
                                   train_task_type: str, val_task_type: str, test_task_type: str,
                                   target_type: str, transform: Optional[callable],
                                   gender_transform: Optional[callable], age_transform: Optional[callable],
                                   combined_transform: Optional[callable],
                                   user_identification_transform: Optional[UserIdentificationTransform] = None,
                                   batch_size: int = 32, num_workers: int = 4, shuffle_train: bool = True,
                                   random_seed: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create loaders from HDF5 file(s).
        
        Args:
            train_file: Path(s) to train HDF5 file(s) (single Path or list of Paths)
            val_file: Path(s) to val HDF5 file(s) (single Path or list of Paths)
            test_file: Path(s) to test HDF5 file(s) (single Path or list of Paths)
            train_task_type: Task type for training ("active", "passive", "both")
            val_task_type: Task type for validation ("active", "passive", "both")
            test_task_type: Task type for testing ("active", "passive", "both")
            target_type: Type of target to return ("gender", "age", "both", "combined", "user_identification")
            transform: Optional transform to be applied on EEG data
            gender_transform: Optional transform to be applied on gender targets
            age_transform: Optional transform to be applied on age targets
            combined_transform: Optional transform to be applied on combined targets
            user_identification_transform: Optional transform for user identification
            batch_size: Batch size for all loaders
            num_workers: Number of workers for all loaders
            shuffle_train: Whether to shuffle training data (default: True)
            random_seed: Random seed for shuffling (None = use current random state)
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_dataset = EEGDataLoader._create_dataset_from_hdf5(
            hdf5_file_or_files=train_file,
            task_type=train_task_type,
            target_type=target_type,
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            user_identification_transform=user_identification_transform,
            shuffle=shuffle_train,  # Shuffle training data to avoid participant-level batch correlations
            random_seed=random_seed
        )
        
        val_dataset = EEGDataLoader._create_dataset_from_hdf5(
            hdf5_file_or_files=val_file,
            task_type=val_task_type,
            target_type=target_type,
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            user_identification_transform=user_identification_transform,
            shuffle=False  # Don't shuffle validation data
        )
        
        test_dataset = EEGDataLoader._create_dataset_from_hdf5(
            hdf5_file_or_files=test_file,
            task_type=test_task_type,
            target_type=target_type,
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            user_identification_transform=user_identification_transform,
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
            # For very large batch sizes (6144), we can use higher prefetch since each worker handles fewer batches
            # With maximum batches, memory per prefetched batch is manageable
            loader_kwargs['prefetch_factor'] = max(prefetch_factor, 16)  # Increased to 16 for maximum GPU utilization
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
        if not batch:
            raise ValueError("Cannot collate empty batch")
        
        # Stack EEG data
        eeg_data = torch.stack([sample['eeg_data'] for sample in batch])
        
        # Determine which fields are present in the batch
        has_gender = 'gender' in batch[0] and batch[0]['gender'] is not None
        has_age = 'age' in batch[0] and batch[0]['age'] is not None
        has_user_identification = 'user_identification' in batch[0]
        has_combined = 'combined' in batch[0]
        
        result = {
            'eeg_data': eeg_data,  # Shape: (batch_size, 60, timepoints), dtype: float32
            'participant_ids': [sample['participant_id'] for sample in batch],
            'task_types': [sample['task_type'] for sample in batch],
            'task_numbers': [sample.get('task_number', -1) for sample in batch],
            'sample_ids': [sample['sample_id'] for sample in batch]
        }
        
        # Stack gender if present
        if has_gender:
            gender_list = []
            for sample in batch:
                if 'gender' not in sample or sample['gender'] is None:
                    raise ValueError(
                        f"Inconsistent batch: some samples have 'gender' and others don't. "
                        f"All samples in a batch must have the same fields."
                    )
                if isinstance(sample['gender'], torch.Tensor):
                    gender_list.append(sample['gender'])
                else:
                    # Gender is typically used for classification, so use long dtype
                    gender_list.append(torch.tensor(sample['gender'], dtype=torch.long))
            result['gender'] = torch.stack(gender_list)
        
        # Stack age if present
        if has_age:
            age_list = []
            for sample in batch:
                if 'age' not in sample or sample['age'] is None:
                    raise ValueError(
                        f"Inconsistent batch: some samples have 'age' and others don't. "
                        f"All samples in a batch must have the same fields."
                    )
                if isinstance(sample['age'], torch.Tensor):
                    age_list.append(sample['age'])
                else:
                    # Age is typically used for regression, so use float32 dtype
                    age_list.append(torch.tensor(sample['age'], dtype=torch.float32))
            result['age'] = torch.stack(age_list)
        
        # Add combined target if present
        if 'combined' in batch[0]:
            combined_list = []
            for sample in batch:
                if isinstance(sample['combined'], torch.Tensor):
                    combined_list.append(sample['combined'])
                else:
                    combined_list.append(torch.tensor(sample['combined'], dtype=torch.long))
            result['combined'] = torch.stack(combined_list)
        
        # Add user identification target if present
        if 'user_identification' in batch[0]:
            user_id_list = []
            for sample in batch:
                if isinstance(sample['user_identification'], torch.Tensor):
                    user_id_list.append(sample['user_identification'])
                else:
                    user_id_list.append(torch.tensor(sample['user_identification'], dtype=torch.long))
            result['user_identification'] = torch.stack(user_id_list)
        
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
                                    user_identification_transform: Optional[UserIdentificationTransform] = None,
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
        train_files, val_files, test_files = EEGDataLoader._get_hdf5_file_paths(
            hdf5_dir_path, segment_length
        )
        EEGDataLoader._verify_hdf5_files(train_files, val_files, test_files)
        
        # Create loaders using helper method
        train_loader, val_loader, test_loader = EEGDataLoader._create_loaders_from_files(
            train_file=train_files,
            val_file=val_files,
            test_file=test_files,
            train_task_type=task_type,
            val_task_type=task_type,
            test_task_type=task_type,
            target_type=target_type,
            transform=transform,
            gender_transform=gender_transform,
            age_transform=age_transform,
            combined_transform=combined_transform,
            user_identification_transform=user_identification_transform,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle_train=shuffle_train,
            random_seed=random_seed
        )
        
        # Log file information
        logger.info(f"Created data loaders from HDF5 files for {segment_length} segments:")
        if isinstance(train_files, (list, tuple)):
            logger.info(f"  Train: {len(train_files)} file(s) (multi-file structure)")
            logger.info(f"  Val: {len(val_files)} file(s) (multi-file structure)")
            logger.info(f"  Test: {len(test_files)} file(s) (multi-file structure)")
        else:
            logger.info(f"  Train: {train_files.name if isinstance(train_files, Path) else train_files}")
            logger.info(f"  Val: {val_files.name if isinstance(val_files, Path) else val_files}")
            logger.info(f"  Test: {test_files.name if isinstance(test_files, Path) else test_files}")
        
        return train_loader, val_loader, test_loader
