#!/usr/bin/env python3
"""
Multi-File EEG Dataset Support

This module provides support for loading EEG data from multiple HDF5 files.
This is particularly useful for 1s segments which can have ~955K samples,
where a single large HDF5 file becomes slow due to B-tree traversal overhead.

By splitting into multiple files (e.g., 10 files with ~95K samples each),
we can significantly improve data loading performance.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator
from torch.utils.data import IterableDataset, get_worker_info
import numpy as np

from eeg_dataset import EEGDataset, UserIdentificationTransform

logger = logging.getLogger(__name__)

# Maximum samples per file to prevent B-tree performance degradation
# For 1s segments, we target ~100K samples per file
MAX_SAMPLES_PER_FILE = 100000

class MultiFileEEGDataset(IterableDataset):
    """
    Dataset that loads from multiple HDF5 files.
    
    This is a wrapper around multiple EEGDataset instances, concatenating
    their outputs. Useful for large datasets (like 1s segments) that are
    split across multiple files for better performance.
    """
    
    def __init__(self,
                 hdf5_files: List[str],
                 task_type: str = "both",
                 transform: Optional[callable] = None,
                 gender_transform: Optional[callable] = None,
                 age_transform: Optional[callable] = None,
                 combined_transform: Optional[callable] = None,
                 user_identification_transform: Optional[UserIdentificationTransform] = None,
                 target_type: str = "both",
                 participant_filter: Optional[List[str]] = None,
                 shuffle: bool = False,
                 random_seed: Optional[int] = None):
        """
        Initialize MultiFileEEGDataset.
        
        Args:
            hdf5_files: List of paths to HDF5 files
            task_type: Type of tasks to include ("active", "passive", or "both")
            transform: Optional transform to be applied on EEG data
            gender_transform: Optional transform to be applied on gender targets
            age_transform: Optional transform to be applied on age targets
            combined_transform: Optional transform to be applied on combined targets
            user_identification_transform: Optional transform for user identification
            target_type: Type of target to return
            participant_filter: Optional list of participant IDs to include
            shuffle: Whether to shuffle samples during iteration
            random_seed: Random seed for shuffling
        """
        if not hdf5_files:
            raise ValueError("hdf5_files list cannot be empty")
        
        # Verify all files exist
        for hdf5_file in hdf5_files:
            if not Path(hdf5_file).exists():
                raise FileNotFoundError(f"HDF5 file not found: {hdf5_file}")
        
        self.hdf5_files = hdf5_files
        self.task_type = task_type
        self.transform = transform
        self.gender_transform = gender_transform
        self.age_transform = age_transform
        self.combined_transform = combined_transform
        self.user_identification_transform = user_identification_transform
        self.target_type = target_type
        self.participant_filter = participant_filter
        self.shuffle = shuffle
        self.random_seed = random_seed
        
        # Create individual datasets for each file
        self.datasets = []
        for hdf5_file in hdf5_files:
            dataset = EEGDataset(
                hdf5_file=hdf5_file,
                task_type=task_type,
                transform=transform,
                gender_transform=gender_transform,
                age_transform=age_transform,
                combined_transform=combined_transform,
                user_identification_transform=user_identification_transform,
                target_type=target_type,
                participant_filter=participant_filter,
                shuffle=shuffle,
                random_seed=random_seed
            )
            self.datasets.append(dataset)
        
        logger.info(f"Initialized MultiFileEEGDataset with {len(hdf5_files)} files")
        logger.info(f"Files: {[Path(f).name for f in hdf5_files]}")
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate through all datasets, interleaving samples from multiple files.
        
        CRITICAL: Uses round-robin interleaving to ensure each batch contains
        samples from multiple files, preventing batch-level correlations.
        
        If shuffle=True, files are shuffled first, then samples are interleaved.
        This ensures better mixing and diversity in each batch.
        """
        # Handle worker sharding
        worker_info = get_worker_info()
        
        # Determine which files this worker should process
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            
            # Distribute files across workers
            files_per_worker = len(self.datasets) // num_workers
            start_file_idx = worker_id * files_per_worker
            end_file_idx = start_file_idx + files_per_worker if worker_id < num_workers - 1 else len(self.datasets)
            
            datasets_to_process = self.datasets[start_file_idx:end_file_idx]
            logger.debug(f"Worker {worker_id}/{num_workers}: processing files {start_file_idx}-{end_file_idx} ({len(datasets_to_process)} files)")
        else:
            datasets_to_process = self.datasets
        
        # Shuffle file order if requested (for better randomization across files)
        if self.shuffle:
            rng = np.random.RandomState(self.random_seed)
            if worker_info is not None:
                # Use worker_id in seed for different shuffling per worker
                seed = (self.random_seed + worker_info.id) if self.random_seed is not None else None
                rng = np.random.RandomState(seed)
            rng.shuffle(datasets_to_process)
        
        # CRITICAL: Use round-robin interleaving instead of sequential processing
        # This ensures each batch contains samples from multiple files
        # Better mixing prevents batch-level correlations
        if len(datasets_to_process) == 1:
            # Single file: just yield all samples (no interleaving needed)
            for sample in datasets_to_process[0]:
                yield sample
        else:
            # Multiple files: interleave samples using round-robin
            # Create iterators for each dataset
            iterators = [iter(dataset) for dataset in datasets_to_process]
            active_iterators = list(range(len(iterators)))
            
            # Round-robin: take one sample from each file in rotation
            # Continue until all files are exhausted
            while active_iterators:
                for i in list(active_iterators):  # Copy list to allow modification during iteration
                    try:
                        sample = next(iterators[i])
                        yield sample
                    except StopIteration:
                        # This file is exhausted, remove it from active list
                        active_iterators.remove(i)
                
                # If we have active iterators, continue round-robin
                # This ensures we cycle through all files until all are exhausted

