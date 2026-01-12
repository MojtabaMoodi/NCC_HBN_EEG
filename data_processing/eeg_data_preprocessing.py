#!/usr/bin/env python3
"""
EEG Data Preprocessing Script

This script processes EEG data from the preprocessed_new directory structure:
- Reads participant directories from multiple releases
- Processes EEG data files (ccd = active task, sus = passive task)
- Handles variable dimensions:
  * (N, 60, 240): Drop first 40 time points, extract all N runs -> N runs of shape (60, 200)
  * (N, 60, 200): Already correct shape, extract all N runs -> N runs of shape (60, 200)
- Creates 1-second, 2-second, and 4-second segments for different model requirements
- Organizes data by task type and participant demographics
- Saves data in HDF5 format with train/val/test splits
- Extracts and saves task numbers and participant IDs as metadata for each segment
"""

import numpy as np
import pandas as pd
import json
import h5py
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
import logging

from constants import get_stratify_label, validate_stratify_by

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SEGMENT_LENGTHS = ['1s', '2s', '4s']
SPLIT_NAMES = ['train', 'val', 'test']
TASK_TYPES = ['active', 'passive']

# Maximum samples per file to prevent HDF5 B-tree performance degradation
# For all segment lengths, we split into multiple files when needed
MAX_SAMPLES_PER_FILE = 100000  # ~100K samples per file for optimal performance
SHUFFLE_BUFFER_SIZE = 25000  # Buffer size for global shuffling (25K samples = ~4.5GB for 4s segments)
# Reduced from 100K to 25K to prevent OOM while still providing good global shuffling

class EEGDataPreprocessor:
    """Class to handle EEG data preprocessing and organization."""
    
    def __init__(self, data_root: str, output_dir: str):
        """
        Initialize the preprocessor.
        
        Args:
            data_root: Path to the preprocessed_new directory
            output_dir: Directory to save HDF5 files
        """
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_demographics(self, participant_dir: Path) -> Dict[str, Any]:
        """
        Load demographic information for a participant.
        
        Args:
            participant_dir: Path to participant directory
            
        Returns:
            Dictionary with demographic information, or None if demographics file doesn't exist
            
        Raises:
            ValueError: If age or gender is missing or invalid when demographics file exists
        """
        demo_file = participant_dir / "demographics.csv"
        if not demo_file.exists():
            logger.warning(f"No demographics file found for {participant_dir.name}")
            return None
            
        try:
            df = pd.read_csv(demo_file)
            if len(df) == 0:
                raise ValueError(f"Empty demographics file for {participant_dir.name}")
                
            # Extract age and gender - these are required
            age = df.iloc[0]['age']
            gender = df.iloc[0]['sex']
            
            # Validate required fields
            if pd.isna(age) or age is None:
                raise ValueError(f"Missing age for {participant_dir.name}")
            if pd.isna(gender) or gender is None:
                raise ValueError(f"Missing gender for {participant_dir.name}")
                
            return {
                'age': float(age),
                'gender': int(gender),  # 1.0 = male, 0.0 = female
                'handedness': float(df.iloc[0]['handedness']) if not pd.isna(df.iloc[0]['handedness']) else None,
                'p_factor': float(df.iloc[0]['p_factor']) if not pd.isna(df.iloc[0]['p_factor']) else None,
                'attention': float(df.iloc[0]['attention']) if not pd.isna(df.iloc[0]['attention']) else None,
                'internalizing': float(df.iloc[0]['internalizing']) if not pd.isna(df.iloc[0]['internalizing']) else None,
                'externalizing': float(df.iloc[0]['externalizing']) if not pd.isna(df.iloc[0]['externalizing']) else None
            }
        except Exception as e:
            logger.error(f"Error loading demographics for {participant_dir.name}: {e}")
            raise
    
    def extract_task_number(self, filename: str) -> Optional[int]:
        """
        Extract task number from filename.
        
        Handles patterns like:
        - sus_1_data_trial_0.npy -> task number 1
        - ccd_2_data_trial_0.npy -> task number 2
        
        Args:
            filename: Filename (with or without extension)
            
        Returns:
            Task number if found, None otherwise
        """
        # Remove extension if present
        stem = Path(filename).stem
        
        # Try to match pattern with number: "sus_1", "ccd_2", etc.
        match = re.match(r'^(sus|ccd)_(\d+)(?:_|$)', stem)
        if match:
            return int(match.group(2))
        
        raise ValueError(f"No task number found in filename: {filename}")
    
    def process_eeg_file_to_segments(self, file_path: Path, segment_length: int) -> Tuple[List[np.ndarray], Optional[int]]:
        """
        Process a single EEG file and create segments of specified length.
        
        This method merges file loading and segment creation:
        1. Loads the .npy file (shape: N, 60, timepoints)
        2. Concatenates all runs along the time axis
        3. Checks if total timepoints are divisible by segment_length * 200
        4. Handles remainder using unified logic for all segment lengths:
           - Calculate remainder = total_timepoints % (segment_length * 200)
           - Calculate half_segment = (segment_length * 200) / 2
           - If remainder < half_segment: drop remainder from beginning
           - If remainder >= half_segment: zero pad at end to complete a full segment
        5. Splits into segments of the specified length
        
        Examples:
            - 1s (segment_samples=200, half=100):
              * If remainder < 100: drop from beginning
              * If remainder >= 100: zero pad at end
            - 2s (segment_samples=400, half=200):
              * If remainder < 200: drop from beginning
              * If remainder >= 200: zero pad at end
            - 4s (segment_samples=800, half=400):
              * If remainder < 400: drop from beginning
              * If remainder >= 400: zero pad at end
        
        Args:
            file_path: Path to the .npy file
            segment_length: Length of segments in seconds (1, 2, or 4)
            
        Returns:
            Tuple of (list of segments, task_number)
            - For 1s: list of (60, 200) arrays
            - For 2s: list of (60, 400) arrays
            - For 4s: list of (60, 800) arrays
        """
        try:
            data = np.load(file_path)
            num_runs, num_channels, num_timepoints = data.shape
            
            # Validate expected dimensions
            if num_channels != 60:
                logger.error(f"Unexpected number of channels {num_channels} (expected 60) for file {file_path}")
                raise ValueError(f"Unexpected number of channels {num_channels} (expected 60) for file {file_path}")
            
            # Extract task number from filename
            task_number = self.extract_task_number(file_path.name)
            
            # Concatenate all runs along time axis
            # Reshape to (num_channels, num_runs * num_timepoints)
            concatenated = data.transpose(1, 0, 2).reshape(num_channels, -1)  # Shape: (60, total_timepoints)
            total_timepoints = concatenated.shape[1]
            segment_samples = segment_length * 200  # 200 for 1s, 400 for 2s, 800 for 4s
            half_segment = segment_samples // 2  # 100 for 1s, 200 for 2s, 400 for 4s
            
            # Check divisibility and handle remainder using unified logic
            remainder = total_timepoints % segment_samples
            
            if remainder > 0:
                if remainder < half_segment:
                    # Drop remainder from beginning
                    concatenated = concatenated[:, remainder:]
                    logger.debug(f"Dropped {remainder} timepoints from beginning for {segment_length}s segments in {file_path.name} (remainder < {half_segment})")
                else:
                    # Zero pad at end to complete a full segment
                    padding_needed = segment_samples - remainder
                    padding = np.zeros((num_channels, padding_needed), dtype=concatenated.dtype)
                    concatenated = np.concatenate([concatenated, padding], axis=1)
                    logger.debug(f"Zero padded {padding_needed} timepoints at the end for {segment_length}s segments in {file_path.name} (remainder >= {half_segment}, remainder was {remainder})")
            
            # Split into segments
            num_segments = concatenated.shape[1] // segment_samples
            segments = []
            for i in range(num_segments):
                segment = concatenated[:, i*segment_samples:(i+1)*segment_samples]
                segments.append(segment)
            
            logger.debug(f"Created {len(segments)} {segment_length}-second segments from {file_path.name}")
            return segments, task_number
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise ValueError(f"Error processing file {file_path}: {e}")
    
    def process_participant(self, participant_dir: Path) -> Dict[str, Any]:
        """
        Process all EEG files for a single participant.
        
        Args:
            participant_dir: Path to participant directory
            
        Returns:
            Dictionary containing processed participant data, or None if demographics missing
            
        Raises:
            ValueError: If age or gender is missing or invalid when demographics file exists
        """
        participant_id = participant_dir.name
        
        # Load demographics - this will return None if file doesn't exist
        demographics = self.load_demographics(participant_dir)
        
        # Skip participant if no demographics
        if demographics is None:
            return None
        
        # Initialize data structures using nested dictionaries
        # Structure: {task_type: {segment_length: {'segments': [], 'task_numbers': []}}}
        data = {
            'active': {1: {'segments': [], 'task_numbers': []},
                      2: {'segments': [], 'task_numbers': []},
                      4: {'segments': [], 'task_numbers': []}},
            'passive': {1: {'segments': [], 'task_numbers': []},
                       2: {'segments': [], 'task_numbers': []},
                       4: {'segments': [], 'task_numbers': []}}
        }
        
        # Process all .npy files
        npy_files = list(participant_dir.glob("*.npy"))

        for file_path in npy_files:
            filename = file_path.stem  # e.g., "ccd_data_trial_0", "ccd_1_data_trial_0", "sus_data_trial_0", "sus_2_data_trial_0"
            
            # Determine task type - check if starts with "ccd" (with or without number) or "sus" (with or without number)
            if filename.startswith("ccd"):
                task_type = "active"
            elif filename.startswith("sus"):
                task_type = "passive"
            else:
                raise ValueError(f"Unknown file type: {filename}")
            
            # Process EEG file to create segments for all lengths (1s, 2s, 4s)
            segment_lengths = [1, 2, 4]
            
            for seg_len in segment_lengths:
                segments, task_number = self.process_eeg_file_to_segments(file_path, segment_length=seg_len)
                if segments:
                    data[task_type][seg_len]['segments'].extend(segments)
                    # Store task number for each segment (same task number for all segments from this file)
                    data[task_type][seg_len]['task_numbers'].extend([task_number] * len(segments))

        
        return {
            'participant_id': participant_id,
            'gender': demographics['gender'],
            'age': demographics['age'],
            'passive_eeg_1s': data['passive'][1]['segments'],  # 1-second segments (list of arrays)
            'active_eeg_1s': data['active'][1]['segments'],    # 1-second segments (list of arrays)
            'passive_eeg_2s': data['passive'][2]['segments'],  # 2-second segments (list of arrays)
            'active_eeg_2s': data['active'][2]['segments'],    # 2-second segments (list of arrays)
            'passive_eeg_4s': data['passive'][4]['segments'],  # 4-second segments (list of arrays)
            'active_eeg_4s': data['active'][4]['segments'],    # 4-second segments (list of arrays)
            'passive_task_numbers_1s': data['passive'][1]['task_numbers'],  # Task numbers for 1s segments
            'active_task_numbers_1s': data['active'][1]['task_numbers'],    # Task numbers for 1s segments
            'passive_task_numbers_2s': data['passive'][2]['task_numbers'],  # Task numbers for 2s segments
            'active_task_numbers_2s': data['active'][2]['task_numbers'],    # Task numbers for 2s segments
            'passive_task_numbers_4s': data['passive'][4]['task_numbers'],  # Task numbers for 4s segments
            'active_task_numbers_4s': data['active'][4]['task_numbers'],    # Task numbers for 4s segments
            'demographics': demographics
        }
    
    def _collect_participant_metadata(self, participant_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Collect only metadata (demographics) for a participant without loading EEG data.
        This is memory-efficient for creating splits.
        
        Args:
            participant_dir: Path to participant directory
            
        Returns:
            Dictionary with participant_id and demographics, or None if demographics missing
        """
        participant_id = participant_dir.name
        demographics = self.load_demographics(participant_dir)
        
        if demographics is None:
            return None
        
        return {
            'participant_id': participant_id,
            'gender': demographics['gender'],
            'age': demographics['age'],
            'demographics': demographics
        }
    
    def process_all_participants(self, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                                 test_ratio: float = 0.15, 
                                 random_seed: int = 42, stratify_by: str = "gender") -> None:
        """
        Process all participants and save in HDF5 format with train/val/test splits.
        
        Each EEG segment is stored with its participant_id in the metadata, enabling
        participant-level isolation or mixing during training.
        
        Uses a two-pass approach to minimize memory usage:
        1. First pass: Collect only metadata to create splits
        2. Second pass: Process participants and save directly to split files
        
        Args:
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            random_seed: Random seed for reproducible splits
            stratify_by: Field to stratify by ("gender", "age", "both")
        """
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
        
        # Validate stratify_by
        validate_stratify_by(stratify_by)
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Get all release directories
        release_dirs = [d for d in self.data_root.iterdir() if d.is_dir() and d.name.startswith("cmi_bids_R")]
        release_dirs.sort()  # Process in order
        
        logger.info(f"Found {len(release_dirs)} release directories")
        
        # Collect all participant directories first
        all_participant_dirs = []
        for release_dir in release_dirs:
            participant_dirs = [d for d in release_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")]
            all_participant_dirs.extend(participant_dirs)
        
        total_participants = len(all_participant_dirs)
        logger.info(f"Found {total_participants} total participants")
        
        # FIRST PASS: Collect only metadata (lightweight, memory-efficient)
        logger.info("First pass: Collecting participant metadata for split creation...")
        participant_metadata = []
        skipped_count = 0
        missing_demographics_count = 0
        
        for participant_dir in tqdm(all_participant_dirs, desc="Collecting metadata", unit="participant"):
            try:
                metadata = self._collect_participant_metadata(participant_dir)
                if metadata is not None:
                    participant_metadata.append(metadata)
                else:
                    skipped_count += 1
                    missing_demographics_count += 1
            except ValueError as e:
                tqdm.write(f"Warning: Skipping participant {participant_dir.name}: {e}")
                skipped_count += 1
            except Exception as e:
                tqdm.write(f"Error: Unexpected error collecting metadata for {participant_dir.name}: {e}")
                skipped_count += 1

        logger.info(f"Metadata collection completed!")
        logger.info(f"Total participants with metadata: {len(participant_metadata)}")
        logger.info(f"Total participants skipped: {skipped_count}")
        logger.info(f"Participants with missing demographics files: {missing_demographics_count}")
        
        if len(participant_metadata) == 0:
            logger.error("No valid participants found!")
            return
        
        # Create train/val/test splits based on metadata
        logger.info("Creating train/val/test splits...")
        participant_ids = [p['participant_id'] for p in participant_metadata]
        stratify_labels = [get_stratify_label(p, stratify_by) for p in participant_metadata]
        
        # Create a mapping from participant_id to label for efficient lookup
        id_to_label = dict(zip(participant_ids, stratify_labels))
        
        # Create mapping from participant_id to participant_dir for second pass
        participant_dir_map = {d.name: d for d in all_participant_dirs}
        
        # Convert to numpy arrays for sklearn compatibility
        participant_ids_arr = np.array(participant_ids)
        stratify_labels_arr = np.array(stratify_labels)
        
        # First split: train+val vs test using StratifiedShuffleSplit
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed)
        train_val_indices, test_indices = next(sss_test.split(participant_ids_arr, stratify_labels_arr))
        test_ids = set(participant_ids_arr[test_indices].tolist())
        
        # Second split: train vs val
        val_size = val_ratio / (train_ratio + val_ratio)
        train_val_labels_arr = stratify_labels_arr[train_val_indices]
        train_val_ids_arr = participant_ids_arr[train_val_indices]
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_seed)
        train_indices, val_indices = next(sss_val.split(train_val_ids_arr, train_val_labels_arr))
        train_ids = set(train_val_ids_arr[train_indices].tolist())
        val_ids = set(train_val_ids_arr[val_indices].tolist())
        
        # Log split sizes and verify ratios
        total_participants = len(participant_ids)
        train_count = len(train_ids)
        val_count = len(val_ids)
        test_count = len(test_ids)
        
        logger.info(f"Train: {train_count} participants ({train_count/total_participants:.1%}, target: {train_ratio:.1%})")
        logger.info(f"Val: {val_count} participants ({val_count/total_participants:.1%}, target: {val_ratio:.1%})")
        logger.info(f"Test: {test_count} participants ({test_count/total_participants:.1%}, target: {test_ratio:.1%})")
        
        # Log stratification distribution
        total_dist = Counter(stratify_labels)
        train_dist = Counter([id_to_label[pid] for pid in train_ids])
        val_dist = Counter([id_to_label[pid] for pid in val_ids])
        test_dist = Counter([id_to_label[pid] for pid in test_ids])
        
        logger.info(f"Stratification distribution (stratify_by='{stratify_by}'):")
        logger.info(f"  Total: {dict(total_dist)}")
        logger.info(f"  Train: {dict(train_dist)}")
        logger.info(f"  Val:   {dict(val_dist)}")
        logger.info(f"  Test:  {dict(test_dist)}")
        
        # Initialize HDF5 files for train/val/test splits (create empty files with structure)
        logger.info("Initializing HDF5 files...")
        for segment_length in SEGMENT_LENGTHS:
            for split_name in SPLIT_NAMES:
                self._initialize_hdf5_file(split_name, segment_length)
        
        # SECOND PASS: Process participants and save directly to split files (memory-efficient)
        logger.info("Second pass: Processing participants and saving to split files...")
        processed_count = 0
        
        for participant_id in tqdm(participant_ids, desc="Processing participants", unit="participant"):
            # Safety check: ensure participant directory exists in map
            if participant_id not in participant_dir_map:
                tqdm.write(f"Warning: Participant {participant_id} not found in directory map. Skipping.")
                continue
            
            participant_dir = participant_dir_map[participant_id]
            
            try:
                # Process participant (loads EEG data)
                participant_data = self.process_participant(participant_dir)
                if participant_data is None:
                    tqdm.write(f"Warning: Participant {participant_id} returned None from process_participant. Skipping.")
                    continue
                
                # Determine which split this participant belongs to
                if participant_id in train_ids:
                    split_name = 'train'
                elif participant_id in val_ids:
                    split_name = 'val'
                elif participant_id in test_ids:
                    split_name = 'test'
                else:
                    # This should never happen, but log a warning if it does
                    tqdm.write(f"Warning: Participant {participant_id} not found in any train/val/test split. Skipping.")
                    continue
                
                # Save to split file (participant_id is stored with each segment in metadata)
                for segment_length in SEGMENT_LENGTHS:
                    self._append_participant_to_hdf5(participant_data, split_name, segment_length)
                processed_count += 1
                
            except Exception as e:
                tqdm.write(f"Error processing participant {participant_id}: {e}")
                continue
        
        # Update participant counts in HDF5 files
        logger.info("Updating participant counts in HDF5 files...")
        for segment_length in SEGMENT_LENGTHS:
            for split_name in SPLIT_NAMES:
                self._update_participant_count(split_name, segment_length)
        
        logger.info(f"Processing completed! Processed {processed_count} participants.")
        logger.info("All splits saved successfully!")
    
    @staticmethod
    def _get_segment_keys(segment_length: str) -> Tuple[str, str, str, str]:
        """
        Get the dictionary keys for EEG data and task numbers based on segment length.
        
        Args:
            segment_length: Length of segments ('1s', '2s', or '4s')
            
        Returns:
            Tuple of (active_key, passive_key, active_task_key, passive_task_key)
        """
        if segment_length == '1s':
            return ('active_eeg_1s', 'passive_eeg_1s', 'active_task_numbers_1s', 'passive_task_numbers_1s')
        elif segment_length == '2s':
            return ('active_eeg_2s', 'passive_eeg_2s', 'active_task_numbers_2s', 'passive_task_numbers_2s')
        elif segment_length == '4s':
            return ('active_eeg_4s', 'passive_eeg_4s', 'active_task_numbers_4s', 'passive_task_numbers_4s')
        else:
            raise ValueError(f"Invalid segment_length: {segment_length}. Must be '1s', '2s', or '4s'")
    
    def _get_hdf5_file_path(self, split_name: str, segment_length: str, part_idx: Optional[int] = None) -> Path:
        """
        Get the HDF5 file path for a given split and segment length.
        
        Args:
            split_name: Name of the split ('train', 'val', 'test')
            segment_length: Length of segments ('1s', '2s', or '4s')
            part_idx: Part index for multi-file splits (None for single file or first part)
            
        Returns:
            Path to the HDF5 file
        """
        if part_idx is not None:
            hdf5_filename = f"eeg_data_{split_name}_{segment_length}_part{part_idx:02d}.h5"
        else:
            hdf5_filename = f"eeg_data_{split_name}_{segment_length}.h5"
        return self.output_dir / hdf5_filename
    
    def _initialize_hdf5_file(self, split_name: str, segment_length: str) -> None:
        """
        Initialize an empty HDF5 file with the proper structure.
        
        Args:
            split_name: Name of the split ('train', 'val', 'test')
            segment_length: Length of segments ('1s', '2s', or '4s')
        """
        hdf5_path = self._get_hdf5_file_path(split_name, segment_length)
        with h5py.File(hdf5_path, 'w') as f:
            f.create_group('active')
            f.create_group('passive')
            f.attrs['split_name'] = split_name
            f.attrs['segment_length'] = segment_length
            f.attrs['num_participants'] = 0
            f.attrs['total_samples'] = 0
    
    def _initialize_hdf5_file_part(self, split_name: str, segment_length: str, part_idx: int) -> None:
        """
        Initialize an empty HDF5 file part with the proper structure.
        
        Args:
            split_name: Name of the split ('train', 'val', 'test')
            segment_length: Length of segments ('1s', '2s', or '4s')
            part_idx: Part index for multi-file splits
        """
        hdf5_path = self._get_hdf5_file_path(split_name, segment_length, part_idx=part_idx)
        with h5py.File(hdf5_path, 'w') as f:
            f.create_group('active')
            f.create_group('passive')
            f.attrs['split_name'] = split_name
            f.attrs['segment_length'] = segment_length
            f.attrs['part_idx'] = part_idx
            f.attrs['num_participants'] = 0
            f.attrs['total_samples'] = 0
    
    def _update_participant_count(self, split_name: str, segment_length: str) -> None:
        """
        Update the participant count in HDF5 file(s) by counting unique participants.
        For all segment lengths with multiple files, counts across all part files.
        
        Uses a two-pass approach:
        1. First pass: Collect all unique participants across all part files
        2. Second pass: Update all files with the final total count
        
        Args:
            split_name: Name of the split ('train', 'val', 'test')
            segment_length: Length of segments ('1s', '2s', or '4s')
        """
        # FIRST PASS: Collect all unique participants across all part files
        participant_set = set()
        part_files = []  # Track which files exist
        part_idx = 0
        
        while True:
            hdf5_path = self._get_hdf5_file_path(split_name, segment_length, part_idx=part_idx)
            if not hdf5_path.exists():
                break
            
            part_files.append(hdf5_path)
            
            try:
                with h5py.File(hdf5_path, 'r') as f:
                    for group_name in TASK_TYPES:
                        if group_name in f:
                            for key in f[group_name].keys():
                                if key.startswith('metadata_'):
                                    metadata = f[group_name][key]
                                    pid = metadata.attrs['participant_id']
                                    if isinstance(pid, bytes):
                                        pid = pid.decode()
                                    participant_set.add(str(pid))
            except Exception as e:
                logger.error(f"Error reading participant data from {hdf5_path}: {e}")
                raise RuntimeError(f"Failed to read participant data from {hdf5_path}") from e
            
            part_idx += 1
        
        if not part_files:
            logger.warning(f"No HDF5 files found for {split_name}/{segment_length}, skipping participant count update")
            return
        
        total_participants = len(participant_set)
        
        # SECOND PASS: Update all files with the final total count
        for hdf5_path in part_files:
            try:
                with h5py.File(hdf5_path, 'r+') as f:
                    f.attrs['num_participants'] = total_participants
            except Exception as e:
                logger.error(f"Error updating participant count in {hdf5_path}: {e}")
                raise RuntimeError(f"Failed to update participant count in {hdf5_path}") from e
        
        logger.debug(f"Updated participant count for {split_name}/{segment_length}: {total_participants} unique participants across {len(part_files)} file(s)")
    
    def _get_current_hdf5_file_and_part(self, split_name: str, segment_length: str, 
                                       current_sample_idx: int) -> Tuple[Path, int]:
        """
        Get the current HDF5 file path and part index based on sample count.
        Rotates to next file when MAX_SAMPLES_PER_FILE is reached for all segment lengths.
        
        Args:
            split_name: Name of the split ('train', 'val', 'test')
            segment_length: Length of segments ('1s', '2s', or '4s')
            current_sample_idx: Current sample index across all files
            
        Returns:
            Tuple of (file_path, part_idx)
        """
        # For all segment lengths, split into multiple files when needed
        part_idx = current_sample_idx // MAX_SAMPLES_PER_FILE
        file_path = self._get_hdf5_file_path(split_name, segment_length, part_idx=part_idx)
        
        return file_path, part_idx
    
    def _get_current_global_sample_idx_for_task(self, split_name: str, segment_length: str, task_type: str) -> int:
        """
        Get the current global sample index for a specific task_type across all part files.
        Counts samples in the specified task_type group.
        
        Args:
            split_name: Name of the split ('train', 'val', 'test')
            segment_length: Length of segments ('1s', '2s', or '4s')
            task_type: Task type ('active' or 'passive')
            
        Returns:
            Total number of samples for this task_type across all part files
        """
        total_samples = 0
        part_idx = 0
        
        while True:
            file_path = self._get_hdf5_file_path(split_name, segment_length, part_idx=part_idx)
            if not file_path.exists():
                break
            
            try:
                with h5py.File(file_path, 'r') as f:
                    if task_type not in f:
                        # Task group doesn't exist in this file, we're done
                        break
                    
                    task_group = f[task_type]
                    # Count samples in this task group
                    file_samples = sum(1 for key in task_group.keys() if key.startswith('sample_'))
                    total_samples += file_samples
                    
                    # If file has less than MAX_SAMPLES_PER_FILE for this task, we're done
                    if file_samples < MAX_SAMPLES_PER_FILE:
                        break
            except (OSError, BlockingIOError) as e:
                # File access error - log and stop counting (file may be locked or corrupted)
                # This is acceptable during initialization counting, but we log it explicitly
                logger.warning(
                    f"Error accessing {file_path} for sample count: {e}. "
                    f"Stopping count at {total_samples} samples. "
                    f"This may indicate a file locking issue or corrupted file. "
                    f"Count may be incomplete."
                )
                break
            except Exception as e:
                # Unexpected error - log and re-raise (don't silently ignore)
                logger.error(f"Unexpected error reading {file_path} for sample count: {e}")
                raise RuntimeError(f"Failed to read sample count from {file_path}") from e
            
            part_idx += 1
        
        return total_samples
    
    def _get_current_total_sample_idx(self, split_name: str, segment_length: str) -> int:
        """
        Get the current total sample index across all task types for file rotation.
        This counts samples from both 'active' and 'passive' task types.
        
        CRITICAL: This is used for file rotation to ensure MAX_SAMPLES_PER_FILE
        is enforced across ALL task types, not per task type.
        
        Args:
            split_name: Name of the split ('train', 'val', 'test')
            segment_length: Length of segments ('1s', '2s', or '4s')
            
        Returns:
            Total number of samples across all task types for file rotation
        """
        total_samples = 0
        part_idx = 0
        
        while True:
            file_path = self._get_hdf5_file_path(split_name, segment_length, part_idx=part_idx)
            if not file_path.exists():
                break
            
            try:
                with h5py.File(file_path, 'r') as f:
                    # Count samples from both task types
                    file_total_samples = 0
                    for task_type in TASK_TYPES:
                        if task_type in f:
                            task_group = f[task_type]
                            task_samples = sum(1 for key in task_group.keys() if key.startswith('sample_'))
                            file_total_samples += task_samples
                    
                    total_samples += file_total_samples
                    
                    # If file has less than MAX_SAMPLES_PER_FILE total, we're done
                    if file_total_samples < MAX_SAMPLES_PER_FILE:
                        break
            except (OSError, BlockingIOError) as e:
                # File access error - log and stop counting (file may be locked or corrupted)
                logger.warning(
                    f"Error accessing {file_path} for total sample count: {e}. "
                    f"Stopping count at {total_samples} samples. "
                    f"This may indicate a file locking issue or corrupted file. "
                    f"Count may be incomplete."
                )
                break
            except Exception as e:
                # Unexpected error - log and re-raise (don't silently ignore)
                logger.error(f"Unexpected error reading {file_path} for total sample count: {e}")
                raise RuntimeError(f"Failed to read total sample count from {file_path}") from e
            
            part_idx += 1
        
        return total_samples
    
    def _flush_sample_buffer(self, buffer: List[Dict[str, Any]], split_name: str, segment_length: str,
                             task_type: str, sample_counters: Dict[str, int], total_counters: Dict[str, int],
                             random_seed: int) -> None:
        """
        Shuffle and save samples from buffer to HDF5 files.
        
        CRITICAL: This method performs GLOBAL shuffling across all participants in the buffer,
        ensuring samples from different participants are mixed before saving.
        This prevents participant-level clustering that can cause model collapse.
        
        Args:
            buffer: List of sample dictionaries, each containing:
                - 'segment': numpy array (EEG data)
                - 'task_number': int
                - 'participant_data': dict with participant metadata
            split_name: Name of the split ('train', 'val', 'test')
            segment_length: Length of segments ('1s', '2s', or '4s')
            task_type: Task type ('active' or 'passive')
            sample_counters: Dictionary tracking per-task-type sample counts (for local indexing)
            total_counters: Dictionary tracking total sample counts per split/segment_length (for file rotation)
            random_seed: Random seed for shuffling
        """
        if len(buffer) == 0:
            return
        
        # GLOBAL SHUFFLE: Shuffle all samples in buffer across all participants
        # This ensures samples from different participants are mixed
        rng = np.random.RandomState(random_seed)
        indices = np.arange(len(buffer))
        rng.shuffle(indices)
        
        # Get current counters
        counter_key = f"{split_name}_{segment_length}_{task_type}"
        if counter_key not in sample_counters:
            raise KeyError(
                f"Missing counter key '{counter_key}' in sample_counters. "
                f"This indicates a logic error in counter initialization."
            )
        current_global_idx = sample_counters[counter_key]
        
        total_key = f"{split_name}_{segment_length}_total"
        if total_key not in total_counters:
            raise KeyError(
                f"Missing total counter key '{total_key}' in total_counters. "
                f"This indicates a logic error in counter initialization."
            )
        current_total_idx = total_counters[total_key]
        
        # Save shuffled samples
        for idx in indices:
            sample = buffer[idx]
            
            # Get current file path based on total_idx (for file rotation)
            file_path, part_idx = self._get_current_hdf5_file_and_part(
                split_name, segment_length, current_total_idx
            )
            
            # Initialize file if it doesn't exist
            if not file_path.exists():
                self._initialize_hdf5_file_part(split_name, segment_length, part_idx)
            
            with h5py.File(file_path, 'r+') as f:
                # Ensure task group exists
                if task_type not in f:
                    f.create_group(task_type)
                task_group = f[task_type]
                
                # Get local sample index within this file for this task type
                local_sample_idx = current_global_idx % MAX_SAMPLES_PER_FILE
                
                # Save segment
                dataset_name = f'sample_{local_sample_idx:06d}'
                
                # Check if dataset already exists (shouldn't happen, but check for logic errors)
                if dataset_name in task_group:
                    error_msg = (
                        f"Dataset {dataset_name} already exists in {file_path}. "
                        f"Participant: {sample['participant_data']['participant_id']}, "
                        f"Task type: {task_type}, "
                        f"Sample index (task): {current_global_idx}, "
                        f"Total index: {current_total_idx}, "
                        f"Local index: {local_sample_idx}. "
                        f"This indicates a logic error in sample indexing."
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                task_group.create_dataset(dataset_name, data=sample['segment'], compression=None)
                
                # Store metadata
                sample_group = task_group.create_group(f'metadata_{local_sample_idx:06d}')
                sample_group.attrs['participant_id'] = sample['participant_data']['participant_id']
                sample_group.attrs['task_type'] = task_type
                sample_group.attrs['task_number'] = sample['task_number']
            
            # Increment both counters
            current_global_idx += 1
            current_total_idx += 1
        
        # Update counters
        sample_counters[counter_key] = current_global_idx
        total_counters[total_key] = current_total_idx
        
        # Clear buffer
        buffer.clear()
    
    def _add_samples_to_buffer(self, segments: List[np.ndarray], task_numbers: List[int],
                               participant_data: Dict[str, Any], split_indices: np.ndarray,
                               buffer: List[Dict[str, Any]]) -> None:
        """
        Add samples to buffer for later global shuffling.
        
        Args:
            segments: List of EEG segment arrays
            task_numbers: List of task numbers corresponding to segments
            participant_data: Dictionary containing participant metadata
            split_indices: Indices of segments to add to buffer
            buffer: Buffer list to append samples to
        """
        for idx in split_indices:
            buffer.append({
                'segment': segments[idx],
                'task_number': task_numbers[idx],
                'participant_data': participant_data
            })
    
    def _save_samples_to_multiple_files(self, split_name: str, segment_length: str, task_type: str,
                                        segments: List[np.ndarray], task_numbers: List[int],
                                        split_indices: np.ndarray, participant_data: Dict[str, Any],
                                        current_global_idx: int, current_total_idx: int) -> int:
        """
        Save samples to multiple HDF5 files for all segment lengths.
        Rotates to next file when MAX_SAMPLES_PER_FILE is reached (based on total samples across all task types).
        
        Args:
            split_name: Name of the split ('train', 'val', 'test')
            segment_length: Length of segments ('1s', '2s', or '4s')
            task_type: Task type ('active' or 'passive')
            segments: List of EEG segment arrays
            task_numbers: List of task numbers
            split_indices: Indices of segments to save
            participant_data: Dictionary containing participant metadata
            current_global_idx: Current global sample index for this task_type (for local indexing)
            current_total_idx: Current total sample index across all task types (for file rotation)
            
        Returns:
            Number of samples saved
        """
        
        # Save each segment, rotating files as needed
        for idx in split_indices:
            # CRITICAL: Use total_idx for file rotation to ensure MAX_SAMPLES_PER_FILE
            # is enforced across ALL task types, not per task type
            file_path, part_idx = self._get_current_hdf5_file_and_part(
                split_name, segment_length, current_total_idx
            )
            
            # Initialize file if it doesn't exist
            if not file_path.exists():
                self._initialize_hdf5_file_part(split_name, segment_length, part_idx)
            
            with h5py.File(file_path, 'r+') as f:
                # Ensure task group exists (should exist from initialization, but check to be safe)
                if task_type not in f:
                    f.create_group(task_type)
                task_group = f[task_type]
                
                # Get local sample index within this file for this task type
                # Use current_global_idx (per-task-type) for local indexing within the task group
                local_sample_idx = current_global_idx % MAX_SAMPLES_PER_FILE
                
                # Save segment
                dataset_name = f'sample_{local_sample_idx:06d}'
                
                # Check if dataset already exists (shouldn't happen in sequential processing)
                # If it does exist, this indicates a serious logic error (duplicate sample indices)
                if dataset_name in task_group:
                    error_msg = (
                        f"Dataset {dataset_name} already exists in {file_path}. "
                        f"Participant: {participant_data['participant_id']}, "
                        f"Task type: {task_type}, "
                        f"Sample index (task): {current_global_idx}, "
                        f"Total index: {current_total_idx}, "
                        f"Local index: {local_sample_idx}. "
                        f"This indicates a logic error in sample indexing - duplicate indices detected. "
                        f"This should not happen in sequential processing."
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                task_group.create_dataset(dataset_name, data=segments[idx], compression=None)
                
                # Store metadata
                sample_group = task_group.create_group(f'metadata_{local_sample_idx:06d}')
                sample_group.attrs['participant_id'] = participant_data['participant_id']
                sample_group.attrs['task_type'] = task_type
                sample_group.attrs['task_number'] = task_numbers[idx]
            
            # Increment both counters
            current_global_idx += 1  # Per-task-type counter for local indexing
            current_total_idx += 1    # Total counter for file rotation
        
        # Return number of samples saved
        return len(split_indices)
    
    def _split_and_save_participant_samples(self, segments: List[np.ndarray], task_numbers: List[int],
                                            participant_data: Dict[str, Any], task_type: str,
                                            segment_length: str, train_ratio: float, val_ratio: float,
                                            test_ratio: float, random_seed: int,
                                            sample_counters: Dict[str, int], 
                                            total_counters: Dict[str, int],
                                            buffers: Dict[str, List[Dict[str, Any]]],
                                            buffer_flush_counts: Dict[str, int]) -> Tuple[int, int, int]:
        """
        Split a participant's samples and add to buffers for global shuffling.
        
        CRITICAL: Samples are added to buffers instead of being saved immediately.
        Buffers are flushed when they reach SHUFFLE_BUFFER_SIZE, performing global
        shuffling across all participants. This prevents participant-level clustering
        that can cause model collapse during training.
        
        Args:
            segments: List of EEG segment arrays for this participant and task type
            task_numbers: List of task numbers corresponding to segments
            participant_data: Dictionary containing participant metadata
            task_type: Task type ('active' or 'passive')
            segment_length: Length of segments ('1s', '2s', or '4s')
            train_ratio: Ratio for training split
            val_ratio: Ratio for validation split
            test_ratio: Ratio for test split
            random_seed: Random seed for shuffling
            sample_counters: Dictionary tracking per-task-type sample counts (for local indexing)
            total_counters: Dictionary tracking total sample counts per split/segment_length (for file rotation)
            buffers: Dictionary of buffers for each split/segment_length/task_type combination
            buffer_flush_counts: Dictionary tracking flush count per buffer (for seed variation)
            
        Returns:
            Tuple of (train_samples_added, val_samples_added, test_samples_added)
        """
        if len(segments) == 0:
            return (0, 0, 0)
        
        # Verify segments and task_numbers have same length
        if len(segments) != len(task_numbers):
            raise ValueError(
                f"Data inconsistency for participant {participant_data['participant_id']}, "
                f"task_type {task_type}, segment_length {segment_length}: "
                f"segments has {len(segments)} items but task_numbers has {len(task_numbers)} items."
            )
        
        # Create indices and shuffle
        num_samples = len(segments)
        indices = np.arange(num_samples)
        rng = np.random.RandomState(random_seed)
        rng.shuffle(indices)
        
        # Calculate split sizes
        train_size = int(num_samples * train_ratio)
        val_size = int(num_samples * val_ratio)
        # test_size = remaining samples
        
        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Add samples to buffers for global shuffling (instead of saving immediately)
        # This prevents participant-level clustering that can cause model collapse
        samples_added = {}
        for split_name, split_indices in [('train', train_indices), ('val', val_indices), ('test', test_indices)]:
            buffer_key = f"{split_name}_{segment_length}_{task_type}"
            if buffer_key not in buffers:
                buffers[buffer_key] = []
            
            # Add samples to buffer
            self._add_samples_to_buffer(
                segments=segments,
                task_numbers=task_numbers,
                participant_data=participant_data,
                split_indices=split_indices,
                buffer=buffers[buffer_key]
            )
            
            samples_added[split_name] = len(split_indices)
            
            # Flush buffer if it reaches SHUFFLE_BUFFER_SIZE
            if len(buffers[buffer_key]) >= SHUFFLE_BUFFER_SIZE:
                # Vary random seed per buffer flush for better randomization
                # Use buffer key hash + flush count to ensure different seeds per flush
                # Ensure seed is within valid range [0, 2^32 - 1] for numpy RandomState
                buffer_hash = abs(hash(buffer_key)) % (2**32)
                flush_count = buffer_flush_counts.get(buffer_key, 0)
                flush_seed = (random_seed + buffer_hash + flush_count) % (2**32)
                buffer_flush_counts[buffer_key] = flush_count + 1
                
                self._flush_sample_buffer(
                    buffer=buffers[buffer_key],
                    split_name=split_name,
                    segment_length=segment_length,
                    task_type=task_type,
                    sample_counters=sample_counters,
                    total_counters=total_counters,
                    random_seed=flush_seed
                )
        
        # Verify all splits are present (should always be the case, but check for logic errors)
        if 'train' not in samples_added or 'val' not in samples_added or 'test' not in samples_added:
            raise RuntimeError(
                f"Missing split in samples_added: {samples_added}. "
                f"This indicates a logic error - all splits should always be present."
            )
        
        return samples_added['train'], samples_added['val'], samples_added['test']
    
    def _save_participant_id_mapping(self, participant_id_to_class_idx: Dict[str, int]) -> None:
        """
        Save participant_id to class_idx mapping to a JSON file.
        
        Args:
            participant_id_to_class_idx: Dictionary mapping participant_id to class index
        """
        # Sort by class index for consistency
        sorted_mapping = dict(sorted(participant_id_to_class_idx.items(), key=lambda x: x[1]))
        
        mapping_file = self.output_dir / 'participant_id_to_class_idx.json'
        with open(mapping_file, 'w') as f:
            json.dump(sorted_mapping, f, indent=2)
        
        logger.info(f"Saved participant_id to class_idx mapping to: {mapping_file}")
        logger.info(f"Total number of user classes: {len(sorted_mapping)}")
    
    def process_participant_user_identification(self, participant_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Process all EEG files for a single participant for user identification.
        This version does not require demographics - uses default values if missing.
        
        Args:
            participant_dir: Path to participant directory
            
        Returns:
            Dictionary containing processed participant data, or None if no valid data found
        """
        participant_id = participant_dir.name
        
        # Try to load demographics, but don't require it for user identification
        demographics = self.load_demographics(participant_dir)
        
        # Use explicit default values if demographics missing
        # These values are explicitly chosen to indicate "unknown" and are NOT silently assumed
        if demographics is None:
            logger.debug(f"No demographics found for participant {participant_id}, using explicit default values (gender=-1, age=-1.0)")
            demographics = {
                'gender': -1,  # Explicitly set to -1 to indicate "unknown" (not a valid gender value)
                'age': -1.0,   # Explicitly set to -1.0 to indicate "unknown" (not a valid age value)
            }
        
        # Initialize data structures using nested dictionaries
        # Structure: {task_type: {segment_length: {'segments': [], 'task_numbers': []}}}
        data = {
            'active': {1: {'segments': [], 'task_numbers': []},
                      2: {'segments': [], 'task_numbers': []},
                      4: {'segments': [], 'task_numbers': []}},
            'passive': {1: {'segments': [], 'task_numbers': []},
                       2: {'segments': [], 'task_numbers': []},
                       4: {'segments': [], 'task_numbers': []}}
        }
        
        # Process all .npy files
        npy_files = list(participant_dir.glob("*.npy"))
        
        if not npy_files:
            logger.warning(f"No .npy files found for participant {participant_id}")
            return None

        for file_path in npy_files:
            filename = file_path.stem  # e.g., "ccd_data_trial_0", "ccd_1_data_trial_0", "sus_data_trial_0", "sus_2_data_trial_0"
            
            # Determine task type - check if starts with "ccd" (with or without number) or "sus" (with or without number)
            if filename.startswith("ccd"):
                task_type = "active"
            elif filename.startswith("sus"):
                task_type = "passive"
            else:
                logger.warning(f"Unknown file type: {filename} for participant {participant_id}. Skipping.")
                continue
            
            # Process EEG file to create segments for all lengths (1s, 2s, 4s)
            segment_lengths = [1, 2, 4]
            
            for seg_len in segment_lengths:
                try:
                    segments, task_number = self.process_eeg_file_to_segments(file_path, segment_length=seg_len)
                    if segments:
                        data[task_type][seg_len]['segments'].extend(segments)
                        # Store task number for each segment (same task number for all segments from this file)
                        data[task_type][seg_len]['task_numbers'].extend([task_number] * len(segments))
                except Exception as e:
                    # Log error but continue processing other segment lengths for this file
                    # This is acceptable since we want to process as much data as possible
                    logger.warning(
                        f"Error processing file {file_path.name} for participant {participant_id}, "
                        f"segment_length {seg_len}s: {e}. Continuing with other segment lengths."
                    )
                    import traceback
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                    continue
        
        # Check if we have any data
        has_data = any(
            len(data[task_type][seg_len]['segments']) > 0
            for task_type in ['active', 'passive']
            for seg_len in [1, 2, 4]
        )
        
        if not has_data:
            logger.warning(f"No valid EEG segments found for participant {participant_id}")
            return None
        
        return {
            'participant_id': participant_id,
            'gender': demographics['gender'],
            'age': demographics['age'],
            'passive_eeg_1s': data['passive'][1]['segments'],
            'active_eeg_1s': data['active'][1]['segments'],
            'passive_eeg_2s': data['passive'][2]['segments'],
            'active_eeg_2s': data['active'][2]['segments'],
            'passive_eeg_4s': data['passive'][4]['segments'],
            'active_eeg_4s': data['active'][4]['segments'],
            'passive_task_numbers_1s': data['passive'][1]['task_numbers'],
            'active_task_numbers_1s': data['active'][1]['task_numbers'],
            'passive_task_numbers_2s': data['passive'][2]['task_numbers'],
            'active_task_numbers_2s': data['active'][2]['task_numbers'],
            'passive_task_numbers_4s': data['passive'][4]['task_numbers'],
            'active_task_numbers_4s': data['active'][4]['task_numbers'],
            'demographics': demographics
        }
    
    def process_all_participants_user_identification(self, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                                                      test_ratio: float = 0.15, 
                                                      random_seed: int = 42) -> None:
        """
        Process all participants for user identification task.
        
        CRITICAL: This method splits each participant's data at the SAMPLE level (not participant level):
        - 70% of each participant's samples -> train
        - 15% of each participant's samples -> val
        - 15% of each participant's samples -> test
        
        This ensures ALL participants appear in ALL splits, which is correct for closed-set user identification:
        - Model learns features for all 3145 users during training
        - Validation/test check if model can identify these known users on new samples
        - This is different from open-set identification where test users are completely unseen
        
        Args:
            train_ratio: Ratio of each participant's samples for training (default: 0.7)
            val_ratio: Ratio of each participant's samples for validation (default: 0.15)
            test_ratio: Ratio of each participant's samples for testing (default: 0.15)
            random_seed: Random seed for reproducible splits
        """
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Get all release directories
        release_dirs = [d for d in self.data_root.iterdir() if d.is_dir() and d.name.startswith("cmi_bids_R")]
        release_dirs.sort()  # Process in order
        
        logger.info(f"Found {len(release_dirs)} release directories")
        
        # Collect all participant directories
        all_participant_dirs = []
        for release_dir in release_dirs:
            participant_dirs = [d for d in release_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")]
            all_participant_dirs.extend(participant_dirs)
        
        total_participants = len(all_participant_dirs)
        logger.info(f"Found {total_participants} total participants")
        
        # For user identification, we process ALL participants regardless of demographics
        logger.info("Processing all participants for user identification")
        logger.info(f"Total participants to process: {total_participants}")
        logger.info("CRITICAL: All participants will appear in ALL splits (train/val/test)")
        logger.info("  This allows model to learn features for all users during training")
        
        if total_participants == 0:
            raise ValueError("No participants found!")
        
        # Initialize HDF5 files for train/val/test splits
        # For all segment lengths, we'll create multiple files to improve performance
        logger.info("Initializing HDF5 files...")
        for segment_length in SEGMENT_LENGTHS:
            for split_name in SPLIT_NAMES:
                # For all segment lengths, initialize first part file
                self._initialize_hdf5_file_part(split_name, segment_length, part_idx=0)
        
        # Sequential processing (parallel processing removed due to HDF5 file locking issues)
        logger.info("Processing participants sequentially...")
        processed_count = 0
        participant_id_to_class_idx = {}
        next_class_idx = 0
        
        # Track current sample indices per split/segment_length/task_type to avoid expensive recounting
        # This is much faster than counting all samples from scratch for each participant
        # CRITICAL: We need both per-task-type counters (for local indexing) and total counters (for file rotation)
        sample_counters = {}
        total_counters = {}  # Track total samples per split/segment_length across all task types
        
        # Initialize buffers for global shuffling (prevents participant-level clustering)
        # Key format: "{split_name}_{segment_length}_{task_type}"
        buffers = {}
        # Track flush count per buffer for seed variation
        buffer_flush_counts = {}
        
        for segment_length in SEGMENT_LENGTHS:
            for split_name in SPLIT_NAMES:
                # Initialize total counter for file rotation (across all task types)
                total_key = f"{split_name}_{segment_length}_total"
                total_counters[total_key] = self._get_current_total_sample_idx(
                    split_name, segment_length
                )
                
                # Initialize per-task-type counters for local indexing
                for task_type in TASK_TYPES:
                    key = f"{split_name}_{segment_length}_{task_type}"
                    # Initialize by counting existing samples (only once at start)
                    sample_counters[key] = self._get_current_global_sample_idx_for_task(
                        split_name, segment_length, task_type
                    )
                    # Initialize buffer for this split/segment_length/task_type
                    buffers[key] = []
                    # Initialize flush count for seed variation
                    buffer_flush_counts[key] = 0
        
        for participant_dir in tqdm(all_participant_dirs, desc="Processing participants", unit="participant"):
            participant_id = participant_dir.name
            
            try:
                # Process participant for user identification
                participant_data = self.process_participant_user_identification(participant_dir)
                if participant_data is None:
                    tqdm.write(f"Warning: Participant {participant_id} returned None. Skipping.")
                    continue
                
                # Assign class index to this participant (sorted by participant_id for consistency)
                if participant_id not in participant_id_to_class_idx:
                    participant_id_to_class_idx[participant_id] = next_class_idx
                    next_class_idx += 1
                else:
                    raise ValueError(f"Participant {participant_id} already has a class index. This should not happen.")
                
                # Split each participant's samples across train/val/test
                # Process each segment length separately
                for segment_length in SEGMENT_LENGTHS:
                    active_key, passive_key, active_task_key, passive_task_key = self._get_segment_keys(segment_length)
                    
                    # Split active task samples
                    # process_participant_user_identification always returns these keys, so raise error if missing
                    if active_key not in participant_data:
                        raise KeyError(
                            f"Missing key '{active_key}' in participant_data for participant {participant_id}. "
                            f"This indicates a bug in process_participant_user_identification."
                        )
                    if active_task_key not in participant_data:
                        raise KeyError(
                            f"Missing key '{active_task_key}' in participant_data for participant {participant_id}. "
                            f"This indicates a bug in process_participant_user_identification."
                        )
                    active_segments = participant_data[active_key]
                    active_task_numbers = participant_data[active_task_key]
                    if active_segments:
                        self._split_and_save_participant_samples(
                            segments=active_segments,
                            task_numbers=active_task_numbers,
                            participant_data=participant_data,
                            task_type='active',
                            segment_length=segment_length,
                            train_ratio=train_ratio,
                            val_ratio=val_ratio,
                            test_ratio=test_ratio,
                            random_seed=random_seed,
                            sample_counters=sample_counters,
                            total_counters=total_counters,
                            buffers=buffers,
                            buffer_flush_counts=buffer_flush_counts
                        )
                    
                    # Split passive task samples
                    # process_participant_user_identification always returns these keys, so raise error if missing
                    if passive_key not in participant_data:
                        raise KeyError(
                            f"Missing key '{passive_key}' in participant_data for participant {participant_id}. "
                            f"This indicates a bug in process_participant_user_identification."
                        )
                    if passive_task_key not in participant_data:
                        raise KeyError(
                            f"Missing key '{passive_task_key}' in participant_data for participant {participant_id}. "
                            f"This indicates a bug in process_participant_user_identification."
                        )
                    passive_segments = participant_data[passive_key]
                    passive_task_numbers = participant_data[passive_task_key]
                    if passive_segments:
                        self._split_and_save_participant_samples(
                            segments=passive_segments,
                            task_numbers=passive_task_numbers,
                            participant_data=participant_data,
                            task_type='passive',
                            segment_length=segment_length,
                            train_ratio=train_ratio,
                            val_ratio=val_ratio,
                            test_ratio=test_ratio,
                            random_seed=random_seed,
                            sample_counters=sample_counters,
                            total_counters=total_counters,
                            buffers=buffers,
                            buffer_flush_counts=buffer_flush_counts
                        )
                
                processed_count += 1
                
            except Exception as e:
                tqdm.write(f"Error processing participant {participant_id}: {e}")
                import traceback
                tqdm.write(traceback.format_exc())
                continue
        
        # Flush all remaining buffers (samples that didn't reach SHUFFLE_BUFFER_SIZE)
        logger.info("Flushing remaining sample buffers with global shuffling...")
        for buffer_key, buffer in buffers.items():
            if len(buffer) > 0:
                # Parse buffer key: "{split_name}_{segment_length}_{task_type}"
                # Example: "train_4s_active" -> split_name="train", segment_length="4s", task_type="active"
                parts = buffer_key.split('_')
                if len(parts) >= 3:
                    split_name = parts[0]
                    # segment_length is "1s", "2s", or "4s" (single part)
                    segment_length = parts[1]
                    task_type = parts[2]
                    
                    # Vary random seed per buffer flush for better randomization
                    # Use buffer key hash + flush count to ensure different seeds per flush
                    # Ensure seed is within valid range [0, 2^32 - 1] for numpy RandomState
                    buffer_hash = abs(hash(buffer_key)) % (2**32)
                    flush_count = buffer_flush_counts.get(buffer_key, 0)
                    flush_seed = (random_seed + buffer_hash + flush_count) % (2**32)
                    buffer_flush_counts[buffer_key] = flush_count + 1
                    
                    self._flush_sample_buffer(
                        buffer=buffer,
                        split_name=split_name,
                        segment_length=segment_length,
                        task_type=task_type,
                        sample_counters=sample_counters,
                        total_counters=total_counters,
                        random_seed=flush_seed
                    )
        
        # Save participant_id to class_idx mapping
        self._save_participant_id_mapping(participant_id_to_class_idx)
        
        # Update participant counts in HDF5 files
        logger.info("Updating participant counts in HDF5 files...")
        for segment_length in SEGMENT_LENGTHS:
            for split_name in SPLIT_NAMES:
                self._update_participant_count(split_name, segment_length)
        
        logger.info(f"Processing completed! Processed {processed_count} participants.")
        logger.info(f"Total number of user classes: {len(participant_id_to_class_idx)}")
        logger.info("All splits saved successfully with global shuffling to prevent participant-level clustering!")
    
    def _save_segments_to_hdf5_group(self, group: h5py.Group, eeg_list: List[np.ndarray], 
                                     task_numbers: List[int], participant_data: Dict[str, Any],
                                     task_type: str, segment_length: str, start_sample_idx: int) -> int:
        """
        Save segments to an HDF5 group and return the next sample index.
        
        Args:
            group: HDF5 group to save to ('active' or 'passive')
            eeg_list: List of EEG segment arrays
            task_numbers: List of task numbers corresponding to segments
            participant_data: Dictionary containing participant metadata
            task_type: Task type ('active' or 'passive')
            segment_length: Length of segments ('1s', '2s', or '4s')
            start_sample_idx: Starting sample index
            
        Returns:
            Next sample index after saving all segments
        """
        participant_id = participant_data['participant_id']
        
        # Verify that eeg_list and task_numbers have the same length
        if len(eeg_list) != len(task_numbers):
            raise ValueError(
                f"Data inconsistency for participant {participant_id}, task_type {task_type}, "
                f"segment_length {segment_length}: eeg_list has {len(eeg_list)} segments but "
                f"task_numbers has {len(task_numbers)} entries. They must have the same length."
            )
        
        sample_idx = start_sample_idx
        for seg_idx, eeg_segment in enumerate(eeg_list):
            dataset_name = f'sample_{sample_idx:06d}'
            # No compression for faster data loading during training
            # Trade-off: ~8-10% larger files but 3-5x faster loading
            group.create_dataset(dataset_name, data=eeg_segment, compression=None)
            
            # Store metadata for this sample
            sample_group = group.create_group(f'metadata_{sample_idx:06d}')
            sample_group.attrs['participant_id'] = participant_id
            sample_group.attrs['gender'] = participant_data['gender']
            sample_group.attrs['age'] = participant_data['age']
            sample_group.attrs['task_type'] = task_type
            
            # Get task number for this segment
            task_num = task_numbers[seg_idx]
            if task_num is None:
                raise ValueError(
                    f"Data inconsistency for participant {participant_id}, task_type {task_type}, "
                    f"segment_length {segment_length}, segment {seg_idx}: task_number is None"
                )
            
            sample_group.attrs['task_number'] = task_num
            sample_idx += 1
        
        return sample_idx
    
    def _append_participant_to_hdf5(self, participant_data: Dict[str, Any], 
                                    split_name: str, segment_length: str) -> None:
        """
        Append a single participant's data to an HDF5 file.
        This is memory-efficient as it processes one participant at a time.
        
        Each segment is stored with participant_id in its metadata, enabling
        participant-level filtering during training (isolation or mixing).
        
        Args:
            participant_data: Dictionary containing processed participant data
            split_name: Name of the split ('train', 'val', 'test')
            segment_length: Length of segments ('1s', '2s', or '4s')
        """
        hdf5_path = self._get_hdf5_file_path(split_name, segment_length)
        
        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        
        active_key, passive_key, active_task_key, passive_task_key = self._get_segment_keys(segment_length)
        
        with h5py.File(hdf5_path, 'r+') as f:
            active_group = f['active']
            passive_group = f['passive']
            
            # Get current sample index from total_samples attribute
            # Default to 0 if attribute doesn't exist (newly created file)
            if 'total_samples' not in f.attrs:
                logger.debug(f"total_samples attribute not found in {hdf5_path}, defaulting to 0 (new file)")
            sample_idx = f.attrs.get('total_samples', 0)
            
            # Process active and passive tasks
            # process_participant always returns these keys, so raise error if missing
            for task_type, group, eeg_key, task_key in [
                ('active', active_group, active_key, active_task_key),
                ('passive', passive_group, passive_key, passive_task_key)
            ]:
                if eeg_key not in participant_data:
                    raise KeyError(
                        f"Missing key '{eeg_key}' in participant_data. "
                        f"This indicates a bug in process_participant."
                    )
                if task_key not in participant_data:
                    raise KeyError(
                        f"Missing key '{task_key}' in participant_data. "
                        f"This indicates a bug in process_participant."
                    )
                eeg_list = participant_data[eeg_key]
                task_numbers = participant_data[task_key]
                
                sample_idx = self._save_segments_to_hdf5_group(
                    group, eeg_list, task_numbers, participant_data,
                    task_type, segment_length, sample_idx
                )
            
            # Update total_samples attribute
            f.attrs['total_samples'] = sample_idx

def main():
    """Main function to run the preprocessing pipeline."""
    data_root = "/home/mojtabam/projects/aip-aghodsib/mojtabam/preprocessed_new"
    output_dir = "/home/mojtabam/projects/aip-aghodsib/mojtabam/EEG/data_processing/processed_eeg_data_hdf5_no_compression"
    
    # Record start time
    start_time = time.time()
    
    # Create preprocessor
    preprocessor = EEGDataPreprocessor(data_root, output_dir)
    
    # Process all participants and create train/val/test splits
    logger.info("Starting EEG data preprocessing...")
    logger.info(f"Data will be saved to: {output_dir}")
    logger.info("Creating 1s, 2s, and 4s segments with train/val/test splits")
    logger.info("Each segment will have participant_id stored in metadata for participant-level isolation/mixing")
    preprocessor.process_all_participants(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        stratify_by="gender"
    )
    
    # Calculate and log elapsed time
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    logger.info("Preprocessing completed successfully!")
    logger.info(f"All data saved to: {output_dir}")
    logger.info(f"Total processing time: {hours:02d}:{minutes:02d}:{seconds:02d} ({elapsed_time:.2f} seconds)")

if __name__ == "__main__":
    main()
