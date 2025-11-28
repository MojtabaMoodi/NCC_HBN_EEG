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
    
    def _get_hdf5_file_path(self, split_name: str, segment_length: str) -> Path:
        """
        Get the HDF5 file path for a given split and segment length.
        
        Args:
            split_name: Name of the split ('train', 'val', 'test')
            segment_length: Length of segments ('1s', '2s', or '4s')
            
        Returns:
            Path to the HDF5 file
        """
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
    
    def _update_participant_count(self, split_name: str, segment_length: str) -> None:
        """
        Update the participant count in an HDF5 file by counting unique participants.
        
        Args:
            split_name: Name of the split ('train', 'val', 'test')
            segment_length: Length of segments ('1s', '2s', or '4s')
        """
        hdf5_path = self._get_hdf5_file_path(split_name, segment_length)
        if not hdf5_path.exists():
            return
        
        with h5py.File(hdf5_path, 'r+') as f:
            participant_set = set()
            for group_name in TASK_TYPES:
                if group_name in f:
                    for key in f[group_name].keys():
                        if key.startswith('metadata_'):
                            metadata = f[group_name][key]
                            pid = metadata.attrs['participant_id']
                            if isinstance(pid, bytes):
                                pid = pid.decode()
                            participant_set.add(str(pid))
            f.attrs['num_participants'] = len(participant_set)
    
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
            sample_idx = f.attrs.get('total_samples', 0)
            
            # Process active and passive tasks
            for task_type, group, eeg_key, task_key in [
                ('active', active_group, active_key, active_task_key),
                ('passive', passive_group, passive_key, passive_task_key)
            ]:
                eeg_list = participant_data.get(eeg_key, [])
                task_numbers = participant_data.get(task_key, [])
                
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
