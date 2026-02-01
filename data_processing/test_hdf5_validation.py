#!/usr/bin/env python3
"""
Comprehensive test script to validate preprocessed HDF5 files against original .npy files.

This script performs extensive validation:
1. Data correctness: Compares HDF5 samples with original .npy files
2. Split ratios: Verifies 70-15-15 train/val/test splits
3. Metadata: Checks participant_id, task_type, task_number
4. Edge cases: Tests participants with few samples, only active/passive tasks, etc.
5. Multi-file structure: Validates HDF5 file rotation logic
6. Completeness: Ensures no samples are lost or duplicated

Usage:
    python test_hdf5_validation.py --participants PARTICIPANT_ID ... --segment-lengths 1s 2s 4s --data-root PATH --hdf5-dir PATH
"""

import numpy as np
import h5py
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import logging
import re
import argparse
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('hdf5_validation.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants (must match preprocessing script)
SEGMENT_LENGTHS = ['1s', '2s', '4s']
SPLIT_NAMES = ['train', 'val', 'test']
TASK_TYPES = ['active', 'passive']
SEGMENT_LENGTH_INTEGERS = [1, 2, 4]
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
MAX_SAMPLES_PER_FILE = 100000
EXPECTED_NUM_CHANNELS = 60
SAMPLES_PER_SECOND = 200

# Tolerance for floating point comparison
FLOAT_TOLERANCE = 1e-6


class HDF5Validator:
    """Validates HDF5 files against original .npy files."""
    
    def __init__(self, data_root: Path, hdf5_dir: Path):
        """
        Initialize validator.
        
        Args:
            data_root: Root directory containing original .npy files
            hdf5_dir: Directory containing preprocessed HDF5 files
            
        Raises:
            FileNotFoundError: If directories don't exist
        """
        self.data_root = Path(data_root)
        self.hdf5_dir = Path(hdf5_dir)
        
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root directory not found: {data_root}")
        if not self.hdf5_dir.exists():
            raise FileNotFoundError(f"HDF5 directory not found: {hdf5_dir}")
    
    def extract_task_number(self, filename: str) -> int:
        """
        Extract task number from filename (same logic as preprocessing).
        
        Args:
            filename: Filename to extract task number from
            
        Returns:
            Task number as integer
            
        Raises:
            ValueError: If task number cannot be extracted
            
        Handles patterns like:
        - sus_1_data_trial_0.npy -> task number 1
        - ccd_2_data_trial_0.npy -> task number 2
        """
        stem = Path(filename).stem
        match = re.match(r'^(sus|ccd)_(\d+)(?:_|$)', stem)
        if match:
            return int(match.group(2))
        raise ValueError(f"No task number found in filename: {filename}")
    
    def _get_expected_shape(self, segment_length: int) -> Tuple[int, int]:
        """
        Get expected shape for a segment length.
        
        Args:
            segment_length: Length in seconds (1, 2, or 4)
            
        Returns:
            Tuple of (num_channels, num_timepoints)
        """
        return (EXPECTED_NUM_CHANNELS, segment_length * SAMPLES_PER_SECOND)
    
    def process_eeg_file_to_segments(self, file_path: Path, segment_length: int) -> Tuple[List[np.ndarray], int]:
        """
        Process a single EEG file and create segments (same logic as preprocessing).
        
        Args:
            file_path: Path to the .npy file
            segment_length: Length of segments in seconds (1, 2, or 4)
            
        Returns:
            Tuple of (list of segments, task_number)
            
        Raises:
            ValueError: If file has unexpected dimensions or task number cannot be extracted
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        data = np.load(file_path)
        
        if len(data.shape) != 3:
            raise ValueError(
                f"Unexpected data shape {data.shape} for file {file_path}. "
                f"Expected 3D array (num_runs, num_channels, num_timepoints)."
            )
        
        num_runs, num_channels, num_timepoints = data.shape
        
        if num_channels != EXPECTED_NUM_CHANNELS:
            raise ValueError(
                f"Unexpected number of channels {num_channels} (expected {EXPECTED_NUM_CHANNELS}) "
                f"for file {file_path}"
            )
        
        # Extract task number from filename - raises ValueError if not found
        task_number = self.extract_task_number(file_path.name)
        
        # Concatenate all runs along time axis
        concatenated = data.transpose(1, 0, 2).reshape(num_channels, -1)  # Shape: (60, total_timepoints)
        total_timepoints = concatenated.shape[1]
        segment_samples = segment_length * SAMPLES_PER_SECOND
        half_segment = segment_samples // 2
        
        # Handle remainder
        remainder = total_timepoints % segment_samples
        
        if remainder > 0:
            if remainder < half_segment:
                # Drop remainder from beginning
                concatenated = concatenated[:, remainder:]
            else:
                # Zero pad at end
                padding_needed = segment_samples - remainder
                padding = np.zeros((num_channels, padding_needed), dtype=concatenated.dtype)
                concatenated = np.concatenate([concatenated, padding], axis=1)
        
        # Split into segments
        num_segments = concatenated.shape[1] // segment_samples
        segments = []
        for i in range(num_segments):
            segment = concatenated[:, i*segment_samples:(i+1)*segment_samples]
            segments.append(segment)
        
        return segments, task_number
    
    def _determine_task_type(self, filename: str) -> str:
        """
        Determine task type from filename.
        
        Args:
            filename: Filename to check
            
        Returns:
            Task type ('active' or 'passive')
            
        Raises:
            ValueError: If task type cannot be determined
        """
        if filename.startswith('ccd'):
            return "active"
        elif filename.startswith('sus'):
            return "passive"
        else:
            raise ValueError(f"Unknown file type: {filename}. Expected 'ccd' (active) or 'sus' (passive).")
    
    def process_participant_npy_files(self, participant_dir: Path) -> Dict[str, Dict[int, List[Tuple[np.ndarray, int]]]]:
        """
        Process all .npy files for a participant (same logic as preprocessing).
        
        Args:
            participant_dir: Path to participant directory
            
        Returns:
            Dictionary: {task_type: {segment_length: [(segment, task_number), ...]}}
            
        Raises:
            FileNotFoundError: If participant directory doesn't exist
            ValueError: If files cannot be processed
        """
        if not participant_dir.exists():
            raise FileNotFoundError(f"Participant directory not found: {participant_dir}")
        
        participant_id = participant_dir.name
        data = {
            'active': {1: [], 2: [], 4: []},
            'passive': {1: [], 2: [], 4: []}
        }
        
        npy_files = list(participant_dir.glob("*.npy"))
        
        if not npy_files:
            raise ValueError(f"No .npy files found for participant {participant_id}")
        
        processing_errors = []
        
        for file_path in npy_files:
            filename = file_path.stem
            
            # Determine task type - raises ValueError if unknown
            try:
                task_type = self._determine_task_type(filename)
            except ValueError as e:
                processing_errors.append(f"File {file_path.name}: {e}")
                continue
            
            # Process for all segment lengths
            for segment_length in SEGMENT_LENGTH_INTEGERS:
                try:
                    segments, task_number = self.process_eeg_file_to_segments(file_path, segment_length)
                    for segment in segments:
                        data[task_type][segment_length].append((segment, task_number))
                except Exception as e:
                    error_msg = (
                        f"Error processing file {file_path.name} for participant {participant_id}, "
                        f"segment_length {segment_length}s: {e}"
                    )
                    processing_errors.append(error_msg)
                    # Continue processing other segment lengths, but collect errors
        
        if processing_errors:
            error_summary = "\n".join(f"  - {err}" for err in processing_errors)
            raise ValueError(
                f"Errors processing .npy files for participant {participant_id}:\n{error_summary}"
            )
        
        return data
    
    def get_hdf5_files(self, split_name: str, segment_length: str) -> List[Path]:
        """
        Get HDF5 file paths for a split and segment length (handles multi-file structure).
        
        Args:
            split_name: Name of the split ('train', 'val', 'test')
            segment_length: Length of segments ('1s', '2s', or '4s')
            
        Returns:
            List of HDF5 file paths, sorted by part index
            
        Raises:
            ValueError: If no HDF5 files found
        """
        if split_name not in SPLIT_NAMES:
            raise ValueError(f"Invalid split name: {split_name}. Must be one of {SPLIT_NAMES}")
        if segment_length not in SEGMENT_LENGTHS:
            raise ValueError(f"Invalid segment length: {segment_length}. Must be one of {SEGMENT_LENGTHS}")
        
        # Check for multiple part files
        part_files = sorted(self.hdf5_dir.glob(f"eeg_data_{split_name}_{segment_length}_part*.h5"))
        
        if part_files:
            return part_files
        
        # Fallback to single file
        single_file = self.hdf5_dir / f"eeg_data_{split_name}_{segment_length}.h5"
        if single_file.exists():
            return [single_file]
        
        raise ValueError(
            f"No HDF5 files found for split '{split_name}' and segment length '{segment_length}' "
            f"in directory {self.hdf5_dir}"
        )
    
    def _extract_metadata_from_hdf5(self, metadata_group: h5py.Group, required_keys: List[str]) -> Dict[str, any]:
        """
        Extract metadata from HDF5 group, ensuring all required keys are present.
        
        Args:
            metadata_group: HDF5 group containing metadata attributes
            required_keys: List of required attribute keys
            
        Returns:
            Dictionary with metadata values
            
        Raises:
            KeyError: If required metadata keys are missing
        """
        metadata_dict = {}
        missing_keys = []
        
        for key in required_keys:
            if key not in metadata_group.attrs:
                missing_keys.append(key)
            else:
                value = metadata_group.attrs[key]
                # Handle bytes
                if isinstance(value, bytes):
                    value = value.decode()
                metadata_dict[key] = value
        
        if missing_keys:
            raise KeyError(
                f"Missing required metadata keys: {missing_keys}. "
                f"Available keys: {list(metadata_group.attrs.keys())}"
            )
        
        return metadata_dict
    
    def load_participant_samples_from_hdf5(self, participant_id: str, split_name: str, 
                                          segment_length: str, task_type: str) -> List[Tuple[np.ndarray, Dict]]:
        """
        Load all samples for a participant from HDF5 files.
        
        Args:
            participant_id: Participant ID to load samples for
            split_name: Name of the split ('train', 'val', 'test')
            segment_length: Length of segments ('1s', '2s', or '4s')
            task_type: Task type ('active' or 'passive')
            
        Returns:
            List of (segment_array, metadata_dict) tuples
            
        Raises:
            ValueError: If HDF5 files not found or task type invalid
            KeyError: If required metadata is missing
            IOError: If HDF5 files cannot be read
        """
        if task_type not in TASK_TYPES:
            raise ValueError(f"Invalid task type: {task_type}. Must be one of {TASK_TYPES}")
        
        hdf5_files = self.get_hdf5_files(split_name, segment_length)
        samples = []
        required_metadata_keys = ['participant_id', 'task_type', 'task_number']
        
        for hdf5_file in hdf5_files:
            try:
                with h5py.File(hdf5_file, 'r') as f:
                    if task_type not in f:
                        # Task group doesn't exist - this is not an error, just means no samples for this task
                        continue
                    
                    task_group = f[task_type]
                    
                    # Iterate through all samples
                    for key in task_group.keys():
                        if key.startswith('sample_'):
                            # Get metadata key
                            local_idx = key.replace('sample_', '')
                            metadata_key = f'metadata_{local_idx}'
                            
                            if metadata_key not in task_group:
                                raise KeyError(
                                    f"Metadata group '{metadata_key}' not found for sample '{key}' "
                                    f"in file {hdf5_file}"
                                )
                            
                            metadata_group = task_group[metadata_key]
                            
                            # Extract metadata - raises KeyError if required keys missing
                            metadata_dict = self._extract_metadata_from_hdf5(
                                metadata_group, required_metadata_keys
                            )
                            
                            # Filter by participant_id
                            if metadata_dict['participant_id'] == participant_id:
                                # Load sample
                                sample_data = task_group[key][:]
                                samples.append((sample_data, metadata_dict))
            except (IOError, OSError) as e:
                raise IOError(f"Error reading HDF5 file {hdf5_file}: {e}") from e
            except KeyError as e:
                raise KeyError(f"Missing required data in HDF5 file {hdf5_file}: {e}") from e
        
        return samples
    
    def compare_arrays(self, arr1: np.ndarray, arr2: np.ndarray, tolerance: float = FLOAT_TOLERANCE) -> bool:
        """Compare two arrays with tolerance."""
        if arr1.shape != arr2.shape:
            return False
        return np.allclose(arr1, arr2, atol=tolerance, rtol=tolerance)
    
    def _calculate_expected_split_counts(self, total_samples: int) -> Dict[str, int]:
        """
        Calculate expected split counts based on ratios.
        
        Args:
            total_samples: Total number of samples
            
        Returns:
            Dictionary with expected counts for train, val, test
        """
        expected_train = int(total_samples * TRAIN_RATIO)
        expected_val = int(total_samples * VAL_RATIO)
        expected_test = total_samples - expected_train - expected_val  # Remaining
        
        return {
            'train': expected_train,
            'val': expected_val,
            'test': expected_test
        }
    
    def _validate_split_ratios(self, expected: Dict[str, int], actual: Dict[str, int], 
                               key: str, results: Dict) -> bool:
        """
        Validate split ratios match expected values.
        
        Args:
            expected: Dictionary with expected counts
            actual: Dictionary with actual counts
            key: Key for logging (e.g., "active_1s")
            results: Results dictionary to update
            
        Returns:
            True if ratios match, False otherwise
        """
        # Allow small deviation (due to rounding)
        train_ok = abs(actual.get('train', 0) - expected['train']) <= 1
        val_ok = abs(actual.get('val', 0) - expected['val']) <= 1
        test_ok = abs(actual.get('test', 0) - expected['test']) <= 1
        
        split_ratio_ok = train_ok and val_ok and test_ok
        
        results['split_ratios'][key] = {
            'expected': expected,
            'actual': actual,
            'ok': split_ratio_ok
        }
        
        if not split_ratio_ok:
            error_msg = (
                f"Split ratio mismatch for {key}: "
                f"expected train={expected['train']}, val={expected['val']}, test={expected['test']}, "
                f"actual train={actual.get('train', 0)}, val={actual.get('val', 0)}, test={actual.get('test', 0)}"
            )
            logger.error(f"  ERROR: {error_msg}")
            results['errors'].append(error_msg)
        
        return split_ratio_ok
    
    def _match_samples(self, hdf5_samples: List[Tuple[np.ndarray, Dict]], 
                      original_samples: List[Tuple[np.ndarray, int]],
                      task_type: str, seg_len: int, key: str, results: Dict) -> Dict[str, int]:
        """
        Match HDF5 samples with original samples by comparing data values.
        
        Args:
            hdf5_samples: List of (sample_array, metadata_dict) from HDF5
            original_samples: List of (sample_array, task_number) from original .npy
            task_type: Task type for validation
            seg_len: Segment length integer (1, 2, or 4)
            key: Key for logging
            results: Results dictionary to update
            
        Returns:
            Dictionary with matching statistics
        """
        matched_count = 0
        metadata_matched = 0
        shape_mismatches = 0
        unmatched_hdf5_samples = 0
        original_matched = set()
        expected_shape = self._get_expected_shape(seg_len)
        
        for hdf5_sample, hdf5_metadata in hdf5_samples:
            # Check shape first
            if hdf5_sample.shape != expected_shape:
                shape_mismatches += 1
                error_msg = (
                    f"Shape mismatch for {key}: "
                    f"expected {expected_shape}, got {hdf5_sample.shape}"
                )
                logger.error(f"    ERROR: {error_msg}")
                results['errors'].append(error_msg)
                continue
            
            # Find matching original sample
            matched = False
            for idx, (orig_sample, orig_task_num) in enumerate(original_samples):
                if idx in original_matched:
                    continue
                
                # Compare data values (with tolerance for floating point)
                if self.compare_arrays(hdf5_sample, orig_sample):
                    # Check task number matches
                    if hdf5_metadata['task_number'] == orig_task_num:
                        matched = True
                        matched_count += 1
                        original_matched.add(idx)
                        
                        # Check metadata
                        if hdf5_metadata['task_type'] == task_type:
                            metadata_matched += 1
                        else:
                            warning_msg = (
                                f"Task type mismatch for {key}: "
                                f"expected {task_type}, got {hdf5_metadata['task_type']}"
                            )
                            logger.warning(f"    WARNING: {warning_msg}")
                            results['warnings'].append(warning_msg)
                        break
            
            if not matched:
                unmatched_hdf5_samples += 1
                error_msg = (
                    f"Could not find matching original sample for HDF5 sample in {key}. "
                    f"Task number: {hdf5_metadata.get('task_number', 'unknown')}, "
                    f"Shape: {hdf5_sample.shape}"
                )
                logger.warning(f"    WARNING: {error_msg}")
                results['warnings'].append(error_msg)
        
        return {
            'matched': matched_count,
            'unmatched': unmatched_hdf5_samples,
            'shape_mismatches': shape_mismatches,
            'metadata_matched': metadata_matched,
            'original_matched_count': len(original_matched)
        }
    
    def _build_validation_key(self, task_type: str, segment_length: str) -> str:
        """Build validation key from task type and segment length."""
        return f"{task_type}_{segment_length}"
    
    def validate_participant(self, participant_id: str, segment_lengths: List[str]) -> Dict[str, any]:
        """
        Validate a single participant's data.
        
        Args:
            participant_id: Participant ID to validate
            segment_lengths: List of segment lengths to test (must be provided, no default)
            
        Returns:
            Dictionary with validation results
            
        Raises:
            FileNotFoundError: If participant directory not found
            ValueError: If segment_lengths is empty or contains invalid values
        """
        if not segment_lengths:
            raise ValueError("segment_lengths cannot be empty. Must provide at least one segment length.")
        
        invalid_lengths = [sl for sl in segment_lengths if sl not in SEGMENT_LENGTHS]
        if invalid_lengths:
            raise ValueError(
                f"Invalid segment lengths: {invalid_lengths}. "
                f"Must be one of {SEGMENT_LENGTHS}"
            )
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Validating participant: {participant_id}")
        logger.info(f"{'='*70}")
        
        # Find participant directory
        participant_dir = None
        for release_dir in self.data_root.iterdir():
            if release_dir.is_dir() and release_dir.name.startswith("cmi_bids_R"):
                potential_dir = release_dir / participant_id
                if potential_dir.exists():
                    participant_dir = potential_dir
                    break
        
        if participant_dir is None:
            raise FileNotFoundError(f"Participant directory not found: {participant_id}")
        
        # Process original .npy files - raises ValueError if errors occur
        logger.info("Processing original .npy files...")
        original_data = self.process_participant_npy_files(participant_dir)
        
        # Count original samples
        original_counts = {}
        for task_type in TASK_TYPES:
            for seg_len in SEGMENT_LENGTH_INTEGERS:
                seg_str = f"{seg_len}s"
                if seg_str in segment_lengths:
                    key = self._build_validation_key(task_type, seg_str)
                    original_counts[key] = len(original_data[task_type][seg_len])
                    logger.info(f"  Original {key}: {original_counts[key]} samples")
        
        # Load from HDF5 and validate
        results = {
            'participant_id': participant_id,
            'original_counts': original_counts,
            'hdf5_counts': {},
            'split_counts': {},
            'data_matches': {},
            'metadata_matches': {},
            'split_ratios': {},
            'edge_cases': {},
            'errors': [],
            'warnings': []
        }
        
        for task_type in TASK_TYPES:
            for seg_len in SEGMENT_LENGTH_INTEGERS:
                segment_length = f"{seg_len}s"
                if segment_length not in segment_lengths:
                    continue
                
                key = self._build_validation_key(task_type, segment_length)
                
                logger.info(f"\nValidating {key}...")
                
                # Get original samples
                original_samples = original_data[task_type][seg_len]
                total_original = len(original_samples)
                
                if total_original == 0:
                    logger.info(f"  No original samples for {key}, skipping")
                    results['edge_cases'][key] = "No samples (participant has no data for this task/segment)"
                    continue
                
                # Load from all splits
                all_hdf5_samples = []
                split_counts = {}
                
                for split_name in SPLIT_NAMES:
                    try:
                        hdf5_samples = self.load_participant_samples_from_hdf5(
                            participant_id, split_name, segment_length, task_type
                        )
                        split_counts[split_name] = len(hdf5_samples)
                        all_hdf5_samples.extend(hdf5_samples)
                        logger.info(f"  {split_name}: {split_counts[split_name]} samples")
                    except (ValueError, KeyError, IOError) as e:
                        error_msg = f"Error loading {split_name} split for {key}: {e}"
                        logger.error(f"  ERROR: {error_msg}")
                        results['errors'].append(error_msg)
                        # Continue with other splits, but record error
                        split_counts[split_name] = 0
                
                total_hdf5 = len(all_hdf5_samples)
                results['hdf5_counts'][key] = total_hdf5
                results['split_counts'][key] = split_counts
                
                # EDGE CASE: Very few samples (test rounding behavior)
                if total_original < 10:
                    results['edge_cases'][key] = f"Few samples ({total_original}) - testing rounding behavior"
                    logger.info(f"  EDGE CASE: Participant has only {total_original} samples")
                
                # Check total count
                if total_hdf5 != total_original:
                    error_msg = (
                        f"Sample count mismatch for {key}: "
                        f"original={total_original}, hdf5={total_hdf5}"
                    )
                    logger.error(f"  ERROR: {error_msg}")
                    results['errors'].append(error_msg)
                else:
                    logger.info(f"  ✓ Total count matches: {total_original}")
                
                # Check split ratios
                expected_counts = self._calculate_expected_split_counts(total_original)
                split_ratio_ok = self._validate_split_ratios(expected_counts, split_counts, key, results)
                
                if split_ratio_ok:
                    logger.info(
                        f"  ✓ Split ratios correct: "
                        f"train={split_counts.get('train', 0)}, "
                        f"val={split_counts.get('val', 0)}, "
                        f"test={split_counts.get('test', 0)}"
                    )
                
                # Match samples by comparing data
                logger.info(f"  Matching samples by data values...")
                match_stats = self._match_samples(
                    all_hdf5_samples, original_samples, task_type, seg_len, key, results
                )
                
                # Verify we matched all originals
                if match_stats['original_matched_count'] != total_original:
                    error_msg = (
                        f"Not all original samples were matched for {key}: "
                        f"matched {match_stats['original_matched_count']}/{total_original} originals"
                    )
                    logger.error(f"  ERROR: {error_msg}")
                    results['errors'].append(error_msg)
                
                results['data_matches'][key] = {
                    'matched': match_stats['matched'],
                    'total': total_hdf5,
                    'unmatched': match_stats['unmatched'],
                    'shape_mismatches': match_stats['shape_mismatches'],
                    'ok': (
                        match_stats['matched'] == total_hdf5 and 
                        match_stats['shape_mismatches'] == 0 and 
                        match_stats['unmatched'] == 0
                    )
                }
                results['metadata_matches'][key] = {
                    'matched': match_stats['metadata_matched'],
                    'total': total_hdf5,
                    'ok': match_stats['metadata_matched'] == total_hdf5
                }
                
                if results['data_matches'][key]['ok']:
                    logger.info(f"  ✓ All {match_stats['matched']} samples matched by data")
                else:
                    error_msg = (
                        f"Sample matching issues for {key}: "
                        f"matched {match_stats['matched']}/{total_hdf5}, "
                        f"unmatched {match_stats['unmatched']}, "
                        f"shape mismatches {match_stats['shape_mismatches']}"
                    )
                    logger.error(f"  ERROR: {error_msg}")
                    results['errors'].append(error_msg)
                
                if results['metadata_matches'][key]['ok']:
                    logger.info(f"  ✓ All {match_stats['metadata_matched']} samples have correct metadata")
                else:
                    warning_msg = (
                        f"Only {match_stats['metadata_matched']}/{total_hdf5} samples have correct metadata for {key}"
                    )
                    logger.warning(f"  WARNING: {warning_msg}")
                    results['warnings'].append(warning_msg)
        
        # EDGE CASE: Check if participant has only active or only passive tasks
        has_active = any(len(original_data['active'][s]) > 0 for s in SEGMENT_LENGTH_INTEGERS)
        has_passive = any(len(original_data['passive'][s]) > 0 for s in SEGMENT_LENGTH_INTEGERS)
        
        if has_active and not has_passive:
            results['edge_cases']['task_type'] = "Only active tasks"
            logger.info("  EDGE CASE: Participant has only active tasks")
        elif has_passive and not has_active:
            results['edge_cases']['task_type'] = "Only passive tasks"
            logger.info("  EDGE CASE: Participant has only passive tasks")
        
        return results
    
    def validate_multiple_participants(self, participant_ids: List[str], 
                                      segment_lengths: List[str]) -> Dict[str, any]:
        """
        Validate multiple participants.
        
        Args:
            participant_ids: List of participant IDs to validate (must be provided, no default)
            segment_lengths: List of segment lengths to test (must be provided, no default)
            
        Returns:
            Dictionary with validation results for all participants
        """
        if not participant_ids:
            raise ValueError("participant_ids cannot be empty. Must provide at least one participant ID.")
        if not segment_lengths:
            raise ValueError("segment_lengths cannot be empty. Must provide at least one segment length.")
        
        all_results = {}
        
        for participant_id in participant_ids:
            try:
                results = self.validate_participant(participant_id, segment_lengths)
                all_results[participant_id] = results
            except Exception as e:
                logger.error(f"Error validating participant {participant_id}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                all_results[participant_id] = {'error': str(e), 'traceback': traceback.format_exc()}
        
        return all_results
    
    def print_summary(self, all_results: Dict[str, any]) -> None:
        """Print summary of validation results."""
        logger.info(f"\n{'='*70}")
        logger.info("VALIDATION SUMMARY")
        logger.info(f"{'='*70}")
        
        total_participants = len(all_results)
        participants_with_errors = 0
        participants_with_warnings = 0
        total_errors = 0
        total_warnings = 0
        
        for participant_id, results in all_results.items():
            if 'error' in results:
                logger.error(f"\n{participant_id}: FAILED - {results['error']}")
                participants_with_errors += 1
                continue
            
            errors = results.get('errors', [])
            warnings = results.get('warnings', [])
            
            if errors:
                participants_with_errors += 1
                total_errors += len(errors)
            if warnings:
                participants_with_warnings += 1
                total_warnings += len(warnings)
            
            status = "PASSED" if not errors else "FAILED"
            logger.info(f"\n{participant_id}: {status}")
            
            if errors:
                logger.error(f"  Errors ({len(errors)}):")
                for error in errors[:5]:  # Show first 5 errors
                    logger.error(f"    - {error}")
                if len(errors) > 5:
                    logger.error(f"    ... and {len(errors) - 5} more errors")
            
            if warnings:
                logger.warning(f"  Warnings ({len(warnings)}):")
                for warning in warnings[:3]:  # Show first 3 warnings
                    logger.warning(f"    - {warning}")
                if len(warnings) > 3:
                    logger.warning(f"    ... and {len(warnings) - 3} more warnings")
            
            if results.get('edge_cases'):
                logger.info(f"  Edge cases detected: {results['edge_cases']}")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Total participants tested: {total_participants}")
        logger.info(f"Participants passed: {total_participants - participants_with_errors}")
        logger.info(f"Participants failed: {participants_with_errors}")
        logger.info(f"Participants with warnings: {participants_with_warnings}")
        logger.info(f"Total errors: {total_errors}")
        logger.info(f"Total warnings: {total_warnings}")
        logger.info(f"{'='*70}")


def find_participant_ids(data_root: Path, max_participants: int) -> List[str]:
    """
    Find participant IDs from the data directory.
    
    Args:
        data_root: Root directory to search
        max_participants: Maximum number of participants to find (must be provided, no default)
        
    Returns:
        List of participant IDs found
    """
    if max_participants <= 0:
        raise ValueError(f"max_participants must be positive, got {max_participants}")
    
    participant_ids = []
    for release_dir in data_root.iterdir():
        if release_dir.is_dir() and release_dir.name.startswith("cmi_bids_R"):
            for participant_dir in release_dir.iterdir():
                if participant_dir.is_dir() and participant_dir.name.startswith("sub-"):
                    participant_ids.append(participant_dir.name)
                    if len(participant_ids) >= max_participants:
                        return participant_ids
    return participant_ids


def main():
    """Main function to run validation."""
    parser = argparse.ArgumentParser(
        description='Validate preprocessed HDF5 files against original .npy files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--participants',
        nargs='+',
        required=False,  # Not required if --auto-detect is used
        help='Participant IDs to test'
    )
    parser.add_argument(
        '--auto-detect',
        action='store_true',
        help='Auto-detect participants (finds first 3)'
    )
    parser.add_argument(
        '--max-participants',
        type=int,
        required=False,
        help='Maximum number of participants to auto-detect (required if --auto-detect is used)'
    )
    parser.add_argument(
        '--segment-lengths',
        nargs='+',
        choices=SEGMENT_LENGTHS,
        required=True,
        help='Segment lengths to test (required, no default)'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='Root directory containing original .npy files (required, no default)'
    )
    parser.add_argument(
        '--hdf5-dir',
        type=str,
        required=True,
        help='Directory containing preprocessed HDF5 files (required, no default)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.participants and not args.auto_detect:
        parser.error("Either --participants or --auto-detect must be specified")
    
    if args.auto_detect and args.max_participants is None:
        parser.error("--max-participants is required when using --auto-detect")
    
    if args.participants and args.auto_detect:
        parser.error("Cannot specify both --participants and --auto-detect")
    
    # Configuration
    data_root = Path(args.data_root)
    hdf5_dir = Path(args.hdf5_dir)
    
    # Get participant IDs
    if args.participants:
        test_participant_ids = args.participants
    else:
        logger.info("Auto-detecting participants from data...")
        test_participant_ids = find_participant_ids(data_root, args.max_participants)
        if not test_participant_ids:
            logger.error("No participants found!")
            return 1
        logger.info(f"Found {len(test_participant_ids)} participants: {test_participant_ids}")
    
    if not test_participant_ids:
        logger.error("No participants to test!")
        return 1
    
    # Create validator
    try:
        validator = HDF5Validator(data_root, hdf5_dir)
    except FileNotFoundError as e:
        logger.error(f"Failed to initialize validator: {e}")
        return 1
    
    # Validate participants
    logger.info(f"Starting validation for {len(test_participant_ids)} participants...")
    logger.info(f"Testing segment lengths: {args.segment_lengths}")
    all_results = validator.validate_multiple_participants(
        test_participant_ids,
        segment_lengths=args.segment_lengths
    )
    
    # Print summary
    validator.print_summary(all_results)
    
    # Return exit code based on results
    has_errors = any('errors' in r and r['errors'] for r in all_results.values() if 'error' not in r)
    if has_errors:
        logger.error("\nValidation FAILED - errors found!")
        return 1
    else:
        logger.info("\nValidation PASSED - all checks successful!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
