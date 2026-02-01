#!/usr/bin/env python3
"""
Validation script for 128-channel EEG data preprocessing.

This script validates the output of the preprocessing pipeline to ensure:
- HDF5 files are created correctly
- Data structure matches expectations (128 channels, correct timepoints)
- Train/val/test splits are correct
- Metadata is properly stored
- Multi-part files are handled correctly
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_hdf5_file(hdf5_path: Path, expected_channels: int, expected_timepoints: int, 
                      split_name: str, segment_length: str) -> dict:
    """
    Validate a single HDF5 file.
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'file': str(hdf5_path),
        'exists': False,
        'valid': False,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    if not hdf5_path.exists():
        results['errors'].append(f"File does not exist: {hdf5_path}")
        return results
    
    results['exists'] = True
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # Check file attributes
            file_split = f.attrs.get('split_name', 'unknown')
            file_segment = f.attrs.get('segment_length', 'unknown')
            num_participants = f.attrs.get('num_participants', 0)
            
            if file_split != split_name:
                results['warnings'].append(f"Split name mismatch: expected '{split_name}', got '{file_split}'")
            if file_segment != segment_length:
                results['warnings'].append(f"Segment length mismatch: expected '{segment_length}', got '{file_segment}'")
            
            # Check structure
            if 'active' not in f and 'passive' not in f:
                results['errors'].append("File missing both 'active' and 'passive' groups")
                return results
            
            total_samples = 0
            participant_ids = set()
            task_types_found = []
            sample_shapes = []
            metadata_keys_checked = []
            
            for task_type in ['active', 'passive']:
                if task_type not in f:
                    continue
                
                task_group = f[task_type]
                task_samples = []
                
                # Count samples
                for key in task_group.keys():
                    if key.startswith('sample_'):
                        sample_name = key
                        metadata_name = key.replace('sample_', 'metadata_', 1)
                        
                        # Load sample
                        sample_data = task_group[sample_name]
                        sample_shape = sample_data.shape
                        sample_shapes.append(sample_shape)
                        
                        # Validate shape
                        if len(sample_shape) != 2:
                            results['errors'].append(
                                f"{task_type}/{sample_name}: Expected 2D array, got shape {sample_shape}"
                            )
                            continue
                        
                        channels, timepoints = sample_shape
                        if channels != expected_channels:
                            results['errors'].append(
                                f"{task_type}/{sample_name}: Expected {expected_channels} channels, got {channels}"
                            )
                        if timepoints != expected_timepoints:
                            results['errors'].append(
                                f"{task_type}/{sample_name}: Expected {expected_timepoints} timepoints, got {timepoints}"
                            )
                        
                        # Check metadata
                        if metadata_name in task_group:
                            metadata = task_group[metadata_name]
                            pid = metadata.attrs.get('participant_id', '')
                            if isinstance(pid, bytes):
                                pid = pid.decode()
                            participant_ids.add(str(pid))
                            
                            # Check required metadata
                            if 'task_type' not in metadata.attrs:
                                results['warnings'].append(f"{task_type}/{sample_name}: Missing 'task_type' in metadata")
                            if 'task_number' not in metadata.attrs:
                                results['warnings'].append(f"{task_type}/{sample_name}: Missing 'task_number' in metadata")
                            
                            # Check data quality
                            sample_array = sample_data[:]
                            if np.any(np.isnan(sample_array)):
                                results['errors'].append(f"{task_type}/{sample_name}: Contains NaN values")
                            if np.any(np.isinf(sample_array)):
                                results['errors'].append(f"{task_type}/{sample_name}: Contains Inf values")
                            
                            task_samples.append(sample_name)
                            metadata_keys_checked.append(metadata_name)
                        else:
                            results['errors'].append(
                                f"{task_type}/{sample_name}: Missing metadata group '{metadata_name}'"
                            )
                
                total_samples += len(task_samples)
                if len(task_samples) > 0:
                    task_types_found.append(task_type)
            
            # Store statistics
            results['stats'] = {
                'total_samples': total_samples,
                'num_participants': len(participant_ids),
                'num_participants_attr': num_participants,
                'task_types': task_types_found,
                'sample_shapes': list(set(sample_shapes)),
                'metadata_count': len(metadata_keys_checked)
            }
            
            # Validate participant count
            if num_participants != len(participant_ids):
                results['warnings'].append(
                    f"Participant count mismatch: attribute says {num_participants}, "
                    f"but found {len(participant_ids)} unique participants in metadata"
                )
            
            # Check if file is valid
            results['valid'] = len(results['errors']) == 0
            
    except Exception as e:
        results['errors'].append(f"Error reading file: {e}")
        import traceback
        results['errors'].append(traceback.format_exc())
    
    return results

def find_all_part_files(output_dir: Path, split_name: str, segment_length: str) -> list:
    """Find all part files for a given split and segment length."""
    part_files = []
    part_idx = 0
    
    while True:
        # Try part file first
        part_file = output_dir / f"eeg_data_{split_name}_{segment_length}_part{part_idx:02d}.h5"
        if part_file.exists():
            part_files.append(part_file)
            part_idx += 1
        else:
            # Try single file
            if part_idx == 0:
                single_file = output_dir / f"eeg_data_{split_name}_{segment_length}.h5"
                if single_file.exists():
                    part_files.append(single_file)
            break
    
    return part_files

def validate_preprocessing(output_dir: str, segment_length: str = '4s', 
                          expected_channels: int = 128, num_samples_to_check: int = 10):
    """
    Validate the preprocessing output.
    
    Args:
        output_dir: Directory containing preprocessed HDF5 files
        segment_length: Segment length to validate ('1s', '2s', or '4s')
        expected_channels: Expected number of channels (128 for new data)
        num_samples_to_check: Number of random samples to check in detail
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        logger.error(f"Output directory does not exist: {output_dir}")
        return False
    
    logger.info("=" * 80)
    logger.info("Validating 128-Channel EEG Data Preprocessing")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Segment length: {segment_length}")
    logger.info(f"Expected channels: {expected_channels}")
    logger.info("=" * 80)
    
    # Expected timepoints based on segment length
    expected_timepoints = {
        '1s': 200,
        '2s': 400,
        '4s': 800
    }
    
    if segment_length not in expected_timepoints:
        logger.error(f"Invalid segment length: {segment_length}. Must be '1s', '2s', or '4s'")
        return False
    
    expected_tp = expected_timepoints[segment_length]
    
    # Validate each split
    splits = ['train', 'val', 'test']
    all_valid = True
    all_stats = {}
    
    for split_name in splits:
        logger.info(f"\n{'='*80}")
        logger.info(f"Validating {split_name.upper()} split")
        logger.info(f"{'='*80}")
        
        # Find all part files
        part_files = find_all_part_files(output_path, split_name, segment_length)
        
        if not part_files:
            logger.error(f"No files found for {split_name}/{segment_length}")
            all_valid = False
            continue
        
        logger.info(f"Found {len(part_files)} file(s) for {split_name}/{segment_length}")
        for pf in part_files:
            logger.info(f"  - {pf.name}")
        
        split_stats = {
            'total_files': len(part_files),
            'total_samples': 0,
            'total_participants': set(),
            'task_type_counts': Counter(),
            'errors': [],
            'warnings': []
        }
        
        # Validate each file
        for part_file in part_files:
            logger.info(f"\nValidating: {part_file.name}")
            results = validate_hdf5_file(
                part_file, expected_channels, expected_tp, split_name, segment_length
            )
            
            if not results['exists']:
                logger.error(f"  ❌ File does not exist")
                all_valid = False
                continue
            
            if results['valid']:
                logger.info(f"  ✅ File is valid")
            else:
                logger.error(f"  ❌ File has errors")
                all_valid = False
            
            # Log errors
            for error in results['errors']:
                logger.error(f"    ERROR: {error}")
                split_stats['errors'].append(f"{part_file.name}: {error}")
            
            # Log warnings
            for warning in results['warnings']:
                logger.warning(f"    WARNING: {warning}")
                split_stats['warnings'].append(f"{part_file.name}: {warning}")
            
            # Accumulate statistics
            stats = results['stats']
            split_stats['total_samples'] += stats.get('total_samples', 0)
            split_stats['task_type_counts'].update(stats.get('task_types', []))
            
            # Get participant IDs from this file
            try:
                with h5py.File(part_file, 'r') as f:
                    for task_type in ['active', 'passive']:
                        if task_type not in f:
                            continue
                        task_group = f[task_type]
                        for key in task_group.keys():
                            if key.startswith('metadata_'):
                                metadata = task_group[key]
                                pid = metadata.attrs.get('participant_id', '')
                                if isinstance(pid, bytes):
                                    pid = pid.decode()
                                split_stats['total_participants'].add(str(pid))
            except Exception as e:
                logger.warning(f"  Could not extract participant IDs from {part_file.name}: {e}")
            
            # Log file statistics
            logger.info(f"  Samples: {stats.get('total_samples', 0)}")
            logger.info(f"  Participants: {stats.get('num_participants', 0)}")
            logger.info(f"  Task types: {stats.get('task_types', [])}")
            logger.info(f"  Sample shapes: {stats.get('sample_shapes', [])}")
        
        # Log split summary
        logger.info(f"\n{split_name.upper()} Split Summary:")
        logger.info(f"  Total files: {split_stats['total_files']}")
        logger.info(f"  Total samples: {split_stats['total_samples']:,}")
        logger.info(f"  Unique participants: {len(split_stats['total_participants'])}")
        logger.info(f"  Task type distribution: {dict(split_stats['task_type_counts'])}")
        logger.info(f"  Errors: {len(split_stats['errors'])}")
        logger.info(f"  Warnings: {len(split_stats['warnings'])}")
        
        all_stats[split_name] = split_stats
    
    # Overall summary
    logger.info(f"\n{'='*80}")
    logger.info("OVERALL SUMMARY")
    logger.info(f"{'='*80}")
    
    total_samples_all = sum(s['total_samples'] for s in all_stats.values())
    total_participants_all = set()
    for s in all_stats.values():
        total_participants_all.update(s['total_participants'])
    
    logger.info(f"Total samples across all splits: {total_samples_all:,}")
    logger.info(f"Total unique participants: {len(total_participants_all)}")
    
    # Check train/val/test participant overlap (should be minimal for participant-level splits)
    train_participants = all_stats.get('train', {}).get('total_participants', set())
    val_participants = all_stats.get('val', {}).get('total_participants', set())
    test_participants = all_stats.get('test', {}).get('total_participants', set())
    
    train_val_overlap = train_participants & val_participants
    train_test_overlap = train_participants & test_participants
    val_test_overlap = val_participants & test_participants
    
    if train_val_overlap:
        logger.warning(f"Train/Val participant overlap: {len(train_val_overlap)} participants")
    if train_test_overlap:
        logger.warning(f"Train/Test participant overlap: {len(train_test_overlap)} participants")
    if val_test_overlap:
        logger.warning(f"Val/Test participant overlap: {len(val_test_overlap)} participants")
    
    # Check sample distribution
    train_samples = all_stats.get('train', {}).get('total_samples', 0)
    val_samples = all_stats.get('val', {}).get('total_samples', 0)
    test_samples = all_stats.get('test', {}).get('total_samples', 0)
    
    if total_samples_all > 0:
        train_ratio = train_samples / total_samples_all
        val_ratio = val_samples / total_samples_all
        test_ratio = test_samples / total_samples_all
        
        logger.info(f"\nSample distribution:")
        logger.info(f"  Train: {train_samples:,} ({train_ratio:.1%})")
        logger.info(f"  Val:   {val_samples:,} ({val_ratio:.1%})")
        logger.info(f"  Test:  {test_samples:,} ({test_ratio:.1%})")
        
        # Check if ratios are close to expected (70/15/15)
        if abs(train_ratio - 0.7) > 0.05:
            logger.warning(f"Train ratio ({train_ratio:.1%}) differs significantly from expected 70%")
        if abs(val_ratio - 0.15) > 0.05:
            logger.warning(f"Val ratio ({val_ratio:.1%}) differs significantly from expected 15%")
        if abs(test_ratio - 0.15) > 0.05:
            logger.warning(f"Test ratio ({test_ratio:.1%}) differs significantly from expected 15%")
    
    # Final validation result
    total_errors = sum(len(s.get('errors', [])) for s in all_stats.values())
    
    logger.info(f"\n{'='*80}")
    if all_valid and total_errors == 0:
        logger.info("✅ VALIDATION PASSED")
        logger.info(f"{'='*80}")
        return True
    else:
        logger.error("❌ VALIDATION FAILED")
        logger.error(f"Total errors: {total_errors}")
        logger.info(f"{'='*80}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Validate 128-channel EEG preprocessing output')
    parser.add_argument('--output_dir', type=str,
                       default='/home/mojtabam/scratch/processed_eeg_data_128ch',
                       help='Path to output directory containing HDF5 files')
    parser.add_argument('--segment_length', type=str, default='4s',
                       choices=['1s', '2s', '4s'],
                       help='Segment length to validate (default: 4s)')
    parser.add_argument('--expected_channels', type=int, default=128,
                       help='Expected number of channels (default: 128)')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of random samples to check in detail (default: 10)')
    
    args = parser.parse_args()
    
    success = validate_preprocessing(
        output_dir=args.output_dir,
        segment_length=args.segment_length,
        expected_channels=args.expected_channels,
        num_samples_to_check=args.num_samples
    )
    
    exit(0 if success else 1)

if __name__ == "__main__":
    main()
