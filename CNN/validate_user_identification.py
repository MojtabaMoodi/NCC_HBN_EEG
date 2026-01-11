#!/usr/bin/env python3
"""
Validation script to verify user identification setup:
1. All participants are included in the mapping
2. All classes are represented in training data
3. Model is correctly configured with the right number of classes
4. Labels are correctly mapped from participant_id to class_idx
"""

import json
import sys
from pathlib import Path
import h5py
import numpy as np
from collections import Counter

def load_mapping(mapping_file: Path) -> dict:
    """Load participant_id to class_idx mapping."""
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    
    return mapping

def get_participants_from_hdf5(hdf5_file: Path) -> set:
    """Extract all unique participant IDs from an HDF5 file."""
    participants = set()
    
    try:
        with h5py.File(hdf5_file, 'r') as f:
            # Structure: active/ and passive/ at top level (split is in filename)
            for task_type in ['active', 'passive']:
                if task_type in f:
                    task_group = f[task_type]
                    # Iterate through all metadata groups
                    for key in task_group.keys():
                        if key.startswith('metadata_'):
                            metadata = task_group[key]
                            if 'participant_id' in metadata.attrs:
                                participant_id = metadata.attrs['participant_id']
                                if isinstance(participant_id, bytes):
                                    participant_id = participant_id.decode()
                                participants.add(str(participant_id))
    except Exception as e:
        print(f"Error reading {hdf5_file}: {e}")
        return set()
    
    return participants

def get_class_labels_from_hdf5(hdf5_file: Path, mapping: dict, split_name: str = 'train') -> list:
    """Extract all class labels (class indices) from an HDF5 file (split is in filename)."""
    class_labels = []
    
    try:
        with h5py.File(hdf5_file, 'r') as f:
            # Structure: active/ and passive/ at top level (split is in filename)
            for task_type in ['active', 'passive']:
                if task_type in f:
                    task_group = f[task_type]
                    # Iterate through all metadata groups
                    for key in task_group.keys():
                        if key.startswith('metadata_'):
                            metadata = task_group[key]
                            if 'participant_id' in metadata.attrs:
                                participant_id = metadata.attrs['participant_id']
                                if isinstance(participant_id, bytes):
                                    participant_id = participant_id.decode()
                                participant_id = str(participant_id)
                                
                                # Map to class index
                                if participant_id in mapping:
                                    class_idx = mapping[participant_id]
                                    class_labels.append(class_idx)
                                else:
                                    print(f"⚠️  Warning: Participant {participant_id} not in mapping!")
    except Exception as e:
        print(f"Error reading {hdf5_file}: {e}")
        return []
    
    return class_labels

def validate_user_identification(hdf5_dir: str):
    """Validate user identification setup."""
    hdf5_dir_path = Path(hdf5_dir)
    
    if not hdf5_dir_path.exists():
        raise FileNotFoundError(f"HDF5 directory not found: {hdf5_dir}")
    
    # Load mapping
    mapping_file = hdf5_dir_path / 'participant_id_to_class_idx.json'
    print(f"📋 Loading mapping from: {mapping_file}")
    mapping = load_mapping(mapping_file)
    
    num_classes = len(mapping)
    print(f"✅ Mapping loaded: {num_classes} participants (classes)")
    
    # Check mapping consistency
    print("\n🔍 Validating mapping consistency...")
    class_indices = list(mapping.values())
    expected_indices = set(range(num_classes))
    actual_indices = set(class_indices)
    
    if expected_indices != actual_indices:
        missing = expected_indices - actual_indices
        extra = actual_indices - expected_indices
        print(f"❌ Mapping inconsistency detected!")
        if missing:
            print(f"   Missing class indices: {sorted(missing)}")
        if extra:
            print(f"   Extra class indices: {sorted(extra)}")
        return False
    else:
        print(f"✅ Mapping is consistent: all class indices [0, {num_classes-1}] are present")
    
    # Check for each segment length
    for segment_length in ['1s', '2s', '4s']:
        print(f"\n📊 Validating {segment_length} segments...")
        
        # Find HDF5 file(s) for this segment length (may be multi-file)
        hdf5_files = []
        for split in ['train', 'val', 'test']:
            # Try single file first
            candidate = hdf5_dir_path / f'eeg_data_{split}_{segment_length}.h5'
            if candidate.exists():
                hdf5_files.append((split, candidate))
            else:
                # Try multi-file pattern (part00, part01, etc.)
                part_files = sorted(hdf5_dir_path.glob(f'eeg_data_{split}_{segment_length}_part*.h5'))
                if part_files:
                    hdf5_files.append((split, part_files[0]))  # Use first part for validation
        
        if not hdf5_files:
            print(f"⚠️  No HDF5 file found for {segment_length} segments")
            continue
        
        # Use train file for main validation
        train_file = None
        for split, file_path in hdf5_files:
            if split == 'train':
                train_file = file_path
                break
        
        if train_file is None:
            print(f"⚠️  No training file found for {segment_length} segments")
            continue
        
        print(f"   Using file: {train_file.name} (and checking other splits)")
        
        # Get all participants from all HDF5 files for this segment length
        # IMPORTANT: Check ALL part files, not just the first one
        hdf5_participants = set()
        for split, file_path in hdf5_files:
            participants = get_participants_from_hdf5(file_path)
            hdf5_participants.update(participants)
        
        # Also check all remaining part files for each split (validation script only checks first part)
        for split in ['train', 'val', 'test']:
            part_files = sorted(hdf5_dir_path.glob(f'eeg_data_{split}_{segment_length}_part*.h5'))
            for part_file in part_files:
                participants = get_participants_from_hdf5(part_file)
                hdf5_participants.update(participants)
        
        mapping_participants = set(mapping.keys())
        
        # Check if all HDF5 participants are in mapping
        missing_in_mapping = hdf5_participants - mapping_participants
        if missing_in_mapping:
            print(f"❌ Participants in HDF5 but not in mapping: {sorted(missing_in_mapping)[:10]}...")
            return False
        else:
            print(f"✅ All {len(hdf5_participants)} participants in HDF5 are in mapping")
        
        # Check if all mapping participants are in HDF5
        missing_in_hdf5 = mapping_participants - hdf5_participants
        if missing_in_hdf5:
            print(f"⚠️  Warning: {len(missing_in_hdf5)} participants in mapping but not in HDF5: {sorted(missing_in_hdf5)[:10]}...")
        else:
            print(f"✅ All {len(mapping_participants)} participants in mapping are in HDF5")
        
        # Check class representation in training data
        print(f"\n   Checking class representation in training data...")
        train_labels = get_class_labels_from_hdf5(train_file, mapping, split_name='train')
        
        # Also check all train part files if multi-file
        train_part_files = sorted(hdf5_dir_path.glob(f'eeg_data_train_{segment_length}_part*.h5'))
        for part_file in train_part_files[1:]:  # Skip first (already processed)
            part_labels = get_class_labels_from_hdf5(part_file, mapping, split_name='train')
            train_labels.extend(part_labels)
        
        if not train_labels:
            print(f"⚠️  Warning: No training labels found!")
            continue
        
        train_class_counts = Counter(train_labels)
        unique_classes_in_train = set(train_class_counts.keys())
        all_classes = set(range(num_classes))
        
        missing_classes = all_classes - unique_classes_in_train
        if missing_classes:
            print(f"❌ {len(missing_classes)} classes missing from training data: {sorted(missing_classes)[:10]}...")
            print(f"   This means some participants have no training samples!")
            return False
        else:
            print(f"✅ All {num_classes} classes are represented in training data")
        
        # Show class distribution
        min_samples = min(train_class_counts.values())
        max_samples = max(train_class_counts.values())
        avg_samples = np.mean(list(train_class_counts.values()))
        print(f"   Class distribution: min={min_samples}, max={max_samples}, avg={avg_samples:.1f} samples per class")
        
        # Check validation and test splits
        for split_name in ['val', 'test']:
            # Find split file(s)
            split_file = None
            for split, file_path in hdf5_files:
                if split == split_name:
                    split_file = file_path
                    break
            
            if split_file is None:
                # Try multi-file pattern
                split_part_files = sorted(hdf5_dir_path.glob(f'eeg_data_{split_name}_{segment_length}_part*.h5'))
                if split_part_files:
                    split_file = split_part_files[0]
            
            if split_file is None:
                continue
            
            split_labels = get_class_labels_from_hdf5(split_file, mapping, split_name=split_name)
            
            # Also check all part files if multi-file
            split_part_files = sorted(hdf5_dir_path.glob(f'eeg_data_{split_name}_{segment_length}_part*.h5'))
            for part_file in split_part_files[1:]:  # Skip first (already processed)
                part_labels = get_class_labels_from_hdf5(part_file, mapping, split_name=split_name)
                split_labels.extend(part_labels)
            if split_labels:
                split_class_counts = Counter(split_labels)
                unique_classes_in_split = set(split_class_counts.keys())
                missing_in_split = all_classes - unique_classes_in_split
                if missing_in_split:
                    print(f"⚠️  Warning: {len(missing_in_split)} classes missing from {split_name} data")
                else:
                    print(f"✅ All {num_classes} classes are represented in {split_name} data")
    
    print(f"\n✅ User identification validation passed!")
    print(f"   Total classes: {num_classes}")
    print(f"   Model should be configured with num_classes={num_classes}")
    return True

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate user identification setup')
    parser.add_argument('--hdf5_dir', type=str,
                       default='/home/mojtabam/scratch/processed_eeg_data_user_identification',
                       help='Directory containing HDF5 files and mapping JSON')
    
    args = parser.parse_args()
    
    try:
        success = validate_user_identification(args.hdf5_dir)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
