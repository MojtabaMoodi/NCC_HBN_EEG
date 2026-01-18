#!/usr/bin/env python3
"""
Helper script to generate train and test manifests from HDF5 files.

This script extracts subject-level information from HDF5 files and creates
CSV manifests in the format required by evaluate_models_vs_random.py.
"""

import argparse
import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict
import sys


def extract_subjects_from_hdf5(hdf5_file: Path, task_type: str = "both") -> Dict[str, Dict]:
    """
    Extract subject information from HDF5 file.
    
    Args:
        hdf5_file: Path to HDF5 file
        task_type: "active", "passive", or "both"
        
    Returns:
        Dictionary mapping subject_id to {
            'label': int (age class 0/1/2),
            'segments': list of (sample_id, hdf5_file) tuples
        }
    """
    subjects = defaultdict(lambda: {'label': None, 'segments': []})
    
    with h5py.File(hdf5_file, 'r') as f:
        task_types = []
        if task_type in ["active", "both"] and 'active' in f:
            task_types.append('active')
        if task_type in ["passive", "both"] and 'passive' in f:
            task_types.append('passive')
        
        for tt in task_types:
            task_group = f[tt]
            
            # Iterate through samples
            for key in task_group.keys():
                if key.startswith('sample_'):
                    sample_id = key
                    sample_data = task_group[sample_id]
                    
                    # Get metadata
                    # Use same logic as existing dataset code: replace 'sample_' with 'metadata_'
                    if sample_id.startswith('sample_'):
                        metadata_key = sample_id.replace('sample_', 'metadata_', 1)  # Only replace first occurrence
                    else:
                        # Fallback: extract number and construct metadata key
                        sample_num = sample_id.split('_')[-1] if '_' in sample_id else sample_id.replace('sample_', '')
                        metadata_key = f'metadata_{sample_num}'
                    
                    if metadata_key in task_group:
                        metadata = task_group[metadata_key]
                    else:
                        # Try alternative metadata format (shouldn't happen with correct structure)
                        metadata_keys = [k for k in task_group.keys() if k.startswith('metadata_')]
                        if not metadata_keys:
                            continue
                        # Use first metadata as fallback (may not be perfect)
                        metadata = task_group[metadata_keys[0]]
                    
                    # Get participant_id
                    participant_id = metadata.attrs.get('participant_id', '')
                    if isinstance(participant_id, bytes):
                        participant_id = participant_id.decode()
                    participant_id = str(participant_id)
                    
                    # Get age and compute label
                    age = metadata.attrs.get('age', None)
                    if age is not None:
                        # Bin age into classes: 0 (<8.5), 1 (8.5-12.5), 2 (>12.5)
                        if age < 8.5:
                            label = 0
                        elif age <= 12.5:
                            label = 1
                        else:
                            label = 2
                    else:
                        # Skip if age is missing
                        continue
                    
                    # Store segment info
                    # Store just the filename (not full path) for portability
                    # Full path can be reconstructed using --hdf5_base_dir
                    hdf5_filename = Path(hdf5_file).name
                    subjects[participant_id]['label'] = label
                    subjects[participant_id]['segments'].append((
                        sample_id,
                        hdf5_filename
                    ))
    
    return dict(subjects)


def generate_train_manifest(hdf5_files: List[Path], output_path: Path, task_type: str = "both"):
    """
    Generate train manifest (subject-level) from HDF5 files.
    
    Args:
        hdf5_files: List of HDF5 file paths
        output_path: Output CSV path
        task_type: Task type to include
    """
    all_subjects = {}
    
    for hdf5_file in hdf5_files:
        print(f"Processing {hdf5_file.name}...")
        subjects = extract_subjects_from_hdf5(hdf5_file, task_type)
        
        # Merge subjects (handle duplicates)
        for subject_id, info in subjects.items():
            if subject_id in all_subjects:
                # Check label consistency
                if all_subjects[subject_id]['label'] != info['label']:
                    print(f"Warning: Subject {subject_id} has inconsistent labels "
                          f"({all_subjects[subject_id]['label']} vs {info['label']})")
                # Merge segments
                all_subjects[subject_id]['segments'].extend(info['segments'])
            else:
                all_subjects[subject_id] = info
    
    # Create DataFrame
    rows = []
    for subject_id, info in all_subjects.items():
        rows.append({
            'subject_id': subject_id,
            'label': info['label']
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('subject_id')
    df.to_csv(output_path, index=False)
    
    print(f"\nGenerated train manifest: {output_path}")
    print(f"  Subjects: {len(df)}")
    print(f"  Label distribution:")
    print(df['label'].value_counts().sort_index())


def generate_test_manifest(hdf5_files: List[Path], output_path: Path, task_type: str = "both"):
    """
    Generate test manifest (segment-level) from HDF5 files.
    
    Args:
        hdf5_files: List of HDF5 file paths
        output_path: Output CSV path
        task_type: Task type to include
    """
    all_segments = []
    
    for hdf5_file in hdf5_files:
        print(f"Processing {hdf5_file.name}...")
        subjects = extract_subjects_from_hdf5(hdf5_file, task_type)
        
        for subject_id, info in subjects.items():
            for sample_id, hdf5_path in info['segments']:
                all_segments.append({
                    'subject_id': subject_id,
                    'label': info['label'],
                    'hdf5_file': hdf5_path,
                    'sample_id': sample_id
                })
    
    df = pd.DataFrame(all_segments)
    df = df.sort_values(['subject_id', 'sample_id'])
    df.to_csv(output_path, index=False)
    
    print(f"\nGenerated test manifest: {output_path}")
    print(f"  Segments: {len(df)}")
    print(f"  Subjects: {df['subject_id'].nunique()}")
    print(f"  Label distribution:")
    print(df['label'].value_counts().sort_index())
    
    # Validate subject label consistency
    inconsistent = []
    for subject_id, group in df.groupby('subject_id'):
        if group['label'].nunique() > 1:
            inconsistent.append(subject_id)
    
    if inconsistent:
        print(f"\n⚠️  Warning: {len(inconsistent)} subjects have inconsistent labels:")
        for subj in inconsistent[:10]:
            print(f"  {subj}: {df[df['subject_id'] == subj]['label'].unique()}")
        if len(inconsistent) > 10:
            print(f"  ... and {len(inconsistent) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Generate train and test manifests from HDF5 files"
    )
    parser.add_argument('--hdf5_dir', type=str, required=True,
                        help='Directory containing HDF5 files')
    parser.add_argument('--split', type=str, required=True,
                        choices=['train', 'test'],
                        help='Split to generate manifest for')
    parser.add_argument('--segment_length', type=str, required=True,
                        choices=['1s', '2s', '4s'],
                        help='Segment length')
    parser.add_argument('--task_type', type=str, default='both',
                        choices=['active', 'passive', 'both'],
                        help='Task type to include')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV path')
    parser.add_argument('--pattern', type=str, default=None,
                        help='File pattern to match (e.g., "eeg_data_train_4s*.h5")')
    
    args = parser.parse_args()
    
    hdf5_dir = Path(args.hdf5_dir)
    if not hdf5_dir.exists():
        raise ValueError(f"HDF5 directory not found: {hdf5_dir}")
    
    # Find HDF5 files
    if args.pattern:
        import glob
        pattern = str(hdf5_dir / args.pattern)
        hdf5_files = [Path(f) for f in glob.glob(pattern)]
    else:
        # Default pattern
        pattern = f"eeg_data_{args.split}_{args.segment_length}*.h5"
        hdf5_files = list(hdf5_dir.glob(pattern))
    
    if not hdf5_files:
        raise ValueError(
            f"No HDF5 files found matching pattern: {pattern}\n"
            f"  Directory: {hdf5_dir}\n"
            f"  Try specifying --pattern explicitly"
        )
    
    print(f"Found {len(hdf5_files)} HDF5 file(s)")
    
    # Generate manifest
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.split == 'train':
        generate_train_manifest(hdf5_files, output_path, args.task_type)
    else:
        generate_test_manifest(hdf5_files, output_path, args.task_type)
    
    print(f"\n✅ Manifest saved to: {output_path}")


if __name__ == "__main__":
    main()
