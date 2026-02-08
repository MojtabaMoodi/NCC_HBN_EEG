"""
I/O utilities for loading manifests and aggregating data by subject.
"""

import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import warnings


def load_train_manifest(manifest_path: str) -> pd.DataFrame:
    """
    Load training manifest (subject-level).
    
    Expected columns: subject_id, label (0/1/2)
    
    Args:
        manifest_path: Path to CSV file
        
    Returns:
        DataFrame with columns: subject_id, label
    """
    df = pd.read_csv(manifest_path)
    
    # Validate required columns
    required_cols = ['subject_id', 'label']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Train manifest missing required columns: {missing}. Found: {list(df.columns)}")
    
    # Ensure subject_id is string
    df['subject_id'] = df['subject_id'].astype(str)
    
    # Validate labels
    if not df['label'].isin([0, 1, 2]).all():
        raise ValueError(f"Train manifest contains invalid labels. Expected 0, 1, or 2. Found: {df['label'].unique()}")
    
    # Check for duplicate subjects
    if df['subject_id'].duplicated().any():
        dup_subjects = df[df['subject_id'].duplicated()]['subject_id'].unique()
        raise ValueError(f"Train manifest contains duplicate subject_ids: {dup_subjects}")
    
    return df[['subject_id', 'label']].copy()


def load_test_manifest(manifest_path: str) -> pd.DataFrame:
    """
    Load test manifest (segment-level).
    
    Expected columns: subject_id, label (0/1/2), and either:
    - segment_path: path to HDF5 file
    - segment_index: index within HDF5 file
    OR
    - hdf5_file: path to HDF5 file
    - sample_id: sample identifier within HDF5
    
    Args:
        manifest_path: Path to CSV file
        
    Returns:
        DataFrame with columns: subject_id, label, and segment identifiers
    """
    df = pd.read_csv(manifest_path)
    
    # Validate required columns
    required_cols = ['subject_id', 'label']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Test manifest missing required columns: {missing}. Found: {list(df.columns)}")
    
    # Ensure subject_id is string
    df['subject_id'] = df['subject_id'].astype(str)
    
    # Validate labels
    if not df['label'].isin([0, 1, 2]).all():
        raise ValueError(f"Test manifest contains invalid labels. Expected 0, 1, or 2. Found: {df['label'].unique()}")
    
    # Check for segment identifier columns
    has_path = 'segment_path' in df.columns
    has_index = 'segment_index' in df.columns
    has_hdf5 = 'hdf5_file' in df.columns
    has_sample_id = 'sample_id' in df.columns
    
    if not (has_path or (has_hdf5 and has_sample_id)):
        raise ValueError(
            "Test manifest must have either:\n"
            "  - 'segment_path' column, OR\n"
            "  - 'hdf5_file' and 'sample_id' columns\n"
            f"Found columns: {list(df.columns)}"
        )
    
    return df.copy()


def validate_subject_labels(test_manifest: pd.DataFrame) -> Dict[str, int]:
    """
    Validate that each subject has exactly one unique label across all segments.
    
    Args:
        test_manifest: Test manifest DataFrame
        
    Returns:
        Dictionary mapping subject_id to label
        
    Raises:
        ValueError: If any subject has inconsistent labels
    """
    subject_labels = {}
    inconsistent = []
    
    for subject_id, group in test_manifest.groupby('subject_id'):
        labels = group['label'].unique()
        if len(labels) > 1:
            inconsistent.append((subject_id, labels.tolist()))
        else:
            subject_labels[subject_id] = labels[0]
    
    if inconsistent:
        error_msg = "Found subjects with inconsistent labels across segments:\n"
        for subj, labels in inconsistent[:10]:  # Show first 10
            error_msg += f"  {subj}: {labels}\n"
        if len(inconsistent) > 10:
            error_msg += f"  ... and {len(inconsistent) - 10} more\n"
        raise ValueError(error_msg)
    
    return subject_labels


def check_train_test_overlap(train_manifest: pd.DataFrame, test_manifest: pd.DataFrame) -> Set[str]:
    """
    Check for overlap between train and test subject IDs.
    
    Args:
        train_manifest: Train manifest DataFrame
        test_manifest: Test manifest DataFrame
        
    Returns:
        Set of overlapping subject IDs (empty if no overlap)
    """
    train_subjects = set(train_manifest['subject_id'].unique())
    test_subjects = set(test_manifest['subject_id'].unique())
    overlap = train_subjects & test_subjects
    
    if overlap:
        warnings.warn(
            f"⚠️  WARNING: Found {len(overlap)} overlapping subjects between train and test sets!\n"
            f"   Overlapping subjects: {sorted(list(overlap))[:10]}{'...' if len(overlap) > 10 else ''}\n"
            f"   This violates subject-level split assumption. Results may be invalid.",
            UserWarning
        )
    
    return overlap


def aggregate_segments_by_subject(
    segment_predictions: np.ndarray,
    segment_labels: np.ndarray,
    segment_subject_ids: List[str],
    aggregation: str = "mean_prob",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Aggregate segment-level predictions to subject-level predictions.

    Args:
        segment_predictions: Array of shape (N_segments, num_classes) with probabilities
        segment_labels: Array of shape (N_segments,) with segment labels
        segment_subject_ids: List of length N_segments with subject IDs
        aggregation: 'mean_prob' = mean probability then argmax; 'majority_vote' = mode
            of per-segment argmax predictions (no retraining required).

    Returns:
        Tuple of:
        - subjects: Array of unique subject IDs, shape (S,)
        - y_true_subject: Array of subject labels, shape (S,)
        - y_pred_subject: Array of subject predictions, shape (S,)
        - p_subject: Dictionary mapping subject_id to mean probability vector, shape (num_classes,)
    """
    if aggregation == "majority_vote":
        try:
            from data_processing.aggregation import aggregate_segment_probs_by_subject_majority_vote
        except ImportError:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from data_processing.aggregation import aggregate_segment_probs_by_subject_majority_vote
        return aggregate_segment_probs_by_subject_majority_vote(
            segment_predictions,
            segment_labels,
            segment_subject_ids,
        )

    # mean_prob (default): original behavior
    subject_data = defaultdict(lambda: {'probs': [], 'labels': []})

    for i, subject_id in enumerate(segment_subject_ids):
        subject_data[subject_id]['probs'].append(segment_predictions[i])
        subject_data[subject_id]['labels'].append(segment_labels[i])

    subjects = []
    y_true_subject = []
    y_pred_subject = []
    p_subject_dict = {}

    for subject_id in sorted(subject_data.keys()):
        probs_list = subject_data[subject_id]['probs']
        labels_list = subject_data[subject_id]['labels']

        unique_labels = np.unique(labels_list)
        if len(unique_labels) > 1:
            raise ValueError(
                f"Subject {subject_id} has inconsistent labels: {unique_labels.tolist()}"
            )

        mean_prob = np.mean(probs_list, axis=0)
        pred = np.argmax(mean_prob)

        subjects.append(subject_id)
        y_true_subject.append(unique_labels[0])
        y_pred_subject.append(pred)
        p_subject_dict[subject_id] = mean_prob

    return (
        np.array(subjects),
        np.array(y_true_subject),
        np.array(y_pred_subject),
        p_subject_dict
    )


def compute_train_subject_proportions(train_manifest: pd.DataFrame) -> Tuple[np.ndarray, int]:
    """
    Compute subject-level class proportions from training set.
    
    Args:
        train_manifest: Train manifest DataFrame with columns: subject_id, label
        
    Returns:
        Tuple of:
        - proportions: Array of shape (3,) with proportions for classes 0, 1, 2
        - majority_class: Class with highest count
    """
    label_counts = train_manifest['label'].value_counts().sort_index()
    
    # Ensure all 3 classes are present
    for cls in [0, 1, 2]:
        if cls not in label_counts.index:
            warnings.warn(
                f"⚠️  WARNING: Class {cls} not found in training set. "
                f"This may cause issues with stratified random baseline.",
                UserWarning
            )
            label_counts[cls] = 0
    
    total = label_counts.sum()
    proportions = label_counts.values / total if total > 0 else np.array([1/3, 1/3, 1/3])
    
    majority_class = label_counts.idxmax()
    
    return proportions, majority_class
