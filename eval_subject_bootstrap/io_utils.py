"""
I/O utilities for loading manifests and aggregating data by subject.
"""

from __future__ import annotations

import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data_processing.aggregation import (  # noqa: E402
    AGGREGATION_MEDIAN,
    aggregate_predictions_by_group,
)


LabelValue = Union[int, float]


def _normalize_labels(df: pd.DataFrame, allowed_labels: Optional[Set[LabelValue]]) -> pd.DataFrame:
    out = df.copy()
    if allowed_labels is not None and all(isinstance(x, int) for x in allowed_labels):
        out["label"] = out["label"].astype(int)
    return out


def _validate_label_column(df: pd.DataFrame, allowed_labels: Optional[Set[LabelValue]]) -> None:
    if allowed_labels is None:
        return
    found = set(df["label"].unique().tolist())
    unexpected = found - allowed_labels
    if unexpected:
        raise ValueError(
            f"Manifest contains labels outside allowed set {sorted(allowed_labels)}: "
            f"{sorted(unexpected)}"
        )


def load_train_manifest(
    manifest_path: str,
    allowed_labels: Optional[Set[LabelValue]] = None,
) -> pd.DataFrame:
    """
    Load training manifest (subject-level).

    Expected columns: subject_id, label
    """
    df = pd.read_csv(manifest_path)
    required_cols = ["subject_id", "label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Train manifest missing required columns: {missing}. Found: {list(df.columns)}"
        )

    df = _normalize_labels(df, allowed_labels)
    df["subject_id"] = df["subject_id"].astype(str)
    _validate_label_column(df, allowed_labels)

    if df["subject_id"].duplicated().any():
        dup_subjects = df[df["subject_id"].duplicated()]["subject_id"].unique()
        raise ValueError(f"Train manifest contains duplicate subject_ids: {dup_subjects}")

    return df[["subject_id", "label"]].copy()


def load_test_manifest(
    manifest_path: str,
    allowed_labels: Optional[Set[LabelValue]] = None,
) -> pd.DataFrame:
    """
    Load test manifest (segment-level).

    Expected columns: subject_id, label, and either segment_path or (hdf5_file, sample_id).
    """
    df = pd.read_csv(manifest_path)
    required_cols = ["subject_id", "label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Test manifest missing required columns: {missing}. Found: {list(df.columns)}"
        )

    df = _normalize_labels(df, allowed_labels)
    df["subject_id"] = df["subject_id"].astype(str)
    _validate_label_column(df, allowed_labels)

    has_path = "segment_path" in df.columns
    has_hdf5 = "hdf5_file" in df.columns and "sample_id" in df.columns
    if not (has_path or has_hdf5):
        raise ValueError(
            "Test manifest must have either 'segment_path' or ('hdf5_file', 'sample_id'). "
            f"Found columns: {list(df.columns)}"
        )

    return df.copy()


def validate_subject_labels(test_manifest: pd.DataFrame) -> Dict[str, LabelValue]:
    """Ensure each subject has exactly one label across segments."""
    subject_labels: Dict[str, LabelValue] = {}
    inconsistent = []

    for subject_id, group in test_manifest.groupby("subject_id"):
        labels = group["label"].unique()
        if len(labels) > 1:
            inconsistent.append((subject_id, labels.tolist()))
        else:
            subject_labels[str(subject_id)] = labels[0]

    if inconsistent:
        error_msg = "Found subjects with inconsistent labels across segments:\n"
        for subj, labels in inconsistent[:10]:
            error_msg += f"  {subj}: {labels}\n"
        if len(inconsistent) > 10:
            error_msg += f"  ... and {len(inconsistent) - 10} more\n"
        raise ValueError(error_msg)

    return subject_labels


def check_train_test_overlap(
    train_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
) -> Set[str]:
    train_subjects = set(train_manifest["subject_id"].unique())
    test_subjects = set(test_manifest["subject_id"].unique())
    overlap = train_subjects & test_subjects

    if overlap:
        warnings.warn(
            f"Found {len(overlap)} overlapping subjects between train and test sets. "
            f"Overlapping subjects: {sorted(list(overlap))[:10]}"
            f"{'...' if len(overlap) > 10 else ''}",
            UserWarning,
            stacklevel=2,
        )

    return overlap


def aggregate_classification_by_subject(
    segment_probs: np.ndarray,
    segment_labels: np.ndarray,
    segment_subject_ids: List[str],
    aggregation: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Aggregate segment-level classification probabilities to subject level."""
    if aggregation == "majority_vote":
        from data_processing.aggregation import aggregate_segment_probs_by_subject_majority_vote

        return aggregate_segment_probs_by_subject_majority_vote(
            segment_probs,
            segment_labels,
            segment_subject_ids,
        )

    if aggregation == "mean_prob":
        subject_data: Dict[str, Dict[str, list]] = defaultdict(lambda: {"probs": [], "labels": []})
        for i, subject_id in enumerate(segment_subject_ids):
            subject_data[subject_id]["probs"].append(segment_probs[i])
            subject_data[subject_id]["labels"].append(segment_labels[i])

        subjects: List[str] = []
        y_true_subject: List[LabelValue] = []
        y_pred_subject: List[int] = []
        p_subject_dict: Dict[str, np.ndarray] = {}

        for subject_id in sorted(subject_data.keys()):
            probs_list = subject_data[subject_id]["probs"]
            labels_list = subject_data[subject_id]["labels"]
            unique_labels = np.unique(labels_list)
            if len(unique_labels) > 1:
                raise ValueError(
                    f"Subject {subject_id} has inconsistent labels: {unique_labels.tolist()}"
                )

            mean_prob = np.mean(probs_list, axis=0)
            pred = int(np.argmax(mean_prob))

            subjects.append(subject_id)
            y_true_subject.append(unique_labels[0])
            y_pred_subject.append(pred)
            p_subject_dict[subject_id] = mean_prob

        return (
            np.array(subjects),
            np.array(y_true_subject),
            np.array(y_pred_subject),
            p_subject_dict,
        )

    raise ValueError(
        f"Unknown classification aggregation {aggregation!r}. "
        "Expected 'majority_vote' or 'mean_prob'."
    )


def aggregate_regression_by_subject(
    segment_predictions_years: np.ndarray,
    segment_labels_years: np.ndarray,
    segment_subject_ids: List[str],
    aggregation: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate segment-level regression predictions to subject level (years)."""
    if aggregation not in (AGGREGATION_MEDIAN, "mean"):
        raise ValueError(
            f"Regression aggregation must be 'median' or 'mean', got {aggregation!r}"
        )

    regression_aggregation = aggregation
    groups, y_true, y_pred, _ = aggregate_predictions_by_group(
        segment_predictions_years,
        segment_labels_years,
        segment_subject_ids,
        prediction_type="regression",
        regression_aggregation=regression_aggregation,
    )
    return groups, y_true, y_pred


def compute_train_subject_proportions(
    train_manifest: pd.DataFrame,
    class_labels: Sequence[int],
) -> Tuple[np.ndarray, int]:
    """
    Compute subject-level class proportions for stratified random baseline.

    Returns:
        proportions aligned with class_labels order, and majority class index.
    """
    labels = list(class_labels)
    if len(labels) == 0:
        raise ValueError("class_labels must be non-empty")

    label_counts = train_manifest["label"].value_counts()
    counts = []
    for cls in labels:
        if cls not in label_counts.index:
            raise ValueError(
                f"Class {cls} not found in training manifest. "
                f"Present labels: {sorted(label_counts.index.tolist())}"
            )
        counts.append(int(label_counts[cls]))

    total = sum(counts)
    if total == 0:
        raise ValueError("Training manifest has zero subjects")

    proportions = np.array(counts, dtype=float) / float(total)
    majority_class = int(labels[int(np.argmax(counts))])
    return proportions, majority_class


def compute_train_regression_baseline(
    train_manifest: pd.DataFrame,
    baseline_stat: str,
) -> float:
    """
    Compute train-set baseline age (years) for regression comparison.

    Args:
        baseline_stat: 'median' or 'mean'
    """
    if baseline_stat not in ("median", "mean"):
        raise ValueError(f"baseline_stat must be 'median' or 'mean', got {baseline_stat!r}")

    ages = train_manifest["label"].astype(float).values
    if len(ages) == 0:
        raise ValueError("Training manifest has zero subjects for regression baseline")

    if baseline_stat == "median":
        return float(np.median(ages))
    return float(np.mean(ages))
