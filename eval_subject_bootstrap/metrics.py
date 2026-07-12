"""
Metrics utilities for subject-level evaluation (classification and regression).
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Sequence[int],
) -> Dict[str, float]:
    """
    Compute classification metrics for subject-level predictions.

    Args:
        y_true: True labels, shape (N,)
        y_pred: Predicted labels, shape (N,)
        class_labels: Ordered class indices present in the task (e.g. (0, 1, 2))

    Returns:
        Dictionary with accuracy, balanced_accuracy, macro_f1, per_class_recall
    """
    labels = list(class_labels)
    if len(labels) == 0:
        raise ValueError("class_labels must be non-empty for classification metrics")

    unexpected_true = set(np.unique(y_true).tolist()) - set(labels)
    unexpected_pred = set(np.unique(y_pred).tolist()) - set(labels)
    if unexpected_true:
        raise ValueError(
            f"y_true contains labels outside class_labels {labels}: {sorted(unexpected_true)}"
        )
    if unexpected_pred:
        raise ValueError(
            f"y_pred contains labels outside class_labels {labels}: {sorted(unexpected_pred)}"
        )

    metrics: Dict[str, float] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    per_class_recall = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    # sklearn 1.7.x balanced_accuracy_score has no `labels` arg; mean recall matches
    # balanced accuracy when all classes are listed explicitly.
    metrics["balanced_accuracy"] = float(np.mean(per_class_recall))
    metrics["macro_f1"] = float(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )

    metrics["per_class_recall"] = {
        int(cls): float(recall) for cls, recall in zip(labels, per_class_recall)
    }
    return metrics


def compute_regression_metrics(
    y_true_years: np.ndarray,
    y_pred_years: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression metrics in years.

    Args:
        y_true_years: True ages in years, shape (N,)
        y_pred_years: Predicted ages in years, shape (N,)
    """
    if len(y_true_years) != len(y_pred_years):
        raise ValueError(
            f"y_true_years and y_pred_years length mismatch: "
            f"{len(y_true_years)} vs {len(y_pred_years)}"
        )
    if len(y_true_years) == 0:
        raise ValueError("Cannot compute regression metrics on empty arrays")

    mae = float(mean_absolute_error(y_true_years, y_pred_years))
    mse = float(mean_squared_error(y_true_years, y_pred_years))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true_years, y_pred_years))

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Sequence[int],
) -> np.ndarray:
    """Compute confusion matrix with explicit label ordering."""
    labels = list(class_labels)
    if len(labels) == 0:
        raise ValueError("class_labels must be non-empty")
    return confusion_matrix(y_true, y_pred, labels=labels)


def get_primary_metric(
    metrics: Dict[str, float],
    primary_metric: str,
) -> float:
    """Return the primary metric value or raise if missing."""
    if primary_metric not in metrics:
        raise ValueError(
            f"Primary metric {primary_metric!r} not found in metrics. "
            f"Available: {list(metrics.keys())}"
        )
    return float(metrics[primary_metric])


def scalar_metric_names(metrics: Dict[str, float]) -> Iterable[str]:
    """Metric keys suitable for bootstrap arrays (exclude nested dicts)."""
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating, np.integer)):
            yield key
