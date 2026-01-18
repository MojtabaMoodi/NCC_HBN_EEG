"""
Metrics utilities for subject-level evaluation.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    recall_score
)
from typing import Dict, Tuple


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute all metrics for subject-level predictions.
    
    Args:
        y_true: True labels, shape (N,)
        y_pred: Predicted labels, shape (N,)
        
    Returns:
        Dictionary with metrics:
        - accuracy
        - balanced_accuracy
        - macro_f1
        - per_class_recall: dict mapping class to recall
    """
    metrics = {}
    
    # Overall metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Per-class recall
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    metrics['per_class_recall'] = {
        int(cls): float(recall)
        for cls, recall in enumerate(per_class_recall)
    }
    
    return metrics


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels, shape (N,)
        y_pred: Predicted labels, shape (N,)
        
    Returns:
        Confusion matrix, shape (3, 3)
    """
    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2])


def get_primary_metric(metrics: Dict[str, float], primary_metric: str = 'balanced_accuracy') -> float:
    """
    Get primary metric value.
    
    Args:
        metrics: Dictionary of metrics
        primary_metric: Name of primary metric
        
    Returns:
        Primary metric value
    """
    if primary_metric not in metrics:
        raise ValueError(
            f"Primary metric '{primary_metric}' not found in metrics. "
            f"Available: {list(metrics.keys())}"
        )
    return metrics[primary_metric]
