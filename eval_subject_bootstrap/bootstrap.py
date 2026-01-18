"""
Subject-level bootstrap utilities for uncertainty estimation.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable

# Import metrics (handle both relative and absolute imports)
try:
    from .metrics import compute_metrics, get_primary_metric
except ImportError:
    from metrics import compute_metrics, get_primary_metric


def subject_bootstrap(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_majority: np.ndarray,
    train_proportions: np.ndarray,
    B: int = 2000,
    seed: int = 123,
    primary_metric: str = 'balanced_accuracy',
    metrics_func: Callable = compute_metrics
) -> Dict[str, np.ndarray]:
    """
    Perform subject-level bootstrap resampling.
    
    For each bootstrap replicate:
    1. Sample subjects with replacement
    2. Compute metrics for model, majority, and random baseline
    3. Compute delta (model - random)
    
    Args:
        y_true: True labels, shape (S,) where S = number of subjects
        y_pred_model: Model predictions, shape (S,)
        y_pred_majority: Majority class predictions, shape (S,)
        train_proportions: Training set class proportions, shape (3,)
        B: Number of bootstrap replicates
        seed: Random seed
        primary_metric: Primary metric name
        metrics_func: Function to compute metrics
        
    Returns:
        Dictionary with bootstrap distributions:
        - model_metrics: shape (B, num_metrics) - metrics for model
        - majority_metrics: shape (B, num_metrics) - metrics for majority baseline
        - random_metrics: shape (B, num_metrics) - metrics for random baseline
        - delta_metrics: shape (B, num_metrics) - delta (model - random)
        - metric_names: list of metric names
    """
    np.random.seed(seed)
    S = len(y_true)
    
    # Initialize storage
    model_metrics_list = []
    majority_metrics_list = []
    random_metrics_list = []
    delta_metrics_list = []
    
    # Get metric names from first computation
    sample_metrics = metrics_func(y_true, y_pred_model)
    metric_names = list(sample_metrics.keys())
    if 'per_class_recall' in metric_names:
        metric_names.remove('per_class_recall')  # Handle separately if needed
    
    # Bootstrap loop
    for b in range(B):
        # Sample subject indices with replacement
        idx_b = np.random.choice(S, size=S, replace=True)
        
        # Get sampled data
        y_true_b = y_true[idx_b]
        y_pred_model_b = y_pred_model[idx_b]
        y_pred_majority_b = y_pred_majority[idx_b]
        
        # Generate random baseline predictions for this replicate
        y_pred_random_b = np.random.choice(
            [0, 1, 2],
            size=S,
            p=train_proportions
        )
        
        # Compute metrics
        model_metrics_b = metrics_func(y_true_b, y_pred_model_b)
        majority_metrics_b = metrics_func(y_true_b, y_pred_majority_b)
        random_metrics_b = metrics_func(y_true_b, y_pred_random_b)
        
        # Extract metric values (excluding per_class_recall)
        model_vals = [model_metrics_b[m] for m in metric_names]
        majority_vals = [majority_metrics_b[m] for m in metric_names]
        random_vals = [random_metrics_b[m] for m in metric_names]
        
        # Compute delta
        delta_vals = [m - r for m, r in zip(model_vals, random_vals)]
        
        model_metrics_list.append(model_vals)
        majority_metrics_list.append(majority_vals)
        random_metrics_list.append(random_vals)
        delta_metrics_list.append(delta_vals)
    
    # Convert to arrays
    model_metrics = np.array(model_metrics_list)  # (B, num_metrics)
    majority_metrics = np.array(majority_metrics_list)  # (B, num_metrics)
    random_metrics = np.array(random_metrics_list)  # (B, num_metrics)
    delta_metrics = np.array(delta_metrics_list)  # (B, num_metrics)
    
    return {
        'model_metrics': model_metrics,
        'majority_metrics': majority_metrics,
        'random_metrics': random_metrics,
        'delta_metrics': delta_metrics,
        'metric_names': metric_names
    }


def compute_bootstrap_ci(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval.
    
    Args:
        values: Bootstrap distribution, shape (B,)
        confidence: Confidence level (default: 0.95)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower = np.percentile(values, lower_percentile)
    upper = np.percentile(values, upper_percentile)
    
    return float(lower), float(upper)


def compute_p_value(values: np.ndarray, null_value: float = 0.0) -> float:
    """
    Compute p-value for testing if values are greater than null_value.
    
    P(delta > null_value) = proportion of bootstrap samples where delta > null_value
    
    Args:
        values: Bootstrap distribution, shape (B,)
        null_value: Null hypothesis value (default: 0.0)
        
    Returns:
        P-value (proportion of values > null_value)
    """
    return float(np.mean(values > null_value))


def summarize_bootstrap(
    bootstrap_results: Dict[str, np.ndarray],
    point_metrics: Dict[str, float],
    confidence: float = 0.95
) -> Dict[str, Dict[str, float]]:
    """
    Summarize bootstrap results with CIs and p-values.
    
    Args:
        bootstrap_results: Results from subject_bootstrap()
        point_metrics: Point estimates from full dataset
        confidence: Confidence level
        
    Returns:
        Dictionary with summary statistics per metric:
        {
            'metric_name': {
                'point_estimate': float,
                'ci_low': float,
                'ci_high': float,
                'delta_point': float,
                'delta_ci_low': float,
                'delta_ci_high': float,
                'p_delta_gt_0': float
            }
        }
    """
    metric_names = bootstrap_results['metric_names']
    model_metrics = bootstrap_results['model_metrics']  # (B, num_metrics)
    delta_metrics = bootstrap_results['delta_metrics']  # (B, num_metrics)
    
    summary = {}
    
    for i, metric_name in enumerate(metric_names):
        model_dist = model_metrics[:, i]
        delta_dist = delta_metrics[:, i]
        
        # CIs
        model_ci_low, model_ci_high = compute_bootstrap_ci(model_dist, confidence)
        delta_ci_low, delta_ci_high = compute_bootstrap_ci(delta_dist, confidence)
        
        # P-value
        p_delta_gt_0 = compute_p_value(delta_dist, null_value=0.0)
        
        summary[metric_name] = {
            'point_estimate': point_metrics[metric_name],
            'ci_low': model_ci_low,
            'ci_high': model_ci_high,
            'delta_point': point_metrics[metric_name] - point_metrics.get('random_' + metric_name, 0.0),
            'delta_ci_low': delta_ci_low,
            'delta_ci_high': delta_ci_high,
            'p_delta_gt_0': p_delta_gt_0
        }
    
    return summary
