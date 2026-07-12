"""
Subject-level bootstrap utilities for uncertainty estimation.
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import numpy as np

try:
    from .metrics import get_primary_metric, scalar_metric_names
except ImportError:
    from metrics import get_primary_metric, scalar_metric_names


def _metric_value_vector(metrics: Dict[str, float], metric_names: Sequence[str]) -> list:
    values = []
    for name in metric_names:
        value = metrics[name]
        if not isinstance(value, (int, float, np.floating, np.integer)):
            raise TypeError(
                f"Bootstrap metric {name!r} must be scalar, got {type(value).__name__}"
            )
        values.append(float(value))
    return values


def subject_bootstrap_classification(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_majority: np.ndarray,
    train_proportions: np.ndarray,
    class_labels: Sequence[int],
    B: int,
    seed: int,
    metrics_func: Callable,
) -> Dict[str, np.ndarray]:
    """
    Subject-level bootstrap for classification tasks.

    For each replicate: resample subjects, compute model / majority / stratified-random
  metrics, and delta (model - random).
    """
    if B <= 0:
        raise ValueError(f"Bootstrap replicate count B must be positive, got {B}")

    labels = list(class_labels)
    if len(labels) == 0:
        raise ValueError("class_labels must be non-empty")
    if len(train_proportions) != len(labels):
        raise ValueError(
            f"train_proportions length {len(train_proportions)} != "
            f"class_labels length {len(labels)}"
        )

    np.random.seed(seed)
    subject_count = len(y_true)

    sample_metrics = metrics_func(y_true, y_pred_model)
    metric_names = list(scalar_metric_names(sample_metrics))
    if not metric_names:
        raise ValueError("metrics_func returned no scalar metrics for bootstrap")

    model_metrics_list = []
    majority_metrics_list = []
    random_metrics_list = []
    delta_metrics_list = []

    for _ in range(B):
        idx_b = np.random.choice(subject_count, size=subject_count, replace=True)
        y_true_b = y_true[idx_b]
        y_pred_model_b = y_pred_model[idx_b]
        y_pred_majority_b = y_pred_majority[idx_b]

        y_pred_random_b = np.random.choice(labels, size=subject_count, p=train_proportions)

        model_metrics_b = metrics_func(y_true_b, y_pred_model_b)
        majority_metrics_b = metrics_func(y_true_b, y_pred_majority_b)
        random_metrics_b = metrics_func(y_true_b, y_pred_random_b)

        model_vals = _metric_value_vector(model_metrics_b, metric_names)
        majority_vals = _metric_value_vector(majority_metrics_b, metric_names)
        random_vals = _metric_value_vector(random_metrics_b, metric_names)
        delta_vals = [m - r for m, r in zip(model_vals, random_vals)]

        model_metrics_list.append(model_vals)
        majority_metrics_list.append(majority_vals)
        random_metrics_list.append(random_vals)
        delta_metrics_list.append(delta_vals)

    return {
        "model_metrics": np.array(model_metrics_list),
        "majority_metrics": np.array(majority_metrics_list),
        "random_metrics": np.array(random_metrics_list),
        "delta_metrics": np.array(delta_metrics_list),
        "metric_names": metric_names,
    }


def subject_bootstrap_regression(
    y_true_years: np.ndarray,
    y_pred_model_years: np.ndarray,
    y_pred_baseline_years: np.ndarray,
    B: int,
    seed: int,
    metrics_func: Callable,
) -> Dict[str, np.ndarray]:
    """
    Subject-level bootstrap for regression (MAE primary; lower is better).

    Baseline predictions are constant per subject (e.g. train-set median age).
    Delta = model_metric - baseline_metric (negative delta => model better for MAE).
    """
    if B <= 0:
        raise ValueError(f"Bootstrap replicate count B must be positive, got {B}")

    np.random.seed(seed)
    subject_count = len(y_true_years)

    sample_metrics = metrics_func(y_true_years, y_pred_model_years)
    metric_names = list(scalar_metric_names(sample_metrics))
    if not metric_names:
        raise ValueError("metrics_func returned no scalar metrics for bootstrap")

    model_metrics_list = []
    baseline_metrics_list = []
    delta_metrics_list = []

    for _ in range(B):
        idx_b = np.random.choice(subject_count, size=subject_count, replace=True)
        y_true_b = y_true_years[idx_b]
        y_pred_model_b = y_pred_model_years[idx_b]
        y_pred_baseline_b = y_pred_baseline_years[idx_b]

        model_metrics_b = metrics_func(y_true_b, y_pred_model_b)
        baseline_metrics_b = metrics_func(y_true_b, y_pred_baseline_b)

        model_vals = _metric_value_vector(model_metrics_b, metric_names)
        baseline_vals = _metric_value_vector(baseline_metrics_b, metric_names)
        delta_vals = [m - b for m, b in zip(model_vals, baseline_vals)]

        model_metrics_list.append(model_vals)
        baseline_metrics_list.append(baseline_vals)
        delta_metrics_list.append(delta_vals)

    return {
        "model_metrics": np.array(model_metrics_list),
        "baseline_metrics": np.array(baseline_metrics_list),
        "delta_metrics": np.array(delta_metrics_list),
        "metric_names": metric_names,
    }


def compute_bootstrap_ci(values: np.ndarray, confidence: float) -> Tuple[float, float]:
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    lower = float(np.percentile(values, lower_percentile))
    upper = float(np.percentile(values, upper_percentile))
    return lower, upper


def compute_proportion(values: np.ndarray, comparator: Callable[[np.ndarray, float], np.ndarray], null_value: float) -> float:
    return float(np.mean(comparator(values, null_value)))


def summarize_bootstrap_classification(
    bootstrap_results: Dict[str, np.ndarray],
    point_metrics: Dict[str, float],
    higher_is_better: bool,
    confidence: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """Summarize classification bootstrap with CIs and one-sided p-value on delta."""
    metric_names = bootstrap_results["metric_names"]
    model_metrics = bootstrap_results["model_metrics"]
    delta_metrics = bootstrap_results["delta_metrics"]

    summary: Dict[str, Dict[str, float]] = {}
    for i, metric_name in enumerate(metric_names):
        model_dist = model_metrics[:, i]
        delta_dist = delta_metrics[:, i]

        model_ci_low, model_ci_high = compute_bootstrap_ci(model_dist, confidence)
        delta_ci_low, delta_ci_high = compute_bootstrap_ci(delta_dist, confidence)

        if higher_is_better:
            p_value = float(np.mean(delta_dist > 0.0))
            p_value_direction = "gt_0"
        else:
            p_value = float(np.mean(delta_dist < 0.0))
            p_value_direction = "lt_0"

        random_key = "random_" + metric_name
        if random_key not in point_metrics:
            raise KeyError(
                f"point_metrics missing {random_key!r} required for delta_point"
            )

        summary[metric_name] = {
            "point_estimate": float(point_metrics[metric_name]),
            "ci_low": model_ci_low,
            "ci_high": model_ci_high,
            "delta_point": float(point_metrics[metric_name]) - float(point_metrics[random_key]),
            "delta_ci_low": delta_ci_low,
            "delta_ci_high": delta_ci_high,
            "bootstrap_prob_better": p_value,
            "bootstrap_prob_direction": p_value_direction,
        }

    return summary


def summarize_bootstrap_regression(
    bootstrap_results: Dict[str, np.ndarray],
    point_metrics: Dict[str, float],
    higher_is_better: bool,
    confidence: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """Summarize regression bootstrap (delta = model - baseline)."""
    metric_names = bootstrap_results["metric_names"]
    model_metrics = bootstrap_results["model_metrics"]
    delta_metrics = bootstrap_results["delta_metrics"]

    summary: Dict[str, Dict[str, float]] = {}
    for i, metric_name in enumerate(metric_names):
        model_dist = model_metrics[:, i]
        delta_dist = delta_metrics[:, i]

        model_ci_low, model_ci_high = compute_bootstrap_ci(model_dist, confidence)
        delta_ci_low, delta_ci_high = compute_bootstrap_ci(delta_dist, confidence)

        if higher_is_better:
            p_value = float(np.mean(delta_dist > 0.0))
            p_value_direction = "gt_0"
        else:
            p_value = float(np.mean(delta_dist < 0.0))
            p_value_direction = "lt_0"

        baseline_key = "baseline_" + metric_name
        if baseline_key not in point_metrics:
            raise KeyError(
                f"point_metrics missing {baseline_key!r} required for delta_point"
            )

        summary[metric_name] = {
            "point_estimate": float(point_metrics[metric_name]),
            "ci_low": model_ci_low,
            "ci_high": model_ci_high,
            "delta_point": float(point_metrics[metric_name]) - float(point_metrics[baseline_key]),
            "delta_ci_low": delta_ci_low,
            "delta_ci_high": delta_ci_high,
            "bootstrap_prob_better": p_value,
            "bootstrap_prob_direction": p_value_direction,
        }

    return summary


def permutation_test_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    primary_metric: str,
    class_labels: Sequence[int],
    M: int,
    seed: int,
    metrics_func: Callable,
    higher_is_better: bool,
) -> float:
    """Backward-compatible wrapper; implementation lives in nhst.py."""
    try:
        from .nhst import permutation_p_value_label_association_classification
    except ImportError:
        from nhst import permutation_p_value_label_association_classification

    del class_labels
    out = permutation_p_value_label_association_classification(
        y_true,
        y_pred,
        primary_metric,
        M,
        seed,
        metrics_func,
        higher_is_better,
    )
    return out["p_value"]
