"""
Classical null-hypothesis significance tests (NHST) for subject-level evaluation.

These p-values follow the standard Monte Carlo form (Davison & Hinkley, 1997):

    p = (1 + #{T_null >= T_obs}) / (M + 1)   [upper-tail, higher is better]
    p = (1 + #{T_null <= T_obs}) / (M + 1)   [lower-tail, lower is better]

where T is the test statistic under the null hypothesis.
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np
from scipy import stats

try:
    from .metrics import get_primary_metric
except ImportError:
    from metrics import get_primary_metric


def _monte_carlo_p_value(null_statistics: np.ndarray, observed: float, *, tail: str) -> float:
    if null_statistics.ndim != 1:
        raise ValueError("null_statistics must be 1-D")
    m = len(null_statistics)
    if m <= 0:
        raise ValueError("null_statistics must be non-empty")

    if tail == "greater":
        count = int(np.sum(null_statistics >= observed))
    elif tail == "less":
        count = int(np.sum(null_statistics <= observed))
    elif tail == "two_sided":
        count = int(np.sum(np.abs(null_statistics) >= abs(observed)))
    else:
        raise ValueError(f"tail must be 'greater', 'less', or 'two_sided', got {tail!r}")

    return float((1 + count) / (m + 1))


def permutation_p_value_vs_random_baseline_classification(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    train_proportions: np.ndarray,
    class_labels: Sequence[int],
    primary_metric: str,
    M: int,
    seed: int,
    metrics_func: Callable,
    higher_is_better: bool,
) -> Dict[str, float]:
    """
    One-sided permutation NHST: model vs stratified random baseline.

    H0: subject-level metric for the model is drawn from the same distribution as
    a stratified random classifier (labels i.i.d. from train class proportions).

    Test statistic: T = primary_metric(y_true, y_pred_model).

    Null distribution: T_m = primary_metric(y_true, y_pred_random_m) with fresh
    stratified random predictions per replicate (model predictions fixed).
    """
    if M <= 0:
        raise ValueError(f"Permutation count M must be positive, got {M}")

    labels = list(class_labels)
    if len(labels) == 0:
        raise ValueError("class_labels must be non-empty")
    if len(train_proportions) != len(labels):
        raise ValueError("train_proportions length must match class_labels")

    rng = np.random.default_rng(seed)
    observed = get_primary_metric(metrics_func(y_true, y_pred_model), primary_metric)

    null_statistics = np.empty(M, dtype=float)
    for m in range(M):
        y_pred_random = rng.choice(labels, size=len(y_true), p=train_proportions)
        null_statistics[m] = get_primary_metric(
            metrics_func(y_true, y_pred_random), primary_metric
        )

    tail = "greater" if higher_is_better else "less"
    p_value = _monte_carlo_p_value(null_statistics, observed, tail=tail)
    return {
        "p_value": p_value,
        "observed_statistic": float(observed),
        "null_mean": float(np.mean(null_statistics)),
        "null_std": float(np.std(null_statistics, ddof=1)),
        "M": float(M),
        "tail": tail,
        "test": "permutation_stratified_random_baseline",
    }


def permutation_p_value_label_association_classification(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    primary_metric: str,
    M: int,
    seed: int,
    metrics_func: Callable,
    higher_is_better: bool,
) -> Dict[str, float]:
    """
    One-sided permutation NHST: are model predictions associated with true labels?

    H0: predictions are independent of labels (labels permuted, predictions fixed).

    Test statistic: T = primary_metric(y_true, y_pred_model).
  Null: T_m = primary_metric(y_perm, y_pred_model).
    """
    if M <= 0:
        raise ValueError(f"Permutation count M must be positive, got {M}")

    rng = np.random.default_rng(seed)
    observed = get_primary_metric(metrics_func(y_true, y_pred_model), primary_metric)

    null_statistics = np.empty(M, dtype=float)
    for _ in range(M):
        y_perm = rng.permutation(y_true)
        null_statistics[_] = get_primary_metric(metrics_func(y_perm, y_pred_model), primary_metric)

    tail = "greater" if higher_is_better else "less"
    p_value = _monte_carlo_p_value(null_statistics, observed, tail=tail)
    return {
        "p_value": p_value,
        "observed_statistic": float(observed),
        "null_mean": float(np.mean(null_statistics)),
        "null_std": float(np.std(null_statistics, ddof=1)),
        "M": float(M),
        "tail": tail,
        "test": "permutation_label_shuffle",
    }


def wilcoxon_p_value_vs_baseline_regression(
    y_true_years: np.ndarray,
    y_pred_model_years: np.ndarray,
    y_pred_baseline_years: np.ndarray,
    primary_metric: str,
) -> Dict[str, float]:
    """
    One-sided Wilcoxon signed-rank NHST on paired per-subject absolute errors.

    H0: median(|y - pred_model|) >= median(|y - pred_baseline|)  (model not better)

    Uses scipy.stats.wilcoxon with alternative='less' on paired absolute errors.
    Only valid when primary_metric is 'mae' (mean absolute error at subject level
    is a monotone function of per-subject absolute errors when aggregated as mean).
    """
    if primary_metric != "mae":
        raise ValueError(
            f"Wilcoxon NHST for regression is defined for primary_metric='mae', "
            f"got {primary_metric!r}"
        )
    if len(y_true_years) != len(y_pred_model_years) or len(y_true_years) != len(y_pred_baseline_years):
        raise ValueError("y_true, y_pred_model, and y_pred_baseline must have the same length")
    if len(y_true_years) < 2:
        raise ValueError("Wilcoxon test requires at least 2 paired subjects")

    abs_err_model = np.abs(y_true_years - y_pred_model_years)
    abs_err_baseline = np.abs(y_true_years - y_pred_baseline_years)

    if np.allclose(abs_err_model, abs_err_baseline):
        return {
            "p_value": 1.0,
            "observed_statistic": float(np.mean(abs_err_model)),
            "test": "wilcoxon_signed_rank_paired_abs_error",
            "alternative": "less",
            "note": "identical_errors",
        }

    result = stats.wilcoxon(
        abs_err_model,
        abs_err_baseline,
        alternative="less",
        method="auto",
    )
    return {
        "p_value": float(result.pvalue),
        "observed_statistic": float(np.mean(abs_err_model)),
        "baseline_statistic": float(np.mean(abs_err_baseline)),
        "wilcoxon_statistic": float(result.statistic),
        "test": "wilcoxon_signed_rank_paired_abs_error",
        "alternative": "less",
    }


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
