"""
Random classifier outputs for LaBraM age/gender evaluation baselines.

Strategies (must be chosen explicitly by the caller):
- ``uniform``: each class equally likely (LaBraM ``random_baseline = 1.0 / nb_classes`` for age;
  gender uses two classes with p=0.5 each).
- ``train_segment_frequency``: sample labels using training-split **segment** class counts
  (same metadata scan as ``compute_class_weights_from_train_hdf5``).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from data_processing.eeg_dataset import compute_train_class_counts_from_hdf5

RANDOM_STRATEGY_UNIFORM = "uniform"
RANDOM_STRATEGY_TRAIN_SEGMENT_FREQUENCY = "train_segment_frequency"
RANDOM_STRATEGIES = (
    RANDOM_STRATEGY_UNIFORM,
    RANDOM_STRATEGY_TRAIN_SEGMENT_FREQUENCY,
)


def resolve_class_proportions(
    *,
    random_strategy: str,
    hdf5_dir: str,
    segment_length: str,
    target_type: str,
    task_type: str,
    num_classes: int,
) -> np.ndarray:
    if random_strategy not in RANDOM_STRATEGIES:
        raise ValueError(
            f"random_strategy must be one of {RANDOM_STRATEGIES}, got {random_strategy!r}."
        )
    if target_type not in ("gender", "age"):
        raise ValueError(
            f"target_type must be 'gender' or 'age' for random classifier, got {target_type!r}."
        )

    if random_strategy == RANDOM_STRATEGY_UNIFORM:
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")
        return np.full(num_classes, 1.0 / float(num_classes), dtype=np.float64)

    counts = compute_train_class_counts_from_hdf5(
        hdf5_dir,
        segment_length,
        target_type,
        task_type=task_type,
    )
    if len(counts) != num_classes:
        raise ValueError(
            f"Train class count length {len(counts)} != num_classes {num_classes}."
        )
    total = float(sum(counts))
    if total <= 0.0:
        raise ValueError(
            f"No training segments found for target_type={target_type!r}, task_type={task_type!r}."
        )
    return np.asarray(counts, dtype=np.float64) / total


def generate_random_segment_predictions(
    n_samples: int,
    *,
    num_classes: int,
    is_binary: bool,
    class_proportions: Sequence[float],
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample independent random predictions per segment.

    Returns:
        pred_for_metrics: segment outputs in the same shape/type as ``evaluate`` stores
            (sigmoid probabilities for binary, logits for multiclass).
        logits_for_loss: raw logits passed to BCE / CE (same convention as ``evaluate``).
        sampled_labels: integer class labels drawn for each segment.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}.")
    proportions = np.asarray(class_proportions, dtype=np.float64).reshape(-1)
    if len(proportions) != num_classes:
        raise ValueError(
            f"class_proportions length {len(proportions)} != num_classes {num_classes}."
        )
    if np.any(proportions < 0):
        raise ValueError("class_proportions must be non-negative.")
    prop_sum = float(proportions.sum())
    if prop_sum <= 0.0:
        raise ValueError("class_proportions must sum to a positive value.")
    proportions = proportions / prop_sum

    rng = np.random.default_rng(random_seed)
    sampled = rng.choice(num_classes, size=n_samples, p=proportions).astype(np.int64)

    if is_binary:
        if num_classes != 2:
            raise ValueError(
                f"Binary random classifier expects num_classes=2, got {num_classes}."
            )
        logits = np.where(sampled.reshape(-1, 1) == 1, 10.0, -10.0)
        pred_for_metrics = 1.0 / (1.0 + np.exp(-logits))
        return pred_for_metrics, logits, sampled

    one_hot = np.zeros((n_samples, num_classes), dtype=np.float64)
    one_hot[np.arange(n_samples), sampled] = 1.0
    logits = np.log(one_hot + 1e-9)
    return logits, logits, sampled
