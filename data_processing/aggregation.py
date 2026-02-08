#!/usr/bin/env python3
"""
Aggregation utilities for segment-level predictions to participant/subject level.

Provides majority vote for classification and median/mean for regression,
so the same logic can be reused in CNN evaluation and eval_subject_bootstrap.
"""

import numpy as np
from collections import defaultdict
from typing import List, Tuple, Optional, Union

# Aggregation strategies
AGGREGATION_MEAN_PROB = "mean_prob"  # Mean probability then argmax (classification only)
AGGREGATION_MAJORITY_VOTE = "majority_vote"  # Mode of class predictions (classification)
AGGREGATION_MEDIAN = "median"  # Median of values (regression)
AGGREGATION_MEAN = "mean"  # Mean of values (regression)


def _mode_per_group(class_indices: np.ndarray) -> int:
    """Return the most frequent value; on tie, return the smallest (deterministic)."""
    if len(class_indices) == 0:
        raise ValueError("Cannot compute mode of empty array")
    values, counts = np.unique(class_indices, return_counts=True)
    max_count = np.max(counts)
    # Take smallest class index among those with max count (deterministic tie-break)
    candidates = values[counts == max_count]
    return int(np.min(candidates))


def aggregate_predictions_by_group(
    predictions: np.ndarray,
    labels: np.ndarray,
    group_ids: List[str],
    prediction_type: str = "classification",
    aggregation: str = AGGREGATION_MAJORITY_VOTE,
    regression_aggregation: str = AGGREGATION_MEDIAN,
    probabilities: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Aggregate segment-level predictions to group (participant/subject) level.
    Each segment/window (e.g. 4s) yields one prediction; aggregation is over
    all such predictions per participant.

    Classification: use majority vote (mode of predicted class over all segments
    per participant) or mean probability then argmax, depending on `aggregation`.
    Regression: use median or mean of predicted values per group.

    Args:
        predictions: Segment-level predictions. Shape (N,) for class indices or
            continuous values.
        labels: Segment-level labels. Shape (N,). Must be consistent per group.
        group_ids: Group identifier per segment. Length N.
        prediction_type: 'classification' or 'regression'.
        aggregation: For classification: AGGREGATION_MAJORITY_VOTE or
            AGGREGATION_MEAN_PROB. Ignored for regression.
        regression_aggregation: For regression: AGGREGATION_MEDIAN or AGGREGATION_MEAN.
        probabilities: Optional (N, num_classes) for classification; used only
            when aggregation is AGGREGATION_MEAN_PROB to compute mean prob per group.

    Returns:
        groups: Unique group ids, shape (G,).
        y_true_agg: One label per group, shape (G,).
        y_pred_agg: One prediction per group, shape (G,).
        prob_agg: If probabilities provided and aggregation is mean_prob, shape (G, num_classes);
            otherwise None.
    """
    if len(predictions) != len(labels) or len(predictions) != len(group_ids):
        raise ValueError(
            "predictions, labels, and group_ids must have the same length. "
            f"Got {len(predictions)}, {len(labels)}, {len(group_ids)}."
        )
    if prediction_type not in ("classification", "regression"):
        raise ValueError(
            f"prediction_type must be 'classification' or 'regression'. Got {prediction_type}."
        )

    # Group by group_id
    by_group = defaultdict(lambda: {"preds": [], "labels": [], "probs": []})
    for i, gid in enumerate(group_ids):
        by_group[gid]["preds"].append(predictions[i])
        by_group[gid]["labels"].append(labels[i])
        if probabilities is not None and i < probabilities.shape[0]:
            by_group[gid]["probs"].append(probabilities[i])

    groups = []
    y_true_agg = []
    y_pred_agg = []
    prob_agg_list = [] if probabilities is not None else None

    for gid in sorted(by_group.keys()):
        data = by_group[gid]
        preds_arr = np.array(data["preds"])
        labels_arr = np.array(data["labels"])

        # Label should be unique per group (subject/participant has one true label)
        unique_labels = np.unique(labels_arr)
        if len(unique_labels) > 1:
            raise ValueError(
                f"Group {gid} has inconsistent labels: {unique_labels.tolist()}. "
                "All segments of the same participant/subject must have the same label."
            )
        true_label = unique_labels[0]

        if prediction_type == "regression":
            if regression_aggregation == AGGREGATION_MEDIAN:
                pred_agg = float(np.median(preds_arr))
            elif regression_aggregation == AGGREGATION_MEAN:
                pred_agg = float(np.mean(preds_arr))
            else:
                raise ValueError(
                    f"regression_aggregation must be '{AGGREGATION_MEDIAN}' or '{AGGREGATION_MEAN}'. "
                    f"Got {regression_aggregation}."
                )
            groups.append(gid)
            y_true_agg.append(true_label)
            y_pred_agg.append(pred_agg)
        else:
            # Classification
            if aggregation == AGGREGATION_MAJORITY_VOTE:
                pred_agg = _mode_per_group(preds_arr)
                groups.append(gid)
                y_true_agg.append(true_label)
                y_pred_agg.append(pred_agg)
                if prob_agg_list is not None and data["probs"]:
                    prob_agg_list.append(np.mean(data["probs"], axis=0))
                elif prob_agg_list is not None:
                    prob_agg_list.append(None)
            elif aggregation == AGGREGATION_MEAN_PROB:
                if not data["probs"] or probabilities is None:
                    raise ValueError(
                        "aggregation='mean_prob' requires probabilities to be provided."
                    )
                mean_prob = np.mean(data["probs"], axis=0)
                pred_agg = int(np.argmax(mean_prob))
                groups.append(gid)
                y_true_agg.append(true_label)
                y_pred_agg.append(pred_agg)
                if prob_agg_list is not None:
                    prob_agg_list.append(mean_prob)
            else:
                raise ValueError(
                    f"aggregation must be '{AGGREGATION_MAJORITY_VOTE}' or '{AGGREGATION_MEAN_PROB}'. "
                    f"Got {aggregation}."
                )

    if prob_agg_list is not None and all(p is not None for p in prob_agg_list):
        prob_agg_out = np.array(prob_agg_list)
    else:
        prob_agg_out = None

    return (
        np.array(groups),
        np.array(y_true_agg),
        np.array(y_pred_agg),
        prob_agg_out,
    )


def aggregate_segment_probs_by_subject_majority_vote(
    segment_probs: np.ndarray,
    segment_labels: np.ndarray,
    segment_subject_ids: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Aggregate segment-level probabilities to subject-level using majority vote.

    For each subject: segment predictions = argmax(probs); subject prediction = mode(segment predictions).
    Convenience wrapper for eval_subject_bootstrap that matches the existing
    aggregate_segments_by_subject return shape.

    Args:
        segment_probs: (N_segments, num_classes).
        segment_labels: (N_segments,).
        segment_subject_ids: Length N_segments.

    Returns:
        subjects: (S,), y_true_subject: (S,), y_pred_subject: (S,),
        p_subject_dict: subject_id -> mean probability vector (for compatibility).
    """
    segment_preds = np.argmax(segment_probs, axis=1)
    subjects, y_true, y_pred, _ = aggregate_predictions_by_group(
        segment_preds,
        segment_labels,
        segment_subject_ids,
        prediction_type="classification",
        aggregation=AGGREGATION_MAJORITY_VOTE,
        probabilities=segment_probs,
    )
    # Mean probability per subject (for compatibility with existing callers)
    by_subject = defaultdict(list)
    for i, sid in enumerate(segment_subject_ids):
        by_subject[sid].append(segment_probs[i])
    p_subject_dict = {sid: np.mean(by_subject[sid], axis=0) for sid in subjects}
    return subjects, y_true, y_pred, p_subject_dict
