"""
Task definitions for subject-level bootstrap evaluation.

Aggregation and metrics match final_logs / LaBraM eval conventions:
- classification: participant majority vote (confidence-weighted when probabilities available)
- regression: participant median of segment predictions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Tuple


TASK_AGE_CLASSIFICATION = "age_classification"
TASK_GENDER_CLASSIFICATION = "gender_classification"
TASK_AGE_REGRESSION = "age_regression"

TASK_IDS: FrozenSet[str] = frozenset(
    {TASK_AGE_CLASSIFICATION, TASK_GENDER_CLASSIFICATION, TASK_AGE_REGRESSION}
)

WINDOW_SIZES: FrozenSet[str] = frozenset({"1s", "2s", "4s"})

BACKEND_CNN = "cnn"
BACKEND_RESNET18 = "resnet18"
BACKEND_RESNET34 = "resnet34"
BACKEND_RESNET50 = "resnet50"
BACKEND_LABRAM = "labram"

BACKEND_IDS: FrozenSet[str] = frozenset(
    {
        BACKEND_CNN,
        BACKEND_RESNET18,
        BACKEND_RESNET34,
        BACKEND_RESNET50,
        BACKEND_LABRAM,
    }
)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    prediction_type: str
    num_classes: int | None
    class_labels: Tuple[int, ...] | None
    primary_metric: str
    higher_is_better: bool
    aggregation: str
    manifest_label_kind: str


@dataclass(frozen=True)
class WindowSpec:
    segment_length: str
    num_workers: int


WINDOW_SPECS = {
    "1s": WindowSpec(segment_length="1s", num_workers=0),
    "2s": WindowSpec(segment_length="2s", num_workers=4),
    "4s": WindowSpec(segment_length="4s", num_workers=4),
}


TASK_SPECS = {
    TASK_AGE_CLASSIFICATION: TaskSpec(
        task_id=TASK_AGE_CLASSIFICATION,
        prediction_type="classification",
        num_classes=3,
        class_labels=(0, 1, 2),
        primary_metric="balanced_accuracy",
        higher_is_better=True,
        aggregation="majority_vote",
        manifest_label_kind="age_bin",
    ),
    TASK_GENDER_CLASSIFICATION: TaskSpec(
        task_id=TASK_GENDER_CLASSIFICATION,
        prediction_type="classification",
        num_classes=2,
        class_labels=(0, 1),
        primary_metric="balanced_accuracy",
        higher_is_better=True,
        aggregation="majority_vote",
        manifest_label_kind="gender",
    ),
    TASK_AGE_REGRESSION: TaskSpec(
        task_id=TASK_AGE_REGRESSION,
        prediction_type="regression",
        num_classes=1,
        class_labels=None,
        primary_metric="mae",
        higher_is_better=False,
        aggregation="median",
        manifest_label_kind="age_years",
    ),
}


def get_task_spec(task_id: str) -> TaskSpec:
    if task_id not in TASK_SPECS:
        raise ValueError(
            f"Unknown task_id {task_id!r}. Expected one of: {sorted(TASK_IDS)}"
        )
    return TASK_SPECS[task_id]


def get_window_spec(segment_length: str) -> WindowSpec:
    if segment_length not in WINDOW_SPECS:
        raise ValueError(
            f"Unknown segment_length {segment_length!r}. Expected one of: {sorted(WINDOW_SPECS)}"
        )
    return WINDOW_SPECS[segment_length]
