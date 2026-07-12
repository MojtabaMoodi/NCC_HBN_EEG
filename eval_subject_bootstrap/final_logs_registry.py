"""
Resolve finetuned checkpoint paths and model configs under ``final_logs/``.

Paths are derived from the on-disk layout produced by final_logs training runs.
Each resolved path is verified to exist; missing checkpoints raise FileNotFoundError.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

try:
    from .task_specs import (
        BACKEND_CNN,
        BACKEND_IDS,
        BACKEND_LABRAM,
        BACKEND_RESNET18,
        BACKEND_RESNET34,
        BACKEND_RESNET50,
        TASK_AGE_CLASSIFICATION,
        TASK_AGE_REGRESSION,
        TASK_GENDER_CLASSIFICATION,
        TASK_IDS,
        WINDOW_SIZES,
        get_task_spec,
    )
except ImportError:
    from task_specs import (
        BACKEND_CNN,
        BACKEND_IDS,
        BACKEND_LABRAM,
        BACKEND_RESNET18,
        BACKEND_RESNET34,
        BACKEND_RESNET50,
        TASK_AGE_CLASSIFICATION,
        TASK_AGE_REGRESSION,
        TASK_GENDER_CLASSIFICATION,
        TASK_IDS,
        WINDOW_SIZES,
        get_task_spec,
    )

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# LaBraM architecture flags — must match LaBraM/confusion_matrix/run_all_confusion_matrices.sh
LABRAM_CTOR_KWARGS: Dict[str, Any] = {
    "drop_rate": 0.0,
    "drop_path_rate": 0.1,
    "attn_drop_rate": 0.0,
    "use_mean_pooling": True,
    "init_scale": 0.001,
    "use_rel_pos_bias": True,
    "use_abs_pos_emb": True,
    "init_values": 0.1,
    "qkv_bias": True,
    "multi_output": False,
}

RESNET_DEPTHS = {
    BACKEND_RESNET18: 18,
    BACKEND_RESNET34: 34,
    BACKEND_RESNET50: 50,
}

CNN_MODEL_TYPES = {
    TASK_AGE_CLASSIFICATION: "age_cnn",
    TASK_GENDER_CLASSIFICATION: "gender_cnn",
    TASK_AGE_REGRESSION: "age_regression_cnn",
}

RESNET_MODEL_TYPES = {
    18: {
        TASK_AGE_CLASSIFICATION: "age_resnet",
        TASK_GENDER_CLASSIFICATION: "gender_resnet",
        TASK_AGE_REGRESSION: "age_regression_resnet",
    },
    34: {
        TASK_AGE_CLASSIFICATION: "age_resnet34",
        TASK_GENDER_CLASSIFICATION: "gender_resnet34",
        TASK_AGE_REGRESSION: "age_regression_resnet34",
    },
    50: {
        TASK_AGE_CLASSIFICATION: "age_resnet50",
        TASK_GENDER_CLASSIFICATION: "gender_resnet50",
        TASK_AGE_REGRESSION: "age_regression_resnet50",
    },
}


@dataclass(frozen=True)
class ModelEntry:
    backend: str
    display_name: str
    ckpt_path: Path
    model_config: Dict[str, Any]


def _require_exists(path: Path, description: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint for {description}: {resolved}"
        )
    return resolved


def resolve_checkpoint_path(
    eeg_root: Path,
    task_id: str,
    segment_length: str,
    backend: str,
) -> Path:
    if task_id not in TASK_IDS:
        raise ValueError(f"Unknown task_id {task_id!r}. Expected one of: {sorted(TASK_IDS)}")
    if segment_length not in WINDOW_SIZES:
        raise ValueError(
            f"Unknown segment_length {segment_length!r}. Expected one of: {sorted(WINDOW_SIZES)}"
        )
    if backend not in BACKEND_IDS:
        raise ValueError(f"Unknown backend {backend!r}. Expected one of: {sorted(BACKEND_IDS)}")

    root = eeg_root.resolve()
    desc = f"{task_id} {segment_length} {backend}"

    if backend == BACKEND_CNN:
        if task_id == TASK_AGE_CLASSIFICATION:
            rel = (
                f"final_logs/CNN/age_classification_{segment_length}/checkpoints/"
                f"age_classification_{segment_length}_best.pth"
            )
        elif task_id == TASK_GENDER_CLASSIFICATION:
            rel = (
                f"final_logs/CNN/gender_baseline_{segment_length}/checkpoints/"
                f"gender_baseline_{segment_length}_best.pth"
            )
        elif task_id == TASK_AGE_REGRESSION:
            rel = (
                f"final_logs/CNN/age_regression_{segment_length}/"
                f"age_regression_{segment_length}_best.pth"
            )
        else:
            raise ValueError(f"Unsupported task for CNN: {task_id}")
        return _require_exists(root / rel, desc)

    if backend in RESNET_DEPTHS:
        depth = RESNET_DEPTHS[backend]
        if task_id == TASK_AGE_CLASSIFICATION:
            rel = (
                f"final_logs/RESNET/RESNET{depth}_age_{segment_length}/checkpoints/"
                f"age_resnet{depth}_{segment_length}_best.pth"
            )
        elif task_id == TASK_GENDER_CLASSIFICATION:
            rel = (
                f"final_logs/RESNET/RESNET{depth}_gender_{segment_length}/checkpoints/"
                f"gender_resnet{depth}_{segment_length}_best.pth"
            )
        elif task_id == TASK_AGE_REGRESSION:
            rel = (
                f"final_logs/RESNET/RESNET{depth}_age_regression_{segment_length}/checkpoints/"
                f"age_regression_resnet{depth}_{segment_length}_best.pth"
            )
        else:
            raise ValueError(f"Unsupported task for ResNet: {task_id}")
        return _require_exists(root / rel, desc)

    if backend == BACKEND_LABRAM:
        if task_id == TASK_AGE_CLASSIFICATION:
            rel = (
                f"final_logs/LaBraM/labram_age_{segment_length}/output/"
                f"age_labram_{segment_length}_best.pth"
            )
        elif task_id == TASK_GENDER_CLASSIFICATION:
            rel = (
                f"final_logs/LaBraM/labram_gender_{segment_length}/output/"
                f"gender_labram_{segment_length}_best.pth"
            )
        elif task_id == TASK_AGE_REGRESSION:
            rel = (
                f"final_logs/LaBraM/LABRAM_base_age_regression_{segment_length}/output/"
                f"checkpoint-best.pth"
            )
        else:
            raise ValueError(f"Unsupported task for LaBraM: {task_id}")
        return _require_exists(root / rel, desc)

    raise ValueError(f"Unhandled backend: {backend}")


def resolve_model_num_classes(task_id: str, backend: str) -> int:
    """Output head size for checkpoint loading (may differ from label space for LaBraM binary)."""
    if task_id == TASK_GENDER_CLASSIFICATION and backend == BACKEND_LABRAM:
        return 1
    spec = get_task_spec(task_id)
    if spec.num_classes is None:
        raise ValueError(f"task {task_id} has no num_classes for model loading")
    return int(spec.num_classes)


def build_model_config(
    eeg_root: Path,
    task_id: str,
    segment_length: str,
    backend: str,
) -> ModelEntry:
    task_spec = get_task_spec(task_id)
    ckpt_path = resolve_checkpoint_path(eeg_root, task_id, segment_length, backend)
    model_num_classes = resolve_model_num_classes(task_id, backend)

    if backend == BACKEND_CNN:
        model_type = CNN_MODEL_TYPES[task_id]
        config: Dict[str, Any] = {
            "name": "cnn",
            "model_type": model_type,
            "ckpt_path": str(ckpt_path),
            "num_classes": model_num_classes,
            "prediction_type": task_spec.prediction_type,
            "ctor_kwargs": {},
        }
        display_name = f"cnn_{task_id}_{segment_length}"

    elif backend in RESNET_DEPTHS:
        depth = RESNET_DEPTHS[backend]
        model_type = RESNET_MODEL_TYPES[depth][task_id]
        config = {
            "name": "resnet",
            "model_type": model_type,
            "ckpt_path": str(ckpt_path),
            "num_classes": model_num_classes,
            "prediction_type": task_spec.prediction_type,
            "ctor_kwargs": {},
        }
        display_name = f"resnet{depth}_{task_id}_{segment_length}"

    elif backend == BACKEND_LABRAM:
        config = {
            "name": "labram",
            "model_type": "labram_base_patch200_200",
            "ckpt_path": str(ckpt_path),
            "num_classes": model_num_classes,
            "prediction_type": task_spec.prediction_type,
            "ctor_kwargs": dict(LABRAM_CTOR_KWARGS),
        }
        display_name = f"labram_{task_id}_{segment_length}"

    else:
        raise ValueError(f"Unhandled backend: {backend}")

    return ModelEntry(
        backend=backend,
        display_name=display_name,
        ckpt_path=ckpt_path,
        model_config=config,
    )


def list_model_entries(
    eeg_root: Path,
    task_id: str,
    segment_length: str,
    backends: Sequence[str],
) -> List[ModelEntry]:
    unknown = [b for b in backends if b not in BACKEND_IDS]
    if unknown:
        raise ValueError(f"Unknown backends: {unknown}. Expected subset of {sorted(BACKEND_IDS)}")

    entries: List[ModelEntry] = []
    for backend in backends:
        entries.append(
            build_model_config(eeg_root, task_id, segment_length, backend)
        )
    return entries


def manifest_paths(manifest_dir: Path, task_id: str, segment_length: str) -> tuple[Path, Path]:
    base = manifest_dir / task_id
    return (
        base / f"train_manifest_{segment_length}.csv",
        base / f"test_manifest_{segment_length}.csv",
    )
