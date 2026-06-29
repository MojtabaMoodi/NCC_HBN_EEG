#!/usr/bin/env python3
"""
Load a finetuned LaBraM checkpoint and compute confusion matrices for age/gender test eval.

Uses the same test loader, participant aggregation, and label semantics as
``run_class_finetuning.py --eval --aggregate_by_participant majority_vote``.

Checkpoints from training runs are typically::

    final_logs/LaBraM/labram_{age,gender}_{1s,2s,4s}/output/{age,gender}_labram_{1s,2s,4s}_best.pth

Example (age, 4s)::

    python LaBraM/eval_confusion_matrix.py \\
      --checkpoint final_logs/LaBraM/labram_age_4s/output/age_labram_4s_best.pth \\
      --dataset age_baseline \\
      --segment_length 4s \\
      --data_path /home/mojtabam/scratch/processed_eeg_data_hdf5 \\
      --task_type both \\
      --aggregate_by_participant majority_vote \\
      --batch_size 64 \\
      --num_workers 4 \\
      --device cuda \\
      --model labram_base_patch200_200 \\
      --qkv_bias true \\
      --rel_pos_bias true \\
      --abs_pos_emb true \\
      --use_mean_pooling true \\
      --layer_scale_init_value 0.1 \\
      --init_scale 0.001 \\
      --drop 0.0 \\
      --attn_drop_rate 0.0 \\
      --drop_path 0.1 \\
      --output_json LaBraM/confusion_matrix/results/age_4s_confusion_matrix.json

See ``LaBraM/confusion_matrix/README.md`` for batch commands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from timm.models import create_model
from torch.utils.data import DataLoader

_labram_dir = Path(__file__).resolve().parent
_project_root = _labram_dir.parent
_labram_dir_str = str(_labram_dir)
try:
    sys.path.remove(_labram_dir_str)
except ValueError:
    pass
sys.path.insert(0, _labram_dir_str)
for _p in (_project_root, _project_root / "data_processing"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import modeling_finetune  # noqa: F401,E402  registers labram_* with timm

from classification_eval_utils import (  # noqa: E402
    LABRAM_CHANNEL_NAMES,
    collect_model_segment_predictions,
    compute_classification_confusion_matrices,
)
from dataset_config import (  # noqa: E402
    CUSTOM_DATASET_CONFIGS,
    get_dataset_config,
    get_dataset_type_and_params,
)
from labram_dataset import collate_labram_with_participant_ids, prepare_labram_dataset  # noqa: E402

SUPPORTED_DATASET_TYPES = frozenset({"age", "gender"})


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(
        f"Expected true/false boolean string; got {value!r}."
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to finetuned checkpoint (e.g. .../output/age_labram_4s_best.pth).",
    )
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(CUSTOM_DATASET_CONFIGS.keys()),
        help="LaBraM dataset name (age_baseline or gender_baseline for standard runs).",
    )
    p.add_argument(
        "--segment_length",
        type=str,
        required=True,
        choices=["1s", "2s", "4s"],
        help="Segment length (must match HDF5 file names under --data_path).",
    )
    p.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="HDF5 directory (train/val/test multipart files for the chosen segment length).",
    )
    p.add_argument(
        "--task_type",
        type=str,
        required=True,
        choices=["active", "passive", "both"],
        help="Task filter for the test loader (LaBraM final logs use 'both').",
    )
    p.add_argument(
        "--aggregate_by_participant",
        type=str,
        required=True,
        choices=["majority_vote"],
        help="Participant aggregation (LaBraM eval uses majority_vote).",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Test DataLoader batch size (LaBraM eval uses 64).",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        required=True,
        help="Test DataLoader worker count (LaBraM eval uses 4 for 2s/4s, 0 for 1s).",
    )
    p.add_argument(
        "--device",
        type=str,
        required=True,
        help="Torch device for inference (e.g. cuda or cpu).",
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="Timm model name (must match the trained checkpoint).",
    )
    p.add_argument("--qkv_bias", type=_parse_bool, required=True, help="Model qkv_bias flag.")
    p.add_argument("--rel_pos_bias", type=_parse_bool, required=True, help="Model rel_pos_bias flag.")
    p.add_argument("--abs_pos_emb", type=_parse_bool, required=True, help="Model abs_pos_emb flag.")
    p.add_argument("--use_mean_pooling", type=_parse_bool, required=True, help="Model use_mean_pooling flag.")
    p.add_argument(
        "--layer_scale_init_value",
        type=float,
        required=True,
        help="Model layer_scale_init_value (init_values).",
    )
    p.add_argument("--init_scale", type=float, required=True, help="Model init_scale.")
    p.add_argument("--drop", type=float, required=True, help="Model drop_rate.")
    p.add_argument("--attn_drop_rate", type=float, required=True, help="Model attn_drop_rate.")
    p.add_argument("--drop_path", type=float, required=True, help="Model drop_path_rate.")
    p.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to write JSON results (confusion matrices + metadata).",
    )
    return p.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"--device {device_str!r} requested but CUDA is not available.")
    return device


def _build_test_loader(
    *,
    dataset_type: str,
    hdf5_dir: str,
    segment_length: str,
    task_type: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    _, _, test_ds = prepare_labram_dataset(
        dataset_type=dataset_type,
        hdf5_dir=hdf5_dir,
        segment_length=segment_length,
        task_type=task_type,
        random_seed=42,
    )
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_labram_with_participant_ids,
    )


def _build_model(args: argparse.Namespace, *, nb_classes: int) -> torch.nn.Module:
    return create_model(
        args.model,
        pretrained=False,
        num_classes=nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        use_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        qkv_bias=args.qkv_bias,
        multi_output=False,
        prediction_type="classification",
    )


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" not in checkpoint:
        raise KeyError(
            f"Checkpoint {checkpoint_path} has no 'model' key; "
            f"found keys: {sorted(checkpoint.keys())}."
        )
    model.load_state_dict(checkpoint["model"])


def _eval_confusion_matrices(
    *,
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    is_binary: bool,
    nb_classes_config: int,
    aggregate_by_participant: str,
) -> Dict[str, Any]:
    pred_all, true_all, participant_ids = collect_model_segment_predictions(
        model,
        data_loader,
        device,
        ch_names=LABRAM_CHANNEL_NAMES,
        is_binary=is_binary,
    )
    confusion = compute_classification_confusion_matrices(
        pred_all,
        true_all,
        participant_ids,
        is_binary=is_binary,
        nb_classes_config=nb_classes_config,
        aggregate_by_participant=aggregate_by_participant,
    )
    return {
        "confusion_matrices": confusion,
        "n_test_segments": int(len(true_all)),
        "n_participants": int(len(set(participant_ids))) if participant_ids else 0,
    }


def _print_summary(confusion: Dict[str, Any], header: str) -> None:
    print(f"\n{header}")
    for level in ("segment", "participant_mean_prob", "participant_majority_vote"):
        payload = confusion.get(level)
        if payload is None:
            continue
        labels = payload["labels"]
        matrix = payload["matrix"]
        print(f"  {level} (n={payload['n_samples']}) labels={labels}")
        for row_idx, row in enumerate(matrix):
            print(f"    true {labels[row_idx]}: {row}")


def main() -> None:
    args = _parse_args()
    dataset_type, _ = get_dataset_type_and_params(args.dataset)
    if dataset_type not in SUPPORTED_DATASET_TYPES:
        raise SystemExit(
            f"Dataset {args.dataset!r} has type {dataset_type!r}. "
            f"This script supports only {sorted(SUPPORTED_DATASET_TYPES)}."
        )

    config = get_dataset_config(args.dataset)
    nb_classes = int(config["nb_classes"])
    is_binary = nb_classes == 1

    hdf5_dir = str(Path(args.data_path).expanduser().resolve())
    if not Path(hdf5_dir).is_dir():
        raise SystemExit(f"--data_path is not a directory: {hdf5_dir}")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    device = _resolve_device(args.device)

    model = _build_model(args, nb_classes=nb_classes)
    model.to(device)
    _load_checkpoint(model, checkpoint_path, device)

    loader = _build_test_loader(
        dataset_type=dataset_type,
        hdf5_dir=hdf5_dir,
        segment_length=args.segment_length,
        task_type=args.task_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    eval_stats = _eval_confusion_matrices(
        model=model,
        data_loader=loader,
        device=device,
        is_binary=is_binary,
        nb_classes_config=nb_classes,
        aggregate_by_participant=args.aggregate_by_participant,
    )
    _print_summary(eval_stats["confusion_matrices"], f"LaBraM confusion matrices — {args.dataset} {args.segment_length}")

    payload: Dict[str, Any] = {
        "model": "labram_finetuned",
        "checkpoint": str(checkpoint_path),
        "dataset": args.dataset,
        "dataset_type": dataset_type,
        "segment_length": args.segment_length,
        "data_path": hdf5_dir,
        "task_type": args.task_type,
        "aggregate_by_participant": args.aggregate_by_participant,
        "nb_classes_config": nb_classes,
        "model_architecture": {
            "model": args.model,
            "qkv_bias": args.qkv_bias,
            "rel_pos_bias": args.rel_pos_bias,
            "abs_pos_emb": args.abs_pos_emb,
            "use_mean_pooling": args.use_mean_pooling,
            "layer_scale_init_value": args.layer_scale_init_value,
            "init_scale": args.init_scale,
            "drop": args.drop,
            "attn_drop_rate": args.attn_drop_rate,
            "drop_path": args.drop_path,
        },
        "device": str(device),
        "test": eval_stats,
    }

    out_path = Path(args.output_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
