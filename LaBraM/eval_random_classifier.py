#!/usr/bin/env python3
"""
Evaluate a random classifier on the LaBraM age/gender test split with the same metrics
and participant aggregation as ``run_class_finetuning.py --eval``.

Reports segment-level metrics plus participant mean-probability and majority-vote metrics
when ``--aggregate_by_participant majority_vote`` (same as LaBraM final logs).

Examples
--------
Age, uniform random labels (matches LaBraM ``random_baseline = 1/nb_classes``)::

    python LaBraM/eval_random_classifier.py \\
      --dataset age_baseline \\
      --segment_length 4s \\
      --data_path /home/mojtabam/scratch/processed_eeg_data_hdf5 \\
      --task_type both \\
      --random_seed 42 \\
      --random_strategy uniform \\
      --aggregate_by_participant majority_vote \\
      --batch_size 64 \\
      --num_workers 4 \\
      --output_json LaBraM/random_classifier/results/age_4s_uniform.json

Gender, train segment frequency::

    python LaBraM/eval_random_classifier.py \\
      --dataset gender_baseline \\
      --segment_length 2s \\
      --data_path /home/mojtabam/scratch/processed_eeg_data_hdf5 \\
      --task_type both \\
      --random_seed 42 \\
      --random_strategy train_segment_frequency \\
      --aggregate_by_participant majority_vote \\
      --batch_size 64 \\
      --num_workers 4 \\
      --output_json LaBraM/random_classifier/results/gender_2s_train_segment_frequency.json

See ``LaBraM/random_classifier/README.md`` for the full results layout and
``run_train_segment_frequency_eval.sh`` to run all age/gender window lengths.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

_labram_dir = Path(__file__).resolve().parent
_project_root = _labram_dir.parent
for _p in (_project_root, _labram_dir, _project_root / "data_processing"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from classification_eval_utils import (  # noqa: E402
    collect_test_labels_and_participant_ids,
    compute_classification_eval_stats,
    compute_mean_classification_loss,
)
from dataset_config import (  # noqa: E402
    CUSTOM_DATASET_CONFIGS,
    get_dataset_config,
    get_dataset_type_and_params,
)
from labram_dataset import collate_labram_with_participant_ids, prepare_labram_dataset  # noqa: E402
from random_classifier import (  # noqa: E402
    RANDOM_STRATEGIES,
    generate_random_segment_predictions,
    resolve_class_proportions,
)

SUPPORTED_DATASET_TYPES = frozenset({"age", "gender"})


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(CUSTOM_DATASET_CONFIGS.keys()),
        help="LaBraM dataset name (use age_baseline or gender_baseline for standard runs).",
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
        "--random_seed",
        type=int,
        required=True,
        help="Seed for random label sampling (independent per segment).",
    )
    p.add_argument(
        "--random_strategy",
        type=str,
        required=True,
        choices=list(RANDOM_STRATEGIES),
        help=(
            "'uniform': equal class probability (LaBraM random_baseline=1/C for age). "
            "'train_segment_frequency': train-split segment class frequencies."
        ),
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
        "--output_json",
        type=str,
        required=True,
        help="Path to write JSON results.",
    )
    p.add_argument(
        "--include_task_type_breakdown",
        action="store_true",
        help="Also evaluate active and passive separately (optional; not in LaBraM final logs table).",
    )
    return p.parse_args()


def _num_sampling_classes(dataset_type: str) -> int:
    if dataset_type == "gender":
        return 2
    if dataset_type == "age":
        return 3
    raise ValueError(f"Unsupported dataset_type {dataset_type!r} for random classifier.")


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


def _eval_one_loader(
    data_loader: DataLoader,
    *,
    metrics: List[str],
    is_binary: bool,
    num_sampling_classes: int,
    hdf5_dir: str,
    segment_length: str,
    target_type: str,
    task_type: str,
    random_strategy: str,
    random_seed: int,
    aggregate_by_participant: str,
) -> Dict[str, Any]:
    true_all, participant_ids = collect_test_labels_and_participant_ids(
        data_loader, is_binary=is_binary
    )
    proportions = resolve_class_proportions(
        random_strategy=random_strategy,
        hdf5_dir=hdf5_dir,
        segment_length=segment_length,
        target_type=target_type,
        task_type=task_type,
        num_classes=num_sampling_classes,
    )
    pred_for_metrics, logits_for_loss, sampled_labels = generate_random_segment_predictions(
        len(true_all),
        num_classes=num_sampling_classes,
        is_binary=is_binary,
        class_proportions=proportions,
        random_seed=random_seed,
    )
    mean_loss = compute_mean_classification_loss(
        logits_for_loss, true_all, is_binary=is_binary
    )
    stats = compute_classification_eval_stats(
        pred_for_metrics,
        true_all,
        participant_ids,
        metrics=metrics,
        is_binary=is_binary,
        aggregate_by_participant=aggregate_by_participant,
        loss=mean_loss,
    )
    stats["n_test_segments"] = int(len(true_all))
    stats["n_participants"] = int(len(set(participant_ids))) if participant_ids else 0
    stats["class_proportions"] = [float(x) for x in proportions]
    stats["sampled_label_counts"] = {
        str(int(k)): int(v)
        for k, v in zip(*np.unique(sampled_labels, return_counts=True))
    }
    return stats


def _print_summary(stats: Dict[str, Any], header: str) -> None:
    seg = stats.get("segment_metrics") or stats
    part_mp = stats.get("participant_mean_prob_metrics") or {}
    part_mv = stats.get("participant_metrics") or {}
    print(f"\n{header}")
    print(
        f"  Segment accuracy: {seg.get('accuracy', float('nan')):.4f} | "
        f"participant (mean prob): {part_mp.get('accuracy', float('nan')):.4f} | "
        f"participant (majority vote): {part_mv.get('accuracy', float('nan')):.4f}"
    )
    if "f1_weighted" in seg:
        print(
            f"  Segment F1 (weighted): {seg.get('f1_weighted', float('nan')):.4f} | "
            f"participant MV F1: {part_mv.get('f1_weighted', float('nan')):.4f}"
        )
    if "balanced_accuracy" in seg:
        print(
            f"  Segment balanced accuracy: {seg.get('balanced_accuracy', float('nan')):.4f} | "
            f"participant MV: {part_mv.get('balanced_accuracy', float('nan')):.4f}"
        )


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
    metrics = list(config["metrics"])
    is_binary = (nb_classes == 1)
    num_sampling_classes = _num_sampling_classes(dataset_type)
    hdf5_dir = str(Path(args.data_path).expanduser().resolve())
    if not Path(hdf5_dir).is_dir():
        raise SystemExit(f"--data_path is not a directory: {hdf5_dir}")

    loader = _build_test_loader(
        dataset_type=dataset_type,
        hdf5_dir=hdf5_dir,
        segment_length=args.segment_length,
        task_type=args.task_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    main_stats = _eval_one_loader(
        loader,
        metrics=metrics,
        is_binary=is_binary,
        num_sampling_classes=num_sampling_classes,
        hdf5_dir=hdf5_dir,
        segment_length=args.segment_length,
        target_type=dataset_type,
        task_type=args.task_type,
        random_strategy=args.random_strategy,
        random_seed=args.random_seed,
        aggregate_by_participant=args.aggregate_by_participant,
    )
    _print_summary(main_stats, f"Random classifier — {args.dataset} {args.segment_length} ({args.task_type})")

    payload: Dict[str, Any] = {
        "model": "random_classifier",
        "dataset": args.dataset,
        "dataset_type": dataset_type,
        "segment_length": args.segment_length,
        "data_path": hdf5_dir,
        "task_type": args.task_type,
        "random_seed": args.random_seed,
        "random_strategy": args.random_strategy,
        "aggregate_by_participant": args.aggregate_by_participant,
        "metrics_requested": metrics,
        "nb_classes_config": nb_classes,
        "num_sampling_classes": num_sampling_classes,
        "test": main_stats,
    }

    if args.include_task_type_breakdown:
        breakdown: Dict[str, Any] = {}
        for tt in ("active", "passive"):
            tt_loader = _build_test_loader(
                dataset_type=dataset_type,
                hdf5_dir=hdf5_dir,
                segment_length=args.segment_length,
                task_type=tt,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            tt_stats = _eval_one_loader(
                tt_loader,
                metrics=metrics,
                is_binary=is_binary,
                num_sampling_classes=num_sampling_classes,
                hdf5_dir=hdf5_dir,
                segment_length=args.segment_length,
                target_type=dataset_type,
                task_type=tt,
                random_strategy=args.random_strategy,
                random_seed=args.random_seed,
                aggregate_by_participant=args.aggregate_by_participant,
            )
            breakdown[tt] = tt_stats
            _print_summary(tt_stats, f"  task_type={tt}")
        payload["task_type_breakdown"] = breakdown

    out_path = Path(args.output_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
