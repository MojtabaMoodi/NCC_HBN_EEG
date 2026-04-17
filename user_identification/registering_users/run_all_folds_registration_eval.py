#!/usr/bin/env python3
"""
Run unknown-only registration evaluation across folds without retraining.

For each fold:
- uses checkpoint at <cv_dir>/fold_k/best_model.pth
- uses unknown split from <hdf5_root>/fold_k
- enrolls a fraction of unknown participants, evaluates open-world user classification
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from user_identification.registering_users.run_registration_eval import (  # noqa: E402
    run_single_task,
)


def _resolve_best_checkpoint(cv_dir: Path, fold_idx: int) -> Path:
    ckpt = (cv_dir / f"fold_{fold_idx}" / "best_model.pth").resolve()
    if not ckpt.is_file():
        raise SystemExit(
            f"Missing best checkpoint for fold {fold_idx}: {ckpt}. "
            "Expected fold-specific best model at <cv_dir>/fold_k/best_model.pth."
        )
    return ckpt


def _std_sample(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def _fold_metrics(payload: Dict[str, Any]) -> Dict[str, float]:
    ow = payload["open_world_user_classification"]
    sel = ow["selected_threshold"]["metrics"]
    ref = ow["reference_max_plain_accuracy"]["metrics"]
    closed = payload["registered_user_classification_probe_only"]
    return {
        "closed_set_probe_accuracy": float(closed["accuracy"]),
        "closed_set_probe_balanced_accuracy": float(closed["balanced_accuracy"]),
        "closed_set_probe_f1_macro": float(closed["f1_macro"]),
        "open_world_accuracy_selected": float(sel["accuracy"]),
        "open_world_balanced_accuracy_selected": float(sel["balanced_accuracy"]),
        "open_world_f1_macro_selected": float(sel["f1_macro"]),
        "open_world_mcc_selected": float(sel["mcc"]),
        "unknown_recall_selected": float(sel["unknown_recall"]),
        "mean_registered_recall_selected": float(sel["mean_registered_recall"]),
        "open_world_accuracy_max_plain": float(ref["accuracy"]),
        "open_world_balanced_accuracy_max_plain": float(ref["balanced_accuracy"]),
    }


def _aggregate_metrics_across_folds(
    summary: Dict[str, Any],
    folds: Dict[str, Any],
    n_folds: int,
    *,
    task_key: Optional[str],
) -> None:
    fk0 = "fold_0"
    metric_names = list(
        _fold_metrics(
            folds[fk0] if task_key is None else folds[fk0][task_key]
        ).keys()
    )
    prefix = f"{task_key}_" if task_key else ""
    for name in metric_names:
        vals: List[float] = []
        for kk in range(n_folds):
            fk = f"fold_{kk}"
            payload = folds[fk] if task_key is None else folds[fk][task_key]
            vals.append(_fold_metrics(payload)[name])
        summary[f"{prefix}{name}_mean"] = float(sum(vals) / len(vals))
        summary[f"{prefix}{name}_std"] = _std_sample(vals)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cv_dir", type=str, required=True)
    p.add_argument("--hdf5_root", type=str, required=True)
    p.add_argument("--segment_length", type=str, required=True, choices=["1s", "2s", "4s"])
    p.add_argument("--task_type", type=str, default="both", choices=["active", "passive", "both"])
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=192)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--dropout_rate", type=float, default=0.1)
    p.add_argument("--model_name", type=str, default="labram_base_patch200_200")
    p.add_argument("--arcface_margin", type=float, default=0.5)
    p.add_argument("--arcface_scale", type=float, default=256.0)
    p.add_argument("--arcface_easy_margin", action="store_true")
    p.add_argument("--register_fraction_of_unknown", type=float, default=0.5)
    p.add_argument("--enrollment_fraction_per_user", type=float, default=0.5)
    p.add_argument(
        "--threshold_criterion",
        type=str,
        default="f1_macro",
        choices=[
            "f1_macro",
            "balanced_accuracy",
            "accuracy",
            "mcc",
            "weighted_recall_balance",
            "harmonic_recall_balance",
        ],
        help="Metric used to pick open-world threshold.",
    )
    p.add_argument(
        "--unknown_recall_weight",
        type=float,
        default=0.5,
        help="Weight in [0,1] for unknown recall in weighted_recall_balance criterion.",
    )
    p.add_argument("--threshold_grid_points", type=int, default=501)
    p.add_argument(
        "--threshold_on",
        type=str,
        default="distance",
        choices=["distance", "known_user_score"],
        help="Open-world threshold axis: same as run_registration_eval.py --threshold_on.",
    )
    args = p.parse_args()
    if args.threshold_grid_points < 11:
        raise SystemExit("--threshold_grid_points must be >= 11")
    if not (0.0 <= float(args.unknown_recall_weight) <= 1.0):
        raise SystemExit("--unknown_recall_weight must be in [0, 1]")
    return args


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; using CPU.", file=sys.stderr)

    cv_dir = Path(args.cv_dir).resolve()
    hdf5_root = Path(args.hdf5_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not cv_dir.is_dir():
        raise SystemExit(f"cv_dir not found: {cv_dir}")
    if not hdf5_root.is_dir():
        raise SystemExit(f"hdf5_root not found: {hdf5_root}")

    single_args = SimpleNamespace(
        dropout_rate=args.dropout_rate,
        model_name=args.model_name,
        arcface_margin=args.arcface_margin,
        arcface_scale=args.arcface_scale,
        arcface_easy_margin=args.arcface_easy_margin,
        register_fraction_of_unknown=args.register_fraction_of_unknown,
        enrollment_fraction_per_user=args.enrollment_fraction_per_user,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        threshold_criterion=args.threshold_criterion,
        unknown_recall_weight=args.unknown_recall_weight,
        threshold_grid_points=args.threshold_grid_points,
        threshold_on=args.threshold_on,
    )

    folds: Dict[str, Any] = {}
    for k in range(args.n_folds):
        ckpt = _resolve_best_checkpoint(cv_dir, k)
        hdf5_dir = hdf5_root / f"fold_{k}"
        if not hdf5_dir.is_dir():
            raise SystemExit(f"Missing HDF5 directory for fold {k}: {hdf5_dir}")

        fold_out = out_root / f"fold_{k}"
        fold_out.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 64}\nFold {k}\n{'=' * 64}")
        print(f"Using checkpoint: {ckpt}")

        if args.task_type == "both":
            payload: Dict[str, Any] = {}
            for tt in ("active", "passive"):
                task_out = fold_out / tt
                task_out.mkdir(parents=True, exist_ok=True)
                payload[tt] = run_single_task(
                    ckpt_path=ckpt,
                    hdf5_dir=hdf5_dir,
                    segment_length=args.segment_length,
                    task_type=tt,
                    out_dir=task_out,
                    device=device,
                    args=single_args,
                )
            with open(fold_out / "registration_open_set_report_both.json", "w") as f:
                json.dump(payload, f, indent=2)
            folds[f"fold_{k}"] = payload
        else:
            payload = run_single_task(
                ckpt_path=ckpt,
                hdf5_dir=hdf5_dir,
                segment_length=args.segment_length,
                task_type=args.task_type,
                out_dir=fold_out,
                device=device,
                args=single_args,
            )
            folds[f"fold_{k}"] = payload

    summary: Dict[str, Any] = {
        "segment_length": args.segment_length,
        "task_type": args.task_type,
        "n_folds": args.n_folds,
        "threshold_criterion": args.threshold_criterion,
        "unknown_recall_weight": args.unknown_recall_weight,
        "threshold_grid_points": args.threshold_grid_points,
        "threshold_on": args.threshold_on,
        "folds": folds,
    }

    if args.task_type == "both":
        for tt in ("active", "passive"):
            _aggregate_metrics_across_folds(summary, folds, args.n_folds, task_key=tt)
    else:
        _aggregate_metrics_across_folds(summary, folds, args.n_folds, task_key=None)

    out_path = out_root / "registration_eval_cv_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote cross-fold summary: {out_path}")


if __name__ == "__main__":
    main()
