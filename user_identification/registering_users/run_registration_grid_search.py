#!/usr/bin/env python3
"""
Grid search for unknown-only registration/open-world evaluation.

This script sweeps key knobs that strongly affect the trade-off between:
- identifying registered users correctly
- rejecting unregistered users as UNKNOWN

It reuses run_single_task() from run_registration_eval.py and writes:
- full fold-level payload for every config
- compact ranked summary JSON
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple

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


def _parse_float_list(raw: str) -> List[float]:
    vals: List[float] = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        vals.append(float(t))
    if not vals:
        raise ValueError("Expected at least one float value")
    return vals


def _parse_str_list(raw: str) -> List[str]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one string value")
    return vals


def _std_sample(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def _mean_std(values: List[float]) -> Dict[str, float]:
    return {"mean": float(sum(values) / len(values)), "std": _std_sample(values)}


def _aggregate_config_metrics(fold_payloads: List[Dict[str, Any]]) -> Dict[str, float]:
    acc: List[float] = []
    bal: List[float] = []
    f1m: List[float] = []
    mcc: List[float] = []
    urec: List[float] = []
    rrec: List[float] = []
    closed_acc: List[float] = []

    for p in fold_payloads:
        ow = p["open_world_user_classification"]["selected_threshold"]["metrics"]
        closed = p["registered_user_classification_probe_only"]
        acc.append(float(ow["accuracy"]))
        bal.append(float(ow["balanced_accuracy"]))
        f1m.append(float(ow["f1_macro"]))
        mcc.append(float(ow["mcc"]))
        urec.append(float(ow["unknown_recall"]))
        rrec.append(float(ow["mean_registered_recall"]))
        closed_acc.append(float(closed["accuracy"]))

    out = {
        "open_world_accuracy_mean": _mean_std(acc)["mean"],
        "open_world_accuracy_std": _mean_std(acc)["std"],
        "open_world_balanced_accuracy_mean": _mean_std(bal)["mean"],
        "open_world_balanced_accuracy_std": _mean_std(bal)["std"],
        "open_world_f1_macro_mean": _mean_std(f1m)["mean"],
        "open_world_f1_macro_std": _mean_std(f1m)["std"],
        "open_world_mcc_mean": _mean_std(mcc)["mean"],
        "open_world_mcc_std": _mean_std(mcc)["std"],
        "unknown_recall_mean": _mean_std(urec)["mean"],
        "unknown_recall_std": _mean_std(urec)["std"],
        "mean_registered_recall_mean": _mean_std(rrec)["mean"],
        "mean_registered_recall_std": _mean_std(rrec)["std"],
        "registered_probe_accuracy_mean": _mean_std(closed_acc)["mean"],
        "registered_probe_accuracy_std": _mean_std(closed_acc)["std"],
    }
    return out


def _objective_score(metrics: Dict[str, float], objective_unknown_weight: float) -> float:
    u = float(metrics["unknown_recall_mean"])
    r = float(metrics["mean_registered_recall_mean"])
    return float(objective_unknown_weight * u + (1.0 - objective_unknown_weight) * r)


def format_config_tag(
    enrollment_fraction_per_user: float,
    unknown_recall_weight: float,
    register_fraction_of_unknown: float,
    threshold_criterion: str,
) -> str:
    """Subdirectory name for one grid cell (must match main() loop)."""
    return (
        f"enroll_{enrollment_fraction_per_user:.3f}"
        f"__unkw_{unknown_recall_weight:.3f}"
        f"__regfrac_{register_fraction_of_unknown:.3f}"
        f"__crit_{threshold_criterion}"
    )


def format_config_tag_from_row(row: Dict[str, Any]) -> str:
    c = row["config"]
    return format_config_tag(
        float(c["enrollment_fraction_per_user"]),
        float(c["unknown_recall_weight"]),
        float(c["register_fraction_of_unknown"]),
        str(c["threshold_criterion"]),
    )


def _rank_sort_key(row: Dict[str, Any]) -> Tuple[float, float, float]:
    """Primary: objective_score; tie-break: macro-F1 then MCC (higher is better)."""
    agg = row["aggregate"]
    return (
        float(row["objective_score"]),
        float(agg["open_world_f1_macro_mean"]),
        float(agg["open_world_mcc_mean"]),
    )


def _prune_config_dirs(out_root: Path, ranked_rows: List[Dict[str, Any]], keep_top_k: int) -> None:
    if keep_top_k < 1:
        raise ValueError("keep_top_k must be >= 1")
    keep_names = {format_config_tag_from_row(r) for r in ranked_rows[:keep_top_k]}
    removed = 0
    for child in sorted(out_root.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.startswith("enroll_"):
            continue
        if child.name in keep_names:
            continue
        shutil.rmtree(child)
        removed += 1
    print(f"Pruned {removed} config director(y/ies); kept top {keep_top_k}: {sorted(keep_names)}")


def _iter_grid(
    enrollment_fractions: Iterable[float],
    unknown_weights: Iterable[float],
    register_fractions: Iterable[float],
    criteria: Iterable[str],
) -> Iterable[Tuple[float, float, float, str]]:
    return itertools.product(enrollment_fractions, unknown_weights, register_fractions, criteria)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cv_dir", type=str, required=True)
    p.add_argument("--hdf5_root", type=str, required=True)
    p.add_argument("--segment_length", type=str, required=True, choices=["1s", "2s", "4s"])
    p.add_argument("--task_type", type=str, default="active", choices=["active", "passive"])
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
    p.add_argument("--threshold_grid_points", type=int, default=501)
    p.add_argument(
        "--threshold_on",
        type=str,
        default="distance",
        choices=["distance", "known_user_score"],
        help="Open-world threshold axis: same as run_registration_eval.py --threshold_on.",
    )

    p.add_argument(
        "--grid_enrollment_fractions",
        type=str,
        default="0.5,0.6,0.7",
        help="Comma-separated enrollment fractions to sweep.",
    )
    p.add_argument(
        "--grid_unknown_recall_weights",
        type=str,
        default="0.4,0.5,0.6,0.7",
        help="Comma-separated unknown_recall_weight values to sweep.",
    )
    p.add_argument(
        "--grid_register_fractions",
        type=str,
        default="0.5",
        help="Comma-separated register_fraction_of_unknown values to sweep.",
    )
    p.add_argument(
        "--grid_threshold_criteria",
        type=str,
        default="weighted_recall_balance,harmonic_recall_balance",
        help="Comma-separated threshold criteria to sweep.",
    )
    p.add_argument(
        "--objective_unknown_weight",
        type=float,
        default=0.5,
        help="Weight for ranking configs: w*unknown_recall + (1-w)*mean_registered_recall.",
    )
    p.add_argument(
        "--prune_configs",
        action="store_true",
        help="After ranking, delete config subdirectories except the top --prune_keep_top_k.",
    )
    p.add_argument(
        "--prune_keep_top_k",
        type=int,
        default=3,
        help="Number of top-ranked config directories to keep when --prune_configs is set.",
    )
    args = p.parse_args()

    if args.threshold_grid_points < 11:
        raise SystemExit("--threshold_grid_points must be >= 11")
    if not (0.0 <= args.objective_unknown_weight <= 1.0):
        raise SystemExit("--objective_unknown_weight must be in [0,1]")
    if int(args.prune_keep_top_k) < 1:
        raise SystemExit("--prune_keep_top_k must be >= 1")
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

    enrollment_fractions = _parse_float_list(args.grid_enrollment_fractions)
    unknown_weights = _parse_float_list(args.grid_unknown_recall_weights)
    register_fractions = _parse_float_list(args.grid_register_fractions)
    criteria = _parse_str_list(args.grid_threshold_criteria)

    for v in enrollment_fractions:
        if not (0.0 < v < 1.0):
            raise SystemExit(f"Enrollment fraction must be in (0,1), got {v}")
    for v in unknown_weights:
        if not (0.0 <= v <= 1.0):
            raise SystemExit(f"Unknown recall weight must be in [0,1], got {v}")
    for v in register_fractions:
        if not (0.0 < v < 1.0):
            raise SystemExit(f"Register fraction must be in (0,1), got {v}")

    results: List[Dict[str, Any]] = []
    grid = list(_iter_grid(enrollment_fractions, unknown_weights, register_fractions, criteria))
    total = len(grid)
    print(f"Running {total} configuration(s)")

    for idx, (enr_frac, unk_w, reg_frac, criterion) in enumerate(grid, start=1):
        config_tag = format_config_tag(enr_frac, unk_w, reg_frac, criterion)
        config_out = out_root / config_tag
        config_out.mkdir(parents=True, exist_ok=True)
        print(f"\n[{idx}/{total}] {config_tag}")

        single_args = SimpleNamespace(
            dropout_rate=args.dropout_rate,
            model_name=args.model_name,
            arcface_margin=args.arcface_margin,
            arcface_scale=args.arcface_scale,
            arcface_easy_margin=args.arcface_easy_margin,
            register_fraction_of_unknown=reg_frac,
            enrollment_fraction_per_user=enr_frac,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            threshold_criterion=criterion,
            unknown_recall_weight=unk_w,
            threshold_grid_points=args.threshold_grid_points,
            threshold_on=args.threshold_on,
        )

        fold_payloads: List[Dict[str, Any]] = []
        fold_map: Dict[str, Any] = {}
        for k in range(args.n_folds):
            ckpt = _resolve_best_checkpoint(cv_dir, k)
            hdf5_dir = hdf5_root / f"fold_{k}"
            if not hdf5_dir.is_dir():
                raise SystemExit(f"Missing HDF5 dir for fold {k}: {hdf5_dir}")
            print(f"  Fold {k}: using checkpoint {ckpt}")

            fold_out = config_out / f"fold_{k}"
            fold_out.mkdir(parents=True, exist_ok=True)
            payload = run_single_task(
                ckpt_path=ckpt,
                hdf5_dir=hdf5_dir,
                segment_length=args.segment_length,
                task_type=args.task_type,
                out_dir=fold_out,
                device=device,
                args=single_args,
            )
            fold_payloads.append(payload)
            fold_map[f"fold_{k}"] = payload

        aggregate = _aggregate_config_metrics(fold_payloads)
        score = _objective_score(aggregate, args.objective_unknown_weight)
        row: Dict[str, Any] = {
            "config": {
                "enrollment_fraction_per_user": enr_frac,
                "unknown_recall_weight": unk_w,
                "register_fraction_of_unknown": reg_frac,
                "threshold_criterion": criterion,
                "threshold_on": args.threshold_on,
            },
            "objective_unknown_weight": args.objective_unknown_weight,
            "objective_score": score,
            "aggregate": aggregate,
        }
        results.append(row)

        with open(config_out / "grid_search_fold_payloads.json", "w") as f:
            json.dump({"config": row["config"], "folds": fold_map}, f, indent=2)
        with open(config_out / "grid_search_summary.json", "w") as f:
            json.dump(row, f, indent=2)

    results.sort(key=_rank_sort_key, reverse=True)
    out_payload = {
        "search_space": {
            "grid_enrollment_fractions": enrollment_fractions,
            "grid_unknown_recall_weights": unknown_weights,
            "grid_register_fractions": register_fractions,
            "grid_threshold_criteria": criteria,
            "objective_unknown_weight": args.objective_unknown_weight,
            "threshold_on": args.threshold_on,
        },
        "ranking_note": (
            "Sorted by descending objective_score; ties broken by open_world_f1_macro_mean "
            "then open_world_mcc_mean."
        ),
        "best_config": results[0] if results else None,
        "ranked_results": results,
    }
    with open(out_root / "grid_search_ranked_results.json", "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nWrote ranked grid search results: {out_root / 'grid_search_ranked_results.json'}")

    if args.prune_configs:
        _prune_config_dirs(out_root, results, int(args.prune_keep_top_k))


if __name__ == "__main__":
    main()
