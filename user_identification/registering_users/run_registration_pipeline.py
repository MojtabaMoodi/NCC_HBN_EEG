#!/usr/bin/env python3
"""
End-to-end registration evaluation pipeline (unknown-only, open-world).

Steps:
1. Grid search on **active** task (same search space as ``run_registration_grid_search.py`` defaults).
2. Grid search on **passive** task.
3. Write ``registration_pipeline_manifest.json`` with best configs, aggregate metrics, and
   copy-paste commands for a clean all-fold re-run.
4. Optionally run ``run_all_folds_registration_eval.py`` twice (active + passive) using each
   task's best hyperparameters from its grid.

Checkpoints are always ``<cv_dir>/fold_k/best_model.pth`` (enforced by the called scripts).

Example::

    python user_identification/registering_users/run_registration_pipeline.py \\
      --cv_dir final_user_identification/4s \\
      --hdf5_root \"${HOME}/scratch\" \\
      --segment_length 4s \\
      --n_folds 5 \\
      --output_root registering_users_results/pipeline_4s_active_passive \\
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_project_root = Path(__file__).resolve().parent.parent.parent
_GRID_SCRIPT = _project_root / "user_identification/registering_users/run_registration_grid_search.py"
_ALLFOLDS_SCRIPT = _project_root / "user_identification/registering_users/run_all_folds_registration_eval.py"


def _load_ranked_best(grid_out: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    path = grid_out / "grid_search_ranked_results.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing ranked results: {path}")
    with open(path) as f:
        data = json.load(f)
    best = data.get("best_config")
    if not best:
        raise RuntimeError(f"No best_config in {path}")
    return data, best


def _build_grid_cmd(
    *,
    cv_dir: str,
    hdf5_root: str,
    segment_length: str,
    task_type: str,
    n_folds: int,
    output_root: Path,
    device: str,
    seed: int,
    batch_size: int,
    num_workers: int,
    dropout_rate: float,
    model_name: str,
    arcface_margin: float,
    arcface_scale: float,
    arcface_easy_margin: bool,
    threshold_grid_points: int,
    grid_enrollment_fractions: str,
    grid_unknown_recall_weights: str,
    grid_register_fractions: str,
    grid_threshold_criteria: str,
    objective_unknown_weight: float,
    prune_configs: bool,
    prune_keep_top_k: int,
    threshold_on: str,
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        str(_GRID_SCRIPT),
        "--cv_dir",
        cv_dir,
        "--hdf5_root",
        hdf5_root,
        "--segment_length",
        segment_length,
        "--task_type",
        task_type,
        "--n_folds",
        str(n_folds),
        "--output_root",
        str(output_root),
        "--device",
        device,
        "--seed",
        str(seed),
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
        "--dropout_rate",
        str(dropout_rate),
        "--model_name",
        model_name,
        "--arcface_margin",
        str(arcface_margin),
        "--arcface_scale",
        str(arcface_scale),
        "--threshold_grid_points",
        str(threshold_grid_points),
        "--threshold_on",
        threshold_on,
        "--grid_enrollment_fractions",
        grid_enrollment_fractions,
        "--grid_unknown_recall_weights",
        grid_unknown_recall_weights,
        "--grid_register_fractions",
        grid_register_fractions,
        "--grid_threshold_criteria",
        grid_threshold_criteria,
        "--objective_unknown_weight",
        str(objective_unknown_weight),
        "--prune_keep_top_k",
        str(prune_keep_top_k),
    ]
    if arcface_easy_margin:
        cmd.append("--arcface_easy_margin")
    if prune_configs:
        cmd.append("--prune_configs")
    return cmd


def _build_all_folds_cmd(
    *,
    cv_dir: str,
    hdf5_root: str,
    segment_length: str,
    task_type: str,
    n_folds: int,
    output_root: Path,
    device: str,
    seed: int,
    batch_size: int,
    num_workers: int,
    dropout_rate: float,
    model_name: str,
    arcface_margin: float,
    arcface_scale: float,
    arcface_easy_margin: bool,
    threshold_grid_points: int,
    register_fraction_of_unknown: float,
    enrollment_fraction_per_user: float,
    threshold_criterion: str,
    unknown_recall_weight: float,
    threshold_on: str,
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        str(_ALLFOLDS_SCRIPT),
        "--cv_dir",
        cv_dir,
        "--hdf5_root",
        hdf5_root,
        "--segment_length",
        segment_length,
        "--task_type",
        task_type,
        "--n_folds",
        str(n_folds),
        "--output_root",
        str(output_root),
        "--device",
        device,
        "--seed",
        str(seed),
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
        "--dropout_rate",
        str(dropout_rate),
        "--model_name",
        model_name,
        "--arcface_margin",
        str(arcface_margin),
        "--arcface_scale",
        str(arcface_scale),
        "--register_fraction_of_unknown",
        str(register_fraction_of_unknown),
        "--enrollment_fraction_per_user",
        str(enrollment_fraction_per_user),
        "--threshold_criterion",
        threshold_criterion,
        "--unknown_recall_weight",
        str(unknown_recall_weight),
        "--threshold_grid_points",
        str(threshold_grid_points),
        "--threshold_on",
        threshold_on,
    ]
    if arcface_easy_margin:
        cmd.append("--arcface_easy_margin")
    return cmd


def _run(cmd: List[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(_project_root), check=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cv_dir", type=str, required=True)
    p.add_argument("--hdf5_root", type=str, required=True)
    p.add_argument("--segment_length", type=str, default="4s", choices=["1s", "2s", "4s"])
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
        help="Forwarded to run_registration_grid_search.py and run_all_folds_registration_eval.py.",
    )

    p.add_argument(
        "--grid_enrollment_fractions",
        type=str,
        default="0.5,0.6,0.7",
        help="Forwarded to run_registration_grid_search.py",
    )
    p.add_argument(
        "--grid_unknown_recall_weights",
        type=str,
        default="0.4,0.5,0.6,0.7",
        help="Forwarded to run_registration_grid_search.py",
    )
    p.add_argument(
        "--grid_register_fractions",
        type=str,
        default="0.5",
        help="Forwarded to run_registration_grid_search.py",
    )
    p.add_argument(
        "--grid_threshold_criteria",
        type=str,
        default="weighted_recall_balance,harmonic_recall_balance",
        help="Forwarded to run_registration_grid_search.py",
    )
    p.add_argument(
        "--objective_unknown_weight",
        type=float,
        default=0.5,
        help="Ranking weight for grid search objective (same as grid script).",
    )
    p.add_argument(
        "--prune_after_grid",
        action="store_true",
        help="After each grid search, keep only top --prune_keep_top_k config dirs.",
    )
    p.add_argument("--prune_keep_top_k", type=int, default=3)
    p.add_argument(
        "--skip_grid",
        action="store_true",
        help="Skip grid search if grid_search_ranked_results.json already exists under each grid output dir.",
    )
    p.add_argument(
        "--no_final_eval",
        action="store_true",
        help="Do not run run_all_folds_registration_eval.py after grids complete.",
    )
    args = p.parse_args()
    if args.threshold_grid_points < 11:
        raise SystemExit("--threshold_grid_points must be >= 11")
    if not (0.0 <= float(args.objective_unknown_weight) <= 1.0):
        raise SystemExit("--objective_unknown_weight must be in [0,1]")
    if int(args.prune_keep_top_k) < 1:
        raise SystemExit("--prune_keep_top_k must be >= 1")
    return args


def main() -> None:
    args = _parse_args()
    out = Path(args.output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)

    grid_active = out / "grid_search_active"
    grid_passive = out / "grid_search_passive"
    eval_active = out / "eval_all_folds_active_best_from_grid"
    eval_passive = out / "eval_all_folds_passive_best_from_grid"

    common_grid = dict(
        cv_dir=args.cv_dir,
        hdf5_root=args.hdf5_root,
        segment_length=args.segment_length,
        n_folds=args.n_folds,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dropout_rate=args.dropout_rate,
        model_name=args.model_name,
        arcface_margin=args.arcface_margin,
        arcface_scale=args.arcface_scale,
        arcface_easy_margin=bool(args.arcface_easy_margin),
        threshold_grid_points=args.threshold_grid_points,
        grid_enrollment_fractions=args.grid_enrollment_fractions,
        grid_unknown_recall_weights=args.grid_unknown_recall_weights,
        grid_register_fractions=args.grid_register_fractions,
        grid_threshold_criteria=args.grid_threshold_criteria,
        objective_unknown_weight=float(args.objective_unknown_weight),
        prune_configs=bool(args.prune_after_grid),
        prune_keep_top_k=int(args.prune_keep_top_k),
        threshold_on=args.threshold_on,
    )

    def maybe_run_grid(task: str, grid_out: Path) -> None:
        ranked = grid_out / "grid_search_ranked_results.json"
        if args.skip_grid and ranked.is_file():
            print(f"Skipping grid for {task}: found {ranked}")
            return
        grid_out.mkdir(parents=True, exist_ok=True)
        cmd = _build_grid_cmd(task_type=task, output_root=grid_out, **common_grid)
        _run(cmd)

    maybe_run_grid("active", grid_active)
    maybe_run_grid("passive", grid_passive)

    ranked_active, best_active = _load_ranked_best(grid_active)
    ranked_passive, best_passive = _load_ranked_best(grid_passive)

    cfg_a = best_active["config"]
    cfg_p = best_passive["config"]

    manifest: Dict[str, Any] = {
        "segment_length": args.segment_length,
        "n_folds": args.n_folds,
        "cv_dir": str(Path(args.cv_dir).resolve()),
        "hdf5_root": str(Path(args.hdf5_root).resolve()),
        "objective_unknown_weight": float(args.objective_unknown_weight),
        "threshold_on": args.threshold_on,
        "active": {
            "grid_output_dir": str(grid_active),
            "ranked_results": str(grid_active / "grid_search_ranked_results.json"),
            "best_config": best_active,
            "search_space": ranked_active.get("search_space"),
            "ranking_note": ranked_active.get("ranking_note"),
        },
        "passive": {
            "grid_output_dir": str(grid_passive),
            "ranked_results": str(grid_passive / "grid_search_ranked_results.json"),
            "best_config": best_passive,
            "search_space": ranked_passive.get("search_space"),
            "ranking_note": ranked_passive.get("ranking_note"),
        },
        "recommended_next_steps": [
            "Use best hyperparameters below for paper tables and deployment experiments.",
            "Passive best settings may differ from active; always report them separately.",
            "If disk is tight, re-run grid with --prune_after_grid or delete old config folders manually.",
        ],
    }

    common_eval = dict(
        cv_dir=args.cv_dir,
        hdf5_root=args.hdf5_root,
        segment_length=args.segment_length,
        n_folds=args.n_folds,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dropout_rate=args.dropout_rate,
        model_name=args.model_name,
        arcface_margin=args.arcface_margin,
        arcface_scale=args.arcface_scale,
        arcface_easy_margin=bool(args.arcface_easy_margin),
        threshold_grid_points=args.threshold_grid_points,
        threshold_on=args.threshold_on,
    )

    cmd_active_final = _build_all_folds_cmd(
        **common_eval,
        task_type="active",
        output_root=eval_active,
        register_fraction_of_unknown=float(cfg_a["register_fraction_of_unknown"]),
        enrollment_fraction_per_user=float(cfg_a["enrollment_fraction_per_user"]),
        threshold_criterion=str(cfg_a["threshold_criterion"]),
        unknown_recall_weight=float(cfg_a["unknown_recall_weight"]),
    )
    cmd_passive_final = _build_all_folds_cmd(
        **common_eval,
        task_type="passive",
        output_root=eval_passive,
        register_fraction_of_unknown=float(cfg_p["register_fraction_of_unknown"]),
        enrollment_fraction_per_user=float(cfg_p["enrollment_fraction_per_user"]),
        threshold_criterion=str(cfg_p["threshold_criterion"]),
        unknown_recall_weight=float(cfg_p["unknown_recall_weight"]),
    )
    manifest["final_eval_commands"] = {
        "active": " ".join(cmd_active_final),
        "passive": " ".join(cmd_passive_final),
    }

    manifest_path = out / "registration_pipeline_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote pipeline manifest: {manifest_path}")

    if not args.no_final_eval:
        _run(cmd_active_final)
        _run(cmd_passive_final)
        manifest["final_eval_output_roots"] = {
            "active": str(eval_active),
            "passive": str(eval_passive),
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Updated manifest with final_eval_output_roots: {manifest_path}")


if __name__ == "__main__":
    main()
