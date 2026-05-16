#!/usr/bin/env python3
"""
Evaluate saved LaBraM+ArcFace user-identification checkpoints on the **test** split,
separately for **active** and **passive** tasks.

Training uses task_type=\"both\"; this script rebuilds test loaders with task_type
\"active\" or \"passive\" so metrics are comparable to LaBraM's evaluate_by_task_type.

**Metrics (per task, full test set):** accuracy (%), loss, F1 macro and weighted,
and **macro-averaged one-vs-rest** false positive rate (FPR) and false negative rate (FNR)
over user classes that appear in the test split for that task. Cross-fold mean ± std
are in the output JSON ``summary`` when ``--cv_dir`` mode runs multiple folds.

**Per-sample labels:** pass ``--save_predictions_dir`` to write compressed NumPy files
``fold_XX_<segment>_<active|passive>.npz`` with integer arrays ``y_true`` and ``y_pred``
(user class indices, same order as the test DataLoader).

Examples
--------
Single fold::

    python user_identification/eval_active_passive_test.py \\
      --checkpoint final_user_identification/4s/fold_0/best_model.pth \\
      --hdf5_dir ~/scratch/fold_0 \\
      --segment_length 4s

All folds (checkpoint at ``<cv_dir>/fold_k/best_model.pth``, data at ``<hdf5_root>/fold_k``)::

    python user_identification/eval_active_passive_test.py \\
      --cv_dir final_user_identification/4s \\
      --hdf5_root ~/scratch \\
      --segment_length 4s \\
      --n_folds 5

``--hdf5_dir`` / ``--hdf5_root`` must be the **same** preprocessed fold(s) used when
training that checkpoint (same ``segment_length``).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

_project_root = Path(__file__).resolve().parent.parent
_labram_dir = _project_root / "LaBraM"
for _p in (_project_root, _labram_dir):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

import utils as labram_utils  # noqa: E402
from CNN.models.arcface_loss import ArcFaceLoss  # noqa: E402
from data_processing.target_transforms import create_user_identification_transform_from_hdf5  # noqa: E402
from LaBraM.labram_dataset import prepare_labram_dataset  # noqa: E402
from user_identification.labram_model import LaBraMUserIdentificationWrapper  # noqa: E402
from user_identification.labram_trainer import evaluate  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, default=None, help="Path to best_model.pth (single-fold mode)")
    p.add_argument("--hdf5_dir", type=str, default=None, help="HDF5 directory for that fold (single-fold mode)")
    p.add_argument("--cv_dir", type=str, default=None, help="e.g. final_user_identification/4s (multi-fold mode)")
    p.add_argument("--hdf5_root", type=str, default=None, help="Parent of fold_0..fold_{n-1} (multi-fold mode)")
    p.add_argument("--n_folds", type=int, default=5, help="Number of folds (multi-fold mode)")
    p.add_argument("--segment_length", type=str, required=True, choices=["1s", "2s", "4s"])
    p.add_argument("--batch_size", type=int, default=288, help="Test batch size (training uses 1.5× train BS)")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dropout_rate",
        type=float,
        default=0.1,
        help="Must match training (train_labram_arcface default 0.1)",
    )
    p.add_argument("--model_name", type=str, default="labram_base_patch200_200")
    p.add_argument("--arcface_margin", type=float, default=0.5)
    p.add_argument("--arcface_scale", type=float, default=256.0)
    p.add_argument("--arcface_easy_margin", action="store_true")
    p.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Write full results here (default: cv_dir/task_type_test_eval.json or next to checkpoint)",
    )
    p.add_argument(
        "--save_predictions_dir",
        type=str,
        default=None,
        help=(
            "If set, save per-sample true/predicted class indices as compressed NumPy "
            "``fold_XX_<segment>_<active|passive>.npz`` (keys: y_true, y_pred) under this directory."
        ),
    )
    return p.parse_args()


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    needed = ["model_state_dict", "arcface_state_dict", "num_classes"]
    missing = [k for k in needed if k not in ckpt]
    if missing:
        raise KeyError(f"Checkpoint {path} missing keys: {missing}")
    if "embedding_dim" not in ckpt:
        raise KeyError(f"Checkpoint {path} missing 'embedding_dim' (required for ArcFace).")
    return ckpt


def _build_model_and_criterion(
    ckpt: Dict[str, Any],
    device: torch.device,
    dropout_rate: float,
    model_name: str,
    arcface_margin: float,
    arcface_scale: float,
    arcface_easy_margin: bool,
) -> Tuple[torch.nn.Module, ArcFaceLoss]:
    num_classes = int(ckpt["num_classes"])
    embedding_dim = int(ckpt["embedding_dim"])
    model = LaBraMUserIdentificationWrapper(
        num_channels=60,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained_path=None,
        model_name=model_name,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    criterion = ArcFaceLoss(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        margin=arcface_margin,
        scale=arcface_scale,
        easy_margin=arcface_easy_margin,
    ).to(device)
    criterion.load_state_dict(ckpt["arcface_state_dict"], strict=True)
    criterion.eval()
    return model, criterion


def _eval_task_split(
    hdf5_dir: Path,
    segment_length: str,
    task_type: str,
    transform: Any,
    model: torch.nn.Module,
    criterion: ArcFaceLoss,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    predictions_npz: Optional[Path] = None,
) -> Dict[str, Any]:
    _, _, test_ds = prepare_labram_dataset(
        dataset_type="user_identification",
        hdf5_dir=str(hdf5_dir),
        segment_length=segment_length,
        task_type=task_type,
        random_seed=seed,
        user_identification_transform=transform,
        sampling_rate=200,
    )
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    collect = predictions_npz is not None
    stats = evaluate(
        loader,
        model,
        device,
        header=f"Test ({task_type}):",
        ch_names=None,
        metrics=["accuracy"],
        use_arcface=True,
        criterion=criterion,
        collect_predictions=collect,
    )
    predictions_path: Optional[str] = None
    if collect:
        yt = stats.pop("y_true", None)
        yp = stats.pop("y_pred", None)
        if yt is None or yp is None:
            raise RuntimeError("evaluate(..., collect_predictions=True) must return y_true and y_pred.")
        predictions_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(predictions_npz, y_true=yt, y_pred=yp)
        predictions_path = str(predictions_npz.resolve())

    acc = float(stats.get("accuracy", 0.0)) * 100.0
    loss = float(stats.get("loss", 0.0))
    f1_macro = stats.get("f1_macro")
    f1_weighted = stats.get("f1_weighted")
    fpr = stats.get("fpr_macro")
    fnr = stats.get("fnr_macro")
    out: Dict[str, Any] = {
        "accuracy_percent": acc,
        "loss": loss,
        "f1_macro": float(f1_macro) if isinstance(f1_macro, (int, float)) else None,
        "f1_weighted": float(f1_weighted) if isinstance(f1_weighted, (int, float)) else None,
        "fpr_macro": float(fpr) if isinstance(fpr, (int, float)) else None,
        "fnr_macro": float(fnr) if isinstance(fnr, (int, float)) else None,
    }
    if predictions_path is not None:
        out["predictions_npz"] = predictions_path
    return out


def _eval_one_fold(
    checkpoint: Path,
    hdf5_dir: Path,
    segment_length: str,
    args: argparse.Namespace,
    device: torch.device,
    fold_index: int,
) -> Dict[str, Any]:
    ckpt = _load_checkpoint(checkpoint, device)
    model, criterion = _build_model_and_criterion(
        ckpt,
        device,
        args.dropout_rate,
        args.model_name,
        args.arcface_margin,
        args.arcface_scale,
        args.arcface_easy_margin,
    )
    transform, _ = create_user_identification_transform_from_hdf5(str(hdf5_dir))
    out: Dict[str, Any] = {
        "checkpoint": str(checkpoint.resolve()),
        "hdf5_dir": str(hdf5_dir.resolve()),
        "segment_length": segment_length,
        "num_classes": int(ckpt["num_classes"]),
        "checkpoint_val_accuracy_percent": float(ckpt.get("val_accuracy", float("nan"))),
        "checkpoint_test_accuracy_percent": float(ckpt.get("test_accuracy", float("nan"))),
    }
    pred_dir: Optional[Path] = None
    if args.save_predictions_dir:
        pred_dir = Path(args.save_predictions_dir).expanduser().resolve()
        pred_dir.mkdir(parents=True, exist_ok=True)

    for tt in ("active", "passive"):
        npz_path: Optional[Path] = None
        if pred_dir is not None:
            npz_path = pred_dir / f"fold_{fold_index:02d}_{segment_length}_{tt}.npz"
        out[tt] = _eval_task_split(
            hdf5_dir,
            segment_length,
            tt,
            transform,
            model,
            criterion,
            device,
            args.batch_size,
            args.num_workers,
            args.seed,
            predictions_npz=npz_path,
        )
    return out


def _std_sample(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; using CPU.", file=sys.stderr)

    single = args.checkpoint is not None and args.hdf5_dir is not None
    multi = args.cv_dir is not None and args.hdf5_root is not None
    if single == multi:
        raise SystemExit("Specify either (--checkpoint AND --hdf5_dir) OR (--cv_dir AND --hdf5_root), not both/neither.")

    fold_results: List[Dict[str, Any]] = []

    if single:
        checkpoint = Path(args.checkpoint).resolve()
        hdf5_dir = Path(args.hdf5_dir).resolve()
        if not checkpoint.is_file():
            raise SystemExit(f"Missing checkpoint: {checkpoint}")
        if not hdf5_dir.is_dir():
            raise SystemExit(f"Missing hdf5_dir: {hdf5_dir}")
        fold_results.append(
            _eval_one_fold(checkpoint, hdf5_dir, args.segment_length, args, device, fold_index=0)
        )
        default_out = checkpoint.parent / "task_type_test_eval.json"
    else:
        cv_dir = Path(args.cv_dir).resolve()
        hdf5_root = Path(args.hdf5_root).resolve()
        if not cv_dir.is_dir():
            raise SystemExit(f"Missing cv_dir: {cv_dir}")
        for k in range(args.n_folds):
            ckpt = cv_dir / f"fold_{k}" / "best_model.pth"
            hdir = hdf5_root / f"fold_{k}"
            if not ckpt.is_file():
                raise SystemExit(f"Missing checkpoint for fold {k}: {ckpt}")
            if not hdir.is_dir():
                raise SystemExit(f"Missing hdf5 fold directory: {hdir}")
            print(f"\n{'='*60}\nFold {k}: {ckpt.name}\n{'='*60}")
            fold_results.append(_eval_one_fold(ckpt, hdir, args.segment_length, args, device, fold_index=k))
            a = fold_results[-1]["active"]["accuracy_percent"]
            p = fold_results[-1]["passive"]["accuracy_percent"]
            print(f"  active:  {a:.4f}%  |  passive: {p:.4f}%")
        default_out = cv_dir / "task_type_test_eval.json"

    active_accs = [float(r["active"]["accuracy_percent"]) for r in fold_results]
    passive_accs = [float(r["passive"]["accuracy_percent"]) for r in fold_results]
    active_f1m = [float(r["active"]["f1_macro"]) for r in fold_results if r["active"].get("f1_macro") is not None]
    passive_f1m = [float(r["passive"]["f1_macro"]) for r in fold_results if r["passive"].get("f1_macro") is not None]
    active_f1w = [float(r["active"]["f1_weighted"]) for r in fold_results if r["active"].get("f1_weighted") is not None]
    passive_f1w = [float(r["passive"]["f1_weighted"]) for r in fold_results if r["passive"].get("f1_weighted") is not None]
    active_fpr = [float(r["active"]["fpr_macro"]) for r in fold_results if r["active"].get("fpr_macro") is not None]
    passive_fpr = [float(r["passive"]["fpr_macro"]) for r in fold_results if r["passive"].get("fpr_macro") is not None]
    active_fnr = [float(r["active"]["fnr_macro"]) for r in fold_results if r["active"].get("fnr_macro") is not None]
    passive_fnr = [float(r["passive"]["fnr_macro"]) for r in fold_results if r["passive"].get("fnr_macro") is not None]
    summary: Dict[str, Any] = {
        "n_folds": len(fold_results),
        "active_test_accuracy_mean_percent": float(sum(active_accs) / len(active_accs)),
        "active_test_accuracy_std_percent": _std_sample(active_accs),
        "passive_test_accuracy_mean_percent": float(sum(passive_accs) / len(passive_accs)),
        "passive_test_accuracy_std_percent": _std_sample(passive_accs),
    }
    if active_f1m:
        summary["active_test_f1_macro_mean"] = float(sum(active_f1m) / len(active_f1m))
        summary["active_test_f1_macro_std"] = _std_sample(active_f1m)
    if passive_f1m:
        summary["passive_test_f1_macro_mean"] = float(sum(passive_f1m) / len(passive_f1m))
        summary["passive_test_f1_macro_std"] = _std_sample(passive_f1m)
    if active_f1w:
        summary["active_test_f1_weighted_mean"] = float(sum(active_f1w) / len(active_f1w))
        summary["active_test_f1_weighted_std"] = _std_sample(active_f1w)
    if passive_f1w:
        summary["passive_test_f1_weighted_mean"] = float(sum(passive_f1w) / len(passive_f1w))
        summary["passive_test_f1_weighted_std"] = _std_sample(passive_f1w)
    if active_fpr:
        summary["active_test_fpr_macro_mean"] = float(sum(active_fpr) / len(active_fpr))
        summary["active_test_fpr_macro_std"] = _std_sample(active_fpr)
    if passive_fpr:
        summary["passive_test_fpr_macro_mean"] = float(sum(passive_fpr) / len(passive_fpr))
        summary["passive_test_fpr_macro_std"] = _std_sample(passive_fpr)
    if active_fnr:
        summary["active_test_fnr_macro_mean"] = float(sum(active_fnr) / len(active_fnr))
        summary["active_test_fnr_macro_std"] = _std_sample(active_fnr)
    if passive_fnr:
        summary["passive_test_fnr_macro_mean"] = float(sum(passive_fnr) / len(passive_fnr))
        summary["passive_test_fnr_macro_std"] = _std_sample(passive_fnr)

    payload = {
        "segment_length": args.segment_length,
        "device": str(device),
        "folds": fold_results,
        "summary": summary,
    }

    out_path = Path(args.output_json).resolve() if args.output_json else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'='*60}\nSUMMARY ({args.segment_length}, n={len(fold_results)})\n{'='*60}")
    print(
        f"  Active:  {summary['active_test_accuracy_mean_percent']:.4f}% "
        f"± {summary['active_test_accuracy_std_percent']:.4f}%"
    )
    print(
        f"  Passive: {summary['passive_test_accuracy_mean_percent']:.4f}% "
        f"± {summary['passive_test_accuracy_std_percent']:.4f}%"
    )
    if summary.get("active_test_f1_macro_mean") is not None:
        print(
            f"  Active F1 (macro):    {summary['active_test_f1_macro_mean']:.6f} "
            f"± {summary['active_test_f1_macro_std']:.6f}"
        )
        print(
            f"  Active F1 (weighted): {summary['active_test_f1_weighted_mean']:.6f} "
            f"± {summary['active_test_f1_weighted_std']:.6f}"
        )
    if summary.get("passive_test_f1_macro_mean") is not None:
        print(
            f"  Passive F1 (macro):    {summary['passive_test_f1_macro_mean']:.6f} "
            f"± {summary['passive_test_f1_macro_std']:.6f}"
        )
        print(
            f"  Passive F1 (weighted): {summary['passive_test_f1_weighted_mean']:.6f} "
            f"± {summary['passive_test_f1_weighted_std']:.6f}"
        )
    if summary.get("active_test_fpr_macro_mean") is not None:
        print(
            f"  Active FPR (macro OvR): {summary['active_test_fpr_macro_mean']:.6f} "
            f"± {summary['active_test_fpr_macro_std']:.6f}"
        )
        print(
            f"  Active FNR (macro OvR): {summary['active_test_fnr_macro_mean']:.6f} "
            f"± {summary['active_test_fnr_macro_std']:.6f}"
        )
    if summary.get("passive_test_fpr_macro_mean") is not None:
        print(
            f"  Passive FPR (macro OvR): {summary['passive_test_fpr_macro_mean']:.6f} "
            f"± {summary['passive_test_fpr_macro_std']:.6f}"
        )
        print(
            f"  Passive FNR (macro OvR): {summary['passive_test_fnr_macro_mean']:.6f} "
            f"± {summary['passive_test_fnr_macro_std']:.6f}"
        )
    print(f"  Wrote: {out_path}")


if __name__ == "__main__":
    main()
