#!/usr/bin/env python3
"""
Evaluate CNN/ResNet user-identification checkpoints on the **val** or **test** split (``--split``),
separately for **active** and **passive** tasks (same HDF5 layout as training). Default is ``test``.

Uses raw EEG tensors from ``data_processing`` (no LaBraM ``/ 100`` scaling). Checkpoints are
typically produced by ``train_user_identification_backbone.py`` (no prior weights required).

**Checkpoints:** Current training always uses ArcFace for user identification (``use_arcface=True``).
This script loads ``arcface_state_dict`` when present (using saved ArcFace hyperparameters when available).
Older CrossEntropy-only checkpoints (``use_arcface=False``) are still supported for evaluation via ``model(...)``.

**Metrics (per task, full test set):** same fields as ``eval_active_passive_test.py`` for LaBraM — accuracy (%),
mean loss, F1 macro / weighted, macro OvR FPR/FNR — via ``user_identification.labram_trainer.evaluate`` with
``eeg_input_mode="cnn"`` (raw ``(B, C, T)`` tensors, no LaBraM ``/100`` or patch rearrange).

Examples
--------
Single fold::

  python user_identification/cnn_resnet_arcface/eval_active_passive_cnn_resnet.py \\
    --checkpoint /path/to/best_model.pth \\
    --hdf5_dir /path/to/same_fold_hdf5 \\
    --segment_length 4s

All folds::

  python user_identification/cnn_resnet_arcface/eval_active_passive_cnn_resnet.py \\
    --cv_dir /path/to/results/4s \\
    --hdf5_root /path/to/5fold_data \\
    --segment_length 4s \\
    --n_folds 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_project_root = Path(__file__).resolve().parents[2]
_labram_dir = _project_root / "LaBraM"
for _p in (_project_root, _labram_dir):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from CNN.models.arcface_loss import ArcFaceLoss
from CNN.models.model_factory import ModelFactory
from user_identification.cnn_resnet_arcface.user_id_test_loaders import (
    build_user_identification_task_test_loaders,
    build_user_identification_task_loaders,
    verify_hdf5_dir,
)
from user_identification.labram_trainer import evaluate


# When ``factory_model_type`` is absent (older checkpoints), infer from trainer's model_info.
_MODEL_INFO_NAME_TO_FACTORY: Dict[str, str] = {
    "EEGUserIdentificationCNN": "user_identification_cnn",
    "EEGResNet18": "resnet18",
    "EEGResNet34": "resnet34",
    "EEGResNet50": "resnet50",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--hdf5_dir", type=str, default=None)
    p.add_argument("--cv_dir", type=str, default=None)
    p.add_argument("--hdf5_root", type=str, default=None)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--segment_length", type=str, required=True, choices=["1s", "2s", "4s"])
    p.add_argument("--batch_size", type=int, default=288)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Evaluate on the val or test split.",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dropout_rate", type=float, default=0.5, help="Must match training model kwargs")
    p.add_argument("--arcface_margin", type=float, default=0.5)
    p.add_argument("--arcface_scale", type=float, default=256.0)
    p.add_argument("--arcface_easy_margin", action="store_true")
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument(
        "--save_predictions_dir",
        type=str,
        default=None,
        help="If set, save y_true / y_pred per fold and task under this directory.",
    )
    return p.parse_args()


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint {path} missing 'model_state_dict'.")
    return ckpt


def _infer_num_classes(ckpt: Dict[str, Any]) -> int:
    if "num_classes" in ckpt:
        return int(ckpt["num_classes"])
    mi = ckpt.get("model_info") or {}
    if isinstance(mi.get("num_classes"), int):
        return int(mi["num_classes"])
    raise KeyError(
        "Checkpoint has no 'num_classes' and model_info lacks 'num_classes'. "
        "Re-train with an updated CNN trainer, or add num_classes to the checkpoint."
    )


def _resolve_factory_model_type(ckpt: Dict[str, Any]) -> str:
    ft = ckpt.get("factory_model_type")
    if isinstance(ft, str) and ft.strip():
        return ft.strip()
    info = ckpt.get("model_info") or {}
    name = info.get("model_name")
    if isinstance(name, str) and name in _MODEL_INFO_NAME_TO_FACTORY:
        return _MODEL_INFO_NAME_TO_FACTORY[name]
    raise KeyError(
        "Checkpoint has no 'factory_model_type' and model_info['model_name'] is missing "
        f"or unknown (got {name!r}). Known names: {sorted(_MODEL_INFO_NAME_TO_FACTORY.keys())}."
    )


def _build_model_and_optional_arcface(
    ckpt: Dict[str, Any],
    device: torch.device,
    dropout_rate: float,
    arcface_margin: float,
    arcface_scale: float,
    arcface_easy_margin: bool,
) -> tuple[torch.nn.Module, Optional[ArcFaceLoss]]:
    factory_type = _resolve_factory_model_type(ckpt)
    num_classes = _infer_num_classes(ckpt)
    use_layer_norm = factory_type == "user_identification_cnn"

    model = ModelFactory.create_model(
        factory_type,
        num_channels=60,
        num_classes=num_classes,
        dropout_rate=float(dropout_rate),
        use_layer_norm=use_layer_norm,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    if not ckpt.get("use_arcface"):
        return model, None
    if "arcface_state_dict" not in ckpt:
        raise KeyError(
            "Checkpoint has use_arcface=True but no 'arcface_state_dict'; file may be corrupt."
        )

    if "embedding_dim" not in ckpt:
        raise KeyError(
            "ArcFace checkpoint missing 'embedding_dim'. Re-train with the current CNN trainer, "
            "which saves embedding_dim when use_arcface is True."
        )
    margin = float(ckpt.get("arcface_margin", arcface_margin))
    scale = float(ckpt.get("arcface_scale", arcface_scale))
    easy = bool(ckpt.get("arcface_easy_margin", arcface_easy_margin))
    criterion = ArcFaceLoss(
        num_classes=num_classes,
        embedding_dim=int(ckpt["embedding_dim"]),
        margin=margin,
        scale=scale,
        easy_margin=easy,
    ).to(device)
    criterion.load_state_dict(ckpt["arcface_state_dict"], strict=True)
    criterion.eval()
    return model, criterion


def _eval_task_split(
    loader,
    model: torch.nn.Module,
    arcface: Optional[ArcFaceLoss],
    device: torch.device,
    task_type: str,
    predictions_npz: Optional[Path] = None,
) -> Dict[str, Any]:
    use_arcface = arcface is not None
    collect = predictions_npz is not None
    stats = evaluate(
        loader,
        model,
        device,
        header=f"Test ({task_type}):",
        ch_names=None,
        metrics=["accuracy"],
        use_arcface=use_arcface,
        criterion=arcface if use_arcface else None,
        collect_predictions=collect,
        eeg_input_mode="cnn",
    )
    predictions_path: Optional[str] = None
    if collect:
        yt = stats.pop("y_true", None)
        yp = stats.pop("y_pred", None)
        if yt is None or yp is None:
            raise RuntimeError(
                "evaluate(..., collect_predictions=True) must return y_true and y_pred."
            )
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
    verify_hdf5_dir(hdf5_dir, segment_length)
    ckpt = _load_checkpoint(checkpoint, device)
    model, arcface = _build_model_and_optional_arcface(
        ckpt,
        device,
        args.dropout_rate,
        args.arcface_margin,
        args.arcface_scale,
        args.arcface_easy_margin,
    )
    out: Dict[str, Any] = {
        "checkpoint": str(checkpoint.resolve()),
        "hdf5_dir": str(hdf5_dir.resolve()),
        "segment_length": segment_length,
        "factory_model_type": _resolve_factory_model_type(ckpt),
        "num_classes": _infer_num_classes(ckpt),
        "use_arcface": bool(ckpt.get("use_arcface")),
        "embedding_dim": ckpt.get("embedding_dim"),
    }
    pred_dir: Optional[Path] = None
    if args.save_predictions_dir:
        pred_dir = Path(args.save_predictions_dir).expanduser().resolve()
        pred_dir.mkdir(parents=True, exist_ok=True)

    split_name: str = getattr(args, "split", "test")
    _, active_ld, passive_ld = build_user_identification_task_loaders(
        hdf5_dir, segment_length, args.batch_size, args.num_workers, split=split_name
    )
    for tt, loader in (("active", active_ld), ("passive", passive_ld)):
        npz_path: Optional[Path] = None
        if pred_dir is not None:
            npz_path = pred_dir / f"fold_{fold_index:02d}_{segment_length}_{tt}.npz"
        out[tt] = _eval_task_split(
            loader,
            model,
            arcface,
            device,
            task_type=tt,
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
        raise SystemExit(
            "Specify either (--checkpoint AND --hdf5_dir) OR (--cv_dir AND --hdf5_root), not both/neither."
        )

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
        default_out = checkpoint.parent / "task_type_test_eval_cnn_resnet.json"
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
        default_out = cv_dir / "task_type_test_eval_cnn_resnet.json"

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
        "pipeline": "cnn_resnet_arcface",
        "folds": fold_results,
        "summary": summary,
    }

    out_path = Path(args.output_json).resolve() if args.output_json else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
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
