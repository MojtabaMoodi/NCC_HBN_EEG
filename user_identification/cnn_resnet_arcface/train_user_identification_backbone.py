#!/usr/bin/env python3
"""
Train user identification from scratch (no pretrained CNN/ResNet checkpoint required).

Uses the **CNN** pipeline (``CNN.experiment``): ``EEGUserIdentificationCNN`` or ``EEGResNet{18,34,50}``,
HDF5 data from ``data_processing``, and the same participant→class mapping as LaBraM user-ID runs.

**Loss:** For ``target_key='user_identification'``, ``CNN.trainer.EEGTrainer`` **always** uses **ArcFace**
(same idea as ``train_labram_arcface.py``): embeddings from ``extract_features()`` plus ``ArcFaceLoss``.
Models without ``extract_features`` raise a clear error at trainer init.

Outputs (under ``--output_dir``):

- ``best_model.pth`` — copy of the experiment best checkpoint (includes ``model_state_dict``,
  ``arcface_state_dict``, ``num_classes``, ``embedding_dim``, ``factory_model_type`` when applicable).
- ``results.json`` — ``val_accuracy`` / ``test_accuracy`` in **percent** (0–100), compatible with
  ``user_identification/run_5fold_cv.py`` aggregation.
- Per-task evaluation dirs ``{experiment_name}_task_active`` and ``{experiment_name}_task_passive``,
  each with a **compact** ``*_user_identification_metrics.json`` (accuracy, F1 macro/weighted, macro
  FPR/FNR). The CNN pipeline **does not** write ``*_predictions.json`` for user identification
  (full softmax matrices would be enormous).
- ``val_active_passive_metrics.json`` — written by a subprocess call to
  ``eval_active_passive_cnn_resnet.py`` with ``--split val`` (validation active/passive metrics).
- ``user_identification_active_passive_fold_metrics.json`` — fold summary: val/test active and
  passive metrics (same numeric fields as above), paths to task metrics JSON files, ``best_model_path``.
- ``user_identification_fold_metrics.log`` — human-readable copy of those metrics.

This path uses **raw** HDF5 EEG amplitudes in the DataLoader (same as ``CNN.trainer``), not the
LaBraM ``/ 100`` scaling used by ``train_labram_arcface.py``. Do not mix checkpoints across those
two pipelines.

**Stability defaults:** ``EEGTrainer`` uses **two Adam learning rates**: the **backbone** uses
``--learning_rate``; the **ArcFace head** uses ``learning_rate * arcface_lr_multiplier``.
**CNN** defaults: ``--learning_rate 3e-4``, ``--warmup_epochs 20``, ``--arcface_lr_multiplier 0.15``.
**ResNet** defaults (tuned to reduce ArcFace collapse vs. the old 1e-3 + scale 256 recipe):
``--learning_rate 4e-4``, ``--warmup_epochs 18``, ``--arcface_lr_multiplier 0.22``, ``--arcface_scale 96``
(unless you pass overrides). **Global gradient clipping** defaults to ``max_grad_norm=0.2``; use
``--max_grad_norm`` to tune (``0`` disables). Training uses **full fp32** and **no ``torch.compile``**
for ``target_key=user_identification``.

For standalone evaluation of a checkpoint on **test** (default) or **validation**, use
``user_identification/cnn_resnet_arcface/eval_active_passive_cnn_resnet.py`` (``--split test`` or
``--split val``).

Examples
--------
::

  python user_identification/cnn_resnet_arcface/train_user_identification_backbone.py \\
    --hdf5_dir /path/to/fold_0 \\
    --output_dir /path/to/out/fold_0 \\
    --backbone resnet34 \\
    --segment_length 4s
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Tuple
import subprocess

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch

from CNN.config import ExperimentConfig, ModelConfig, SystemConfig, TrainingConfig
from CNN.experiment import Experiment
from CNN.utils import create_data_config_for_segment_length
from data_processing.eeg_dataset import get_recommended_num_workers
from user_identification.cnn_resnet_arcface.model_registry import (
    resolve_model_type,
    use_batch_norm_backbone,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--hdf5_dir", type=str, required=True, help="Directory with train/val/test HDF5")
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for checkpoints, copies, and results.json",
    )
    p.add_argument(
        "--backbone",
        type=str,
        required=True,
        choices=["cnn", "resnet18", "resnet34", "resnet50"],
        help="Backbone: user_identification_cnn or resnet18/34/50",
    )
    p.add_argument("--segment_length", type=str, default="4s", choices=["1s", "2s", "4s"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Backbone Adam LR (ArcFace head uses this × arcface_lr_multiplier). Default: 3e-4 (cnn), 4e-4 (resnet*).",
    )
    p.add_argument(
        "--warmup_epochs",
        type=int,
        default=None,
        help="LR warmup length. Default: 20 (cnn), 18 (resnet*).",
    )
    p.add_argument(
        "--arcface_lr_multiplier",
        type=float,
        default=None,
        help="ArcFace head LR = learning_rate × this. Default: 0.15 (cnn), 0.22 (resnet*).",
    )
    p.add_argument("--batch_size", type=int, default=None, help="Override; default from create_data_config_for_segment_length")
    p.add_argument("--num_workers", type=int, default=None, help="Override; default from get_recommended_num_workers")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout_rate", type=float, default=0.5, help="Passed to the model (CNN and ResNet)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Passed to SystemConfig.device",
    )
    p.add_argument(
        "--arcface_margin",
        type=float,
        default=None,
        help="If set, overrides TrainingConfig.arcface_margin (default: config file default)",
    )
    p.add_argument(
        "--arcface_scale",
        type=float,
        default=None,
        help="If set, overrides TrainingConfig.arcface_scale (lower e.g. 128 can reduce logit sharpness / collapse).",
    )
    p.add_argument(
        "--max_grad_norm",
        type=float,
        default=None,
        help="Global L2 norm clip after backward (backbone + ArcFace). Default: 0.2. Set 0 to disable clipping.",
    )
    return p.parse_args()


def _resolved_training_hyperparams(args: argparse.Namespace) -> Tuple[float, int, float]:
    """Defaults tuned for stable ArcFace user-ID training (ResNet used 1e-3 + scale 256 was collapse-prone)."""
    if args.learning_rate is not None:
        lr = float(args.learning_rate)
    else:
        lr = 3e-4 if args.backbone == "cnn" else 4e-4

    if args.warmup_epochs is not None:
        warmup = int(args.warmup_epochs)
    else:
        warmup = 20 if args.backbone == "cnn" else 18

    if args.arcface_lr_multiplier is not None:
        arcface_m = float(args.arcface_lr_multiplier)
    else:
        arcface_m = 0.15 if args.backbone == "cnn" else 0.22
    return lr, warmup, arcface_m


def main() -> None:
    args = _parse_args()
    hdf5_dir = Path(args.hdf5_dir)
    if not hdf5_dir.is_dir():
        raise FileNotFoundError(f"hdf5_dir is not a directory: {hdf5_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    model_type = resolve_model_type(args.backbone)
    is_resnet = use_batch_norm_backbone(args.backbone)

    data_cfg = create_data_config_for_segment_length(
        args.segment_length,
        batch_size=args.batch_size,
        random_seed=args.seed,
        num_gpus=max(1, n_gpu),
    )
    data_cfg.hdf5_dir = str(hdf5_dir.resolve())
    if args.num_workers is not None:
        data_cfg.num_workers = int(args.num_workers)
    else:
        data_cfg.num_workers = get_recommended_num_workers(args.segment_length, max(1, n_gpu))

    exp_name = f"user_identification_{args.backbone}_{args.segment_length}"
    learning_rate, warmup_epochs, arcface_lr_m = _resolved_training_hyperparams(args)
    if args.max_grad_norm is not None:
        max_grad = float(args.max_grad_norm)
    else:
        max_grad = 0.2
    training_kwargs = dict(
        epochs=args.epochs,
        learning_rate=learning_rate,
        target_key="user_identification",
        prediction_type="classification",
        weight_decay=args.weight_decay,
        warmup_epochs=warmup_epochs,
        max_grad_norm=max_grad,
        scheduler_patience=3,
        use_stratified_train_batches=False,
        arcface_lr_multiplier=float(arcface_lr_m),
    )
    if args.arcface_margin is not None:
        training_kwargs["arcface_margin"] = float(args.arcface_margin)
    if args.arcface_scale is not None:
        training_kwargs["arcface_scale"] = float(args.arcface_scale)
    elif args.backbone != "cnn":
        training_kwargs["arcface_scale"] = 96.0

    _cfg = TrainingConfig()
    arcface_scale_eff = float(training_kwargs.get("arcface_scale", _cfg.arcface_scale))
    print(
        f"Training hyperparams: learning_rate={learning_rate}, warmup_epochs={warmup_epochs}, "
        f"arcface_lr_multiplier={arcface_lr_m}, arcface_scale={arcface_scale_eff}, max_grad_norm={max_grad}"
    )

    experiment = Experiment(
        ExperimentConfig(
            name=exp_name,
            model_type=model_type,
            target_type="user_identification",
            description=(
                f"User identification (ArcFace via CNN trainer): backbone={args.backbone}, "
                f"segment_length={args.segment_length}"
            ),
            data_config=data_cfg,
            model_config=ModelConfig(
                num_channels=60,
                dropout_rate=float(args.dropout_rate),
                use_layer_norm=not is_resnet,
            ),
            training_config=TrainingConfig(**training_kwargs),
        ),
        system_config=SystemConfig(
            results_dir=str(output_dir.resolve()),
            device=args.device,
            num_gpus=max(1, n_gpu),
        ),
    )

    experiment.setup()
    nc = int(experiment.config.model_config.num_classes)
    use_arcface = bool(getattr(experiment.trainer, "use_arcface", False))
    if not use_arcface:
        raise RuntimeError(
            f"Expected ArcFace for user identification (num_classes={nc}), but trainer has "
            f"use_arcface=False. Supported backbones must implement extract_features()."
        )
    print(f"\nUser identification: num_classes={nc}, loss=ArcFace (always for this target).\n")
    result = experiment.run()
    if not result.success:
        raise RuntimeError(f"Experiment failed: {result.error}")

    ckpt_src = Path(experiment._get_checkpoint_path())
    if not ckpt_src.is_file():
        raise FileNotFoundError(f"Expected best checkpoint at {ckpt_src} after training.")
    ckpt_dst = output_dir / "best_model.pth"
    shutil.copy2(ckpt_src, ckpt_dst)

    train_metrics = result.training_metrics or {}
    val_list = train_metrics.get("val_acc") or []
    if not val_list:
        raise RuntimeError("Training did not record val_acc; cannot write results.json.")
    val_accuracy_pct = float(max(val_list)) * 100.0

    metrics = result.metrics or {}
    if "accuracy" not in metrics:
        raise KeyError(
            f"Evaluation metrics missing 'accuracy'; keys={list(metrics.keys())}. "
            "Cannot write results.json for run_5fold_cv compatibility."
        )
    test_accuracy_pct = float(metrics["accuracy"]) * 100.0

    results_payload = {
        "val_accuracy": val_accuracy_pct,
        "test_accuracy": test_accuracy_pct,
        "backbone": args.backbone,
        "factory_model_type": model_type,
        "segment_length": args.segment_length,
        "experiment_name": exp_name,
        "num_classes": nc,
        "use_arcface": use_arcface,
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\nWrote {output_dir / 'results.json'}")
    print(f"Copied best checkpoint to {ckpt_dst}")

    def _load_task_metrics(task_type: str) -> tuple[dict, Path]:
        task_dir = output_dir / f"{exp_name}_task_{task_type}"
        if not task_dir.is_dir():
            raise FileNotFoundError(
                f"Expected task metrics directory not found: {task_dir}. "
                f"Got exp_name={exp_name}, task_type={task_type}."
            )
        candidates = list(task_dir.glob("*_user_identification_metrics.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No *_user_identification_metrics.json found under {task_dir}. "
                f"Contents={list(task_dir.iterdir())[:20]}"
            )
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected exactly 1 user-identification metrics json under {task_dir}, found {len(candidates)}: "
                f"{[c.name for c in candidates]}"
            )
        metrics_path = candidates[0]
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f), metrics_path

    def _extract_compact_metrics(raw_metrics: dict) -> dict:
        metrics = raw_metrics.get("metrics") if isinstance(raw_metrics, dict) else None
        if not isinstance(metrics, dict):
            raise KeyError(f"Expected top-level {raw_metrics} to contain a dict 'metrics' field.")

        required = ("accuracy", "f1_macro", "f1_weighted", "fpr_macro", "fnr_macro")
        missing = [k for k in required if k not in metrics]
        if missing:
            raise KeyError(f"Missing required metrics keys={missing}. Available keys={list(metrics.keys())}")

        # Convert accuracy to percent for consistency with results.json / LaBraM outputs.
        return {
            "accuracy_percent": float(metrics["accuracy"]) * 100.0,
            "f1_macro": float(metrics["f1_macro"]),
            "f1_weighted": float(metrics["f1_weighted"]),
            "fpr_macro": float(metrics["fpr_macro"]),
            "fnr_macro": float(metrics["fnr_macro"]),
        }

    def _extract_eval_task_metrics(task_block: dict) -> dict:
        if not isinstance(task_block, dict):
            raise KeyError(f"Expected task_block to be a dict, got {type(task_block)}")
        required = ("accuracy_percent", "f1_macro", "f1_weighted", "fpr_macro", "fnr_macro")
        missing = [k for k in required if k not in task_block]
        if missing:
            raise KeyError(
                f"Missing required eval task keys={missing}. Available keys={list(task_block.keys())}"
            )
        return {
            "accuracy_percent": float(task_block["accuracy_percent"]),
            "f1_macro": float(task_block["f1_macro"]) if task_block["f1_macro"] is not None else float("nan"),
            "f1_weighted": float(task_block["f1_weighted"]) if task_block["f1_weighted"] is not None else float("nan"),
            "fpr_macro": float(task_block["fpr_macro"]) if task_block["fpr_macro"] is not None else float("nan"),
            "fnr_macro": float(task_block["fnr_macro"]) if task_block["fnr_macro"] is not None else float("nan"),
        }

    # Fold-level summary comparable to LaBraM user-identification logging: active/passive test metrics.
    test_active_raw, test_active_path = _load_task_metrics("active")
    test_passive_raw, test_passive_path = _load_task_metrics("passive")
    test_active = _extract_compact_metrics(test_active_raw)
    test_passive = _extract_compact_metrics(test_passive_raw)

    # Validation active/passive metrics (match LaBraM fields) from the ResNet/CNN eval script.
    # This avoids relying on training's val_acc alone.
    val_out_path = output_dir / "val_active_passive_metrics.json"
    eval_script = _project_root / "user_identification" / "cnn_resnet_arcface" / "eval_active_passive_cnn_resnet.py"
    eval_cmd = [
        sys.executable,
        str(eval_script),
        "--checkpoint",
        str(ckpt_dst),
        "--hdf5_dir",
        str(hdf5_dir.resolve()),
        "--segment_length",
        args.segment_length,
        "--device",
        ("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)),
        "--dropout_rate",
        str(args.dropout_rate),
        "--split",
        "val",
        "--output_json",
        str(val_out_path),
    ]
    # Use training batch sizing defaults from the CNN pipeline (eval script has its own defaults, but
    # we set them consistently when possible).
    try:
        data_cfg_batch = int(getattr(data_cfg, "batch_size", 0) or 0)
        data_cfg_workers = int(getattr(data_cfg, "num_workers", 0) or 0)
        if data_cfg_batch > 0:
            eval_cmd += ["--batch_size", str(data_cfg_batch)]
        if data_cfg_workers > 0:
            eval_cmd += ["--num_workers", str(data_cfg_workers)]
    except Exception as e:
        raise RuntimeError(f"Failed to resolve data_cfg batch/worker settings for validation eval: {e}") from e

    ret = subprocess.run(eval_cmd, cwd=str(_project_root))
    if ret.returncode != 0:
        raise RuntimeError(f"Validation evaluation failed (exit code {ret.returncode}). Cmd: {' '.join(eval_cmd)}")

    with open(val_out_path, "r", encoding="utf-8") as f:
        val_payload = json.load(f)

    if not val_payload.get("folds"):
        raise RuntimeError(f"Validation eval output has no folds. Path: {val_out_path}")
    val_fold0 = val_payload["folds"][0]
    val_active = _extract_eval_task_metrics(val_fold0["active"])
    val_passive = _extract_eval_task_metrics(val_fold0["passive"])

    fold_summary = {
        "val_accuracy_percent": val_accuracy_pct,
        "val_active": val_active,
        "val_passive": val_passive,
        "test_active": test_active,
        "test_passive": test_passive,
        "backbone": args.backbone,
        "segment_length": args.segment_length,
        "num_classes": nc,
        "use_arcface": use_arcface,
        "best_model_path": str(ckpt_dst.resolve()),
        "task_active_metrics_json": str(test_active_path.resolve()),
        "task_passive_metrics_json": str(test_passive_path.resolve()),
    }

    # Write compact fold metrics + a human-readable log.
    fold_summary_path = output_dir / "user_identification_active_passive_fold_metrics.json"
    with open(fold_summary_path, "w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2)

    fold_log_path = output_dir / "user_identification_fold_metrics.log"
    with open(fold_log_path, "w", encoding="utf-8") as f:
        f.write(f"best_model: {ckpt_dst.resolve()}\n")
        f.write(f"val_accuracy_percent: {val_accuracy_pct:.4f}\n")
        f.write("val_active:\n")
        f.write(
            f"  accuracy_percent={val_active['accuracy_percent']:.4f}, "
            f"f1_macro={val_active['f1_macro']:.6f}, "
            f"f1_weighted={val_active['f1_weighted']:.6f}, "
            f"fpr_macro={val_active['fpr_macro']:.8f}, "
            f"fnr_macro={val_active['fnr_macro']:.8f}\n"
        )
        f.write("val_passive:\n")
        f.write(
            f"  accuracy_percent={val_passive['accuracy_percent']:.4f}, "
            f"f1_macro={val_passive['f1_macro']:.6f}, "
            f"f1_weighted={val_passive['f1_weighted']:.6f}, "
            f"fpr_macro={val_passive['fpr_macro']:.8f}, "
            f"fnr_macro={val_passive['fnr_macro']:.8f}\n"
        )
        f.write("test_active:\n")
        f.write(
            f"  accuracy_percent={test_active['accuracy_percent']:.4f}, "
            f"f1_macro={test_active['f1_macro']:.6f}, "
            f"f1_weighted={test_active['f1_weighted']:.6f}, "
            f"fpr_macro={test_active['fpr_macro']:.8f}, "
            f"fnr_macro={test_active['fnr_macro']:.8f}\n"
        )
        f.write("test_passive:\n")
        f.write(
            f"  accuracy_percent={test_passive['accuracy_percent']:.4f}, "
            f"f1_macro={test_passive['f1_macro']:.6f}, "
            f"f1_weighted={test_passive['f1_weighted']:.6f}, "
            f"fpr_macro={test_passive['fpr_macro']:.8f}, "
            f"fnr_macro={test_passive['fnr_macro']:.8f}\n"
        )
        f.write(f"wrote_json: {fold_summary_path}\n")

    print(f"Wrote fold summary: {fold_summary_path}")
    print(f"Wrote fold log: {fold_log_path}")


if __name__ == "__main__":
    main()
