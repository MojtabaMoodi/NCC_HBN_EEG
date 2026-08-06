#!/usr/bin/env python3
"""
Subject-level bootstrap evaluation for final_logs checkpoints.

Loads saved best checkpoints from ``final_logs/`` (no retraining), runs test
inference, aggregates to participant level, and reports bootstrap CIs and
p-values vs a stratified random baseline (classification) or train-set median
age baseline (regression).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

_PROJECT_ROOT = _EVAL_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bootstrap import (  # noqa: E402
    subject_bootstrap_classification,
    subject_bootstrap_regression,
    summarize_bootstrap_classification,
    summarize_bootstrap_regression,
)
from nhst import (  # noqa: E402
    permutation_p_value_label_association_classification,
    permutation_p_value_vs_random_baseline_classification,
    wilcoxon_p_value_vs_baseline_regression,
)
from data_processing.target_transforms import create_age_regression_transform_from_hdf5  # noqa: E402
from final_logs_registry import (  # noqa: E402
    BACKEND_IDS,
    list_model_entries,
    manifest_paths,
)
from generate_manifests import find_hdf5_files, generate_test_manifest, generate_train_manifest  # noqa: E402
from io_utils import (  # noqa: E402
    aggregate_classification_by_subject,
    aggregate_regression_by_subject,
    check_train_test_overlap,
    compute_train_regression_baseline,
    compute_train_subject_proportions,
    load_test_manifest,
    load_train_manifest,
    validate_subject_labels,
)
from metrics import (  # noqa: E402
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_regression_metrics,
)
from model_loader import get_labram_input_chans, get_model_output, load_model_from_config  # noqa: E402
from prob_utils import logits_to_probabilities  # noqa: E402
from task_specs import TASK_IDS, get_task_spec, get_window_spec  # noqa: E402


class ManifestDataset(Dataset):
    """Dataset that loads EEG segments from HDF5 via manifest rows."""

    def __init__(self, manifest: pd.DataFrame, hdf5_base_dir: Path):
        self.manifest = manifest.copy()
        self.hdf5_base_dir = hdf5_base_dir.resolve()
        if not self.hdf5_base_dir.is_dir():
            raise FileNotFoundError(f"HDF5 base directory not found: {self.hdf5_base_dir}")

        if "segment_path" in manifest.columns:
            self.use_segment_path = True
        elif "hdf5_file" in manifest.columns and "sample_id" in manifest.columns:
            self.use_segment_path = False
        else:
            raise ValueError(
                "Manifest must have either 'segment_path' or ('hdf5_file', 'sample_id')"
            )

        self._hdf5_files: Dict[str, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.manifest.iloc[idx]
        subject_id = str(row["subject_id"])
        label = float(row["label"])

        if self.use_segment_path:
            segment_path = Path(row["segment_path"])
            if not segment_path.is_absolute():
                segment_path = self.hdf5_base_dir / segment_path
            hdf5_file = str(segment_path).split(":")[0]
            if ":" in str(segment_path):
                sample_id = str(segment_path).split(":")[1]
            else:
                parts = str(segment_path).split("/")
                sample_id = parts[-1]
                hdf5_file = "/".join(parts[:-2]) if len(parts) > 2 else parts[0]
        else:
            hdf5_file = row["hdf5_file"]
            sample_id = row["sample_id"]

        eeg_data, participant_id = self._load_from_hdf5(str(hdf5_file), str(sample_id))
        return {
            "eeg_data": torch.as_tensor(eeg_data, dtype=torch.float32),
            "label": label,
            "subject_id": subject_id,
            "participant_id": participant_id,
        }

    def _resolve_hdf5_path(self, hdf5_file: str) -> str:
        hdf5_path = Path(hdf5_file)
        if hdf5_path.is_absolute():
            return str(hdf5_path.resolve())
        return str((self.hdf5_base_dir / hdf5_file).resolve())

    def _load_from_hdf5(self, hdf5_file: str, sample_id: str) -> Tuple[np.ndarray, str]:
        hdf5_file = self._resolve_hdf5_path(hdf5_file)
        if hdf5_file not in self._hdf5_files:
            if not Path(hdf5_file).is_file():
                raise FileNotFoundError(f"HDF5 file not found: {hdf5_file}")
            self._hdf5_files[hdf5_file] = h5py.File(hdf5_file, "r")

        f = self._hdf5_files[hdf5_file]
        if sample_id.startswith("sample_"):
            metadata_key = sample_id.replace("sample_", "metadata_", 1)
            sample_num = sample_id.replace("sample_", "", 1)
        else:
            sample_num = sample_id.split("_")[-1]
            metadata_key = f"metadata_{sample_num}"

        for task_type in ("active", "passive"):
            if task_type not in f:
                continue
            task_group = f[task_type]
            if sample_id not in task_group:
                continue

            eeg_data = task_group[sample_id][:]
            if metadata_key not in task_group:
                raise KeyError(
                    f"Metadata {metadata_key!r} not found for {sample_id!r} in {hdf5_file}:{task_type}"
                )
            metadata = task_group[metadata_key]
            participant_id = metadata.attrs.get("participant_id", None)
            if participant_id is None:
                raise KeyError(f"participant_id missing in {hdf5_file}:{task_type}/{metadata_key}")
            if isinstance(participant_id, bytes):
                participant_id = participant_id.decode()
            return eeg_data, str(participant_id)

        available = []
        for task_type in ("active", "passive"):
            if task_type in f:
                available.extend([k for k in f[task_type].keys() if k.startswith("sample_")])
        raise ValueError(
            f"Sample {sample_id!r} not found in {hdf5_file}. "
            f"Available samples (first 10): {available[:10]}"
        )

    def close(self) -> None:
        for handle in self._hdf5_files.values():
            handle.close()
        self._hdf5_files.clear()


def run_classification_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    model_config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    num_classes = int(model_config["num_classes"])
    prediction_type = model_config["prediction_type"]
    if prediction_type != "classification":
        raise ValueError(f"run_classification_inference requires classification, got {prediction_type}")

    output_key = model_config.get("output_key")
    model_name = model_config["name"].lower()
    input_chans = get_labram_input_chans() if model_name == "labram" else None

    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_subject_ids: List[str] = []

    with torch.no_grad():
        for batch in dataloader:
            eeg_data = batch["eeg_data"].to(device)
            labels = np.asarray(batch["label"], dtype=float)
            subject_ids = batch["subject_id"]

            logits = get_model_output(
                model,
                eeg_data,
                output_key=output_key,
                model_name=model_name,
                input_chans=input_chans,
                prediction_type="classification",
                num_classes=num_classes,
            )
            # Pass head width (1 for binary LaBraM), not label-space size.
            probs = logits_to_probabilities(logits, num_classes)

            all_probs.append(probs)
            all_labels.append(labels)
            all_subject_ids.extend(subject_ids)

    return (
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_labels, axis=0),
        all_subject_ids,
    )


def run_regression_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    model_config: Dict[str, Any],
    age_min: float,
    age_max: float,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    num_classes = int(model_config["num_classes"])
    prediction_type = model_config["prediction_type"]
    if prediction_type != "regression":
        raise ValueError(f"run_regression_inference requires regression, got {prediction_type}")

    output_key = model_config.get("output_key")
    model_name = model_config.get("name", "").lower()
    input_chans = get_labram_input_chans() if model_name == "labram" else None

    all_preds_years: List[np.ndarray] = []
    all_labels_years: List[np.ndarray] = []
    all_subject_ids: List[str] = []

    with torch.no_grad():
        for batch in dataloader:
            eeg_data = batch["eeg_data"].to(device)
            labels_years = np.asarray(batch["label"], dtype=float)
            subject_ids = batch["subject_id"]

            normalized = get_model_output(
                model,
                eeg_data,
                output_key=output_key,
                model_name=model_name,
                input_chans=input_chans,
                prediction_type="regression",
                num_classes=num_classes,
            )
            preds_years = normalized.detach().cpu().numpy() * (age_max - age_min) + age_min

            all_preds_years.append(preds_years)
            all_labels_years.append(labels_years)
            all_subject_ids.extend(subject_ids)

    return (
        np.concatenate(all_preds_years, axis=0),
        np.concatenate(all_labels_years, axis=0),
        all_subject_ids,
    )


def ensure_manifests(
    *,
    hdf5_base_dir: Path,
    manifest_dir: Path,
    task_id: str,
    segment_length: str,
    task_type: str,
) -> Tuple[Path, Path]:
    train_path, test_path = manifest_paths(manifest_dir, task_id, segment_length)
    if train_path.is_file() and test_path.is_file():
        return train_path, test_path

    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_files = find_hdf5_files(hdf5_base_dir, "train", segment_length, pattern=None)
    test_files = find_hdf5_files(hdf5_base_dir, "test", segment_length, pattern=None)

    if not train_path.is_file():
        generate_train_manifest(train_files, train_path, task_id, task_type)
    if not test_path.is_file():
        generate_test_manifest(test_files, test_path, task_id, task_type)

    return train_path, test_path


def resolve_age_range_from_train_hdf5(hdf5_base_dir: Path, segment_length: str) -> Tuple[float, float]:
    train_files = find_hdf5_files(hdf5_base_dir, "train", segment_length, pattern=None)
    _, age_min, age_max = create_age_regression_transform_from_hdf5(
        [str(path) for path in train_files]
    )
    return float(age_min), float(age_max)


def get_git_commit_hash() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def parse_backends(backends_arg: str) -> List[str]:
    backends = [part.strip() for part in backends_arg.split(",") if part.strip()]
    if not backends:
        raise ValueError("--backends must list at least one backend")
    unknown = [b for b in backends if b not in BACKEND_IDS]
    if unknown:
        raise ValueError(f"Unknown backends: {unknown}. Expected subset of {sorted(BACKEND_IDS)}")
    return backends


def evaluate_classification(
    *,
    model_entry,
    task_spec,
    train_manifest: pd.DataFrame,
    dataloader: DataLoader,
    device: torch.device,
    bootstrap_B: int,
    permutation_M: int,
    seed: int,
    confidence: float,
) -> Dict[str, Any]:
    model = load_model_from_config(model_entry.model_config, device=str(device))
    model.to(device)

    segment_probs, segment_labels, segment_subject_ids = run_classification_inference(
        model, dataloader, device, model_entry.model_config
    )

    subjects, y_true_subject, y_pred_subject, _ = aggregate_classification_by_subject(
        segment_probs,
        segment_labels.astype(int),
        segment_subject_ids,
        aggregation=task_spec.aggregation,
    )

    class_labels = list(task_spec.class_labels)
    metrics_fn = lambda yt, yp: compute_classification_metrics(  # noqa: E731
        yt.astype(int), yp.astype(int), class_labels
    )

    point_metrics = metrics_fn(y_true_subject, y_pred_subject)
    conf_matrix = compute_confusion_matrix(
        y_true_subject.astype(int), y_pred_subject.astype(int), class_labels
    )

    train_proportions, majority_class = compute_train_subject_proportions(
        train_manifest, class_labels
    )
    y_pred_random = np.random.default_rng(seed).choice(
        class_labels, size=len(y_true_subject), p=train_proportions
    )
    random_metrics = metrics_fn(y_true_subject, y_pred_random)
    y_pred_majority = np.full(len(y_true_subject), majority_class, dtype=int)
    majority_metrics = metrics_fn(y_true_subject, y_pred_majority)

    bootstrap_results = subject_bootstrap_classification(
        y_true_subject.astype(int),
        y_pred_subject.astype(int),
        y_pred_majority,
        train_proportions,
        class_labels,
        B=bootstrap_B,
        seed=seed,
        metrics_func=metrics_fn,
    )

    point_with_random = dict(point_metrics)
    for key, value in random_metrics.items():
        if isinstance(value, (int, float)):
            point_with_random["random_" + key] = value

    summary = summarize_bootstrap_classification(
        bootstrap_results,
        point_with_random,
        higher_is_better=task_spec.higher_is_better,
        confidence=confidence,
    )

    nhst: Dict[str, Any] = {}
    nhst_p_value: Optional[float] = None
    if permutation_M > 0:
        nhst["vs_random_baseline"] = permutation_p_value_vs_random_baseline_classification(
            y_true_subject.astype(int),
            y_pred_subject.astype(int),
            train_proportions,
            class_labels,
            task_spec.primary_metric,
            permutation_M,
            seed,
            metrics_fn,
            task_spec.higher_is_better,
        )
        nhst["label_association"] = permutation_p_value_label_association_classification(
            y_true_subject.astype(int),
            y_pred_subject.astype(int),
            task_spec.primary_metric,
            permutation_M,
            seed,
            metrics_fn,
            task_spec.higher_is_better,
        )
        nhst_p_value = nhst["vs_random_baseline"]["p_value"]
        summary[task_spec.primary_metric]["p_value"] = nhst_p_value
        summary[task_spec.primary_metric]["p_value_test"] = nhst["vs_random_baseline"]["test"]

    return {
        "point_metrics": point_metrics,
        "random_metrics": random_metrics,
        "majority_metrics": majority_metrics,
        "confusion_matrix": conf_matrix.tolist(),
        "bootstrap_summary": summary,
        "nhst": nhst,
        "p_value": nhst_p_value,
        "permutation_p_value": nhst.get("label_association", {}).get("p_value"),
        "num_subjects": int(len(subjects)),
        "num_segments": int(len(segment_subject_ids)),
        "checkpoint_path": str(model_entry.ckpt_path),
    }


def evaluate_regression_run(
    *,
    model_entry,
    task_spec,
    train_manifest: pd.DataFrame,
    dataloader: DataLoader,
    device: torch.device,
    bootstrap_B: int,
    seed: int,
    confidence: float,
    baseline_stat: str,
    age_min: float,
    age_max: float,
) -> Dict[str, Any]:
    model = load_model_from_config(model_entry.model_config, device=str(device))
    model.to(device)

    segment_preds, segment_labels, segment_subject_ids = run_regression_inference(
        model,
        dataloader,
        device,
        model_entry.model_config,
        age_min=age_min,
        age_max=age_max,
    )

    subjects, y_true_subject, y_pred_subject = aggregate_regression_by_subject(
        segment_preds,
        segment_labels,
        segment_subject_ids,
        aggregation=task_spec.aggregation,
    )

    metrics_fn = compute_regression_metrics
    point_metrics = metrics_fn(y_true_subject, y_pred_subject)

    baseline_age = compute_train_regression_baseline(train_manifest, baseline_stat)
    y_pred_baseline = np.full(len(y_true_subject), baseline_age, dtype=float)
    baseline_metrics = metrics_fn(y_true_subject, y_pred_baseline)

    bootstrap_results = subject_bootstrap_regression(
        y_true_subject,
        y_pred_subject,
        y_pred_baseline,
        B=bootstrap_B,
        seed=seed,
        metrics_func=metrics_fn,
    )

    point_with_baseline = dict(point_metrics)
    for key, value in baseline_metrics.items():
        point_with_baseline["baseline_" + key] = value

    summary = summarize_bootstrap_regression(
        bootstrap_results,
        point_with_baseline,
        higher_is_better=task_spec.higher_is_better,
        confidence=confidence,
    )

    nhst = {
        "vs_median_age_baseline": wilcoxon_p_value_vs_baseline_regression(
            y_true_subject,
            y_pred_subject,
            y_pred_baseline,
            task_spec.primary_metric,
        )
    }
    nhst_p_value = nhst["vs_median_age_baseline"]["p_value"]
    summary[task_spec.primary_metric]["p_value"] = nhst_p_value
    summary[task_spec.primary_metric]["p_value_test"] = nhst["vs_median_age_baseline"]["test"]

    return {
        "point_metrics": point_metrics,
        "baseline_metrics": baseline_metrics,
        "train_baseline_age_years": baseline_age,
        "bootstrap_summary": summary,
        "nhst": nhst,
        "p_value": nhst_p_value,
        "permutation_p_value": None,
        "num_subjects": int(len(subjects)),
        "num_segments": int(len(segment_subject_ids)),
        "checkpoint_path": str(model_entry.ckpt_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap evaluation for final_logs checkpoints")
    parser.add_argument("--task", type=str, required=True, choices=sorted(TASK_IDS))
    parser.add_argument("--segment_length", type=str, required=True, choices=["1s", "2s", "4s"])
    parser.add_argument(
        "--backends",
        type=str,
        required=True,
        help="Comma-separated backends: cnn,resnet18,resnet34,resnet50,labram",
    )
    parser.add_argument("--hdf5_base_dir", type=str, required=True)
    parser.add_argument("--eeg_root", type=str, required=True, help="EEG repo root containing final_logs/")
    parser.add_argument("--manifest_dir", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True, choices=["active", "passive", "both"])
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--device", type=str, required=True, choices=["cuda", "cpu"])
    parser.add_argument("--bootstrap_B", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--confidence", type=float, required=True)
    parser.add_argument("--permutation_M", type=int, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--regression_baseline_stat",
        type=str,
        required=False,
        choices=["median", "mean"],
        help="Required when --task age_regression",
    )
    parser.add_argument("--save_npz", action="store_true")

    args = parser.parse_args()

    task_spec = get_task_spec(args.task)
    window_spec = get_window_spec(args.segment_length)
    if task_spec.prediction_type == "regression" and args.regression_baseline_stat is None:
        raise ValueError("--regression_baseline_stat is required for age_regression")

    hdf5_base_dir = Path(args.hdf5_base_dir)
    eeg_root = Path(args.eeg_root)
    manifest_dir = Path(args.manifest_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backends = parse_backends(args.backends)
    model_entries = list_model_entries(eeg_root, args.task, args.segment_length, backends)

    train_manifest_path, test_manifest_path = ensure_manifests(
        hdf5_base_dir=hdf5_base_dir,
        manifest_dir=manifest_dir,
        task_id=args.task,
        segment_length=args.segment_length,
        task_type=args.task_type,
    )

    allowed_labels = set(task_spec.class_labels) if task_spec.class_labels is not None else None
    train_manifest = load_train_manifest(str(train_manifest_path), allowed_labels=allowed_labels)
    test_manifest = load_test_manifest(str(test_manifest_path), allowed_labels=allowed_labels)

    validate_subject_labels(test_manifest)
    check_train_test_overlap(train_manifest, test_manifest)

    if torch.cuda.is_available() and args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        raise RuntimeError("CUDA requested via --device cuda but torch.cuda.is_available() is False")

    dataset = ManifestDataset(test_manifest, hdf5_base_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=window_spec.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    age_min: Optional[float] = None
    age_max: Optional[float] = None
    if task_spec.prediction_type == "regression":
        age_min, age_max = resolve_age_range_from_train_hdf5(hdf5_base_dir, args.segment_length)

    all_results: Dict[str, Any] = {}
    try:
        for entry in model_entries:
            print(f"\n{'=' * 60}\nEvaluating {entry.display_name}\n{'=' * 60}")
            if task_spec.prediction_type == "classification":
                result = evaluate_classification(
                    model_entry=entry,
                    task_spec=task_spec,
                    train_manifest=train_manifest,
                    dataloader=dataloader,
                    device=device,
                    bootstrap_B=args.bootstrap_B,
                    permutation_M=args.permutation_M,
                    seed=args.seed,
                    confidence=args.confidence,
                )
            else:
                assert age_min is not None and age_max is not None
                result = evaluate_regression_run(
                    model_entry=entry,
                    task_spec=task_spec,
                    train_manifest=train_manifest,
                    dataloader=dataloader,
                    device=device,
                    bootstrap_B=args.bootstrap_B,
                    seed=args.seed,
                    confidence=args.confidence,
                    baseline_stat=args.regression_baseline_stat,
                    age_min=age_min,
                    age_max=age_max,
                )
            all_results[entry.display_name] = result
    finally:
        dataset.close()

    final_results = {
        "task": args.task,
        "segment_length": args.segment_length,
        "backends": backends,
        "aggregation": task_spec.aggregation,
        "primary_metric": task_spec.primary_metric,
        "higher_is_better": task_spec.higher_is_better,
        "hdf5_base_dir": str(hdf5_base_dir.resolve()),
        "train_manifest": str(train_manifest_path.resolve()),
        "test_manifest": str(test_manifest_path.resolve()),
        "models": all_results,
        "bootstrap_B": args.bootstrap_B,
        "confidence": args.confidence,
        "seed": args.seed,
        "permutation_M": args.permutation_M,
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit_hash(),
    }
    if task_spec.prediction_type == "regression":
        final_results["age_min"] = age_min
        final_results["age_max"] = age_max
        final_results["regression_baseline_stat"] = args.regression_baseline_stat

    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(final_results, indent=2), encoding="utf-8")
    print(f"\nSaved results to {results_path}")

    summary_rows = []
    nhst_rows = []
    for model_name, model_results in all_results.items():
        nhst = model_results.get("nhst", {})
        for metric_name, metric_summary in model_results["bootstrap_summary"].items():
            row = {
                "model": model_name,
                "metric": metric_name,
                "point_estimate": metric_summary["point_estimate"],
                "ci_low": metric_summary["ci_low"],
                "ci_high": metric_summary["ci_high"],
                "delta_point": metric_summary["delta_point"],
                "delta_ci_low": metric_summary["delta_ci_low"],
                "delta_ci_high": metric_summary["delta_ci_high"],
                "bootstrap_prob_better": metric_summary["bootstrap_prob_better"],
                "bootstrap_prob_direction": metric_summary["bootstrap_prob_direction"],
            }
            if metric_name == task_spec.primary_metric and "p_value" in metric_summary:
                row["p_value"] = metric_summary["p_value"]
                row["p_value_test"] = metric_summary.get("p_value_test")
            summary_rows.append(row)

        if "vs_random_baseline" in nhst:
            nhst_rows.append(
                {
                    "model": model_name,
                    "test": nhst["vs_random_baseline"]["test"],
                    "primary_metric": task_spec.primary_metric,
                    "nhst_p_value": nhst["vs_random_baseline"]["p_value"],
                    "observed_statistic": nhst["vs_random_baseline"]["observed_statistic"],
                    "null_mean": nhst["vs_random_baseline"]["null_mean"],
                    "M": nhst["vs_random_baseline"]["M"],
                    "tail": nhst["vs_random_baseline"]["tail"],
                }
            )
            nhst_rows.append(
                {
                    "model": model_name,
                    "test": nhst["label_association"]["test"],
                    "primary_metric": task_spec.primary_metric,
                    "nhst_p_value": nhst["label_association"]["p_value"],
                    "observed_statistic": nhst["label_association"]["observed_statistic"],
                    "null_mean": nhst["label_association"]["null_mean"],
                    "M": nhst["label_association"]["M"],
                    "tail": nhst["label_association"]["tail"],
                }
            )
        elif "vs_median_age_baseline" in nhst:
            row = nhst["vs_median_age_baseline"]
            nhst_rows.append(
                {
                    "model": model_name,
                    "test": row["test"],
                    "primary_metric": task_spec.primary_metric,
                    "nhst_p_value": row["p_value"],
                    "observed_statistic": row["observed_statistic"],
                    "null_mean": row.get("baseline_statistic"),
                    "M": None,
                    "tail": row.get("alternative"),
                }
            )

    summary_path = out_dir / "summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    if nhst_rows:
        nhst_path = out_dir / "nhst_summary.csv"
        pd.DataFrame(nhst_rows).to_csv(nhst_path, index=False)
        print(f"Saved NHST summary to {nhst_path}")


if __name__ == "__main__":
    main()
