#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing JSON file: {path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e


def _safe_float(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _metric(d: Dict[str, Any], key: str) -> Optional[float]:
    return _safe_float(d.get(key))


def _task_metric(metrics: Dict[str, Any], task: str, key: str) -> Optional[float]:
    ttm = metrics.get("task_type_metrics")
    if not isinstance(ttm, dict):
        return None
    td = ttm.get(task)
    if not isinstance(td, dict):
        return None
    return _metric(td, key)


def _guess_task_kind(metrics: Dict[str, Any]) -> str:
    if any(k in metrics for k in ("mae", "rmse", "mse", "r2")):
        return "regression"
    if any(k in metrics for k in ("accuracy", "f1_weighted", "roc_auc")):
        return "classification"
    return "unknown"


@dataclass(frozen=True)
class FlatExperimentRow:
    source_summary_path: str
    source_dir: str
    experiment_name: str
    model_type: str
    target_type: str
    task_kind: str
    success: bool
    description: Optional[str]
    total_time_s: Optional[float]
    training_time_s: Optional[float]
    evaluation_time_s: Optional[float]
    accuracy: Optional[float]
    f1_weighted: Optional[float]
    roc_auc: Optional[float]
    active_accuracy: Optional[float]
    passive_accuracy: Optional[float]
    mae: Optional[float]
    rmse: Optional[float]
    r2: Optional[float]
    active_mae: Optional[float]
    passive_mae: Optional[float]
    active_rmse: Optional[float]
    passive_rmse: Optional[float]
    active_r2: Optional[float]
    passive_r2: Optional[float]
    checkpoint_path: Optional[str]
    error: Optional[str]
    experiment_report_html: Optional[str]
    experiment_results_csv: Optional[str]

    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "source_summary_path": self.source_summary_path,
            "source_dir": self.source_dir,
            "experiment_name": self.experiment_name,
            "model_type": self.model_type,
            "target_type": self.target_type,
            "task_kind": self.task_kind,
            "success": self.success,
            "description": self.description,
            "total_time_s": self.total_time_s,
            "training_time_s": self.training_time_s,
            "evaluation_time_s": self.evaluation_time_s,
            "accuracy": self.accuracy,
            "f1_weighted": self.f1_weighted,
            "roc_auc": self.roc_auc,
            "active_accuracy": self.active_accuracy,
            "passive_accuracy": self.passive_accuracy,
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "active_mae": self.active_mae,
            "passive_mae": self.passive_mae,
            "active_rmse": self.active_rmse,
            "passive_rmse": self.passive_rmse,
            "active_r2": self.active_r2,
            "passive_r2": self.passive_r2,
            "checkpoint_path": self.checkpoint_path,
            "error": self.error,
            "experiment_report_html": self.experiment_report_html,
            "experiment_results_csv": self.experiment_results_csv,
        }


def _discover_experiment_summaries(final_logs_dir: Path) -> List[Path]:
    return sorted(final_logs_dir.rglob("experiment_summary.json"))


def _discover_labram_jsonl_logs(final_logs_dir: Path) -> List[Path]:
    # LaBraM logs are JSONL with one JSON object per line.
    # Example: final_logs/LaBraM/labram_gender_1s/log.txt
    labram_dir = final_logs_dir / "LaBraM"
    if not labram_dir.exists():
        return []
    return sorted(labram_dir.rglob("log.txt"))


def _parse_jsonl(path: Path) -> List[Dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing log file: {path}") from e

    out: List[Dict[str, Any]] = []
    for idx, line in enumerate(lines, start=1):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON on {path}:{idx}: {e}") from e
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object on {path}:{idx}, got {type(obj).__name__}")
        out.append(obj)
    return out


def _labram_target_from_name(name: str) -> Optional[str]:
    n = name.lower()
    if "gender" in n:
        return "gender"
    if "age" in n:
        return "age"
    return None


def _labram_task_kind_from_name(name: str) -> str:
    if "regression" in name.lower():
        return "regression"
    return "classification"


def _resolve_labram_experiment_dir(log_path: Path) -> Path:
    """Resolve run directory when log.txt lives under an ``output/`` subfolder."""
    exp_dir = log_path.parent
    if exp_dir.name == "output":
        return exp_dir.parent
    return exp_dir


def _discover_labram_train_log(exp_dir: Path) -> Optional[Path]:
    """Return the newest train log for a LaBraM run (train.log, train_slurm-*.log, train_resume_*.log)."""
    candidates = list(exp_dir.glob("train*.log"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _weighted_mean(values: Iterable[Tuple[float, int]]) -> Optional[float]:
    total_n = 0
    weighted = 0.0
    for value, n in values:
        if n <= 0:
            continue
        weighted += value * n
        total_n += n
    if total_n <= 0:
        return None
    return weighted / total_n


_LABRAM_FINAL_TASKTYPE_RE = re.compile(
    r"^\s*(ACTIVE|PASSIVE)\s+tasks:\s+Accuracy\s*=\s*([0-9]+(?:\.[0-9]+)?)%\s*,.*\(\s*n\s*=\s*([0-9,]+)\s*\)\s*$",
    flags=re.IGNORECASE,
)


_LABRAM_EVAL_HEADER_RE = re.compile(r"^Evaluating on (active|passive) tasks \(([0-9,]+) samples\)\.\.\.$", flags=re.IGNORECASE)
_LABRAM_TEST_LINE_RE = re.compile(
    r"^(Active|Passive)\s+Test:\s+\[\s*\d+/\?\]\s+.*?\baccuracy:\s+[0-9.]+\s+\(([0-9.]+)\)",
    flags=re.IGNORECASE,
)
_LABRAM_TOTAL_TIME_RE = re.compile(r"^(Active|Passive)\s+Test:\s+Total time:", flags=re.IGNORECASE)

_LABRAM_MAJORITY_VOTE_RE = re.compile(
    r"participant\s*\(\s*majority\s+vote\s*\)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
    flags=re.IGNORECASE,
)


def _parse_labram_trainlog_last_block(text: str) -> Optional[Tuple[Optional[float], Optional[int], Optional[float], Optional[int]]]:
    """
    Fallback parser: take the final "Active Test"/"Passive Test" blocks and extract:
    - sample counts from "Evaluating on (active|passive) tasks (N samples)..."
    - final running-average accuracy from the last progress line before "... Total time"
    Returns (active_acc, n_active, passive_acc, n_passive), acc in [0,1].
    """
    lines = text.splitlines()

    def parse_block(kind: str) -> Tuple[Optional[float], Optional[int]]:
        # Find last header for this kind
        n_samples: Optional[int] = None
        total_time_idx: Optional[int] = None
        for i, line in enumerate(lines):
            m = _LABRAM_EVAL_HEADER_RE.match(line.strip())
            if m and m.group(1).lower() == kind:
                n_samples = int(m.group(2).replace(",", ""))
            if _LABRAM_TOTAL_TIME_RE.match(line.strip()) and line.strip().lower().startswith(kind):
                total_time_idx = i

        if total_time_idx is None:
            return None, n_samples

        # Walk backwards from total_time to find the last progress line with running avg accuracy.
        acc: Optional[float] = None
        for j in range(total_time_idx - 1, max(-1, total_time_idx - 2000), -1):
            m = _LABRAM_TEST_LINE_RE.match(lines[j].strip())
            if not m:
                continue
            if m.group(1).lower() == kind:
                acc = float(m.group(2))
                break

        return acc, n_samples

    a_acc, n_a = parse_block("active")
    p_acc, n_p = parse_block("passive")
    if a_acc is None and p_acc is None and n_a is None and n_p is None:
        return None
    return a_acc, n_a, p_acc, n_p


def _parse_labram_majority_vote(output_text: str) -> Optional[float]:
    matches = list(_LABRAM_MAJORITY_VOTE_RE.finditer(output_text))
    if not matches:
        return None
    return float(matches[-1].group(1))


def _parse_labram_trainlog_tasktype(train_log_path: Path) -> Optional[Tuple[float, int, float, int]]:
    """
    Parse the final task-type-specific results printed near the end of LaBraM train.log, e.g.
      ACTIVE tasks: Accuracy = 89.8124%, F1-Score = 0.0000 (n=35,284)
      PASSIVE tasks: Accuracy = 86.8481%, F1-Score = 0.0000 (n=204,188)
    Returns (active_acc, n_active, passive_acc, n_passive) with acc in [0,1].
    """
    text = train_log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # Find the last block to be robust to multiple evaluations.
    active: Optional[Tuple[float, int]] = None
    passive: Optional[Tuple[float, int]] = None
    for line in lines:
        m = _LABRAM_FINAL_TASKTYPE_RE.match(line)
        if not m:
            continue
        kind = m.group(1).lower()
        acc_pct = float(m.group(2))
        n = int(m.group(3).replace(",", ""))
        if kind == "active":
            active = (acc_pct / 100.0, n)
        elif kind == "passive":
            passive = (acc_pct / 100.0, n)

    if active is None or passive is None:
        # Fallback to parsing the last active/passive evaluation blocks (running-average accuracy).
        fb = _parse_labram_trainlog_last_block(text)
        if fb is None:
            return None
        a_acc, n_a, p_acc, n_p = fb
        if a_acc is None or p_acc is None or n_a is None or n_p is None:
            return None
        return a_acc, int(n_a), p_acc, int(n_p)
    return active[0], active[1], passive[0], passive[1]


def _best_epoch_by_val(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Pick the epoch with best validation accuracy; break ties by val_f1_weighted.
    best: Optional[Dict[str, Any]] = None
    best_key: Tuple[float, float] = (float("-inf"), float("-inf"))
    for r in records:
        val_acc = _safe_float(r.get("val_accuracy"))
        val_f1w = _safe_float(r.get("val_f1_weighted"))
        key = (val_acc if val_acc is not None else float("-inf"), val_f1w if val_f1w is not None else float("-inf"))
        if best is None or key > best_key:
            best = r
            best_key = key
    if best is None:
        raise ValueError("No valid records to select best epoch from.")
    return best


def _best_epoch_by_val_mae(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    best: Optional[Dict[str, Any]] = None
    best_mae = float("inf")
    for r in records:
        val_mae = _safe_float(r.get("val_mae"))
        if val_mae is None:
            continue
        if best is None or val_mae < best_mae:
            best = r
            best_mae = val_mae
    if best is None:
        raise ValueError("No valid records with val_mae to select best epoch from.")
    return best


def _best_epoch_for_records(records: List[Dict[str, Any]], task_kind: str) -> Dict[str, Any]:
    if task_kind == "regression":
        return _best_epoch_by_val_mae(records)
    return _best_epoch_by_val(records)


def _extract_labram_task_type_metrics(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    # Some epochs include a top-level "task_type_metrics" with active/passive breakdown.
    # Prefer the record chosen for reporting (caller can pass [best] list), otherwise
    # use the last occurrence to avoid silently dropping available information.
    for r in reversed(records):
        ttm = r.get("task_type_metrics")
        if isinstance(ttm, dict):
            return ttm
    return None


def _ttm_counts(ttm: Dict[str, Any]) -> Optional[Tuple[Optional[int], Optional[int]]]:
    def _num_samples(d: Any) -> Optional[int]:
        if not isinstance(d, dict):
            return None
        ns = d.get("num_samples")
        if isinstance(ns, int):
            return ns
        if isinstance(ns, float) and float(ns).is_integer():
            return int(ns)
        return None

    a = _num_samples(ttm.get("active"))
    p = _num_samples(ttm.get("passive"))
    if a is None and p is None:
        return None
    return a, p


def _split_counts(rec: Dict[str, Any], key: str) -> Optional[Tuple[Optional[int], Optional[int]]]:
    v = rec.get(key)
    if not isinstance(v, dict):
        return None
    a = v.get("active")
    p = v.get("passive")
    def _as_int(x: Any) -> Optional[int]:
        if isinstance(x, int):
            return x
        if isinstance(x, float) and float(x).is_integer():
            return int(x)
        return None
    ai = _as_int(a)
    pi = _as_int(p)
    if ai is None and pi is None:
        return None
    return ai, pi


def _flatten_labram_log(log_path: Path, base_dir: Path) -> Tuple[Optional[FlatExperimentRow], List[str]]:
    warnings: List[str] = []
    exp_dir = _resolve_labram_experiment_dir(log_path)
    exp_name = exp_dir.name  # e.g. labram_gender_frozen_1s
    target_type = _labram_target_from_name(exp_name)
    if target_type is None:
        warnings.append(f"{_relpath(log_path, base_dir)}: could not infer target_type from directory name {exp_name!r}")
        return None, warnings

    records = _parse_jsonl(log_path)
    if not records:
        warnings.append(f"{_relpath(log_path, base_dir)}: no JSON records found")
        return None, warnings

    task_kind = _labram_task_kind_from_name(exp_name)
    if task_kind == "regression" and not any(_safe_float(r.get("val_mae")) is not None for r in records):
        task_kind = "classification"

    best = _best_epoch_for_records(records, task_kind)
    ttm = best.get("task_type_metrics")
    if not isinstance(ttm, dict):
        ttm = _extract_labram_task_type_metrics(records)

    frozen = bool(best.get("freeze_backbone"))
    best_metric = "val_mae" if task_kind == "regression" else "val_accuracy"
    description = (
        f"LaBraM {'frozen ' if frozen else ''}fine-tuning ({exp_name}) — best epoch by {best_metric}"
    )

    # Classification metrics from JSONL (segment-level test at best val epoch).
    roc_auc_jsonl = _safe_float(best.get("test_roc_auc"))
    accuracy_jsonl = _safe_float(best.get("test_accuracy"))
    f1w_jsonl = _safe_float(best.get("test_f1_weighted"))

    mae_jsonl = _safe_float(best.get("test_mae"))
    rmse_jsonl = _safe_float(best.get("test_rmse"))
    r2_jsonl = _safe_float(best.get("test_r2"))

    active_acc: Optional[float] = _task_metric({"task_type_metrics": ttm}, "active", "accuracy") if isinstance(ttm, dict) else None
    passive_acc: Optional[float] = _task_metric({"task_type_metrics": ttm}, "passive", "accuracy") if isinstance(ttm, dict) else None
    active_mae: Optional[float] = _task_metric({"task_type_metrics": ttm}, "active", "mae") if isinstance(ttm, dict) else None
    passive_mae: Optional[float] = _task_metric({"task_type_metrics": ttm}, "passive", "mae") if isinstance(ttm, dict) else None
    active_rmse: Optional[float] = _task_metric({"task_type_metrics": ttm}, "active", "rmse") if isinstance(ttm, dict) else None
    passive_rmse: Optional[float] = _task_metric({"task_type_metrics": ttm}, "passive", "rmse") if isinstance(ttm, dict) else None
    active_r2: Optional[float] = _task_metric({"task_type_metrics": ttm}, "active", "r2") if isinstance(ttm, dict) else None
    passive_r2: Optional[float] = _task_metric({"task_type_metrics": ttm}, "passive", "r2") if isinstance(ttm, dict) else None

    train_log = _discover_labram_train_log(exp_dir)
    if train_log is not None and task_kind == "classification":
        parsed = _parse_labram_trainlog_tasktype(train_log)
        if parsed is None:
            warnings.append(
                f"{_relpath(train_log, base_dir)}: could not find final ACTIVE/PASSIVE accuracy lines"
            )
        else:
            a_acc, n_a, p_acc, n_p = parsed
            active_acc, passive_acc = a_acc, p_acc
            overall_from_counts = _weighted_mean([(a_acc, n_a), (p_acc, n_p)])
            if overall_from_counts is not None:
                accuracy_jsonl = overall_from_counts
                f1w_jsonl = None
                roc_auc_jsonl = None
            else:
                warnings.append(
                    f"{_relpath(train_log, base_dir)}: ACTIVE/PASSIVE lines have n=0 sample counts; "
                    "using JSONL test_accuracy for overall and parsed active/passive accuracies"
                )
    elif train_log is None and task_kind == "classification":
        warnings.append(
            f"{_relpath(log_path, base_dir)}: no train*.log under {_relpath(exp_dir, base_dir)}; "
            "using JSONL test metrics and task_type_metrics when present"
        )

    if task_kind == "regression":
        row = FlatExperimentRow(
            source_summary_path=_relpath(log_path, base_dir),
            source_dir=_relpath(exp_dir, base_dir),
            experiment_name=exp_name,
            model_type="labram",
            target_type=target_type,
            task_kind="regression",
            success=True,
            description=description,
            total_time_s=None,
            training_time_s=None,
            evaluation_time_s=None,
            accuracy=None,
            f1_weighted=None,
            roc_auc=None,
            active_accuracy=None,
            passive_accuracy=None,
            mae=mae_jsonl,
            rmse=rmse_jsonl,
            r2=r2_jsonl,
            active_mae=active_mae,
            passive_mae=passive_mae,
            active_rmse=active_rmse,
            passive_rmse=passive_rmse,
            active_r2=active_r2,
            passive_r2=passive_r2,
            checkpoint_path=None,
            error=None,
            experiment_report_html=None,
            experiment_results_csv=None,
        )
        return row, warnings

    row = FlatExperimentRow(
        source_summary_path=_relpath(log_path, base_dir),
        source_dir=_relpath(exp_dir, base_dir),
        experiment_name=exp_name,
        model_type="labram",
        target_type=target_type,
        task_kind="classification",
        success=True,
        description=description,
        total_time_s=None,
        training_time_s=None,
        evaluation_time_s=None,
        accuracy=accuracy_jsonl,
        f1_weighted=f1w_jsonl,
        roc_auc=roc_auc_jsonl,
        active_accuracy=active_acc,
        passive_accuracy=passive_acc,
        mae=None,
        rmse=None,
        r2=None,
        active_mae=None,
        passive_mae=None,
        active_rmse=None,
        passive_rmse=None,
        active_r2=None,
        passive_r2=None,
        checkpoint_path=None,
        error=None,
        experiment_report_html=None,
        experiment_results_csv=None,
    )
    return row, warnings


def _flatten_labram_output_log(output_log: Path, base_dir: Path) -> Tuple[Optional[FlatExperimentRow], List[str]]:
    """LaBraM eval-only runs with output.log (e.g. majority vote) and no JSONL log.txt."""
    warnings: List[str] = []
    exp_dir = output_log.parent
    if exp_dir.name == "output":
        exp_dir = exp_dir.parent
    exp_name = exp_dir.name
    target_type = _labram_target_from_name(exp_name)
    if target_type is None:
        warnings.append(f"{_relpath(output_log, base_dir)}: could not infer target_type from {exp_name!r}")
        return None, warnings

    text = output_log.read_text(encoding="utf-8", errors="replace")
    maj = _parse_labram_majority_vote(text)
    if maj is None:
        warnings.append(
            f"{_relpath(output_log, base_dir)}: no 'participant (majority vote):' line found; "
            "skipping LaBraM output-only row"
        )
        return None, warnings

    train_log = _discover_labram_train_log(exp_dir)
    active_acc: Optional[float] = None
    passive_acc: Optional[float] = None
    overall_acc: Optional[float] = maj
    if train_log is not None:
        parsed = _parse_labram_trainlog_tasktype(train_log)
        if parsed is not None:
            a_acc, n_a, p_acc, n_p = parsed
            active_acc, passive_acc = a_acc, p_acc
            overall_from_counts = _weighted_mean([(a_acc, n_a), (p_acc, n_p)])
            if overall_from_counts is not None:
                overall_acc = overall_from_counts

    row = FlatExperimentRow(
        source_summary_path=_relpath(output_log, base_dir),
        source_dir=_relpath(exp_dir, base_dir),
        experiment_name=exp_name,
        model_type="labram",
        target_type=target_type,
        task_kind="classification",
        success=True,
        description=f"LaBraM ({exp_name}) — participant majority vote from output.log",
        total_time_s=None,
        training_time_s=None,
        evaluation_time_s=None,
        accuracy=overall_acc,
        f1_weighted=None,
        roc_auc=None,
        active_accuracy=active_acc,
        passive_accuracy=passive_acc,
        mae=None,
        rmse=None,
        r2=None,
        active_mae=None,
        passive_mae=None,
        active_rmse=None,
        passive_rmse=None,
        active_r2=None,
        passive_r2=None,
        checkpoint_path=None,
        error=None,
        experiment_report_html=None,
        experiment_results_csv=None,
    )
    return row, warnings


def _relpath(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _summary_sidecar(path: Path, filename: str) -> Optional[Path]:
    for rel in (filename, str(Path("reports") / filename)):
        candidate = path.parent / rel
        if candidate.exists():
            return candidate
    return None


def _flatten_summary(summary_path: Path, base_dir: Path) -> Tuple[List[FlatExperimentRow], List[str]]:
    warnings: List[str] = []
    payload = _read_json(summary_path)
    experiments = payload.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError(f"Expected 'experiments' list in {summary_path}")

    sidecar_html = _summary_sidecar(summary_path, "experiment_report.html")
    sidecar_csv = _summary_sidecar(summary_path, "experiment_results.csv")

    rows: List[FlatExperimentRow] = []
    for i, exp in enumerate(experiments):
        if not isinstance(exp, dict):
            warnings.append(f"{_relpath(summary_path, base_dir)}: experiments[{i}] is not an object")
            continue

        exp_name = exp.get("experiment_name")
        model_type = exp.get("model_type")
        target_type = exp.get("target_type")
        if not all(isinstance(x, str) for x in (exp_name, model_type, target_type)):
            warnings.append(
                f"{_relpath(summary_path, base_dir)}: experiments[{i}] missing required fields "
                f"(experiment_name/model_type/target_type)"
            )
            continue

        success = exp.get("success")
        if not isinstance(success, bool):
            warnings.append(f"{_relpath(summary_path, base_dir)}: {exp_name} has non-bool success={success!r}")
            success = False

        metrics = exp.get("metrics")
        if metrics is None:
            metrics = {}
        if not isinstance(metrics, dict):
            warnings.append(f"{_relpath(summary_path, base_dir)}: {exp_name} has non-object metrics")
            metrics = {}

        task_kind = _guess_task_kind(metrics)
        row = FlatExperimentRow(
            source_summary_path=_relpath(summary_path, base_dir),
            source_dir=_relpath(summary_path.parent, base_dir),
            experiment_name=exp_name,
            model_type=model_type,
            target_type=target_type,
            task_kind=task_kind,
            success=success,
            description=exp.get("description") if isinstance(exp.get("description"), str) else None,
            total_time_s=_safe_float(exp.get("total_time")),
            training_time_s=_safe_float(exp.get("training_time")),
            evaluation_time_s=_safe_float(exp.get("evaluation_time")),
            accuracy=_metric(metrics, "accuracy"),
            f1_weighted=_metric(metrics, "f1_weighted"),
            roc_auc=_metric(metrics, "roc_auc"),
            active_accuracy=_task_metric(metrics, "active", "accuracy"),
            passive_accuracy=_task_metric(metrics, "passive", "accuracy"),
            mae=_metric(metrics, "mae"),
            rmse=_metric(metrics, "rmse"),
            r2=_metric(metrics, "r2"),
            active_mae=_task_metric(metrics, "active", "mae"),
            passive_mae=_task_metric(metrics, "passive", "mae"),
            active_rmse=_task_metric(metrics, "active", "rmse"),
            passive_rmse=_task_metric(metrics, "passive", "rmse"),
            active_r2=_task_metric(metrics, "active", "r2"),
            passive_r2=_task_metric(metrics, "passive", "r2"),
            checkpoint_path=exp.get("model_checkpoint_path") if isinstance(exp.get("model_checkpoint_path"), str) else None,
            error=exp.get("error") if isinstance(exp.get("error"), str) else None,
            experiment_report_html=_relpath(sidecar_html, base_dir) if sidecar_html else None,
            experiment_results_csv=_relpath(sidecar_csv, base_dir) if sidecar_csv else None,
        )
        rows.append(row)

    return rows, warnings


def _flatten_experiment_results_csv(csv_path: Path, base_dir: Path) -> Tuple[List[FlatExperimentRow], List[str]]:
    """Parse one or more experiment rows from a tabular experiment_results.csv (e.g. under reports/)."""
    warnings: List[str] = []
    exp_dir = csv_path.parent.parent
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)
    if not raw_rows:
        warnings.append(f"{_relpath(csv_path, base_dir)}: empty CSV")
        return [], warnings

    html_sidecar = exp_dir / "reports" / "experiment_report.html"
    experiment_report_html = _relpath(html_sidecar, base_dir) if html_sidecar.exists() else None

    out: List[FlatExperimentRow] = []
    for d in raw_rows:
        exp_name = d.get("experiment_name")
        model_type = d.get("model_type")
        target_type = d.get("target_type")
        if not all(isinstance(x, str) and x for x in (exp_name, model_type, target_type)):
            warnings.append(f"{_relpath(csv_path, base_dir)}: row missing experiment_name/model_type/target_type")
            continue

        metrics: Dict[str, Any] = {}
        acc = _safe_float(d.get("combined_accuracy"))
        if acc is not None:
            metrics["accuracy"] = acc
        mae = _safe_float(d.get("mae"))
        if mae is not None:
            metrics["mae"] = mae
        rmse = _safe_float(d.get("rmse"))
        if rmse is not None:
            metrics["rmse"] = rmse
        r2 = _safe_float(d.get("r2"))
        if r2 is not None:
            metrics["r2"] = r2
        f1 = _safe_float(d.get("f1_weighted"))
        if f1 is not None:
            metrics["f1_weighted"] = f1
        roc = _safe_float(d.get("roc_auc"))
        if roc is not None:
            metrics["roc_auc"] = roc

        active_acc = _safe_float(d.get("active_accuracy"))
        passive_acc = _safe_float(d.get("passive_accuracy"))
        if active_acc is not None or passive_acc is not None:
            ttm: Dict[str, Any] = {}
            if active_acc is not None:
                ttm["active"] = {"accuracy": active_acc}
            if passive_acc is not None:
                ttm["passive"] = {"accuracy": passive_acc}
            metrics["task_type_metrics"] = ttm

        task_kind = _guess_task_kind(metrics)
        row = FlatExperimentRow(
            source_summary_path=_relpath(csv_path, base_dir),
            source_dir=_relpath(exp_dir, base_dir),
            experiment_name=exp_name,
            model_type=model_type,
            target_type=target_type,
            task_kind=task_kind,
            success=True,
            description=None,
            total_time_s=_safe_float(d.get("training_time")),
            training_time_s=_safe_float(d.get("training_time")),
            evaluation_time_s=_safe_float(d.get("evaluation_time")),
            accuracy=_metric(metrics, "accuracy"),
            f1_weighted=_metric(metrics, "f1_weighted"),
            roc_auc=_metric(metrics, "roc_auc"),
            active_accuracy=_task_metric(metrics, "active", "accuracy"),
            passive_accuracy=_task_metric(metrics, "passive", "accuracy"),
            mae=_metric(metrics, "mae"),
            rmse=_metric(metrics, "rmse"),
            r2=_metric(metrics, "r2"),
            active_mae=_task_metric(metrics, "active", "mae"),
            passive_mae=_task_metric(metrics, "passive", "mae"),
            active_rmse=_task_metric(metrics, "active", "rmse"),
            passive_rmse=_task_metric(metrics, "passive", "rmse"),
            active_r2=_task_metric(metrics, "active", "r2"),
            passive_r2=_task_metric(metrics, "passive", "r2"),
            checkpoint_path=None,
            error=None,
            experiment_report_html=experiment_report_html,
            experiment_results_csv=_relpath(csv_path, base_dir),
        )
        out.append(row)
    return out, warnings


def _discover_labram_output_logs(final_logs_dir: Path) -> List[Path]:
    """LaBraM eval dirs named labram_* at any depth (e.g. final_logs/LaBraM/labram_gender_1s/)."""
    paths: List[Path] = []
    for p in sorted(final_logs_dir.rglob("labram_*")):
        if not p.is_dir():
            continue
        ol, lj = p / "output.log", p / "log.txt"
        if ol.exists() and not lj.exists():
            paths.append(ol)
    return paths


def _has_any_sources(final_logs_dir: Path) -> bool:
    if list(final_logs_dir.rglob("experiment_summary.json")):
        return True
    if _discover_labram_jsonl_logs(final_logs_dir):
        return True
    if _discover_labram_output_logs(final_logs_dir):
        return True
    if list(final_logs_dir.rglob("reports/experiment_results.csv")):
        return True
    return False


def _majority_expected_experiment_names() -> Set[str]:
    """Full grid for `final_logs_majority/`: 9 CNN + 18 ResNet + 6 LaBraM (33 unique names)."""
    names: Set[str] = set()
    for w in ("1s", "2s", "4s"):
        names.add(f"gender_baseline_{w}")
        names.add(f"age_classification_{w}")
        names.add(f"age_regression_{w}")
    for prefix in ("age_resnet18", "age_resnet34", "age_resnet50"):
        for w in ("1s", "2s", "4s"):
            names.add(f"{prefix}_{w}")
    for prefix in ("gender_resnet18", "gender_resnet34", "gender_resnet50"):
        for w in ("1s", "2s", "4s"):
            names.add(f"{prefix}_{w}")
    for task in ("age", "gender"):
        for w in ("1s", "2s", "4s"):
            names.add(f"labram_{task}_{w}")
    return names


def _write_csv(path: Path, rows: List[FlatExperimentRow]) -> None:
    fieldnames = list(rows[0].to_csv_row().keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r.to_csv_row())


def _fmt(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "_No rows._\n"
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out) + "\n"


def _top_k_classification(
    rows: List[FlatExperimentRow], k: Optional[int] = None, target: Optional[str] = None
) -> List[FlatExperimentRow]:
    xs = [r for r in rows if r.success and r.task_kind == "classification" and r.accuracy is not None]
    if target is not None:
        xs = [r for r in xs if r.target_type == target]
    sorted_xs = sorted(xs, key=lambda r: (r.accuracy or float("-inf")), reverse=True)
    return sorted_xs if k is None else sorted_xs[:k]


def _top_k_regression(
    rows: List[FlatExperimentRow], k: Optional[int] = None, target: Optional[str] = None
) -> List[FlatExperimentRow]:
    xs = [r for r in rows if r.success and r.task_kind == "regression" and r.mae is not None]
    if target is not None:
        xs = [r for r in xs if r.target_type == target]
    sorted_xs = sorted(xs, key=lambda r: (r.mae or float("inf")))
    return sorted_xs if k is None else sorted_xs[:k]


_WINDOW_SUFFIX_RE = re.compile(r"_(1s|2s|4s)$")


def _window_suffix(experiment_name: str) -> Optional[str]:
    m = _WINDOW_SUFFIX_RE.search(experiment_name)
    return m.group(1) if m else None


def _classification_accuracy_matrix(
    rows: List[FlatExperimentRow],
    target: str,
    windows: Tuple[str, ...] = ("1s", "2s", "4s"),
) -> List[List[str]]:
    """One row per model_type with accuracy in each window column (same evaluation target)."""
    xs = [
        r
        for r in rows
        if r.success
        and r.task_kind == "classification"
        and r.target_type == target
        and r.accuracy is not None
    ]
    by_key: Dict[Tuple[str, str], FlatExperimentRow] = {}
    for r in xs:
        w = _window_suffix(r.experiment_name)
        if w is None or w not in windows:
            continue
        by_key[(r.model_type, w)] = r

    model_types = sorted({k[0] for k in by_key.keys()})
    out: List[List[str]] = []
    for mt in model_types:
        row = [f"`{mt}`"]
        for w in windows:
            r = by_key.get((mt, w))
            if r is None:
                row.append("—")
            else:
                row.append(f"{_fmt(r.accuracy)} ({r.experiment_name})")
        out.append(row)
    return out


def _write_markdown(
    path: Path,
    base_dir: Path,
    final_logs_dir: Path,
    rows: List[FlatExperimentRow],
    warnings: List[str],
    leaderboard_limit: Optional[int] = None,
) -> None:
    total = len(rows)
    ok = sum(1 for r in rows if r.success)
    failed = total - ok
    targets = sorted({r.target_type for r in rows})
    models = sorted({r.model_type for r in rows})
    kinds = sorted({r.task_kind for r in rows})

    best_gender = _top_k_classification(rows, k=leaderboard_limit, target="gender")
    best_age_cls = _top_k_classification(rows, k=leaderboard_limit, target="age")
    best_reg = _top_k_regression(rows, k=leaderboard_limit)
    lb_note = (
        "all experiments, sorted by accuracy (descending)"
        if leaderboard_limit is None
        else f"top {leaderboard_limit} by accuracy"
    )
    lb_note_reg = (
        "all experiments, sorted by MAE (ascending)"
        if leaderboard_limit is None
        else f"top {leaderboard_limit} by MAE (lower is better)"
    )

    def link(p: Optional[str]) -> str:
        if not p:
            return "—"
        return f"[link]({_relpath((base_dir / p).resolve(), base_dir)})"

    md: List[str] = []
    md.append("# Final logs report\n")
    md.append(f"- **Generated**: {date.today().isoformat()}\n")
    md.append(f"- **Source**: `{_relpath(final_logs_dir, base_dir)}`\n")
    md.append(f"- **Experiments**: total={total}, success={ok}, failed={failed}\n")
    n_labram = sum(1 for r in rows if r.model_type == "labram")
    n_from_summaries = sum(1 for r in rows if r.source_summary_path.endswith("experiment_summary.json"))
    n_supplemental_csv_rows = sum(
        1 for r in rows if "/reports/experiment_results.csv" in r.source_summary_path.replace("\\", "/")
    )
    n_summary_files = len({r.source_summary_path for r in rows if r.source_summary_path.endswith("experiment_summary.json")})
    n_labram_txt = len({r.source_summary_path for r in rows if r.source_summary_path.endswith("log.txt")})
    n_labram_out = len({r.source_summary_path for r in rows if r.source_summary_path.endswith("output.log")})
    md.append(
        f"- **Coverage**: {n_summary_files} `experiment_summary.json` → {n_from_summaries} rows; "
        f"`reports/experiment_results.csv` → {n_supplemental_csv_rows} rows; "
        f"LaBraM `log.txt` {n_labram_txt}, `output.log` {n_labram_out} (total LaBraM rows: {n_labram})\n"
    )
    md.append(f"- **Target types**: {', '.join(f'`{t}`' for t in targets) if targets else '—'}\n")
    md.append(f"- **Model types**: {', '.join(f'`{m}`' for m in models) if models else '—'}\n")
    md.append(f"- **Task kinds**: {', '.join(f'`{k}`' for k in kinds) if kinds else '—'}\n")
    md.append("\n")

    md.append("## Leaderboards\n\n")
    md.append(f"### Gender (classification) — {lb_note}\n\n")
    md.append(
        _md_table(
            ["experiment", "model", "acc", "f1_w", "auc", "active_acc", "passive_acc", "report"],
            [
                [
                    f"`{r.experiment_name}`",
                    f"`{r.model_type}`",
                    _fmt(r.accuracy),
                    _fmt(r.f1_weighted),
                    _fmt(r.roc_auc),
                    _fmt(r.active_accuracy),
                    _fmt(r.passive_accuracy),
                    link(r.experiment_report_html),
                ]
                for r in best_gender
            ],
        )
    )

    md.append("\n#### Gender — accuracy by window (1s / 2s / 4s)\n\n")
    md.append(
        _md_table(
            ["model", "1s", "2s", "4s"],
            _classification_accuracy_matrix(rows, "gender"),
        )
    )

    md.append(f"\n### Age (classification) — {lb_note}\n\n")
    md.append(
        _md_table(
            ["experiment", "model", "acc", "f1_w", "auc", "active_acc", "passive_acc", "report"],
            [
                [
                    f"`{r.experiment_name}`",
                    f"`{r.model_type}`",
                    _fmt(r.accuracy),
                    _fmt(r.f1_weighted),
                    _fmt(r.roc_auc),
                    _fmt(r.active_accuracy),
                    _fmt(r.passive_accuracy),
                    link(r.experiment_report_html),
                ]
                for r in best_age_cls
            ],
        )
    )

    md.append("\n#### Age — accuracy by window (1s / 2s / 4s)\n\n")
    md.append(
        _md_table(
            ["model", "1s", "2s", "4s"],
            _classification_accuracy_matrix(rows, "age"),
        )
    )

    md.append(f"\n### Regression — {lb_note_reg}\n\n")
    md.append(
        _md_table(
            ["experiment", "model", "mae", "rmse", "r2", "active_mae", "passive_mae", "report"],
            [
                [
                    f"`{r.experiment_name}`",
                    f"`{r.model_type}`",
                    _fmt(r.mae, 4),
                    _fmt(r.rmse, 4),
                    _fmt(r.r2, 4),
                    _fmt(r.active_mae, 4),
                    _fmt(r.passive_mae, 4),
                    link(r.experiment_report_html),
                ]
                for r in best_reg
            ],
        )
    )

    md.append("\n## Full index\n\n")
    md.append(
        _md_table(
            ["experiment", "target", "kind", "success", "primary", "active", "passive", "source"],
            [
                [
                    f"`{r.experiment_name}`",
                    f"`{r.target_type}`",
                    f"`{r.task_kind}`",
                    "✅" if r.success else "❌",
                    _fmt(r.accuracy) if r.task_kind == "classification" else _fmt(r.mae, 4),
                    _fmt(r.active_accuracy) if r.task_kind == "classification" else _fmt(r.active_mae, 4),
                    _fmt(r.passive_accuracy) if r.task_kind == "classification" else _fmt(r.passive_mae, 4),
                    f"`{r.source_summary_path}`",
                ]
                for r in sorted(rows, key=lambda x: (x.target_type, x.model_type, x.experiment_name))
            ],
        )
    )

    if warnings:
        md.append("\n## Parsing warnings\n\n")
        md.extend([f"- {w}\n" for w in warnings])

    if final_logs_dir.resolve().name == "final_logs_majority":
        expected = _majority_expected_experiment_names()
        present = {r.experiment_name for r in rows}
        missing = sorted(expected - present)
        md.append("\n## Missing experiments (expected full grid)\n\n")
        md.append(
            "This tree is expected to contain **33** distinct experiment names: **9** CNN runs "
            "(`gender_baseline_*`, `age_classification_*`, `age_regression_*` at 1s/2s/4s), **18** ResNet runs "
            "(three backbones × three window lengths × age and gender), **6** LaBraM runs (`labram_age_*`, "
            "`labram_gender_*`). The count is **33** (not 34) because `age_resnet18_1s` is listed once "
            "(root `experiment_summary.json` and `RESNET/age_resnet18_1s/` are the same experiment).\n\n"
        )
        md.append(f"- **Present in this report**: {len(present)} rows  \n")
        md.append(f"- **Expected names**: {len(expected)}  \n")
        if missing:
            md.append(
                "\nThe following expected names have **no** ingested row (no `experiment_summary.json` "
                "entry, no row in `reports/experiment_results.csv`, and no LaBraM `log.txt` / `output.log`):\n\n"
            )
            for m in missing:
                md.append(f"- `{m}`\n")
        else:
            md.append("\nAll **33** expected experiment names are present.\n")

    path.write_text("".join(md), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate final_logs/* experiment_summary.json into one report.")
    parser.add_argument(
        "--final-logs-dir",
        type=Path,
        default=(Path(__file__).resolve().parents[1] / "final_logs"),
        help="Path to final_logs directory (default: repo_root/final_logs)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(Path(__file__).resolve().parent / f"final_logs_report_{date.today().isoformat()}"),
        help="Output directory for the generated report",
    )
    parser.add_argument(
        "--leaderboard-limit",
        type=int,
        default=None,
        metavar="N",
        help="Max rows per leaderboard section (default: include every experiment)",
    )
    args = parser.parse_args(argv)

    final_logs_dir: Path = args.final_logs_dir
    out_dir: Path = args.out_dir

    if not final_logs_dir.exists():
        raise FileNotFoundError(f"--final-logs-dir does not exist: {final_logs_dir}")
    if not final_logs_dir.is_dir():
        raise NotADirectoryError(f"--final-logs-dir is not a directory: {final_logs_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(__file__).resolve().parents[1]

    if not _has_any_sources(final_logs_dir):
        raise FileNotFoundError(
            f"No ingestible sources under {final_logs_dir} "
            "(expect experiment_summary.json, LaBraM log.txt, labram_*/output.log, "
            "or reports/experiment_results.csv)"
        )

    summaries = _discover_experiment_summaries(final_logs_dir)
    labram_logs = _discover_labram_jsonl_logs(final_logs_dir)
    labram_outputs = _discover_labram_output_logs(final_logs_dir)
    supplemental_csvs = sorted(final_logs_dir.rglob("reports/experiment_results.csv"))

    all_rows: List[FlatExperimentRow] = []
    warnings: List[str] = []
    for s in summaries:
        rows, ws = _flatten_summary(s, base_dir=base_dir)
        all_rows.extend(rows)
        warnings.extend(ws)

    seen_names: Set[str] = {r.experiment_name for r in all_rows}
    for csv_path in supplemental_csvs:
        rows, ws = _flatten_experiment_results_csv(csv_path, base_dir=base_dir)
        warnings.extend(ws)
        for r in rows:
            if r.experiment_name not in seen_names:
                all_rows.append(r)
                seen_names.add(r.experiment_name)

    for lp in labram_logs:
        r, ws = _flatten_labram_log(lp, base_dir=base_dir)
        warnings.extend(ws)
        if r is not None:
            all_rows.append(r)
            seen_names.add(r.experiment_name)

    for ol in labram_outputs:
        if ol.parent.name in seen_names:
            continue
        r, ws = _flatten_labram_output_log(ol, base_dir=base_dir)
        warnings.extend(ws)
        if r is not None:
            all_rows.append(r)
            seen_names.add(r.experiment_name)

    # Deterministic order in outputs
    all_rows = sorted(all_rows, key=lambda r: (r.target_type, r.task_kind, r.model_type, r.experiment_name, r.source_summary_path))

    # Write machine-friendly outputs
    (_write_csv(out_dir / "final_logs_experiments.csv", all_rows))
    (out_dir / "final_logs_experiments.json").write_text(
        json.dumps([r.to_csv_row() for r in all_rows], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Write human-friendly report
    _write_markdown(
        out_dir / "final_logs_report.md",
        base_dir=base_dir,
        final_logs_dir=final_logs_dir,
        rows=all_rows,
        warnings=warnings,
        leaderboard_limit=args.leaderboard_limit,
    )

    # Print paths for convenience in terminal logs
    print(f"Wrote: {out_dir / 'final_logs_report.md'}")
    print(f"Wrote: {out_dir / 'final_logs_experiments.csv'}")
    print(f"Wrote: {out_dir / 'final_logs_experiments.json'}")
    if warnings:
        print(f"Warnings: {len(warnings)} (see report section 'Parsing warnings')", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

