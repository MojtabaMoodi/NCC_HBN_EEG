#!/usr/bin/env python3
"""Generate inference-time, parameter-count, and hardware report from final_logs."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEVICE_RE = re.compile(
    r"Using device:\s*cuda\s*\((\d+)\s+GPU\(s\):\s*\[(.*?)\]\)",
    re.IGNORECASE,
)
GPU_CONFIG_RE = re.compile(r"GPU\s+(\d+):\s*(NVIDIA\s+\S+)", re.IGNORECASE)
PARAMS_RE = re.compile(r"number of (?:trainable )?params:\s*([\d,]+)", re.IGNORECASE)
SEC_PER_IT_RE = re.compile(
    r"^(Test|Val|Active Test|Passive Test):\s+Total time:.*? \(([0-9.]+)\s+s / it\)",
    re.IGNORECASE,
)
EVAL_TIME_RE = re.compile(r"Evaluation time:\s*([\d.]+)s")
EVAL_BATCH_RE = re.compile(r"batch_size[=:]\s*(\d+)", re.IGNORECASE)


def _parse_eval_times(log_text: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (pooled_test_eval_s, total_eval_s) from evaluation blocks in a log section."""
    times = [float(m.group(1)) for m in EVAL_TIME_RE.finditer(log_text)]
    if not times:
        return None, None
    if len(times) == 1:
        return times[0], times[0]
    return times[-2], times[-1]


def _ms_per_sample(seconds: Optional[float], num_samples: Optional[int]) -> Optional[float]:
    if seconds is None or not num_samples:
        return None
    return seconds / num_samples * 1000.0


def _split_task_eval_ms(
    pooled_s: Optional[float],
    total_s: Optional[float],
    active_n: Optional[int],
    passive_n: Optional[int],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Derive ms/sample for pooled, active-only, and passive-only eval phases."""
    total_n = (active_n + passive_n) if isinstance(active_n, int) and isinstance(passive_n, int) else None
    ms_total = _ms_per_sample(pooled_s, total_n)
    if pooled_s is None or total_s is None or total_n is None or total_s <= pooled_s:
        ms_active = _ms_per_sample(total_s, active_n) if total_s and active_n else None
        ms_passive = _ms_per_sample(total_s, passive_n) if total_s and passive_n else None
        return ms_total, ms_active, ms_passive
    split_s = total_s - pooled_s
    ms_active = _ms_per_sample(split_s * active_n / total_n, active_n)
    ms_passive = _ms_per_sample(split_s * passive_n / total_n, passive_n)
    return ms_total, ms_active, ms_passive


def _log_section_for_experiment(log_text: str, exp_name: str) -> str:
    """Return the log section for one experiment inside a multi-experiment training log."""
    start = log_text.find(f"Experiment: {exp_name}")
    if start == -1:
        start = log_text.find(f"STARTING EXPERIMENT: {exp_name}")
    if start == -1:
        return log_text
    next_exp = re.search(
        r"\nProgress: \d+/\d+\nExperiment: ",
        log_text[start + 1 :],
    )
    end = start + 1 + next_exp.start() if next_exp else len(log_text)
    return log_text[start:end]


@dataclass
class ReportRow:
    task: str
    segment: str
    model_family: str
    model_variant: str
    total_parameters: Optional[int]
    parameters_source: str
    gpu_model: Optional[str]
    num_gpus: Optional[int]
    hardware_source: str
    inference_s_per_batch_test: Optional[float]
    inference_s_per_batch_active: Optional[float]
    inference_s_per_batch_passive: Optional[float]
    inference_ms_per_sample_test: Optional[float]
    inference_ms_per_sample_active: Optional[float]
    inference_ms_per_sample_passive: Optional[float]
    eval_batch_size: Optional[int]
    eval_total_time_s: Optional[float]
    test_num_samples: Optional[int]
    inference_source: str
    log_source: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _last_matches(pattern: re.Pattern[str], text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in text.splitlines():
        m = pattern.match(line.strip())
        if m:
            out[m.group(1).lower()] = m.group(2)
    return out


def _parse_labram_log(log_path: Path) -> Tuple[Optional[int], Optional[str], Optional[int], Dict[str, float]]:
    text = _read_text(log_path)
    params = None
    m = PARAMS_RE.search(text)
    if m:
        params = int(m.group(1).replace(",", ""))

    gpu_model = None
    num_gpus = None
    dm = DEVICE_RE.search(text)
    if dm:
        num_gpus = int(dm.group(1))
        names = [n.strip().strip("'") for n in dm.group(2).split(",") if n.strip()]
        if names:
            gpu_model = names[0]
    else:
        gpus = GPU_CONFIG_RE.findall(text)
        if gpus:
            num_gpus = len({g[0] for g in gpus})
            gpu_model = gpus[0][1]

    timings: Dict[str, float] = {}
    for line in text.splitlines():
        m = SEC_PER_IT_RE.match(line.strip())
        if not m:
            continue
        label = m.group(1).lower().replace(" ", "_")
        timings[label] = float(m.group(2))
    return params, gpu_model, num_gpus, timings


def _canonical_labram_log(run_dir: Path) -> Optional[Path]:
    """Pick the newest log that contains parameter count (prefer resume/eval logs)."""
    candidates = sorted(run_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        if PARAMS_RE.search(_read_text(path)):
            return path
    return candidates[0] if candidates else None


def _labram_rows(final_logs: Path) -> List[ReportRow]:
    specs = [
        ("age_classification", "age", "1s", "labram_age_1s"),
        ("age_classification", "age", "2s", "labram_age_2s"),
        ("age_classification", "age", "4s", "labram_age_4s"),
        ("gender_classification", "gender", "1s", "labram_gender_1s"),
        ("gender_classification", "gender", "2s", "labram_gender_2s"),
        ("gender_classification", "gender", "4s", "labram_gender_4s"),
        ("age_regression", "age_regression", "1s", "LABRAM_base_age_regression_1s"),
        ("age_regression", "age_regression", "2s", "LABRAM_base_age_regression_2s"),
        ("age_regression", "age_regression", "4s", "LABRAM_base_age_regression_4s"),
    ]
    rows: List[ReportRow] = []
    for task, variant, segment, dirname in specs:
        run_dir = final_logs / "LaBraM" / dirname
        log_path = _canonical_labram_log(run_dir)
        if log_path is None:
            continue
        params, gpu_model, num_gpus, timings = _parse_labram_log(log_path)
        rows.append(
            ReportRow(
                task=task,
                segment=segment,
                model_family="LaBraM",
                model_variant=variant,
                total_parameters=params,
                parameters_source=str(log_path.relative_to(final_logs.parent)),
                gpu_model=gpu_model,
                num_gpus=num_gpus,
                hardware_source=str(log_path.relative_to(final_logs.parent)),
                inference_s_per_batch_test=timings.get("test"),
                inference_s_per_batch_active=timings.get("active_test"),
                inference_s_per_batch_passive=timings.get("passive_test"),
                inference_ms_per_sample_test=None,
                inference_ms_per_sample_active=None,
                inference_ms_per_sample_passive=None,
                eval_batch_size=None,
                eval_total_time_s=None,
                test_num_samples=None,
                inference_source="final eval block: '<Phase> Test: Total time: ... (X s / it)'",
                log_source=str(log_path.relative_to(final_logs.parent)),
            )
        )
    return rows


def _parse_cnn_log_hardware(log_path: Path) -> Tuple[Optional[str], Optional[int]]:
    text = _read_text(log_path)
    gpus = GPU_CONFIG_RE.findall(text)
    if not gpus:
        return None, None
    return gpus[0][1], len({g[0] for g in gpus})


def _cnn_param_count() -> int:
    # Architecture constant (same head sizes differ by <=130 params; use gender as representative).
    return 593_522


def _cnn_rows(final_logs: Path) -> List[ReportRow]:
    task_map = {
        "gender_baseline": "gender_classification",
        "age_classification": "age_classification",
        "age_regression": "age_regression",
    }
    rows: List[ReportRow] = []
    for segment in ("1s", "2s", "4s"):
        summary_path = final_logs / "CNN" / f"CNN_{segment}" / "experiment_summary.json"
        log_path = final_logs / "CNN" / f"CNN_{segment}" / f"cnn_{segment}.log"
        if not summary_path.exists():
            continue
        summary = _parse_json(summary_path)
        gpu_model, num_gpus = _parse_cnn_log_hardware(log_path) if log_path.exists() else (None, None)
        log_text = _read_text(log_path) if log_path.exists() else ""
        batch_match = EVAL_BATCH_RE.search(log_text)
        eval_batch_size = int(batch_match.group(1)) if batch_match else None

        for exp in summary.get("experiments", []):
            if not isinstance(exp, dict):
                continue
            exp_name = str(exp.get("experiment_name", ""))
            task = None
            for prefix, mapped in task_map.items():
                if exp_name.startswith(prefix):
                    task = mapped
                    break
            if task is None:
                continue
            metrics = exp.get("metrics") if isinstance(exp.get("metrics"), dict) else {}
            ttm = metrics.get("task_type_metrics") if isinstance(metrics.get("task_type_metrics"), dict) else {}
            active_n = (ttm.get("active") or {}).get("num_samples") if isinstance(ttm.get("active"), dict) else None
            passive_n = (ttm.get("passive") or {}).get("num_samples") if isinstance(ttm.get("passive"), dict) else None
            total_n = active_n + passive_n if isinstance(active_n, int) and isinstance(passive_n, int) else None
            eval_time = exp.get("evaluation_time")
            section = _log_section_for_experiment(log_text, exp_name)
            pooled_s, total_s = _parse_eval_times(section)
            ms_total, ms_active, ms_passive = _split_task_eval_ms(
                pooled_s, float(total_s or eval_time or 0) or None, active_n, passive_n
            )

            rows.append(
                ReportRow(
                    task=task,
                    segment=segment,
                    model_family="CNN",
                    model_variant="eeg_cnn",
                    total_parameters=_cnn_param_count(),
                    parameters_source="computed from CNN model architecture (not logged in final_logs)",
                    gpu_model=gpu_model,
                    num_gpus=num_gpus,
                    hardware_source=str(log_path.relative_to(final_logs.parent)) if log_path.exists() else "",
                    inference_s_per_batch_test=None,
                    inference_s_per_batch_active=None,
                    inference_s_per_batch_passive=None,
                    inference_ms_per_sample_test=ms_total,
                    inference_ms_per_sample_active=ms_active,
                    inference_ms_per_sample_passive=ms_passive,
                    eval_batch_size=eval_batch_size,
                    eval_total_time_s=float(total_s or eval_time) if (total_s or eval_time) is not None else None,
                    test_num_samples=total_n,
                    inference_source="training log Evaluation time lines + sample counts from experiment_summary.json",
                    log_source=str(log_path.relative_to(final_logs.parent)) if log_path.exists() else str(summary_path.relative_to(final_logs.parent)),
                )
            )
    return rows


def _resnet_param_counts(final_logs: Path) -> Dict[int, int]:
    """Load parameter counts from a saved checkpoint (not logged in training output)."""
    counts: Dict[int, int] = {}
    for backbone in (18, 34, 50):
        ckpts = sorted((final_logs / "RESNET").glob(f"RESNET{backbone}_age_1s/**/*.pth"))
        if not ckpts:
            raise FileNotFoundError(
                f"No ResNet{backbone} checkpoint under final_logs/RESNET/RESNET{backbone}_age_1s"
            )
        import torch

        state = torch.load(ckpts[0], map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "state_dict" in state:
                state = state["state_dict"]
        counts[backbone] = sum(v.numel() for v in state.values())
    return counts


def _resnet_rows(final_logs: Path) -> List[ReportRow]:
    rows: List[ReportRow] = []
    param_counts = _resnet_param_counts(final_logs)
    for backbone in (18, 34, 50):
        for task_suffix, task in (
            ("age", "age_classification"),
            ("gender", "gender_classification"),
            ("age_regression", "age_regression"),
        ):
            for segment in ("1s", "2s", "4s"):
                run_dir = final_logs / "RESNET" / f"RESNET{backbone}_{task_suffix}_{segment}"
                summary_path = run_dir / "experiment_summary.json"
                log_path = run_dir / f"resnet{backbone}_{task_suffix}_{segment}.log"
                if not summary_path.exists():
                    continue
                summary = _parse_json(summary_path)
                exp = (summary.get("experiments") or [None])[0]
                if not isinstance(exp, dict):
                    continue
                gpu_model, num_gpus = (None, 1)
                if log_path.exists():
                    gpus = GPU_CONFIG_RE.findall(_read_text(log_path))
                    if gpus:
                        gpu_model = gpus[0][1]
                        num_gpus = len({g[0] for g in gpus})
                    elif "NVIDIA L40S" in _read_text(log_path):
                        gpu_model = "NVIDIA L40S"
                metrics = exp.get("metrics") if isinstance(exp.get("metrics"), dict) else {}
                ttm = metrics.get("task_type_metrics") if isinstance(metrics.get("task_type_metrics"), dict) else {}
                active_n = (ttm.get("active") or {}).get("num_samples") if isinstance(ttm.get("active"), dict) else None
                passive_n = (ttm.get("passive") or {}).get("num_samples") if isinstance(ttm.get("passive"), dict) else None
                total_n = active_n + passive_n if isinstance(active_n, int) and isinstance(passive_n, int) else None
                eval_time = exp.get("evaluation_time")
                log_text = _read_text(log_path) if log_path.exists() else ""
                pooled_s, total_s = _parse_eval_times(log_text)
                ms_total, ms_active, ms_passive = _split_task_eval_ms(
                    pooled_s, float(total_s or eval_time or 0) or None, active_n, passive_n
                )
                rows.append(
                    ReportRow(
                        task=task,
                        segment=segment,
                        model_family="ResNet",
                        model_variant=f"resnet{backbone}",
                        total_parameters=param_counts[backbone],
                        parameters_source="computed from saved ResNet checkpoint (not logged in final_logs)",
                        gpu_model=gpu_model,
                        num_gpus=num_gpus,
                        hardware_source=str(log_path.relative_to(final_logs.parent)) if log_path.exists() else "",
                        inference_s_per_batch_test=None,
                        inference_s_per_batch_active=None,
                        inference_s_per_batch_passive=None,
                        inference_ms_per_sample_test=ms_total,
                        inference_ms_per_sample_active=ms_active,
                        inference_ms_per_sample_passive=ms_passive,
                        eval_batch_size=None,
                        eval_total_time_s=float(total_s or eval_time) if (total_s or eval_time) is not None else None,
                        test_num_samples=total_n,
                        inference_source="training log Evaluation time lines + sample counts from experiment_summary.json",
                        log_source=str(log_path.relative_to(final_logs.parent)) if log_path.exists() else str(summary_path.relative_to(final_logs.parent)),
                    )
                )
    return rows


def _fmt(v: Optional[float], digits: int = 4) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def _write_csv(path: Path, rows: Iterable[ReportRow]) -> None:
    fieldnames = list(asdict(ReportRow("", "", "", "", None, "", None, None, "", None, None, None, None, None, None, None, None, None, "", "")).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(asdict(row))


def _write_markdown(path: Path, rows: List[ReportRow], final_logs: Path, warnings: List[str]) -> None:
    lines: List[str] = [
        "# Inference time, parameters, and hardware — `final_logs`",
        "",
        f"Generated from `{final_logs}` on {date.today().isoformat()}.",
        "",
        "## Notes",
        "",
        "- **LaBraM** logs report per-batch inference as `s/it` from final `Test` / `Active Test` / `Passive Test` blocks.",
        "- **CNN / ResNet** logs report pooled-test `Evaluation time` and total eval time (including active/passive splits). "
        "Per-sample ms uses pooled time for combined test; active/passive ms uses the remainder split by sample counts.",
        "- **Parameter counts**: LaBraM from training logs; CNN from model architecture; ResNet from saved checkpoints.",
        "- All runs in this report used **NVIDIA L40S** GPUs (count varies by experiment).",
        "",
    ]
    for task in ("age_classification", "gender_classification", "age_regression"):
        task_rows = [r for r in rows if r.task == task]
        if not task_rows:
            continue
        title = task.replace("_", " ").title()
        lines.extend([f"## {title}", ""])
        for family in ("LaBraM", "CNN", "ResNet"):
            fam_rows = [r for r in task_rows if r.model_family == family]
            if not fam_rows:
                continue
            lines.append(f"### {family}")
            lines.append("")
            if family == "LaBraM":
                lines.append(
                    "| Segment | Params | GPU | #GPUs | Test s/batch | Active s/batch | Passive s/batch | Source |"
                )
                lines.append("|---------|--------|-----|-------|--------------|----------------|-----------------|--------|")
                for r in sorted(fam_rows, key=lambda x: x.segment):
                    lines.append(
                        f"| {r.segment} | {r.total_parameters:,} | {r.gpu_model or '—'} | {r.num_gpus or '—'} | "
                        f"{_fmt(r.inference_s_per_batch_test)} | {_fmt(r.inference_s_per_batch_active)} | "
                        f"{_fmt(r.inference_s_per_batch_passive)} | `{r.log_source}` |"
                    )
            else:
                lines.append(
                    "| Variant | Segment | Params | GPU | #GPUs | Eval time (s) | ms/sample (all) | ms/sample active | ms/sample passive | Source |"
                )
                lines.append("|---------|---------|--------|-----|-------|---------------|-----------------|------------------|-------------------|--------|")
                for r in sorted(fam_rows, key=lambda x: (x.model_variant, x.segment)):
                    lines.append(
                        f"| {r.model_variant} | {r.segment} | {r.total_parameters:,} | {r.gpu_model or '—'} | {r.num_gpus or '—'} | "
                        f"{_fmt(r.eval_total_time_s, 1)} | {_fmt(r.inference_ms_per_sample_test, 3)} | "
                        f"{_fmt(r.inference_ms_per_sample_active, 3)} | {_fmt(r.inference_ms_per_sample_passive, 3)} | `{r.log_source}` |"
                    )
            lines.append("")

    if warnings:
        lines.extend(["## Parsing warnings", ""])
        lines.extend(f"- {w}" for w in warnings)
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-logs-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "final_logs",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / f"final_logs_inference_report_{date.today().isoformat()}",
    )
    args = parser.parse_args(argv)

    final_logs: Path = args.final_logs_dir
    out_dir: Path = args.out_dir
    if not final_logs.is_dir():
        raise FileNotFoundError(f"Missing final_logs directory: {final_logs}")
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    rows = _labram_rows(final_logs) + _cnn_rows(final_logs) + _resnet_rows(final_logs)
    rows = sorted(rows, key=lambda r: (r.task, r.model_family, r.model_variant, r.segment))

    if not rows:
        raise RuntimeError(f"No report rows extracted from {final_logs}")

    for row in rows:
        if row.total_parameters is None:
            warnings.append(f"{row.model_family} {row.model_variant} {row.segment}: missing parameter count")
        if row.gpu_model is None:
            warnings.append(f"{row.model_family} {row.model_variant} {row.segment}: missing GPU info")

    _write_csv(out_dir / "inference_hardware_report.csv", rows)
    (out_dir / "inference_hardware_report.json").write_text(
        json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8"
    )
    _write_markdown(out_dir / "inference_hardware_report.md", rows, final_logs, warnings)

    print(f"Wrote: {out_dir / 'inference_hardware_report.md'}")
    print(f"Wrote: {out_dir / 'inference_hardware_report.csv'}")
    print(f"Wrote: {out_dir / 'inference_hardware_report.json'}")
    if warnings:
        print(f"Warnings: {len(warnings)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
