#!/usr/bin/env python3
"""Plot confusion-matrix heatmaps from LaBraM/confusion_matrix/results/*.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

_labram_dir = Path(__file__).resolve().parent.parent
if str(_labram_dir) not in sys.path:
    sys.path.insert(0, str(_labram_dir))

from classification_eval_utils import classification_display_labels  # noqa: E402

AGGREGATION_TITLES = {
    "segment": "Segment-level",
    "participant_mean_prob": "Participant-level (mean probability)",
    "participant_majority_vote": "Participant-level (majority vote)",
}
def _plot_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    class_labels: Sequence[str],
    title: str,
    output_path: Path,
    *,
    normalize: bool = False,
) -> None:
    data = np.asarray(matrix, dtype=float)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        display = np.divide(
            data,
            row_sums,
            out=np.zeros_like(data),
            where=row_sums != 0,
        )
        fmt = ".1%"
        cmap = "Blues"
        vmax = 1.0
    else:
        display = data
        fmt = "d"
        cmap = "Blues"
        vmax = None

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(display, interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Row-normalized proportion" if normalize else "Count", rotation=90, va="center")

    tick_positions = np.arange(len(class_labels))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(class_labels, rotation=0, ha="center")
    ax.set_yticklabels(class_labels, rotation=90, va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    threshold = display.max() / 2 if display.size and display.max() > 0 else 0
    for row in range(display.shape[0]):
        for col in range(display.shape[1]):
            value = display[row, col]
            if normalize:
                text = f"{value:.1%}" if row_sums[row, 0] > 0 else "0.0%"
            else:
                text = f"{int(value)}"
            color = "white" if value > threshold else "black"
            ax.text(col, row, text, ha="center", va="center", color=color, fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_results_json(
    json_path: Path,
    output_dir: Path,
    *,
    normalize: bool = False,
) -> List[Path]:
    with json_path.open("r", encoding="utf-8") as handle:
        payload: Dict[str, Any] = json.load(handle)

    dataset_type = payload["dataset_type"]
    segment_length = payload["segment_length"]
    task_type = payload.get("task_type", "both")
    task_name = json_path.stem.replace("_confusion_matrix", "")
    confusion_matrices = payload["test"]["confusion_matrices"]

    written: List[Path] = []
    for aggregation_key, cm_payload in confusion_matrices.items():
        labels = cm_payload["labels"]
        matrix = cm_payload["matrix"]
        n_samples = cm_payload.get("n_samples")
        class_labels = classification_display_labels(dataset_type, labels)
        agg_title = AGGREGATION_TITLES.get(aggregation_key, aggregation_key)
        title = f"LaBraM {dataset_type} {segment_length} ({task_type}) — {agg_title}"
        if n_samples is not None:
            title += f"\n(n={n_samples:,})"

        suffix = "_normalized" if normalize else ""
        output_path = output_dir / f"{task_name}_{aggregation_key}{suffix}.png"
        _plot_confusion_matrix(
            matrix,
            class_labels,
            title,
            output_path,
            normalize=normalize,
        )
        written.append(output_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory containing *_confusion_matrix.json files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for PNG outputs (default: <results_dir>/figures).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Also save row-normalized confusion matrices.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or results_dir / "figures").resolve()
    json_files = sorted(results_dir.glob("*_confusion_matrix.json"))
    if not json_files:
        raise SystemExit(f"No confusion-matrix JSON files found in {results_dir}")

    all_written: List[Path] = []
    for json_path in json_files:
        all_written.extend(plot_results_json(json_path, output_dir, normalize=False))
        if args.normalize:
            all_written.extend(plot_results_json(json_path, output_dir, normalize=True))

    print(f"Wrote {len(all_written)} figure(s) to {output_dir}")
    for path in all_written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
