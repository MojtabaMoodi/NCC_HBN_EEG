#!/usr/bin/env python3
"""Top-10 channel ranking report for final_saliency2/ (vanilla gradients CSVs)."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CSV_NAME = "channel_ranking_vanilla_gradients.csv"
TOP_N = 10

SPLITS: Tuple[Tuple[str, Path], ...] = (
    ("combined (all)", Path(".")),
    ("active", Path("active")),
    ("passive", Path("passive")),
)


def _discover_experiment_dirs(saliency_root: Path) -> List[Path]:
    return sorted(
        p for p in saliency_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )


def _read_top_channels(csv_path: Path, n: int) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    out: List[Dict[str, str]] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            ch = row.get("Channel") or row.get("channel") or ""
            rank = row.get("Rank") or row.get("rank") or str(i + 1)
            mi = row.get("Mean_Importance") or row.get("mean_importance") or ""
            si = row.get("Std_Importance") or row.get("std_importance") or ""
            out.append(
                {
                    "rank": str(rank).strip(),
                    "channel": str(ch).strip(),
                    "mean_importance": str(mi).strip(),
                    "std_importance": str(si).strip(),
                }
            )
    return out


def _fmt_float(s: str, nd: int = 6) -> str:
    s = s.strip()
    if not s:
        return "—"
    try:
        return f"{float(s):.{nd}f}"
    except ValueError:
        return s


def _write_markdown(
    out_path: Path,
    saliency_root: Path,
    repo_root: Path,
    experiments: List[Path],
    rows_flat: List[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("# Saliency: top 10 channels per experiment\n\n")
    lines.append(f"- **Generated**: {date.today().isoformat()}\n")
    lines.append(f"- **Source**: `{saliency_root.relative_to(repo_root)}`\n")
    lines.append(f"- **Experiments**: {len(experiments)}\n")
    lines.append(f"- **Splits**: combined (root CSV), **active/**, **passive/** (when present)\n")
    lines.append(f"- **Metric**: `Mean_Importance` from `{CSV_NAME}` (vanilla gradients)\n\n")

    lines.append("## Summary table (combined split, rank 1 only)\n\n")
    lines.append("| experiment | #1 channel | mean importance |\n")
    lines.append("| --- | --- | --- |\n")
    for exp in experiments:
        key = (exp.name, "combined (all)")
        r1 = next((r for r in rows_flat if r["experiment"] == key[0] and r["split"] == key[1] and r["rank"] == 1), None)
        if r1:
            lines.append(
                f"| `{exp.name}` | {r1['channel']} | {_fmt_float(r1['mean_importance'])} |\n"
            )
        else:
            lines.append(f"| `{exp.name}` | — | — |\n")
    lines.append("\n")

    for exp in experiments:
        lines.append(f"## `{exp.name}`\n\n")
        for split_label, rel in SPLITS:
            csv_path = exp / rel / CSV_NAME if rel != Path(".") else exp / CSV_NAME
            top = _read_top_channels(csv_path, TOP_N)
            if not top:
                lines.append(f"### {split_label}\n\n*No `{CSV_NAME}` at this path.*\n\n")
                continue
            lines.append(f"### {split_label}\n\n")
            lines.append("| rank | channel | mean importance | std |\n")
            lines.append("| ---: | --- | ---: | ---: |\n")
            for row in top:
                lines.append(
                    f"| {row['rank']} | {row['channel']} | {_fmt_float(row['mean_importance'])} | "
                    f"{_fmt_float(row['std_importance'])} |\n"
                )
            lines.append("\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def _collect_flat_rows(experiments: List[Path], saliency_root: Path) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for exp in experiments:
        for split_label, rel in SPLITS:
            csv_path = exp / rel / CSV_NAME if rel != Path(".") else exp / CSV_NAME
            top = _read_top_channels(csv_path, TOP_N)
            for row in top:
                flat.append(
                    {
                        "experiment": exp.name,
                        "split": split_label,
                        "rank": int(row["rank"]) if row["rank"].isdigit() else row["rank"],
                        "channel": row["channel"],
                        "mean_importance": row["mean_importance"],
                        "std_importance": row["std_importance"],
                        "csv_path": str(csv_path.relative_to(saliency_root)) if csv_path.exists() else "",
                    }
                )
    return flat


def _write_csv_flat(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = ["experiment", "split", "rank", "channel", "mean_importance", "std_importance", "csv_path"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def _run_pandoc_pdf(md_path: Path, pdf_path: Path) -> None:
    out_dir = md_path.parent
    cmd = [
        "pandoc",
        str(md_path.name),
        "-o",
        str(pdf_path.name),
        "--pdf-engine=pdflatex",
        "-V",
        "geometry:margin=0.9in",
        "-V",
        "fontsize=10pt",
        "--toc",
        "--toc-depth=2",
    ]
    subprocess.run(cmd, cwd=str(out_dir), check=True, capture_output=True, text=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Top-10 channel report for final_saliency2.")
    parser.add_argument(
        "--saliency-dir",
        type=Path,
        default=(Path(__file__).resolve().parents[1] / "final_saliency2"),
        help="Path to final_saliency2",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(Path(__file__).resolve().parent / f"saliency_top_channels_report_{date.today().isoformat()}"),
        help="Output directory",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF export (Markdown + CSV + JSON only)",
    )
    args = parser.parse_args(argv)

    saliency_dir = args.saliency_dir.resolve()
    out_dir = args.out_dir.resolve()
    repo_root = Path(__file__).resolve().parents[1]

    if not saliency_dir.is_dir():
        raise NotADirectoryError(saliency_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    experiments = _discover_experiment_dirs(saliency_dir)
    if not experiments:
        raise FileNotFoundError(f"No experiment dirs under {saliency_dir}")

    rows_flat = _collect_flat_rows(experiments, saliency_dir)

    md_path = out_dir / "saliency_top_channels_report.md"
    csv_path = out_dir / "saliency_top10_channels.csv"
    json_path = out_dir / "saliency_top10_channels.json"
    pdf_path = out_dir / "saliency_top_channels_report.pdf"

    _write_markdown(md_path, saliency_dir, repo_root, experiments, rows_flat)
    _write_csv_flat(csv_path, rows_flat)
    json_path.write_text(json.dumps(rows_flat, indent=2), encoding="utf-8")

    print(f"Wrote: {md_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")

    if not args.no_pdf:
        try:
            _run_pandoc_pdf(md_path, pdf_path)
            print(f"Wrote: {pdf_path}")
        except FileNotFoundError:
            print("Note: pandoc not found; PDF skipped. Install pandoc + pdflatex for PDF.", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(e.stderr or e.stdout, file=sys.stderr)
            print("Note: PDF build failed; Markdown and CSV are still valid.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
