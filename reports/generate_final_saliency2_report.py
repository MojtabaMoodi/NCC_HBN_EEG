#!/usr/bin/env python3
"""Build a Markdown + PDF report for final_saliency2/ (vanilla gradients saliency outputs)."""
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


TOPO_BASENAME = "topography_vanilla_gradients.png"


def _relpath_md(out_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), out_dir.resolve()).replace("\\", "/")


def _discover_experiment_dirs(saliency_root: Path) -> List[Path]:
    dirs = []
    for p in sorted(saliency_root.iterdir()):
        if p.is_dir() and not p.name.startswith("."):
            dirs.append(p)
    return dirs


def _top3_channels(csv_path: Path) -> Optional[str]:
    if not csv_path.exists():
        return None
    try:
        with csv_path.open(encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            rows = []
            for i, row in enumerate(r):
                if i >= 3:
                    break
                ch = row.get("Channel", row.get("channel", ""))
                mi = row.get("Mean_Importance", row.get("mean_importance", ""))
                rows.append(f"{ch} ({mi})")
        return ", ".join(rows) if rows else None
    except OSError:
        return None


def _list_artifacts(exp_dir: Path) -> Dict[str, Any]:
    pngs = sorted(exp_dir.rglob("*.png"))
    csvs = sorted(exp_dir.rglob("*.csv"))
    return {
        "n_png": len(pngs),
        "n_csv": len(csvs),
        "png_paths": [str(p.relative_to(exp_dir)) for p in pngs],
    }


def _write_markdown(
    out_path: Path,
    saliency_root: Path,
    repo_root: Path,
    experiments: List[Path],
) -> None:
    out_dir = out_path.parent
    lines: List[str] = []
    lines.append("# final_saliency2 report\n\n")
    lines.append(f"- **Generated**: {date.today().isoformat()}\n")
    lines.append(f"- **Source**: `{saliency_root.relative_to(repo_root)}`\n")
    lines.append(f"- **Experiments (runs)**: {len(experiments)}\n")
    lines.append(f"- **Method**: vanilla gradients (see `topography_vanilla_gradients.png` per split)\n\n")

    lines.append("## Index\n\n")
    lines.append("| run | combined topo | active topo | passive topo | top-3 channels (combined) |\n")
    lines.append("| --- | --- | --- | --- | --- |\n")

    rows_data: List[Tuple[Path, Dict[str, Optional[Path]]]] = []
    for exp in experiments:
        name = exp.name
        combined = exp / TOPO_BASENAME
        active = exp / "active" / TOPO_BASENAME
        passive = exp / "passive" / TOPO_BASENAME
        ch_csv = exp / "channel_ranking_vanilla_gradients.csv"
        top3 = _top3_channels(ch_csv) or "-"
        c, a, p = combined.exists(), active.exists(), passive.exists()
        rows_data.append((exp, {"combined": combined if c else None, "active": active if a else None, "passive": passive if p else None}))
        lines.append(
            f"| `{name}` | {'yes' if c else 'no'} | {'yes' if a else 'no'} | {'yes' if p else 'no'} | {top3} |\n"
        )

    lines.append("\n## Topography figures\n\n")
    lines.append(
        "For each run: **combined** (all data), **active**, **passive** task splits. "
        "Images are embedded from `final_saliency2/<run>/`.\n\n"
    )

    for exp, _ in rows_data:
        name = exp.name
        lines.append(f"### `{name}`\n\n")
        for label, sub in (
            ("Combined (train/eval pool)", exp / TOPO_BASENAME),
            ("Active tasks", exp / "active" / TOPO_BASENAME),
            ("Passive tasks", exp / "passive" / TOPO_BASENAME),
        ):
            if not sub.exists():
                lines.append(f"*{label}: file missing (`{sub.name}`)*\n\n")
                continue
            rel = _relpath_md(out_dir, sub)
            lines.append(f"#### {label}\n\n")
            # Constrain width for LaTeX PDF (many figures per document).
            lines.append(f"![{name} — {label}]({rel}){{width=58%}}\n\n")

    lines.append("## File inventory\n\n")
    lines.append("Per-run counts of generated artifacts (full path list: `final_saliency2_index.json`).\n\n")
    lines.append("| run | PNG files | CSV files |\n")
    lines.append("| --- | ---: | ---: |\n")
    for exp in experiments:
        art = _list_artifacts(exp)
        lines.append(f"| `{exp.name}` | {art['n_png']} | {art['n_csv']} |\n")
    lines.append("\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def _write_json_index(out_path: Path, saliency_root: Path, experiments: List[Path], repo_root: Path) -> None:
    payload: List[Dict[str, Any]] = []
    for exp in experiments:
        combined = exp / TOPO_BASENAME
        payload.append(
            {
                "run": exp.name,
                "relative_dir": str(exp.relative_to(repo_root)),
                "topography_combined": str(combined.relative_to(repo_root)) if combined.exists() else None,
                "topography_active": str((exp / "active" / TOPO_BASENAME).relative_to(repo_root))
                if (exp / "active" / TOPO_BASENAME).exists()
                else None,
                "topography_passive": str((exp / "passive" / TOPO_BASENAME).relative_to(repo_root))
                if (exp / "passive" / TOPO_BASENAME).exists()
                else None,
                "artifacts": _list_artifacts(exp),
            }
        )
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_pandoc_pdf(md_path: Path, pdf_path: Path) -> None:
    out_dir = md_path.parent
    cmd = [
        "pandoc",
        str(md_path.name),
        "-o",
        str(pdf_path.name),
        "--pdf-engine=pdflatex",
        "-V",
        "geometry:margin=0.85in",
        "-V",
        "fontsize=10pt",
        "--toc",
        "--toc-depth=2",
    ]
    try:
        subprocess.run(cmd, cwd=str(out_dir), check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(e.stderr or e.stdout or str(e), file=sys.stderr)
        raise


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate final_saliency2 Markdown + PDF report.")
    parser.add_argument(
        "--saliency-dir",
        type=Path,
        default=(Path(__file__).resolve().parents[1] / "final_saliency2"),
        help="Path to final_saliency2 directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(Path(__file__).resolve().parent / f"final_saliency2_report_{date.today().isoformat()}"),
        help="Output directory for report artifacts",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Only write Markdown and JSON; do not run pandoc",
    )
    args = parser.parse_args(argv)

    saliency_dir: Path = args.saliency_dir.resolve()
    out_dir: Path = args.out_dir.resolve()
    repo_root = Path(__file__).resolve().parents[1]

    if not saliency_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {saliency_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    experiments = _discover_experiment_dirs(saliency_dir)
    if not experiments:
        raise FileNotFoundError(f"No experiment subdirectories under {saliency_dir}")

    md_path = out_dir / "final_saliency2_report.md"
    json_path = out_dir / "final_saliency2_index.json"
    pdf_path = out_dir / "final_saliency2_report.pdf"

    _write_markdown(md_path, saliency_dir, repo_root, experiments)
    _write_json_index(json_path, saliency_dir, experiments, repo_root)

    print(f"Wrote: {md_path}")
    print(f"Wrote: {json_path}")

    if args.skip_pdf:
        print("(PDF skipped: --skip-pdf)")
        return 0

    try:
        _run_pandoc_pdf(md_path, pdf_path)
        print(f"Wrote: {pdf_path}")
    except FileNotFoundError:
        print("pandoc not found; install pandoc and a LaTeX engine (pdflatex) for PDF export.", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
