#!/usr/bin/env python3
"""Markdown + JSON + optional PDF report for final_user_identification/ (LaBraM ArcFace CV + saliency)."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TOPO = "topography_vanilla_gradients.png"
CH_CSV = "channel_ranking_vanilla_gradients.csv"
TOP_N = 10

SPLITS: Tuple[Tuple[str, Path], ...] = (
    ("combined", Path(".")),
    ("active", Path("active")),
    ("passive", Path("passive")),
)

# (relative path under uid_root, label for report)
SEGMENT_ROOTS: Tuple[Tuple[str, str], ...] = (
    (".", "1s segments"),
    ("2s", "2s segments"),
    ("4s", "4s segments"),
)


def _relpath_md(out_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), out_dir.resolve()).replace("\\", "/")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


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


def _fmt_num(s: str, nd: int = 6) -> str:
    s = (s or "").strip()
    if not s:
        return "—"
    try:
        return f"{float(s):.{nd}f}"
    except ValueError:
        return s


def _discover_folds(segment_dir: Path) -> List[Path]:
    folds = []
    for p in sorted(segment_dir.iterdir()):
        if p.is_dir() and re.match(r"^fold_\d+$", p.name):
            folds.append(p)
    return folds


def _saliency_dir(fold_dir: Path) -> Path:
    return fold_dir / "saliency"


def _collect_segment_data(uid_root: Path, segment_rel: str, segment_label: str) -> Dict[str, Any]:
    seg_dir = (uid_root / segment_rel).resolve()
    out: Dict[str, Any] = {
        "segment_key": segment_rel,
        "segment_label": segment_label,
        "path": str(seg_dir.relative_to(uid_root)) if seg_dir != uid_root else ".",
        "cv_summary": None,
        "folds": [],
    }
    cv_path = seg_dir / "cv_summary.json"
    if cv_path.exists():
        out["cv_summary"] = _read_json(cv_path)

    for fold_dir in _discover_folds(seg_dir):
        sal = _saliency_dir(fold_dir)
        results = _read_json(fold_dir / "results.json")
        fold_info: Dict[str, Any] = {
            "fold": fold_dir.name,
            "results": results,
            "saliency": {
                "combined": str(sal / CH_CSV) if (sal / CH_CSV).exists() else None,
                "topography_paths": {},
                "top_channels": {},
            },
        }
        for split_label, rel in SPLITS:
            csv_p = sal / rel / CH_CSV if rel != Path(".") else sal / CH_CSV
            topo_p = sal / rel / TOPO if rel != Path(".") else sal / TOPO
            fold_info["saliency"]["top_channels"][split_label] = _read_top_channels(csv_p, TOP_N)
            fold_info["saliency"]["topography_paths"][split_label] = str(topo_p) if topo_p.exists() else None
        out["folds"].append(fold_info)
    return out


def _write_markdown(
    out_path: Path,
    uid_root: Path,
    repo_root: Path,
    segments: List[Dict[str, Any]],
    out_dir: Path,
) -> None:
    lines: List[str] = []
    lines.append("# User identification (LaBraM ArcFace) — full report\n\n")
    lines.append(f"- **Generated**: {date.today().isoformat()}\n")
    lines.append(f"- **Source**: `{uid_root.relative_to(repo_root)}`\n")
    lines.append("- **Content**: cross-validation accuracies, top-10 channels (vanilla gradients), topography figures per fold.\n\n")

    for seg in segments:
        label = seg["segment_label"]
        lines.append(f"## {label} (`{seg['path']}`)\n\n")

        cv = seg.get("cv_summary")
        if cv:
            lines.append("### Cross-validation summary\n\n")
            lines.append("| metric | mean | std |\n")
            lines.append("| --- | ---: | ---: |\n")
            lines.append(
                f"| val accuracy (%) | {cv.get('val_accuracy_mean', '—')} | {cv.get('val_accuracy_std', '—')} |\n"
            )
            lines.append(
                f"| test accuracy (%) | {cv.get('test_accuracy_mean', '—')} | {cv.get('test_accuracy_std', '—')} |\n"
            )
            lines.append("\n**Per-fold (val / test %):**\n\n")
            vf = cv.get("val_accuracy_per_fold") or []
            tf = cv.get("test_accuracy_per_fold") or []
            for i in range(max(len(vf), len(tf))):
                v = vf[i] if i < len(vf) else None
                t = tf[i] if i < len(tf) else None
                lines.append(f"- Fold {i}: val={v}, test={t}\n")
            lines.append("\n")
        else:
            lines.append("*No `cv_summary.json` in this segment directory (per-fold tables below only).*\n\n")

        for fi in seg["folds"]:
            fname = fi["fold"]
            lines.append(f"### {fname}\n\n")
            res = fi.get("results") or {}
            if res:
                lines.append("**Accuracies** (from `results.json`):\n\n")
                lines.append(f"- validation: **{res.get('val_accuracy', '—')}** %\n")
                lines.append(f"- test: **{res.get('test_accuracy', '—')}** %\n\n")
            else:
                lines.append("*No `results.json` (fold may be incomplete).*\n\n")

            for split_label, rel in SPLITS:
                if seg["path"] != ".":
                    sal = uid_root / seg["path"] / fname / "saliency"
                else:
                    sal = uid_root / fname / "saliency"
                csv_p = sal / rel / CH_CSV if rel != Path(".") else sal / CH_CSV
                topo_p = sal / rel / TOPO if rel != Path(".") else sal / TOPO
                top = _read_top_channels(csv_p, TOP_N)
                lines.append(f"#### Top channels — {split_label}\n\n")
                if not top:
                    lines.append(f"*Missing `{CH_CSV}`.*\n\n")
                    continue
                lines.append("| rank | channel | mean importance | std |\n")
                lines.append("| ---: | --- | ---: | ---: |\n")
                for row in top:
                    lines.append(
                        f"| {row['rank']} | {row['channel']} | {_fmt_num(row['mean_importance'])} | "
                        f"{_fmt_num(row['std_importance'])} |\n"
                    )
                lines.append("\n")

            lines.append(f"#### Topography — {fname}\n\n")
            for split_label, rel in SPLITS:
                if seg["path"] != ".":
                    sal = uid_root / seg["path"] / fname / "saliency"
                else:
                    sal = uid_root / fname / "saliency"
                topo_p = sal / rel / TOPO if rel != Path(".") else sal / TOPO
                if not topo_p.exists():
                    lines.append(f"*{split_label}: `{TOPO}` not found.*\n\n")
                    continue
                rel_md = _relpath_md(out_dir, topo_p)
                lines.append(f"**{split_label}**\n\n")
                lines.append(f"![{fname} {split_label}]({rel_md}){{width=52%}}\n\n")

        lines.append("\n---\n\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def _flatten_export(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for seg in segments:
        sk = seg["segment_label"]
        for fi in seg["folds"]:
            res = fi.get("results") or {}
            for split_label, rel in SPLITS:
                for row in fi["saliency"]["top_channels"].get(split_label, []):
                    flat.append(
                        {
                            "segment": sk,
                            "fold": fi["fold"],
                            "split": split_label,
                            "rank": int(row["rank"]) if str(row["rank"]).isdigit() else row["rank"],
                            "channel": row["channel"],
                            "mean_importance": row["mean_importance"],
                            "std_importance": row["std_importance"],
                            "val_accuracy": res.get("val_accuracy"),
                            "test_accuracy": res.get("test_accuracy"),
                        }
                    )
    return flat


def _run_pandoc_pdf(md_path: Path, pdf_path: Path) -> None:
    cmd = [
        "pandoc",
        str(md_path.name),
        "-o",
        str(pdf_path.name),
        "--pdf-engine=pdflatex",
        "-V",
        "geometry:margin=0.8in",
        "-V",
        "fontsize=9pt",
        "--toc",
        "--toc-depth=3",
    ]
    subprocess.run(cmd, cwd=str(md_path.parent), check=True, capture_output=True, text=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Report for final_user_identification/")
    parser.add_argument(
        "--uid-dir",
        type=Path,
        default=(Path(__file__).resolve().parents[1] / "final_user_identification"),
        help="Path to final_user_identification",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(Path(__file__).resolve().parent / f"final_user_identification_report_{date.today().isoformat()}"),
        help="Output directory",
    )
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF export")
    args = parser.parse_args(argv)

    uid_root = args.uid_dir.resolve()
    out_dir = args.out_dir.resolve()
    repo_root = Path(__file__).resolve().parents[1]

    if not uid_root.is_dir():
        raise NotADirectoryError(uid_root)

    out_dir.mkdir(parents=True, exist_ok=True)

    segments: List[Dict[str, Any]] = []
    for seg_rel, seg_label in SEGMENT_ROOTS:
        seg_path = uid_root / seg_rel
        if not seg_path.is_dir():
            continue
        segments.append(_collect_segment_data(uid_root, seg_rel, seg_label))

    if not segments:
        raise FileNotFoundError(f"No segment directories under {uid_root}")

    md_path = out_dir / "final_user_identification_report.md"
    json_path = out_dir / "final_user_identification_report.json"
    csv_path = out_dir / "final_user_identification_top_channels.csv"
    pdf_path = out_dir / "final_user_identification_report.pdf"

    _write_markdown(md_path, uid_root, repo_root, segments, out_dir)

    export = {
        "generated": date.today().isoformat(),
        "source": str(uid_root.relative_to(repo_root)),
        "segments": segments,
        "flat_top_channels": _flatten_export(segments),
    }
    json_path.write_text(json.dumps(export, indent=2), encoding="utf-8")

    if export["flat_top_channels"]:
        keys = list(export["flat_top_channels"][0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in export["flat_top_channels"]:
                w.writerow({k: row.get(k, "") for k in keys})

    print(f"Wrote: {md_path}")
    print(f"Wrote: {json_path}")
    if export["flat_top_channels"]:
        print(f"Wrote: {csv_path}")

    if not args.no_pdf:
        try:
            _run_pandoc_pdf(md_path, pdf_path)
            print(f"Wrote: {pdf_path}")
        except FileNotFoundError:
            print("Note: pandoc not found; PDF skipped.", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(e.stderr or e.stdout, file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
