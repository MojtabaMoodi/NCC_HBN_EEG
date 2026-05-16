#!/usr/bin/env python3
"""
Run full evaluation for every saved best_model.pth under final_user_identification/{1s,2s,4s}/fold_*:

1. **Test** — closed-set accuracy split by **active** vs **passive** (ArcFace),
   via ``eval_active_passive_test.py`` (one JSON per fold).
2. **Test + unknown** — confidence, OOD / known-user score, plots, and ``open_set_summary.json``
   via ``analyze_confidence.py --split both``, run **separately** for ``--task_type active``
   and ``--task_type passive`` (so test and unknown are reported per task type).

HDF5 roots are **per segment length**: each must contain ``fold_0`` … ``fold_{n-1}`` with the
matching ``*_1s_*``, ``*_2s_*``, or ``*_4s_*`` files (same layout as training).

Example (GPU server, repo root = EEG)::

    python user_identification/run_all_folds_active_passive_test_unknown_eval.py \\
      --final_root final_user_identification \\
      --hdf5_root_1s /path/to/preprocessed_1s_parent \\
      --hdf5_root_2s /path/to/preprocessed_2s_parent \\
      --hdf5_root_4s ~/scratch \\
      --n_folds 5 \\
      --device cuda \\
      --log_dir final_user_identification

Omit any ``--hdf5_root_*`` you do not have on disk; that segment is skipped.

Writes ``<final_root>/active_passive_test_unknown_eval_summary.json`` aggregating paths and metrics.

**Log file (stdout + stderr)**

Show output in the terminal and save it::

    python user_identification/run_all_folds_active_passive_test_unknown_eval.py ... \\
      2>&1 | tee final_user_identification/eval_run.log

Save only to a file::

    python user_identification/run_all_folds_active_passive_test_unknown_eval.py ... \\
      > final_user_identification/eval_run.log 2>&1

The script prints a **FOLD RESULT** block after each fold and a **SEGMENT SUMMARY** after each
segment length; the same numbers are in the JSON files below.

**Split logs (optional)** — pass ``--log_dir`` (e.g. ``final_user_identification``) to write::

    <log_dir>/eval_run_<segment>_fold0.log … fold4.log   # full subprocess + FOLD RESULT for that fold
    <log_dir>/eval_run_<segment>.log                     # mean ± std across folds for that segment

When only one segment is evaluated, the cross-segment overview is appended to that segment’s
``eval_run_<segment>.log``. With multiple segments, overview goes to ``<log_dir>/eval_run_overview.log``.

**Where to read results**

- **Per fold:** ``<final_root>/<1s|2s|4s>/fold_<k>/eval_task_type_test_accuracy.json`` (test active/passive
  accuracy); open-set metrics under ``eval_test_unknown_active/open_set_summary.json`` (and passive).
- **All folds:** ``<final_root>/active_passive_test_unknown_eval_summary.json``
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO


SEGMENT_CHOICES = ("1s", "2s", "4s")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _std_sample(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def _emit_fold_report(
    segment: str,
    fold_idx: int,
    fold_entry: Dict[str, Any],
    stream: TextIO = sys.stdout,
) -> None:
    """Print one block per fold so logs are easy to grep (flush for tee/redirection)."""
    lines = [
        "",
        "=" * 72,
        f"FOLD RESULT  segment={segment}  fold_{fold_idx}",
        "=" * 72,
    ]
    t = fold_entry.get("test") or {}
    if t.get("active"):
        a = t["active"]
        loss = a.get("loss")
        loss_s = f"{loss:.6f}" if isinstance(loss, (int, float)) else str(loss)
        f1m = a.get("f1_macro")
        f1w = a.get("f1_weighted")
        f1_extra = ""
        if isinstance(f1m, (int, float)) and f1m == f1m:
            f1_extra = f"   F1_macro={float(f1m):.6f}"
            if isinstance(f1w, (int, float)) and f1w == f1w:
                f1_extra += f"   F1_weighted={float(f1w):.6f}"
        fp = a.get("fpr_macro")
        fn = a.get("fnr_macro")
        if isinstance(fp, (int, float)) and fp == fp:
            f1_extra += f"   FPR_macro={float(fp):.6f}"
        if isinstance(fn, (int, float)) and fn == fn:
            f1_extra += f"   FNR_macro={float(fn):.6f}"
        lines.append(
            f"  Test accuracy (active):   {float(a['accuracy_percent']):.4f}%   loss={loss_s}{f1_extra}"
        )
    else:
        lines.append("  Test accuracy (active):   (missing)")
    if t.get("passive"):
        p = t["passive"]
        loss = p.get("loss")
        loss_s = f"{loss:.6f}" if isinstance(loss, (int, float)) else str(loss)
        f1m = p.get("f1_macro")
        f1w = p.get("f1_weighted")
        f1_extra = ""
        if isinstance(f1m, (int, float)) and f1m == f1m:
            f1_extra = f"   F1_macro={float(f1m):.6f}"
            if isinstance(f1w, (int, float)) and f1w == f1w:
                f1_extra += f"   F1_weighted={float(f1w):.6f}"
        fp = p.get("fpr_macro")
        fn = p.get("fnr_macro")
        if isinstance(fp, (int, float)) and fp == fp:
            f1_extra += f"   FPR_macro={float(fp):.6f}"
        if isinstance(fn, (int, float)) and fn == fn:
            f1_extra += f"   FNR_macro={float(fn):.6f}"
        lines.append(
            f"  Test accuracy (passive): {float(p['accuracy_percent']):.4f}%   loss={loss_s}{f1_extra}"
        )
    else:
        lines.append("  Test accuracy (passive):  (missing)")
    ckt = t.get("checkpoint_test_accuracy_percent")
    if isinstance(ckt, (int, float)) and ckt == ckt:
        lines.append(
            f"  Checkpoint combined test (val-epoch, task_type=both): {float(ckt):.4f}%"
        )
    uo = fold_entry.get("unknown_open_set") or {}
    for key in ("active", "passive"):
        block = uo.get(key)
        if isinstance(block, dict) and "best_open_set_accuracy" in block:
            bt = block.get("best_threshold_known_user_score")
            bt_s = f"{float(bt):.4f}" if isinstance(bt, (int, float)) else str(bt)
            lines.append(
                f"  Open-set accuracy (test+unknown, {key}): {float(block['best_open_set_accuracy']):.4f}  "
                f"(best threshold on known-user score: {bt_s})"
            )
        elif fold_entry.get("analyze_output_dirs"):
            lines.append(
                f"  Open-set ({key}): no open_set_summary.json (unknown split missing or analysis skipped)"
            )
    tpath = fold_entry.get("test_task_accuracy_json", "")
    if tpath:
        lines.append(f"  JSON: {tpath}")
    print("\n".join(lines), file=stream, flush=True)


def _emit_segment_summary(
    segment: str, seg_payload: Dict[str, Any], stream: TextIO = sys.stdout
) -> None:
    print("", file=stream, flush=True)
    print("#" * 72, file=stream, flush=True)
    n = len(seg_payload.get("folds") or {})
    print(f"SEGMENT SUMMARY  {segment}  (folds with results: {n})", file=stream, flush=True)
    print("#" * 72, file=stream, flush=True)
    sta = seg_payload.get("summary_test_accuracy_percent")
    if sta:
        print(
            f"  Test accuracy  mean ± std  — active:  {sta['active_mean']:.4f}% ± {sta['active_std']:.4f}%",
            file=stream,
            flush=True,
        )
        print(
            f"  Test accuracy  mean ± std  — passive: {sta['passive_mean']:.4f}% ± {sta['passive_std']:.4f}%",
            file=stream,
            flush=True,
        )
    stf = seg_payload.get("summary_test_f1_macro")
    if stf:
        print(
            f"  Test F1 (macro) mean ± std — active:  {stf['active_mean']:.6f} ± {stf['active_std']:.6f}",
            file=stream,
            flush=True,
        )
        print(
            f"  Test F1 (macro) mean ± std — passive: {stf['passive_mean']:.6f} ± {stf['passive_std']:.6f}",
            file=stream,
            flush=True,
        )
    stfw = seg_payload.get("summary_test_f1_weighted")
    if stfw:
        print(
            f"  Test F1 (weighted) mean ± std — active:  {stfw['active_mean']:.6f} ± {stfw['active_std']:.6f}",
            file=stream,
            flush=True,
        )
        print(
            f"  Test F1 (weighted) mean ± std — passive: {stfw['passive_mean']:.6f} ± {stfw['passive_std']:.6f}",
            file=stream,
            flush=True,
        )
    stfp = seg_payload.get("summary_test_fpr_macro")
    if stfp:
        print(
            f"  Test FPR (macro OvR) mean ± std — active:  {stfp['active_mean']:.6f} ± {stfp['active_std']:.6f}",
            file=stream,
            flush=True,
        )
        print(
            f"  Test FPR (macro OvR) mean ± std — passive: {stfp['passive_mean']:.6f} ± {stfp['passive_std']:.6f}",
            file=stream,
            flush=True,
        )
    stfn = seg_payload.get("summary_test_fnr_macro")
    if stfn:
        print(
            f"  Test FNR (macro OvR) mean ± std — active:  {stfn['active_mean']:.6f} ± {stfn['active_std']:.6f}",
            file=stream,
            flush=True,
        )
        print(
            f"  Test FNR (macro OvR) mean ± std — passive: {stfn['passive_mean']:.6f} ± {stfn['passive_std']:.6f}",
            file=stream,
            flush=True,
        )
    sos = seg_payload.get("summary_open_set_accuracy")
    if sos:
        print(
            f"  Open-set acc   mean ± std  — active:  {sos['active_mean']:.4f} ± {sos['active_std']:.4f}",
            file=stream,
            flush=True,
        )
        print(
            f"  Open-set acc   mean ± std  — passive: {sos['passive_mean']:.4f} ± {sos['passive_std']:.4f}",
            file=stream,
            flush=True,
        )
    elif seg_payload.get("folds") and not seg_payload.get("summary_open_set_accuracy"):
        print(
            "  Open-set: not in summary (--skip_analyze or no unknown split / open_set_summary.json)",
            file=stream,
            flush=True,
        )
    print("", file=stream, flush=True)


def _emit_final_overview(
    summary: Dict[str, Any], out_path: Path, stream: TextIO = sys.stdout
) -> None:
    print("=" * 72, file=stream, flush=True)
    print("ALL SEGMENTS — overview (full detail in JSON)", file=stream, flush=True)
    print("=" * 72, file=stream, flush=True)
    for seg, payload in sorted((summary.get("segments") or {}).items()):
        sta = (payload or {}).get("summary_test_accuracy_percent") or {}
        if sta:
            print(
                f"  {seg}:  test active  {sta.get('active_mean', float('nan')):.4f}% ± {sta.get('active_std', 0):.4f}%  |  "
                f"passive {sta.get('passive_mean', float('nan')):.4f}% ± {sta.get('passive_std', 0):.4f}%",
                file=stream,
                flush=True,
            )
        sos = (payload or {}).get("summary_open_set_accuracy") or {}
        if sos:
            print(
                f"  {seg}:  open-set active  {sos.get('active_mean', float('nan')):.4f} ± {sos.get('active_std', 0):.4f}  |  "
                f"passive {sos.get('passive_mean', float('nan')):.4f} ± {sos.get('passive_std', 0):.4f}",
                file=stream,
                flush=True,
            )
        stf = (payload or {}).get("summary_test_f1_macro") or {}
        if stf:
            print(
                f"  {seg}:  test F1 (macro) active {stf.get('active_mean', float('nan')):.6f} ± {stf.get('active_std', 0):.6f}  |  "
                f"passive {stf.get('passive_mean', float('nan')):.6f} ± {stf.get('passive_std', 0):.6f}",
                file=stream,
                flush=True,
            )
        stfp = (payload or {}).get("summary_test_fpr_macro") or {}
        if stfp:
            print(
                f"  {seg}:  test FPR (macro OvR) active {stfp.get('active_mean', float('nan')):.6f} ± {stfp.get('active_std', 0):.6f}  |  "
                f"passive {stfp.get('passive_mean', float('nan')):.6f} ± {stfp.get('passive_std', 0):.6f}",
                file=stream,
                flush=True,
            )
        stfn = (payload or {}).get("summary_test_fnr_macro") or {}
        if stfn:
            print(
                f"  {seg}:  test FNR (macro OvR) active {stfn.get('active_mean', float('nan')):.6f} ± {stfn.get('active_std', 0):.6f}  |  "
                f"passive {stfn.get('passive_mean', float('nan')):.6f} ± {stfn.get('passive_std', 0):.6f}",
                file=stream,
                flush=True,
            )
    print(f"\nAggregate JSON: {out_path}", file=stream, flush=True)
    print("=" * 72, file=stream, flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--final_root",
        type=str,
        default="final_user_identification",
        help="Directory containing 1s/, 2s/, 4s/ each with fold_k/best_model.pth",
    )
    p.add_argument("--hdf5_root_1s", type=str, default=None, help="Parent of fold_* for 1s checkpoints")
    p.add_argument("--hdf5_root_2s", type=str, default=None, help="Parent of fold_* for 2s checkpoints")
    p.add_argument("--hdf5_root_4s", type=str, default=None, help="Parent of fold_* for 4s checkpoints")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--eval_batch_size", type=int, default=288)
    p.add_argument("--analyze_batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--skip_analyze",
        action="store_true",
        help="Only run test active/passive accuracy (skip analyze_confidence / unknown)",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned commands without running",
    )
    p.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help=(
            "If set, write eval_run_<segment>_fold<k>.log (subprocess output + FOLD RESULT) and "
            "eval_run_<segment>.log (segment mean±std). See module docstring."
        ),
    )
    p.add_argument(
        "--save_predictions_dir",
        type=str,
        default=None,
        help=(
            "If set, pass to eval_active_passive_test.py as --save_predictions_dir "
            "<this path>/<segment>/ so each segment writes fold_XX_<segment>_<active|passive>.npz "
            "with per-sample y_true and y_pred."
        ),
    )
    return p.parse_args()


def _segment_hdf5_map(args: argparse.Namespace) -> Dict[str, str]:
    m = {}
    if args.hdf5_root_1s:
        m["1s"] = args.hdf5_root_1s
    if args.hdf5_root_2s:
        m["2s"] = args.hdf5_root_2s
    if args.hdf5_root_4s:
        m["4s"] = args.hdf5_root_4s
    return m


def _run(cmd: List[str], dry_run: bool, log_fp: Optional[TextIO] = None) -> None:
    line = "\n$ " + " ".join(cmd) + "\n"
    if dry_run:
        if log_fp is not None:
            log_fp.write(line)
            log_fp.flush()
        else:
            print(line.rstrip(), flush=True)
        return
    if log_fp is not None:
        log_fp.write(line)
        log_fp.flush()
        try:
            subprocess.run(
                cmd,
                check=True,
                cwd=str(_repo_root()),
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except subprocess.CalledProcessError:
            log_fp.flush()
            print(
                "Subprocess failed — traceback is at the end of the fold log file.",
                file=sys.stderr,
                flush=True,
            )
            raise
    else:
        print(line.rstrip(), flush=True)
        subprocess.run(cmd, check=True, cwd=str(_repo_root()))


def main() -> None:
    args = _parse_args()
    root = Path(args.final_root).resolve()
    if not root.is_dir():
        raise SystemExit(f"--final_root is not a directory: {root}")

    hdf5_map = _segment_hdf5_map(args)
    if not hdf5_map:
        raise SystemExit("Provide at least one of --hdf5_root_1s, --hdf5_root_2s, --hdf5_root_4s")

    log_dir_path = Path(args.log_dir).expanduser().resolve() if args.log_dir else None
    if log_dir_path is not None:
        log_dir_path.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    repo = _repo_root()
    eval_script = repo / "user_identification" / "eval_active_passive_test.py"
    analyze_script = repo / "user_identification" / "analyze_confidence.py"

    summary: Dict[str, Any] = {
        "final_root": str(root),
        "n_folds": args.n_folds,
        "segments": {},
    }

    for seg in SEGMENT_CHOICES:
        if seg not in hdf5_map:
            continue
        hdf5_parent = Path(hdf5_map[seg]).expanduser().resolve()
        seg_dir = root / seg
        if not seg_dir.is_dir():
            print(f"[skip] No checkpoint directory: {seg_dir}")
            continue

        seg_payload: Dict[str, Any] = {
            "hdf5_parent": str(hdf5_parent),
            "folds": {},
        }
        active_accs: List[float] = []
        passive_accs: List[float] = []
        active_f1_macro: List[float] = []
        passive_f1_macro: List[float] = []
        active_f1_weighted: List[float] = []
        passive_f1_weighted: List[float] = []
        active_fpr_macro: List[float] = []
        passive_fpr_macro: List[float] = []
        active_fnr_macro: List[float] = []
        passive_fnr_macro: List[float] = []

        for k in range(args.n_folds):
            fold_dir = seg_dir / f"fold_{k}"
            ckpt = fold_dir / "best_model.pth"
            hdf5_dir = hdf5_parent / f"fold_{k}"
            fold_key = f"fold_{k}"

            if not ckpt.is_file():
                print(f"[skip] {seg} {fold_key}: missing {ckpt}")
                continue
            if not hdf5_dir.is_dir():
                print(f"[skip] {seg} {fold_key}: missing hdf5 {hdf5_dir}")
                continue

            acc_json = fold_dir / "eval_task_type_test_accuracy.json"
            out_active = fold_dir / "eval_test_unknown_active"
            out_passive = fold_dir / "eval_test_unknown_passive"

            fold_log_path: Optional[Path] = None
            fold_fp: Optional[TextIO] = None
            if log_dir_path is not None:
                fold_log_path = log_dir_path / f"eval_run_{seg}_fold{k}.log"
                fold_fp = open(fold_log_path, "w", encoding="utf-8")
                fold_fp.write(f"# eval_run_{seg}_fold{k}.log\n")
                fold_fp.write(f"# checkpoint: {ckpt}\n# hdf5_dir: {hdf5_dir}\n\n")
                fold_fp.flush()

            try:
                # 1) Test accuracy: active vs passive (+ F1, FPR/FNR; optional per-sample labels)
                eval_cmd = [
                    py,
                    str(eval_script),
                    "--checkpoint",
                    str(ckpt),
                    "--hdf5_dir",
                    str(hdf5_dir),
                    "--segment_length",
                    seg,
                    "--device",
                    args.device,
                    "--batch_size",
                    str(args.eval_batch_size),
                    "--num_workers",
                    str(args.num_workers),
                    "--output_json",
                    str(acc_json),
                ]
                if args.save_predictions_dir:
                    pred_root = Path(args.save_predictions_dir).expanduser().resolve() / seg
                    eval_cmd.extend(["--save_predictions_dir", str(pred_root)])
                _run(
                    eval_cmd,
                    args.dry_run,
                    log_fp=fold_fp,
                )

                fold_entry: Dict[str, Any] = {
                    "checkpoint": str(ckpt),
                    "hdf5_dir": str(hdf5_dir),
                    "test_task_accuracy_json": str(acc_json),
                }

                acc_data = _load_json(acc_json) if not args.dry_run else None
                if acc_data and acc_data.get("folds"):
                    fr0 = acc_data["folds"][0]
                    fold_entry["test"] = {
                        "active": fr0.get("active"),
                        "passive": fr0.get("passive"),
                        "checkpoint_val_accuracy_percent": fr0.get(
                            "checkpoint_val_accuracy_percent"
                        ),
                        "checkpoint_test_accuracy_percent": fr0.get(
                            "checkpoint_test_accuracy_percent"
                        ),
                    }
                    if fr0.get("active"):
                        active_accs.append(float(fr0["active"]["accuracy_percent"]))
                        am = fr0["active"].get("f1_macro")
                        aw = fr0["active"].get("f1_weighted")
                        if isinstance(am, (int, float)) and am == am:
                            active_f1_macro.append(float(am))
                        if isinstance(aw, (int, float)) and aw == aw:
                            active_f1_weighted.append(float(aw))
                        afp = fr0["active"].get("fpr_macro")
                        afn = fr0["active"].get("fnr_macro")
                        if isinstance(afp, (int, float)) and afp == afp:
                            active_fpr_macro.append(float(afp))
                        if isinstance(afn, (int, float)) and afn == afn:
                            active_fnr_macro.append(float(afn))
                    if fr0.get("passive"):
                        passive_accs.append(float(fr0["passive"]["accuracy_percent"]))
                        pm = fr0["passive"].get("f1_macro")
                        pw = fr0["passive"].get("f1_weighted")
                        if isinstance(pm, (int, float)) and pm == pm:
                            passive_f1_macro.append(float(pm))
                        if isinstance(pw, (int, float)) and pw == pw:
                            passive_f1_weighted.append(float(pw))
                        pfp = fr0["passive"].get("fpr_macro")
                        pfn = fr0["passive"].get("fnr_macro")
                        if isinstance(pfp, (int, float)) and pfp == pfp:
                            passive_fpr_macro.append(float(pfp))
                        if isinstance(pfn, (int, float)) and pfn == pfn:
                            passive_fnr_macro.append(float(pfn))

                if not args.skip_analyze:
                    for task_t, out_dir in (("active", out_active), ("passive", out_passive)):
                        _run(
                            [
                                py,
                                str(analyze_script),
                                "--checkpoint",
                                str(ckpt),
                                "--hdf5_dir",
                                str(hdf5_dir),
                                "--segment_length",
                                seg,
                                "--split",
                                "both",
                                "--task_type",
                                task_t,
                                "--output_dir",
                                str(out_dir),
                                "--device",
                                args.device,
                                "--batch_size",
                                str(args.analyze_batch_size),
                                "--num_workers",
                                str(args.num_workers),
                            ],
                            args.dry_run,
                            log_fp=fold_fp,
                        )

                    fold_entry["analyze_output_dirs"] = {
                        "active": str(out_active),
                        "passive": str(out_passive),
                    }
                    if not args.dry_run:
                        oa = _load_json(out_active / "open_set_summary.json")
                        op = _load_json(out_passive / "open_set_summary.json")
                        fold_entry["unknown_open_set"] = {
                            "active": oa,
                            "passive": op,
                        }
                        fold_entry["unknown_summaries"] = {
                            "active": str(out_active / "unknown" / "unknown_confidence_summary.json"),
                            "passive": str(out_passive / "unknown" / "unknown_confidence_summary.json"),
                        }

                seg_payload["folds"][fold_key] = fold_entry
                if not args.dry_run:
                    report_stream: TextIO = fold_fp if fold_fp is not None else sys.stdout
                    _emit_fold_report(seg, k, fold_entry, stream=report_stream)
            finally:
                if fold_fp is not None:
                    fold_fp.close()
                    if fold_log_path is not None:
                        print(
                            f"[{seg} {fold_key}] detailed log: {fold_log_path}",
                            flush=True,
                        )

        oa_acc: List[float] = []
        op_acc: List[float] = []
        for fe in seg_payload["folds"].values():
            uo = fe.get("unknown_open_set") or {}
            a = uo.get("active") or {}
            p = uo.get("passive") or {}
            if isinstance(a.get("best_open_set_accuracy"), (int, float)):
                oa_acc.append(float(a["best_open_set_accuracy"]))
            if isinstance(p.get("best_open_set_accuracy"), (int, float)):
                op_acc.append(float(p["best_open_set_accuracy"]))
        if oa_acc:
            seg_payload["summary_open_set_accuracy"] = {
                "active_mean": float(sum(oa_acc) / len(oa_acc)),
                "active_std": _std_sample(oa_acc),
                "passive_mean": float(sum(op_acc) / len(op_acc)),
                "passive_std": _std_sample(op_acc),
                "n_folds_in_summary": len(oa_acc),
            }

        if active_accs:
            seg_payload["summary_test_accuracy_percent"] = {
                "active_mean": float(sum(active_accs) / len(active_accs)),
                "active_std": _std_sample(active_accs),
                "passive_mean": float(sum(passive_accs) / len(passive_accs)),
                "passive_std": _std_sample(passive_accs),
                "n_folds_in_summary": len(active_accs),
            }
        if active_f1_macro and passive_f1_macro:
            seg_payload["summary_test_f1_macro"] = {
                "active_mean": float(sum(active_f1_macro) / len(active_f1_macro)),
                "active_std": _std_sample(active_f1_macro),
                "passive_mean": float(sum(passive_f1_macro) / len(passive_f1_macro)),
                "passive_std": _std_sample(passive_f1_macro),
                "n_folds_in_summary": len(active_f1_macro),
            }
        if active_f1_weighted and passive_f1_weighted:
            seg_payload["summary_test_f1_weighted"] = {
                "active_mean": float(sum(active_f1_weighted) / len(active_f1_weighted)),
                "active_std": _std_sample(active_f1_weighted),
                "passive_mean": float(sum(passive_f1_weighted) / len(passive_f1_weighted)),
                "passive_std": _std_sample(passive_f1_weighted),
                "n_folds_in_summary": len(active_f1_weighted),
            }
        if active_fpr_macro and passive_fpr_macro:
            seg_payload["summary_test_fpr_macro"] = {
                "active_mean": float(sum(active_fpr_macro) / len(active_fpr_macro)),
                "active_std": _std_sample(active_fpr_macro),
                "passive_mean": float(sum(passive_fpr_macro) / len(passive_fpr_macro)),
                "passive_std": _std_sample(passive_fpr_macro),
                "n_folds_in_summary": len(active_fpr_macro),
            }
        if active_fnr_macro and passive_fnr_macro:
            seg_payload["summary_test_fnr_macro"] = {
                "active_mean": float(sum(active_fnr_macro) / len(active_fnr_macro)),
                "active_std": _std_sample(active_fnr_macro),
                "passive_mean": float(sum(passive_fnr_macro) / len(passive_fnr_macro)),
                "passive_std": _std_sample(passive_fnr_macro),
                "n_folds_in_summary": len(active_fnr_macro),
            }
        summary["segments"][seg] = seg_payload
        if not args.dry_run and seg_payload.get("folds"):
            if log_dir_path is not None:
                seg_log = log_dir_path / f"eval_run_{seg}.log"
                with open(seg_log, "w", encoding="utf-8") as sf:
                    _emit_segment_summary(seg, seg_payload, stream=sf)
                print(f"[{seg}] summary log: {seg_log}", flush=True)
            else:
                _emit_segment_summary(seg, seg_payload, stream=sys.stdout)

    out_path = root / "active_passive_test_unknown_eval_summary.json"
    if not args.dry_run:
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        n_seg = len(summary.get("segments") or {})
        if log_dir_path is not None and n_seg >= 1:
            if n_seg == 1:
                only_seg = next(iter(summary["segments"].keys()))
                overview_path = log_dir_path / f"eval_run_{only_seg}.log"
                with open(overview_path, "a", encoding="utf-8") as of:
                    of.write("\n")
                    _emit_final_overview(summary, out_path, stream=of)
                print(f"[overview] appended to {overview_path}", flush=True)
            else:
                overview_path = log_dir_path / "eval_run_overview.log"
                with open(overview_path, "w", encoding="utf-8") as of:
                    _emit_final_overview(summary, out_path, stream=of)
                print(f"[overview] wrote {overview_path}", flush=True)
        else:
            _emit_final_overview(summary, out_path, stream=sys.stdout)
    else:
        print("\n(dry_run: no summary file written)", flush=True)


if __name__ == "__main__":
    main()
