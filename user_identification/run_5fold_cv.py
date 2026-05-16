#!/usr/bin/env python3
"""
Run 5-fold (or N-fold) cross-validation for user identification.

Expects data prepared with data_processing/preprocess_user_identification.py --n_folds 5
(each fold in hdf5_root/fold_0, hdf5_root/fold_1, ...). Trains one model per fold,
then aggregates val/test accuracy into mean ± std and other statistics.

``--train_script`` (optional): path to the per-fold training script. If omitted, defaults to
``user_identification/train_labram_arcface.py``. Pass e.g.
``--train_script user_identification/cnn_resnet_arcface/train_user_identification_backbone.py``
for CNN/ResNet + ArcFace training. All arguments after ``run_5fold_cv.py``'s own flags are forwarded
to that script (parsed with ``parse_known_args``), then ``--hdf5_dir`` and ``--output_dir`` are
appended for each fold. Example::

  python user_identification/run_5fold_cv.py \\
    --hdf5_root ~/scratch/1s \\
    --output_root final_user_identification/resnet18_1s \\
    --n_folds 5 \\
    --train_script user_identification/cnn_resnet_arcface/train_user_identification_backbone.py \\
    --segment_length 1s --epochs 200 --seed 42 --backbone resnet18 --batch_size 512

Per-fold logging: for each fold *k*, stdout and stderr of the training subprocess are written to
``output_root/fold_k/train_fold_k.log`` (the log header includes the exact command and start time).

Two workflows:

  A) Single machine: run all folds and aggregate in one go:
     python user_identification/run_5fold_cv.py \\
       --hdf5_root /path/to/5fold_preprocessed \\
       --output_root /path/to/results_5fold \\
       --n_folds 5 \\
       [train_labram_arcface.py args...]

  B) Distributed: train each fold on a separate machine, then aggregate once.
     - Prepare data once: preprocess_user_identification.py --n_folds 5 --output_dir /path/to/user_id_5fold
     - On machine for fold k: copy fold_k data, then:
       python user_identification/train_labram_arcface.py \\
         --hdf5_dir /path/to/fold_k \\
         --output_dir /path/to/results_5fold/fold_k \\
         [other args...]
     - After all folds are done, copy results_5fold/fold_0..fold_4 to one place and run:
       python user_identification/run_5fold_cv.py \\
         --output_root /path/to/results_5fold \\
         --n_folds 5 \\
         --aggregate_only
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _parse_our_args(argv):
    """Parse --hdf5_root, --output_root, --n_folds, --aggregate_only; return (parsed, rest_argv)."""
    parser = argparse.ArgumentParser(description='5-fold CV for user identification')
    parser.add_argument('--hdf5_root', type=str, default=None,
                        help='Root directory containing fold_0, fold_1, ... (required unless --aggregate_only)')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Root directory where fold results are/will be (output_root/fold_0, ...)')
    parser.add_argument('--n_folds', type=int, required=True,
                        help='Number of folds (e.g. 5)')
    parser.add_argument('--aggregate_only', action='store_true',
                        help='Only aggregate existing results from output_root/fold_*/results.json; do not train')
    parser.add_argument(
        '--train_script',
        type=str,
        default=None,
        help=(
            'Path to per-fold training script (default: train_labram_arcface.py). '
            'Use e.g. user_identification/cnn_resnet_arcface/train_user_identification_backbone.py '
            'for CNN/ResNet ArcFace training.'
        ),
    )
    args, rest = parser.parse_known_args(argv)
    return args, rest


def aggregate_fold_results(output_root: Path, n_folds: int) -> dict:
    """
    Read results.json from output_root/fold_0 .. fold_{n_folds-1}, compute mean ± std.
    Returns summary dict and raises if any results.json is missing.
    """
    results_list = []
    for k in range(n_folds):
        results_path = output_root / f"fold_{k}" / "results.json"
        if not results_path.exists():
            raise FileNotFoundError(
                f"Results not found for fold {k}: {results_path}. "
                "Run train_labram_arcface.py for this fold or ensure results are copied here."
            )
        with open(results_path) as f:
            results_list.append(json.load(f))

    val_accs = [r["val_accuracy"] for r in results_list]
    test_accs = [r["test_accuracy"] for r in results_list]
    n = len(results_list)
    mean_val = sum(val_accs) / n
    mean_test = sum(test_accs) / n
    if n > 1:
        std_val_sample = (sum((x - mean_val) ** 2 for x in val_accs) / (n - 1)) ** 0.5
        std_test_sample = (sum((x - mean_test) ** 2 for x in test_accs) / (n - 1)) ** 0.5
    else:
        std_val_sample = 0.0
        std_test_sample = 0.0

    return {
        "n_folds": n_folds,
        "val_accuracy_mean": mean_val,
        "val_accuracy_std": std_val_sample,
        "val_accuracy_per_fold": val_accs,
        "test_accuracy_mean": mean_test,
        "test_accuracy_std": std_test_sample,
        "test_accuracy_per_fold": test_accs,
    }


def main():
    project_root = Path(__file__).resolve().parent.parent
    our_args, rest_argv = _parse_our_args(sys.argv[1:])
    hdf5_root = Path(our_args.hdf5_root) if our_args.hdf5_root else None
    output_root = Path(our_args.output_root)
    n_folds = our_args.n_folds
    aggregate_only = our_args.aggregate_only

    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    if aggregate_only:
        if our_args.hdf5_root:
            raise ValueError("Do not pass --hdf5_root when using --aggregate_only")
        output_root.mkdir(parents=True, exist_ok=True)
        summary = aggregate_fold_results(output_root, n_folds)
        summary_path = output_root / "cv_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        val_accs = summary["val_accuracy_per_fold"]
        test_accs = summary["test_accuracy_per_fold"]
        print(f"\n{'='*60}")
        print("5-FOLD CROSS-VALIDATION SUMMARY (aggregate only)")
        print(f"{'='*60}")
        print(f"  Val accuracy:  {summary['val_accuracy_mean']:.4f}% ± {summary['val_accuracy_std']:.4f}%  (per fold: {[f'{x:.2f}' for x in val_accs]})")
        print(f"  Test accuracy: {summary['test_accuracy_mean']:.4f}% ± {summary['test_accuracy_std']:.4f}%  (per fold: {[f'{x:.2f}' for x in test_accs]})")
        print(f"  Summary saved to: {summary_path}")
        print(f"{'='*60}\n")
        return

    # Training mode: require hdf5_root
    if hdf5_root is None or not hdf5_root.is_dir():
        raise FileNotFoundError(
            f"hdf5_root is required and must be a directory: {our_args.hdf5_root}. "
            "Prepare data with: data_processing/preprocess_user_identification.py --n_folds N --output_dir <hdf5_root>"
        )
    for k in range(n_folds):
        fold_data = hdf5_root / f"fold_{k}"
        if not fold_data.is_dir():
            raise FileNotFoundError(
                f"Fold directory not found: {fold_data}. "
                f"Prepare data with: data_processing/preprocess_user_identification.py --n_folds {n_folds} --output_dir {hdf5_root}"
            )

    output_root.mkdir(parents=True, exist_ok=True)
    if our_args.train_script:
        train_script = Path(our_args.train_script)
        if not train_script.is_file():
            raise FileNotFoundError(f"--train_script is not a file: {train_script}")
    else:
        train_script = project_root / "user_identification" / "train_labram_arcface.py"

    for k in range(n_folds):
        fold_hdf5 = hdf5_root / f"fold_{k}"
        fold_output = output_root / f"fold_{k}"
        fold_output.mkdir(parents=True, exist_ok=True)
        cmd = (
            [sys.executable, str(train_script)]
            + rest_argv
            + ["--hdf5_dir", str(fold_hdf5), "--output_dir", str(fold_output)]
        )
        print(f"\n{'='*60}")
        print(f"Fold {k + 1}/{n_folds}: --hdf5_dir {fold_hdf5} --output_dir {fold_output}")
        print(f"{'='*60}\n")
        # Persist per-fold logs (stdout + stderr) for easier debugging/resume.
        fold_log_path = fold_output / f"train_fold_{k}.log"
        with open(fold_log_path, "w", encoding="utf-8") as fp:
            fp.write(f"# Command: {' '.join(cmd)}\n")
            fp.write(f"# CWD: {project_root}\n")
            fp.write(f"# Start (UTC): {datetime.now(timezone.utc).isoformat()}\n")
            fp.flush()
            ret = subprocess.run(
                cmd,
                cwd=str(project_root),
                stdout=fp,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if ret.returncode != 0:
            raise RuntimeError(
                f"Training failed for fold {k} with exit code {ret.returncode}. "
                f"Command: {' '.join(cmd)}"
            )

    summary = aggregate_fold_results(output_root, n_folds)
    summary_path = output_root / "cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("5-FOLD CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Val accuracy:  {summary['val_accuracy_mean']:.4f}% ± {summary['val_accuracy_std']:.4f}%  (per fold: {[f'{x:.2f}' for x in summary['val_accuracy_per_fold']]})")
    print(f"  Test accuracy: {summary['test_accuracy_mean']:.4f}% ± {summary['test_accuracy_std']:.4f}%  (per fold: {[f'{x:.2f}' for x in summary['test_accuracy_per_fold']]})")
    print(f"  Summary saved to: {summary_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
