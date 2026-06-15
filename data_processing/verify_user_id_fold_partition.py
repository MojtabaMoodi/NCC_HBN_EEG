#!/usr/bin/env python3
"""
Verify user-ID N-fold preprocessing outputs under output_dir/fold_*.

Reads each fold's participant_id_to_class_idx.json (known) and
unknown_participant_ids.json (held-out unknown). Checks:

  - Per fold: known and unknown IDs are disjoint.
  - All folds cover the same participant universe (known | unknown).
  - Across folds: unknown sets are disjoint (no ID is "unknown" in two folds).
  - Union of per-fold unknown sets equals that universe (each participant is
    unknown in exactly one fold — matches preprocess_user_identification design).

Run from project root:

  python data_processing/verify_user_id_fold_partition.py \\
    --output_dir /path/to/user_id_5fold --n_folds 5

Optional: also compare to the partition recomputed from raw data (must match
preprocessing --data_root / --random_seed / --n_folds):

  python data_processing/verify_user_id_fold_partition.py \\
    --output_dir /path/to/user_id_5fold --n_folds 5 \\
    --data_root /path/to/preprocessed_new --random_seed 42

Exit 0 if checks pass; non-zero otherwise. Does not read HDF5 segment files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Set, Tuple


def _load_fold_jsons(fold_dir: Path) -> Tuple[Set[str], Set[str]]:
    mapping_path = fold_dir / "participant_id_to_class_idx.json"
    unknown_path = fold_dir / "unknown_participant_ids.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Missing {mapping_path}")
    with open(mapping_path, encoding="utf-8") as f:
        mapping: Dict[str, int] = json.load(f)
    known = set(mapping.keys())
    if not unknown_path.is_file():
        raise FileNotFoundError(
            f"Missing {unknown_path} (expected for n_folds / explicit unknown split)."
        )
    with open(unknown_path, encoding="utf-8") as f:
        unknown_list = json.load(f)
    if not isinstance(unknown_list, list):
        raise ValueError(f"{unknown_path} must be a JSON list of participant IDs")
    unknown = set(unknown_list)
    if len(unknown) != len(unknown_list):
        raise ValueError(f"{unknown_path} contains duplicate participant IDs")
    return known, unknown


def _verify_output_dir(output_dir: Path, n_folds: int) -> Dict[str, Set[str]]:
    """
    Validate fold_* JSONs; return dict fold_key -> unknown set for optional source check.
    """
    per_fold_unknown: Dict[str, Set[str]] = {}
    universes: list[Set[str]] = []

    for k in range(n_folds):
        fold_dir = output_dir / f"fold_{k}"
        if not fold_dir.is_dir():
            raise FileNotFoundError(f"Expected directory missing: {fold_dir}")
        known, unknown = _load_fold_jsons(fold_dir)
        overlap = known & unknown
        if overlap:
            raise RuntimeError(
                f"fold_{k}: participant(s) in both mapping and unknown list: "
                f"{sorted(overlap)[:5]!r}{'...' if len(overlap) > 5 else ''}"
            )
        universe = known | unknown
        universes.append(universe)
        per_fold_unknown[f"fold_{k}"] = unknown

    u0 = universes[0]
    for k in range(1, n_folds):
        if universes[k] != u0:
            diff = u0.symmetric_difference(universes[k])
            sample = next(iter(diff))
            raise RuntimeError(
                f"fold_0 vs fold_{k}: participant universe differs (example id: {sample!r})"
            )

    all_unknown_union: Set[str] = set()
    for k in range(n_folds):
        uk = per_fold_unknown[f"fold_{k}"]
        inter = all_unknown_union & uk
        if inter:
            raise RuntimeError(
                f"Participant(s) marked unknown in more than one fold: "
                f"{sorted(inter)[:5]!r}{'...' if len(inter) > 5 else ''}"
            )
        all_unknown_union |= uk

    if all_unknown_union != u0:
        missing = u0 - all_unknown_union
        extra = all_unknown_union - u0
        raise RuntimeError(
            "Union of per-fold unknown sets does not match fold universe: "
            f"missing_from_union={len(missing)}, unexpected={len(extra)} "
            f"(example missing: {next(iter(missing), None)!r})"
        )

    return per_fold_unknown


def _verify_matches_source(
    output_dir: Path,
    n_folds: int,
    data_root: str,
    random_seed: int,
    per_fold_unknown: Dict[str, Set[str]],
) -> None:
    from eeg_data_preprocessing import EEGDataPreprocessor
    from preprocess_user_identification import _partition_into_folds

    dummy_out = output_dir / ".verify_partition_scratch_unused"
    pre = EEGDataPreprocessor(data_root, str(dummy_out), window_sizes=["4s"])
    all_dirs = pre.get_all_participant_dirs()
    if not all_dirs:
        raise ValueError(f"No participants under data_root={data_root!r}")
    fold_unknown_paths = _partition_into_folds(all_dirs, n_folds, random_seed)
    for k in range(n_folds):
        expected = {p.name for p in fold_unknown_paths[k]}
        got = per_fold_unknown[f"fold_{k}"]
        if expected != got:
            only_exp = expected - got
            only_got = got - expected
            raise RuntimeError(
                f"fold_{k} unknown_participant_ids.json does not match recomputed partition from "
                f"data_root + random_seed: only_in_source={sorted(list(only_exp))[:5]!r}, "
                f"only_in_json={sorted(list(only_got))[:5]!r}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory containing fold_0, fold_1, ... (preprocessing output)",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=None,
        help="Number of folds (default: infer from fold_* subdirectories under output_dir)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="If set with --random_seed, assert JSON unknown sets match _partition_into_folds from source",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for optional --data_root cross-check (default: 42)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        print(f"output_dir is not a directory: {output_dir}", file=sys.stderr)
        return 1

    if args.n_folds is not None:
        n_folds = args.n_folds
        if n_folds < 2:
            print("n_folds must be >= 2", file=sys.stderr)
            return 2
    else:
        fold_dirs = sorted(
            d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")
        )
        indices = []
        for d in fold_dirs:
            suffix = d.name[len("fold_") :]
            if suffix.isdigit():
                indices.append(int(suffix))
        if not indices:
            print(f"No fold_* directories under {output_dir}", file=sys.stderr)
            return 1
        n_folds = max(indices) + 1
        expected = {f"fold_{k}" for k in range(n_folds)}
        actual = {d.name for d in fold_dirs}
        if not expected.issubset(actual):
            missing = expected - actual
            print(f"Inferred n_folds={n_folds} but missing: {sorted(missing)}", file=sys.stderr)
            return 1

    try:
        per_fold_unknown = _verify_output_dir(output_dir, n_folds)
    except (OSError, ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"VERIFY FAILED: {e}", file=sys.stderr)
        return 1

    print("OK: fold_* JSONs are consistent (disjoint unknowns, full cover, same universe per fold).")
    print(f"  output_dir: {output_dir}")
    print(f"  n_folds: {n_folds}")
    k0_known, k0_unknown = _load_fold_jsons(output_dir / "fold_0")
    print(f"  |universe| = {len(k0_known | k0_unknown)} (from fold_0 known ∪ unknown)")
    for k in range(n_folds):
        uk = per_fold_unknown[f"fold_{k}"]
        print(f"  fold_{k} |unknown| = {len(uk)}")

    if args.data_root is not None:
        try:
            _verify_matches_source(
                output_dir, n_folds, args.data_root, args.random_seed, per_fold_unknown
            )
        except (OSError, ValueError, RuntimeError, FileNotFoundError) as e:
            print(f"SOURCE CROSS-CHECK FAILED: {e}", file=sys.stderr)
            return 1
        print("OK: per-fold unknown IDs match recomputed partition from --data_root / --random_seed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
