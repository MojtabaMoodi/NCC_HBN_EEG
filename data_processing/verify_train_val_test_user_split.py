#!/usr/bin/env python3
"""
Temporary script to verify that participants (users) in the processed EEG HDF5 data
are mutually exclusive across train, validation, and test splits.

- No user in training may appear in validation or test.
- No user in validation may appear in training or test.
- No user in test may appear in training or validation.

Reads only metadata (participant_id) from each split's HDF5 file(s); does not load EEG data.
Supports both single-file and multi-part HDF5 layout.

Usage:
    conda activate eeg_env
    python verify_train_val_test_user_split.py
    python verify_train_val_test_user_split.py --data_dir /path/to/processed_eeg_data_hdf5
"""

import argparse
import sys
from pathlib import Path

try:
    import h5py
except ImportError:
    print("Error: h5py is required. Install with: pip install h5py", file=sys.stderr)
    sys.exit(2)

TASK_TYPES = ("active", "passive")
SEGMENT_LENGTHS = ("1s", "2s", "4s")
SPLIT_NAMES = ("train", "val", "test")


def _get_split_file_paths(hdf5_dir: Path, segment_length: str, split_name: str) -> list[Path]:
    """Return sorted list of HDF5 paths for a split (multi-part or single file)."""
    part_files = sorted(hdf5_dir.glob(f"eeg_data_{split_name}_{segment_length}_part*.h5"))
    if part_files:
        return part_files
    single = hdf5_dir / f"eeg_data_{split_name}_{segment_length}.h5"
    if single.exists():
        return [single]
    return []


def collect_participant_ids_from_file(hdf5_path: Path) -> set[str]:
    """Collect unique participant_id from all metadata_* groups in one HDF5 file."""
    ids = set()
    with h5py.File(hdf5_path, "r") as f:
        for group_name in TASK_TYPES:
            if group_name not in f:
                continue
            group = f[group_name]
            for key in group.keys():
                if not key.startswith("metadata_"):
                    continue
                meta = group[key]
                if "participant_id" not in meta.attrs:
                    continue
                pid = meta.attrs["participant_id"]
                if isinstance(pid, bytes):
                    pid = pid.decode()
                ids.add(str(pid))
    return ids


def collect_participant_ids_for_split(hdf5_dir: Path, segment_length: str, split_name: str) -> set[str]:
    """Collect unique participant_id across all part files for one split."""
    paths = _get_split_file_paths(hdf5_dir, segment_length, split_name)
    all_ids = set()
    for p in paths:
        all_ids |= collect_participant_ids_from_file(p)
    return all_ids


def main():
    parser = argparse.ArgumentParser(
        description="Verify train/val/test have no overlapping participants in processed EEG HDF5 data."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/mojtabam/scratch/processed_eeg_data_hdf5",
        help="Path to directory containing eeg_data_*_1s_part*.h5 (and 2s, 4s) files",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Error: Data directory does not exist or is not a directory: {data_dir}", file=sys.stderr)
        sys.exit(2)

    any_violation = False

    for segment_length in SEGMENT_LENGTHS:
        train_paths = _get_split_file_paths(data_dir, segment_length, "train")
        val_paths = _get_split_file_paths(data_dir, segment_length, "val")
        test_paths = _get_split_file_paths(data_dir, segment_length, "test")

        if not train_paths or not val_paths or not test_paths:
            print(f"[{segment_length}] Skipping: missing split files (train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)})")
            continue

        train_ids = collect_participant_ids_for_split(data_dir, segment_length, "train")
        val_ids = collect_participant_ids_for_split(data_dir, segment_length, "val")
        test_ids = collect_participant_ids_for_split(data_dir, segment_length, "test")

        train_val = train_ids & val_ids
        train_test = train_ids & test_ids
        val_test = val_ids & test_ids

        print(f"\n--- Segment length: {segment_length} ---")
        print(f"  Train participants: {len(train_ids)}")
        print(f"  Val   participants: {len(val_ids)}")
        print(f"  Test  participants: {len(test_ids)}")

        if train_val or train_test or val_test:
            any_violation = True
            if train_val:
                print(f"  VIOLATION: {len(train_val)} participant(s) in BOTH train and val: {sorted(train_val)[:10]}{'...' if len(train_val) > 10 else ''}")
            if train_test:
                print(f"  VIOLATION: {len(train_test)} participant(s) in BOTH train and test: {sorted(train_test)[:10]}{'...' if len(train_test) > 10 else ''}")
            if val_test:
                print(f"  VIOLATION: {len(val_test)} participant(s) in BOTH val and test: {sorted(val_test)[:10]}{'...' if len(val_test) > 10 else ''}")
        else:
            print("  OK: Train, val, and test have mutually exclusive participants.")

    print()
    if any_violation:
        print("Result: FAIL — at least one split has overlapping participants.")
        sys.exit(1)
    print("Result: PASS — train, val, and test are mutually exclusive by participant.")
    sys.exit(0)


if __name__ == "__main__":
    main()
