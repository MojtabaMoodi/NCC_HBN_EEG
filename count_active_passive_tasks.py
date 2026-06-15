#!/usr/bin/env python3
"""
Temporary script: count active and passive task samples per segment length (1s, 2s, 4s)
in the processed EEG HDF5 directory.

Usage (use conda env with h5py, e.g. eeg_env):
  conda activate eeg_env
  python count_active_passive_tasks.py ~/scratch/processed_eeg_data_hdf5
"""
import argparse
from pathlib import Path

import h5py

SEGMENT_LENGTHS = ("1s", "2s", "4s")
SPLIT_NAMES = ("train", "val", "test", "unknown")


def count_samples_in_group(group) -> int:
    """Count datasets that are samples (key.startswith('sample_'))."""
    if group is None:
        return 0
    return sum(1 for key in group.keys() if key.startswith("sample_"))


def count_hdf5_dir(hdf5_dir: Path) -> dict:
    """
    Scan hdf5_dir for eeg_data_*.h5 files and count active/passive per segment length.
    Returns: { segment_length: { 'active': int, 'passive': int, 'by_split': { split: {'active': int, 'passive': int} } } }
    """
    hdf5_dir = Path(hdf5_dir).expanduser().resolve()
    if not hdf5_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {hdf5_dir}")

    # Aggregate: per segment_length -> active, passive, and per-split breakdown
    result = {
        seg: {"active": 0, "passive": 0, "by_split": {s: {"active": 0, "passive": 0} for s in SPLIT_NAMES}}
        for seg in SEGMENT_LENGTHS
    }

    h5_files = sorted(hdf5_dir.glob("eeg_data_*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No eeg_data_*.h5 files found in {hdf5_dir}")

    for p in h5_files:
        try:
            with h5py.File(p, "r") as f:
                seg_len = f.attrs.get("segment_length", None)
                split_name = f.attrs.get("split_name", None)
                if seg_len not in SEGMENT_LENGTHS:
                    continue
                if split_name is None:
                    split_name = "train"  # fallback for old files without attribute
                by_split = result[seg_len]["by_split"]
                if split_name not in by_split:
                    by_split[split_name] = {"active": 0, "passive": 0}

                ag = f.get("active")
                pg = f.get("passive")
                a = count_samples_in_group(ag)
                b = count_samples_in_group(pg)

                result[seg_len]["active"] += a
                result[seg_len]["passive"] += b
                by_split[split_name]["active"] += a
                by_split[split_name]["passive"] += b
        except Exception as e:
            raise RuntimeError(f"Error reading {p}: {e}") from e

    return result


def main():
    parser = argparse.ArgumentParser(description="Count active/passive task samples per 1s/2s/4s in HDF5 dir")
    parser.add_argument(
        "hdf5_dir",
        type=str,
        default="~/scratch/processed_eeg_data_hdf5",
        nargs="?",
        help="Path to processed EEG HDF5 directory (default: ~/scratch/processed_eeg_data_hdf5)",
    )
    args = parser.parse_args()
    hdf5_dir = Path(args.hdf5_dir).expanduser().resolve()

    counts = count_hdf5_dir(hdf5_dir)

    print(f"HDF5 directory: {hdf5_dir}\n")
    print("Total samples by segment length (active / passive):")
    print("-" * 60)
    for seg in SEGMENT_LENGTHS:
        r = counts[seg]
        total = r["active"] + r["passive"]
        print(f"  {seg}:  active = {r['active']:,}   passive = {r['passive']:,}   total = {total:,}")
    print()
    print("By split (active / passive):")
    print("-" * 60)
    for seg in SEGMENT_LENGTHS:
        print(f"  [{seg}]")
        by_split = counts[seg]["by_split"]
        for split in sorted(by_split.keys()):
            s = by_split[split]
            if s["active"] == 0 and s["passive"] == 0:
                continue
            print(f"    {split}:  active = {s['active']:,}   passive = {s['passive']:,}")


if __name__ == "__main__":
    main()
