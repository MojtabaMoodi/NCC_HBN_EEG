#!/usr/bin/env python3
"""
Generate train and test manifests from HDF5 files for bootstrap evaluation.

Supports age classification (3-class bins), gender classification (binary), and
age regression (continuous years).
"""

from __future__ import annotations

import argparse
import glob
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import pandas as pd

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

_PROJECT_ROOT = _EVAL_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data_processing.constants import get_age_class, get_gender_class  # noqa: E402
from task_specs import TASK_IDS, get_task_spec  # noqa: E402


def _label_from_metadata(task_id: str, age: float, gender: float) -> float:
    spec = get_task_spec(task_id)
    if spec.manifest_label_kind == "age_bin":
        return float(get_age_class(age))
    if spec.manifest_label_kind == "gender":
        return float(get_gender_class(gender))
    if spec.manifest_label_kind == "age_years":
        return float(age)
    raise ValueError(f"Unhandled manifest_label_kind: {spec.manifest_label_kind}")


def extract_subjects_from_hdf5(
    hdf5_file: Path,
    task_id: str,
    task_type: str,
) -> Dict[str, Dict]:
    """
    Extract subject information from one HDF5 file.

    Returns mapping subject_id -> {label, segments: [(sample_id, hdf5_filename), ...]}
    """
    subjects: Dict[str, Dict] = defaultdict(lambda: {"label": None, "segments": []})

    with h5py.File(hdf5_file, "r") as f:
        task_types: List[str] = []
        if task_type in ("active", "both") and "active" in f:
            task_types.append("active")
        if task_type in ("passive", "both") and "passive" in f:
            task_types.append("passive")
        if not task_types:
            raise ValueError(
                f"No requested task groups found in {hdf5_file}. "
                f"task_type={task_type!r}, top-level keys={list(f.keys())}"
            )

        for tt in task_types:
            task_group = f[tt]
            for key in task_group.keys():
                if not key.startswith("sample_"):
                    continue

                sample_id = key
                if sample_id.startswith("sample_"):
                    metadata_key = sample_id.replace("sample_", "metadata_", 1)
                else:
                    sample_num = sample_id.split("_")[-1]
                    metadata_key = f"metadata_{sample_num}"

                if metadata_key not in task_group:
                    metadata_keys = [k for k in task_group.keys() if k.startswith("metadata_")]
                    raise KeyError(
                        f"Metadata key {metadata_key!r} not found for sample {sample_id!r} "
                        f"in {hdf5_file} group {tt!r}. "
                        f"Available metadata keys (first 5): {metadata_keys[:5]}"
                    )

                metadata = task_group[metadata_key]
                participant_id = metadata.attrs.get("participant_id", None)
                if participant_id is None:
                    raise KeyError(
                        f"participant_id missing in {hdf5_file}:{tt}/{metadata_key}"
                    )
                if isinstance(participant_id, bytes):
                    participant_id = participant_id.decode()
                participant_id = str(participant_id)

                age = metadata.attrs.get("age", None)
                gender = metadata.attrs.get("gender", None)
                if age is None or gender is None:
                    raise KeyError(
                        f"age/gender missing in {hdf5_file}:{tt}/{metadata_key} "
                        f"(age={age}, gender={gender})"
                    )

                label = _label_from_metadata(task_id, float(age), float(gender))
                hdf5_filename = hdf5_file.name

                if (
                    subjects[participant_id]["label"] is not None
                    and subjects[participant_id]["label"] != label
                ):
                    raise ValueError(
                        f"Subject {participant_id} has inconsistent labels in {hdf5_file}: "
                        f"{subjects[participant_id]['label']} vs {label}"
                    )

                subjects[participant_id]["label"] = label
                subjects[participant_id]["segments"].append((sample_id, hdf5_filename))

    return dict(subjects)


def _merge_subjects(
    all_subjects: Dict[str, Dict],
    subjects: Dict[str, Dict],
    source_file: Path,
) -> None:
    for subject_id, info in subjects.items():
        if subject_id in all_subjects:
            if all_subjects[subject_id]["label"] != info["label"]:
                raise ValueError(
                    f"Subject {subject_id} has inconsistent labels across HDF5 files "
                    f"({all_subjects[subject_id]['label']} vs {info['label']}) "
                    f"while merging {source_file}"
                )
            all_subjects[subject_id]["segments"].extend(info["segments"])
        else:
            all_subjects[subject_id] = {
                "label": info["label"],
                "segments": list(info["segments"]),
            }


def generate_train_manifest(
    hdf5_files: List[Path],
    output_path: Path,
    task_id: str,
    task_type: str,
) -> None:
    all_subjects: Dict[str, Dict] = {}
    for hdf5_file in hdf5_files:
        print(f"Processing {hdf5_file.name}...")
        subjects = extract_subjects_from_hdf5(hdf5_file, task_id, task_type)
        _merge_subjects(all_subjects, subjects, hdf5_file)

    rows = [
        {"subject_id": subject_id, "label": info["label"]}
        for subject_id, info in all_subjects.items()
    ]
    df = pd.DataFrame(rows).sort_values("subject_id")
    df.to_csv(output_path, index=False)

    print(f"\nGenerated train manifest: {output_path}")
    print(f"  Subjects: {len(df)}")
    print("  Label distribution:")
    print(df["label"].value_counts().sort_index())


def generate_test_manifest(
    hdf5_files: List[Path],
    output_path: Path,
    task_id: str,
    task_type: str,
) -> None:
    all_segments = []
    for hdf5_file in hdf5_files:
        print(f"Processing {hdf5_file.name}...")
        subjects = extract_subjects_from_hdf5(hdf5_file, task_id, task_type)
        for subject_id, info in subjects.items():
            for sample_id, hdf5_path in info["segments"]:
                all_segments.append(
                    {
                        "subject_id": subject_id,
                        "label": info["label"],
                        "hdf5_file": hdf5_path,
                        "sample_id": sample_id,
                    }
                )

    df = pd.DataFrame(all_segments).sort_values(["subject_id", "sample_id"])
    df.to_csv(output_path, index=False)

    print(f"\nGenerated test manifest: {output_path}")
    print(f"  Segments: {len(df)}")
    print(f"  Subjects: {df['subject_id'].nunique()}")
    print("  Label distribution:")
    print(df["label"].value_counts().sort_index())

    inconsistent = [
        subject_id
        for subject_id, group in df.groupby("subject_id")
        if group["label"].nunique() > 1
    ]
    if inconsistent:
        raise ValueError(
            f"Test manifest has {len(inconsistent)} subjects with inconsistent labels. "
            f"First examples: {inconsistent[:10]}"
        )


def find_hdf5_files(hdf5_dir: Path, split: str, segment_length: str, pattern: str | None) -> List[Path]:
    if pattern:
        matched = [Path(f) for f in glob.glob(str(hdf5_dir / pattern))]
    else:
        matched = list(hdf5_dir.glob(f"eeg_data_{split}_{segment_length}*.h5"))

    if not matched:
        raise FileNotFoundError(
            f"No HDF5 files found for split={split!r} segment_length={segment_length!r} "
            f"in {hdf5_dir}. pattern={pattern!r}"
        )
    return sorted(matched)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate train and test manifests from HDF5 files"
    )
    parser.add_argument("--hdf5_dir", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"])
    parser.add_argument("--segment_length", type=str, required=True, choices=["1s", "2s", "4s"])
    parser.add_argument("--task", type=str, required=True, choices=sorted(TASK_IDS))
    parser.add_argument("--task_type", type=str, required=True, choices=["active", "passive", "both"])
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--pattern", type=str, default=None)

    args = parser.parse_args()
    get_task_spec(args.task)

    hdf5_dir = Path(args.hdf5_dir)
    if not hdf5_dir.is_dir():
        raise FileNotFoundError(f"HDF5 directory not found: {hdf5_dir}")

    hdf5_files = find_hdf5_files(hdf5_dir, args.split, args.segment_length, args.pattern)
    print(f"Found {len(hdf5_files)} HDF5 file(s)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.split == "train":
        generate_train_manifest(hdf5_files, output_path, args.task, args.task_type)
    else:
        generate_test_manifest(hdf5_files, output_path, args.task, args.task_type)

    print(f"\nManifest saved to: {output_path}")


if __name__ == "__main__":
    main()
