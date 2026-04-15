"""Scan unknown-split HDF5 files and build deterministic participant/sample splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Set, Tuple

import h5py
import numpy as np

from data_processing.eeg_dataset import EEGDataLoader

TaskFilter = Literal["active", "passive"]


@dataclass(frozen=True)
class UnknownSampleRef:
    """Single segment in unknown HDF5."""

    hdf5_path: str
    task_type: str
    sample_key: str
    participant_id: str


def _normalize_unknown_paths(hdf5_dir: Path, segment_length: str) -> List[Path]:
    raw = EEGDataLoader._get_unknown_hdf5_file_paths(hdf5_dir, segment_length)
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [Path(p) for p in raw]
    return [Path(raw)]


def list_unknown_samples(
    hdf5_dir: str | Path,
    segment_length: str,
    task_filter: TaskFilter,
) -> List[UnknownSampleRef]:
    """
    List all samples in the unknown split for one task (active or passive).

    Raises:
        FileNotFoundError: If hdf5_dir is missing.
        ValueError: If no unknown HDF5 exists or the task group is empty.
    """
    root = Path(hdf5_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"hdf5_dir is not a directory: {root}")

    paths = _normalize_unknown_paths(root, segment_length)
    if not paths:
        raise ValueError(
            f"No unknown HDF5 for segment_length={segment_length!r} under {root}. "
            "Expected eeg_data_unknown_<seg>.h5 or eeg_data_unknown_<seg>_part*.h5."
        )

    out: List[UnknownSampleRef] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Unknown HDF5 missing: {path}")
        with h5py.File(path, "r") as f:
            if task_filter not in f:
                continue
            group = f[task_filter]
            sample_keys = sorted(k for k in group.keys() if k.startswith("sample_"))
            for sk in sample_keys:
                meta_key = sk.replace("sample_", "metadata_", 1)
                if meta_key not in group:
                    raise KeyError(f"{path}: metadata {meta_key!r} missing for {sk!r}")
                meta = group[meta_key]
                pid_attr = meta.attrs["participant_id"]
                pid = pid_attr.decode() if isinstance(pid_attr, bytes) else str(pid_attr)
                out.append(
                    UnknownSampleRef(
                        hdf5_path=str(path.resolve()),
                        task_type=task_filter,
                        sample_key=sk,
                        participant_id=pid,
                    )
                )

    if not out:
        raise ValueError(
            f"No samples found for task={task_filter!r} in unknown split under {root}. "
            "Check preprocessing and task_filter."
        )
    return out


def split_unknown_participants_for_registration(
    samples: Sequence[UnknownSampleRef],
    *,
    register_fraction_of_unknown: float,
    enrollment_fraction_per_user: float,
    seed: int,
    min_samples_per_participant: int = 2,
) -> Tuple[
    Set[str],
    Set[str],
    Dict[str, List[UnknownSampleRef]],
    Dict[str, List[UnknownSampleRef]],
    List[UnknownSampleRef],
]:
    """
    Split unknown participants into registered vs unregistered, then enrollment vs probe for registered.

    - ``register_fraction_of_unknown``: fraction of *unknown participants* who are "registered"
      (e.g. 0.5 when the unknown pool is 20% of all users → 10% of all users registered).
    - ``enrollment_fraction_per_user``: fraction of each registered user's segments used to build
      the mean embedding; the rest are probe segments.

    Enrollment/probe split is deterministic: samples are sorted by (path, task, sample_key) then
    the first floor(n * enrollment_fraction_per_user) segments go to enrollment.

    Raises:
        ValueError: On invalid fractions or too few samples per participant.
    """
    if not (0.0 < register_fraction_of_unknown < 1.0):
        raise ValueError(
            f"register_fraction_of_unknown must be in (0, 1), got {register_fraction_of_unknown!r}"
        )
    if not (0.0 < enrollment_fraction_per_user < 1.0):
        raise ValueError(
            f"enrollment_fraction_per_user must be in (0, 1), got {enrollment_fraction_per_user!r}"
        )

    by_pid: Dict[str, List[UnknownSampleRef]] = {}
    for s in samples:
        by_pid.setdefault(s.participant_id, []).append(s)

    for pid, lst in by_pid.items():
        lst.sort(key=lambda r: (r.hdf5_path, r.task_type, r.sample_key))

    participants = sorted(by_pid.keys())
    if len(participants) < 2:
        raise ValueError(
            f"Need at least 2 unknown participants to split registered vs unregistered; got {len(participants)}"
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(participants))
    n_reg = int(round(len(participants) * register_fraction_of_unknown))
    n_reg = max(1, min(n_reg, len(participants) - 1))

    registered = {participants[int(i)] for i in perm[:n_reg]}
    unregistered = {participants[int(i)] for i in perm[n_reg:]}

    enroll: Dict[str, List[UnknownSampleRef]] = {}
    probe: Dict[str, List[UnknownSampleRef]] = {}
    unreg_samples: List[UnknownSampleRef] = []

    for pid in registered:
        items = by_pid[pid]
        if len(items) < min_samples_per_participant:
            raise ValueError(
                f"Participant {pid!r} has only {len(items)} segment(s); need at least "
                f"{min_samples_per_participant} to split enrollment vs probe."
            )
        n_enroll = int(np.floor(len(items) * enrollment_fraction_per_user))
        if n_enroll < 1:
            n_enroll = 1
        if n_enroll >= len(items):
            raise ValueError(
                f"Participant {pid!r}: enrollment leaves no probe segments "
                f"(n={len(items)}, enrollment_fraction={enrollment_fraction_per_user})."
            )
        enroll[pid] = items[:n_enroll]
        probe[pid] = items[n_enroll:]

    for pid in sorted(unregistered):
        unreg_samples.extend(by_pid[pid])

    return registered, unregistered, enroll, probe, unreg_samples
