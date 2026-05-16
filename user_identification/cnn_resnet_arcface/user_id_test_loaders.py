"""
Build **test** DataLoaders for user identification, split by task type (active / passive).

Mirrors ``CNN.experiment.Experiment._evaluate_by_task_type`` for the user_identification
target so evaluation stays consistent with training data (same HDF5, same transform).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader

from data_processing.eeg_dataset import EEGDataLoader, EEGDataset
from data_processing.target_transforms import (
    UserIdentificationTransform,
    create_user_identification_transform_from_hdf5,
)

try:
    from data_processing.multi_file_dataset import MultiFileEEGDataset
except ImportError:  # pragma: no cover
    MultiFileEEGDataset = None  # type: ignore


def build_user_identification_task_test_loaders(
    hdf5_dir: str | Path,
    segment_length: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[UserIdentificationTransform, DataLoader, DataLoader]:
    """
    Return (transform, active_test_loader, passive_test_loader) for the **test** split only.

    Raises:
        RuntimeError: If 1s multi-file layout is required but MultiFileEEGDataset is unavailable.
    """
    return build_user_identification_task_loaders(
        hdf5_dir=hdf5_dir,
        segment_length=segment_length,
        batch_size=batch_size,
        num_workers=num_workers,
        split="test",
    )


def build_user_identification_task_loaders(
    hdf5_dir: str | Path,
    segment_length: str,
    batch_size: int,
    num_workers: int,
    split: str,
) -> Tuple[UserIdentificationTransform, DataLoader, DataLoader]:
    """
    Return (transform, active_loader, passive_loader) for a user-identification split.

    Args:
        split: One of {"train","val","test"}.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be one of {{'train','val','test'}}, got {split!r}")

    hdf5_dir_path = Path(hdf5_dir)
    user_identification_transform, _ = create_user_identification_transform_from_hdf5(str(hdf5_dir_path))
    train_files, val_files, test_files = EEGDataLoader._get_hdf5_file_paths(hdf5_dir_path, segment_length)
    EEGDataLoader._verify_hdf5_files(train_files, val_files, test_files)

    split_files = {"train": train_files, "val": val_files, "test": test_files}[split]

    if isinstance(split_files, (list, tuple)):
        if MultiFileEEGDataset is None:
            raise RuntimeError(
                f"Multi-file {split} HDF5 requires data_processing.multi_file_dataset.MultiFileEEGDataset; "
                "import failed."
            )
        split_paths = [str(f) for f in split_files]
        active_ds = MultiFileEEGDataset(
            hdf5_files=split_paths,
            task_type="active",
            target_type="user_identification",
            user_identification_transform=user_identification_transform,
            shuffle=False,
        )
        passive_ds = MultiFileEEGDataset(
            hdf5_files=split_paths,
            task_type="passive",
            target_type="user_identification",
            user_identification_transform=user_identification_transform,
            shuffle=False,
        )
    else:
        active_ds = EEGDataset(
            hdf5_file=str(split_files),
            task_type="active",
            target_type="user_identification",
            user_identification_transform=user_identification_transform,
            shuffle=False,
        )
        passive_ds = EEGDataset(
            hdf5_file=str(split_files),
            task_type="passive",
            target_type="user_identification",
            user_identification_transform=user_identification_transform,
            shuffle=False,
        )

    active_loader = EEGDataLoader.create_dataloader(active_ds, batch_size=batch_size, num_workers=num_workers)
    passive_loader = EEGDataLoader.create_dataloader(passive_ds, batch_size=batch_size, num_workers=num_workers)
    return user_identification_transform, active_loader, passive_loader


def verify_hdf5_dir(hdf5_dir: str | Path, segment_length: str) -> None:
    """Raise if train/val/test HDF5 layout is missing or invalid."""
    hdf5_dir_path = Path(hdf5_dir)
    tr, va, te = EEGDataLoader._get_hdf5_file_paths(hdf5_dir_path, segment_length)
    EEGDataLoader._verify_hdf5_files(tr, va, te)
