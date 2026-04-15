"""Map-style dataset for unknown-split samples (LaBraM tuple format)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from data_processing.target_transforms import UserIdentificationTransform
from user_identification.registering_users.unknown_hdf5_index import UnknownSampleRef


class UnknownRegistrationMapDataset(Dataset):
    """
    Random-access dataset over explicit unknown-split sample references.

    Yields ``(eeg_tensor, label_tensor, participant_id)`` like ``LaBraMEEGDataset``.
    """

    def __init__(
        self,
        samples: Sequence[UnknownSampleRef],
        user_identification_transform: UserIdentificationTransform,
        *,
        transform: Optional[Any] = None,
    ) -> None:
        if not samples:
            raise ValueError("samples must be non-empty")
        self._samples: List[UnknownSampleRef] = list(samples)
        self._ui = user_identification_transform
        self._transform = transform
        self._cache: Dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self._samples)

    def _open(self, path: str) -> Any:
        if path not in self._cache:
            self._cache[path] = h5py.File(path, "r")
        return self._cache[path]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        ref = self._samples[idx]
        f = self._open(ref.hdf5_path)
        group = f[ref.task_type]
        raw = np.array(group[ref.sample_key][:], dtype=np.float32)
        if self._transform is not None:
            raw = self._transform(raw)
        eeg = torch.from_numpy(np.asarray(raw, dtype=np.float32))
        if eeg.dim() < 2:
            raise ValueError(
                f"Expected EEG (n_channels, n_time) for {ref.sample_key!r}, got shape {tuple(eeg.shape)}"
            )
        label = self._ui(ref.participant_id)
        return eeg, label, ref.participant_id

    def close(self) -> None:
        for fp in self._cache.values():
            try:
                fp.close()
            except Exception as e:
                raise RuntimeError(f"Failed closing HDF5 file: {e}") from e
        self._cache.clear()
